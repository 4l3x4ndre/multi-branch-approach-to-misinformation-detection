from typer import Typer, Option
from loguru import logger
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import nan as torch_nan, save as torch_save
import json
import re
from sentence_transformers import SentenceTransformer

from corpus_truth_manipulation.dataset import MMFakeBenchDataset, collate_fn
from corpus_truth_manipulation.config import CONFIG, MMFAKEBENCH_GRAPHS, XFACTA_GRAPHS, XFACTA_FORMATTED, COSMOS_GRAPHS, COSMOS_FORMATTED

from src.graph_to_pyg_object import convert_nx_to_pyg_data
from src.dbpedia_build_graph import build_entity_graph_from_names_batch
from src.text_to_graph import extract_claim_graphs
from src.utils.embeddings import NLP


app = Typer()

@app.command()
def main(
    split:str, 
    dataset_folder:str,
    start_from_step:int=Option(0, help="Step number to resume from, if interrupted.")
):
    assert dataset_folder in ['MMFAKEBENCH', 'XFACTA', 'COSMOS'], """
    dataset_folder should be in 'MMFAKEBENCH' or 'XFACTA'
    """
    if dataset_folder == 'MMFAKEBENCH':
        data_root = CONFIG.MMFAKEBENCH
        dataset_folder = MMFAKEBENCH_GRAPHS
    elif dataset_folder == 'XFACTA':
        data_root = XFACTA_FORMATTED
        dataset_folder = XFACTA_GRAPHS
    elif dataset_folder == 'COSMOS':
        data_root = COSMOS_FORMATTED
        dataset_folder = COSMOS_GRAPHS
    else:
        raise ValueError(f'Unknown dataset folder <{dataset_folder}>.')

    CONFIG.device = "cpu"

    dataset = MMFakeBenchDataset(
        data_root=data_root,
        split=split,
        transform=None,
        include_img=False,
        shuffle_data=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=CONFIG.data.batch_size,
        shuffle=False,
        num_workers=CONFIG.data.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    os.makedirs(dataset_folder, exist_ok=True)
    os.makedirs(os.path.join(dataset_folder, split), exist_ok=True)

    step = 0
    saved_dataset_json = []
    txt_encoder_model = SentenceTransformer("all-MiniLM-L6-v2", device=CONFIG.device, token=True, local_files_only=True)
    txt_encoder_model.eval()
    
    can_proceed = True if start_from_step == 0 else False
    logger.info("Starting graph extraction and saving...")
    for batch in tqdm(loader):

        if not can_proceed and start_from_step > 0:
            if step < start_from_step:
                step += len(batch['text_str'])
                continue
            else:
                can_proceed = True
                logger.info(f"Resuming from step {step} as requested.")


        texts = batch['text_str']
        graphs, extracted_texts_nans = extract_claim_graphs(texts)

        indices = [i for i, t in enumerate(extracted_texts_nans) if isinstance(t, str) and t.strip()]
        texts_to_process = [extracted_texts_nans[i] for i in indices]

        kg_entities = []
        for doc in NLP.pipe(texts_to_process, batch_size=32): 
            entities = [ent.text for ent in doc.ents]
            nouns = [token.text for token in doc if token.pos_ in ("NOUN", "PROPN")]
            noun_phrases = [chunk.text for chunk in doc.noun_chunks]
            all_candidates = list(dict.fromkeys(entities + nouns + noun_phrases))
            if len(all_candidates) == 0:
                logger.error(f"No entities found in text: {doc.text}")
            kg_entities.append(all_candidates)

        pattern = r'[^\w\s.,?!:;-]'  # keeps letters, numbers, whitespace, and basic punctuation
        for i in range(len(kg_entities)):
            kg_entities[i] = [re.sub(pattern, '', ent).strip() for ent in kg_entities[i] if re.sub(pattern, '', ent).strip() != '']

        kg_graphs = build_entity_graph_from_names_batch(kg_entities, NLP, neighbor_limit=CONFIG.model.kg_neighbor_limit)

        aligned_kg_graphs = [torch_nan] * len(extracted_texts_nans)
        for idx, kgg in zip(indices, kg_graphs):
            aligned_kg_graphs[idx] = kgg

        for idx, (graph, kg) in enumerate(zip(graphs, aligned_kg_graphs)):

            if graph is torch_nan or kg is torch_nan or len(kg.nodes) == 0 or len(graph.nodes) == 0:
                pass
            else:
                true_value = batch['metadata'][idx]['claim_real']
                graph_pyg = convert_nx_to_pyg_data(graph, txt_encoder_model)
                kg_pyg = convert_nx_to_pyg_data(kg, txt_encoder_model)
                torch_save(
                    { 'claim': graph_pyg, 'kg':kg_pyg, 'text':texts[idx], 'claim_real': true_value,
                     'metadata': batch['metadata'][idx],
                     'json_idx':len(saved_dataset_json),
                     'filename_key': f"graph_{step}.pt",},
                    dataset_folder / split / f"graph_{step}.pt"
                )
                saved_dataset_json.append(batch['metadata'][idx])

            step += 1

    if split == "train" or split == "val":
        split_jsonname = "train_val"
    else:
        split_jsonname = split
    with open(dataset_folder / f"{split_jsonname}_dbpedia.json", 'w', encoding='utf-8') as f:
        json.dump(saved_dataset_json, f, indent=2, ensure_ascii=False)
                

if __name__ == "__main__":
    app()
