from tqdm import tqdm
import json
import os
from torch.utils.data import DataLoader
from loguru import logger

from corpus_truth_manipulation.config import CONFIG, COSMOS, COSMOS_FORMATTED
from corpus_truth_manipulation.dataset import MMFakeBenchDataset, collate_fn

def main(split:str='test'):
    os.makedirs(COSMOS_FORMATTED, exist_ok=True)

    # ====================================================================
    # Step 1: Read test set
    # ====================================================================
    test_set = COSMOS / f"{split}_data.json"
    merged_data = []
    with open(test_set, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # skip empty lines
                data = json.loads(line)
                merged_data.append(data)

    # ====================================================================
    # Step 2: Reformat json to match MMFakeBench format.
    # ====================================================================
    formatted_data = []
    for record in tqdm(merged_data, desc="Formatting COSMOS json"):

        """
        Example data:
        {"img_local_path": "test/0.jpg", "caption1": "Julian Castro at his announcement in San Antonio, Tex., on Saturday. Mr. Castro, the former secretary of housing and urban development, would be one of the youngest presidents if elected.", "caption2": "Julian Castro at his announcement in San Antonio, Tex., on Saturday, Jan. 12, 2019.", "context_label": 0, "article_url": "https://www.nytimes.com/2019/06/13/us/politics/julian-castro-fox-town-hall.html", "maskrcnn_bboxes": [[389.9706726074219, 72.9228744506836, 505.0566711425781, 373.24993896484375], [89.46248626708984, 312.29644775390625, 190.55088806152344, 396.4997253417969], [116.25189971923828, 225.38841247558594, 161.36624145507812, 288.41522216796875], [180.07785034179688, 225.37646484375, 207.3575439453125, 271.2514953613281], [579.815185546875, 193.33509826660156, 597.6293334960938, 249.89108276367188], [217.98863220214844, 225.41282653808594, 256.5491638183594, 267.5371398925781], [67.05160522460938, 237.61740112304688, 92.31876373291016, 275.8415222167969], [29.469621658325195, 240.86349487304688, 64.6895980834961, 276.5841369628906], [229.984375, 298.4461669921875, 251.81227111816406, 330.3661804199219], [89.82146453857422, 205.71160888671875, 104.25022888183594, 228.6795196533203]], "caption1_modified": "PERSON at his announcement in GPE, GPE, on DATE. Mr. PERSON, the former secretary of housing and urban development, would be one of the youngest presidents if elected.", "caption1_entities": [["Julian Castro", "PERSON"], ["San Antonio", "GPE"], ["Tex.", "GPE"], ["Saturday", "DATE"], ["Castro", "PERSON"]], "caption2_modified": "PERSON at his announcement in GPE, GPE, on DATE.", "caption2_entities": [["Julian Castro", "PERSON"], ["San Antonio", "GPE"], ["Tex.", "GPE"], ["Saturday, Jan. 12, 2019", "DATE"]], "bert_base_score": "0.5769946", "bert_large_score": "0.60118324"}
        """

        print(record['article_url'])
        source = ''
        if "https://" not in record['article_url']:
            source = record['article_url'].split('/')[0]
        else:
            source = record['article_url'].split('//')[1].split('/')[0]
        formatted_record = {
            "text": record['caption1'],
            "gt_answers": "True" if record['context_label'] == 0 else "mismatch",
            "image_path": record['img_local_path'],
            "fake_cls": "original" if record['context_label'] == 0 else "mismatch",
            "image_source": source,
            "text_source": source,
        }

        formatted_data.append(formatted_record)

    os.makedirs(COSMOS_FORMATTED / split, exist_ok=True)
    os.makedirs(COSMOS_FORMATTED / split / "source", exist_ok=True)
    formatted_output_path = COSMOS_FORMATTED / split / "source" / "cosmos.json"
    with open(formatted_output_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, ensure_ascii=False, indent=4)


    # ====================================================================
    # Step 3: Extract graphs
    # ===================================================================
    CONFIG.device = 'cpu'
    dataset = MMFakeBenchDataset(
        data_root=COSMOS_FORMATTED,
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

    logger.info(f"Length of COSMOS formatted dataset: {len(dataset)}")




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="test", help="Dataset split to process (e.g., test, train, val)")
    args = parser.parse_args()
    main(split=args.split)
