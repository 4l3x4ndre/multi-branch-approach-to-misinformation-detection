import streamlit as st
import subprocess
import sys
import spacy.util
import os
import glob
from PIL import Image

# Check and download spaCy model if not already present
SPACY_MODEL_NAME = "en_core_web_md"
if not spacy.util.is_package(SPACY_MODEL_NAME):
    with st.spinner(f"Downloading spaCy model '{SPACY_MODEL_NAME}'... This may take a moment."):
        try:
            subprocess.check_call([sys.executable, "-m", "spacy", "download", SPACY_MODEL_NAME])
            st.success(f"spaCy model '{SPACY_MODEL_NAME}' downloaded successfully!")
        except subprocess.CalledProcessError as e:
            st.error(f"Failed to download spaCy model '{SPACY_MODEL_NAME}': {e}")
            st.stop()
        except Exception as e:
            st.error(f"An unexpected error occurred during spaCy model download: {e}")
            st.stop()

# Check and download huggingface_hub if not already present
try:
    import huggingface_hub
except ImportError:
    with st.spinner("Installing huggingface_hub..."):
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
            st.success("huggingface_hub installed successfully!")
            import huggingface_hub
        except Exception as e:
            st.error(f"Failed to install huggingface_hub: {e}")
            st.stop()

from huggingface_hub import hf_hub_download

import torch
import spacy
import re
import networkx as nx
from torch_geometric.data import Batch
from sentence_transformers import SentenceTransformer

from corpus_truth_manipulation.config import CONFIG
from src.utils.embeddings import NLP
from src.text_to_graph import extract_claim_graphs
from src.dbpedia_build_graph import build_entity_graph_from_names_batch
from src.graph_to_pyg_object import convert_nx_to_pyg_data
from src.models import MultimodalMisinformationDetector

from loguru import logger
from torch import nan as torch_nan

@torch.no_grad()
def get_graphs_from_text(text: str):
    """
    Process text to extract claim and evidence graphs.
    Returns PyG Batch objects for claim and evidence.
    """
    CONFIG.device = "cpu" # Force CPU for demo

    # Initialize Sentence Transformer for node embeddings
    txt_encoder_model = SentenceTransformer("all-MiniLM-L6-v2", device=CONFIG.device, token=True, local_files_only=True)
    txt_encoder_model.eval()

    # 1. Extract Claim Graph
    graphs, extracted_texts = extract_claim_graphs([text])
    
    if not graphs or graphs[0] is torch_nan or len(graphs[0].nodes) == 0:
        logger.warning("No claim graph extracted from text.")
        return None, None

    claim_graph_nx = graphs[0]
    
    # 2. Extract Entities for KG
    kg_entities = []
    doc = NLP(text)
    entities = [ent.text for ent in doc.ents]
    nouns = [token.text for token in doc if token.pos_ in ("NOUN", "PROPN")]
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]
    all_candidates = list(dict.fromkeys(entities + nouns + noun_phrases))
    
    # Clean entities
    pattern = r'[^\w\s.,?!:;-]'
    cleaned_candidates = [re.sub(pattern, '', ent).strip() for ent in all_candidates if re.sub(pattern, '', ent).strip() != '']
    kg_entities.append(cleaned_candidates)

    # 3. Build Knowledge Graph (Evidence Graph)
    kg_graphs = build_entity_graph_from_names_batch(kg_entities, NLP, neighbor_limit=CONFIG.model.kg_neighbor_limit)
    
    if not kg_graphs or len(kg_graphs[0].nodes) == 0:
         logger.warning("No knowledge graph extracted from text.")
         evidence_graph_nx = nx.DiGraph()
         evidence_graph_nx.add_node("dummy") # Add dummy node
    else:
        evidence_graph_nx = kg_graphs[0]

    # 4. Convert to PyG Data
    claim_pyg = convert_nx_to_pyg_data(claim_graph_nx, txt_encoder_model, device=torch.device(CONFIG.device))
    evidence_pyg = convert_nx_to_pyg_data(evidence_graph_nx, txt_encoder_model, device=torch.device(CONFIG.device))

    # 5. Create Batch (model expects Batch objects)
    claim_batch = Batch.from_data_list([claim_pyg])
    evidence_batch = Batch.from_data_list([evidence_pyg])

    return claim_batch, evidence_batch

def load_model(model_filename: str):
    """
    Load the MultimodalMisinformationDetector model.
    Downloads from Hugging Face if not found locally.
    """
    CONFIG.device = "cpu"
    
    # --- HF Download Logic ---
    HF_REPO = "4l3x4ndre/multimodal-misinformation-detector"
    local_models_dir = "models"
    os.makedirs(local_models_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(local_models_dir, model_filename)
    
    if not os.path.exists(checkpoint_path):
        with st.spinner(f"Downloading model '{model_filename}' from Hugging Face..."):
            try:
                hf_hub_download(repo_id=HF_REPO, filename=model_filename, local_dir=local_models_dir)
                st.success(f"Model downloaded to {checkpoint_path}")
            except Exception as e:
                st.error(f"Failed to download model from Hugging Face: {e}")
                return None

    # --- Dynamic Parameter Calculation ---
    cfg = CONFIG.training
    layer_choice = CONFIG.training.global_layer_sizes
    classifier_input_dim = 0
    if cfg.use_deepfake_detector:
        classifier_input_dim += CONFIG.model.fakedetector_feature_dim
    if cfg.use_clip:
        if cfg.clip_use_image_features:
            classifier_input_dim += CONFIG.model.clip_image_dim
        if cfg.clip_use_text_features:
            classifier_input_dim += CONFIG.model.clip_text_dim
    if cfg.use_encyclopedic_knowledge:
        classifier_input_dim += CONFIG.model.claimverifier_hidden_dims*2 + CONFIG.model.claimverifier_hidden_dims // 2
    global_layer_sizes = [int(x) if x != "input" else classifier_input_dim for x in layer_choice.split("-")]
    
    model = MultimodalMisinformationDetector(
        use_deepfake_detector=CONFIG.training.use_deepfake_detector,
        use_clip=CONFIG.training.use_clip,
        use_encyclopedic_knowledge=CONFIG.training.use_encyclopedic_knowledge,
        classifier_input_dim=classifier_input_dim, # Use calculated dim
        clip_use_image_features=CONFIG.training.clip_use_image_features,
        clip_use_text_features=CONFIG.training.clip_use_text_features,
        ency_in_channels=773,

        ency_int_dims=CONFIG.model.claimverifier_int_dims,
        ency_hidden_dim=CONFIG.model.claimverifier_hidden_dims,
        layer_sizes=global_layer_sizes,
        dropout=CONFIG.training.dropout,
        activation_sigmoid=True,
        finetune_deepfake_detector=False,
    )
    
    # --- Weight Loading with Optimization ---
    try:
        checkpoint = torch.load(checkpoint_path, map_location=CONFIG.device)
        
        state_dict = None
        # Handle different saving mechanisms
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            logger.info(f"Optimizing model checkpoint {checkpoint_path}...")
            state_dict = checkpoint['model_state_dict']
            
            # Remove 'module.' prefix
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace("module.", "")
                new_state_dict[name] = v
            state_dict = new_state_dict

            # Overwrite optimized
            torch.save(state_dict, checkpoint_path)
            logger.success(f"Overwrote {checkpoint_path} with optimized state dict.")
            del checkpoint

        elif isinstance(checkpoint, dict):
             state_dict = checkpoint
        else:
             state_dict = checkpoint.state_dict()
             
        # Cleanup prefixes if still present
        if state_dict is not None and len(state_dict) > 0 and 'module.' in list(state_dict.keys())[0]:
             new_state_dict = {}
             for k, v in state_dict.items():
                name = k.replace("module.", "")
                new_state_dict[name] = v
             state_dict = new_state_dict

        model.load_state_dict(state_dict, strict=False) 
        model.eval()
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None
