import os
import shutil
from pathlib import Path
from loguru import logger
import glob
import json
import torch
from tqdm import tqdm

from corpus_truth_manipulation.config import (
    COSMOS_GRAPHS,
    XFACTA_GRAPHS,
    COSMOS_XFACTA_GRAPHS,
)

def main():
    """
    Fuses the graph directories of COSMOS and XFACTA into a single training set.
    """
    logger.info("Fusing COSMOS and XFACTA graph datasets...")

    # ====================================================================
    # Step 1: Define source and destination directories
    # ====================================================================
    cosmos_graphs_dir = COSMOS_GRAPHS / 'test'
    xfacta_graphs_dir = XFACTA_GRAPHS / 'test'
    dest_dir = COSMOS_XFACTA_GRAPHS / 'train'

    if not cosmos_graphs_dir.exists():
        logger.error(f"COSMOS graph directory not found at: {cosmos_graphs_dir}")
        return
    if not xfacta_graphs_dir.exists():
        logger.error(f"XFACTA graph directory not found at: {xfacta_graphs_dir}")
        return

    os.makedirs(dest_dir, exist_ok=True)
    logger.info(f"Destination directory created at: {dest_dir}")

    # ====================================================================
    # Step 2: Copy and rename graph files
    # ====================================================================
    
    # --- Process COSMOS graphs ---
    cosmos_files = glob.glob(str(cosmos_graphs_dir / '*.pt'))
    logger.info(f"Found {len(cosmos_files)} graph files in COSMOS directory.")
    for i, file_path in enumerate(tqdm(cosmos_files, desc="Copying COSMOS graphs")):
        dest_file_path = dest_dir / f"cosmos_graph_{i}.pt"
        shutil.copy(file_path, dest_file_path)

    # --- Process XFACTA graphs ---
    xfacta_files = glob.glob(str(xfacta_graphs_dir / '*.pt'))
    logger.info(f"Found {len(xfacta_files)} graph files in XFACTA directory.")
    for i, file_path in enumerate(tqdm(xfacta_files, desc="Copying XFACTA graphs")):
        dest_file_path = dest_dir / f"xfacta_graph_{i}.pt"
        shutil.copy(file_path, dest_file_path)

    total_files = len(glob.glob(str(dest_dir / '*.pt')))
    logger.success(f"Graph fusion complete. Total files in destination: {total_files}")

    # ====================================================================
    # Step 3: Create JSON metadata file
    # ====================================================================
    logger.info("Creating JSON metadata file...")
    json_metadata = []
    fused_graph_files = glob.glob(str(dest_dir / '*.pt'))

    for file_path in tqdm(fused_graph_files, desc="Extracting metadata"):
        data = torch.load(file_path, map_location='cpu')
        metadata = data.get('metadata', {})
        
        json_entry = {
            "image_real": metadata.get("image_real"),
            "claim_real": metadata.get("claim_real"),
            "mismatch": metadata.get("mismatch"),
            "overall_truth": metadata.get("overall_truth"),
            "gt_answers": metadata.get("gt_answsers"),
            "fake_cls": metadata.get("fake_cls"),
        }
        json_metadata.append(json_entry)

    json_output_path = COSMOS_XFACTA_GRAPHS / 'train_dbpedia_split.json'
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(json_metadata, f, indent=2, ensure_ascii=False)
    
    logger.success(f"JSON metadata file created at: {json_output_path}")


if __name__ == '__main__':
    main()
