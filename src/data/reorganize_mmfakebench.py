
import os
import json
import random
import shutil
import torch
from tqdm import tqdm
from loguru import logger
from pathlib import Path

def process_and_split_dataset(source_pt_dirs, dest_dir_path, train_ratio=0.8):
    """
    Loads all .pt files from source directories, combines them, splits them
    into train/test sets, and copies them to a new destination with new
    metadata JSON files.
    """
    logger.info(f"Processing dataset for destination: {dest_dir_path}")
    dest_dir_path.mkdir(parents=True, exist_ok=True)

    # 1. Load metadata from all .pt files
    all_metadata = []
    logger.info(f"Loading graph files from: {source_pt_dirs}")
    for source_dir in source_pt_dirs:
        for pt_file in tqdm(list(source_dir.glob("*.pt")), desc=f"Reading files in {source_dir.name}"):
            try:
                data = torch.load(pt_file, map_location='cpu', weights_only=False)
                metadata = data.get('metadata')
                if metadata is not None:
                    # Store original path to find the file later for copying
                    metadata['original_pt_path'] = str(pt_file)
                    all_metadata.append(metadata)
                else:
                    logger.warning(f"File {pt_file} is missing 'metadata' key.")
            except Exception as e:
                logger.error(f"Could not process file {pt_file}: {e}")
    
    logger.info(f"Found {len(all_metadata)} total samples.")

    # 2. Perform random split
    random.shuffle(all_metadata)
    split_point = int(len(all_metadata) * train_ratio)
    train_meta = all_metadata[:split_point]
    test_meta = all_metadata[split_point:]

    logger.info(f"Splitting into {len(train_meta)} train and {len(test_meta)} test samples.")

    # 3. Copy files and create new JSONs
    for split_name, split_metadata in [("train", train_meta), ("test", test_meta)]:
        split_dest_dir = dest_dir_path / split_name
        split_dest_dir.mkdir(exist_ok=True)
        
        new_metadata_list = []
        for i, meta in enumerate(tqdm(split_metadata, desc=f"Copying {split_name} files")):
            original_path = Path(meta.pop('original_pt_path'))
            new_filename = f"graph_{i}.pt"
            new_path = split_dest_dir / new_filename
            
            # update the metadata inside the .pt file before saving it
            pt_data = torch.load(original_path, map_location='cpu', weights_only=False)
            pt_data['metadata'] = meta 
            torch.save(pt_data, new_path)

            new_metadata_list.append(meta)

        # Save the new JSON file for the split
        json_path = dest_dir_path / f"{split_name}_dbpedia_split.json"
        logger.info(f"Saving new metadata to {json_path}")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(new_metadata_list, f, indent=4, ensure_ascii=False)


def main():
    project_root = Path(__file__).parent.parent.parent
    source_dir = project_root / "data" / "processed" / "MMFakeBench_graphs"
    dest_dir = project_root / "data" / "processed" / "MMFakeBenchOriginal_graphs"

    if dest_dir.exists():
        logger.warning(f"Destination directory {dest_dir} already exists. Files may be overwritten.")
    
    # --- Process val_original (current 'test') ---
    val_original_sources = [source_dir / "test"]
    val_original_dest = dest_dir / "val_original"
    process_and_split_dataset(val_original_sources, val_original_dest)

    # --- Process test_original (current 'train' + 'val') ---
    test_original_sources = [source_dir / "train", source_dir / "val"]
    test_original_dest = dest_dir / "test_original"
    process_and_split_dataset(test_original_sources, test_original_dest)

    logger.success("Reorganization complete.")


if __name__ == "__main__":
    main()
