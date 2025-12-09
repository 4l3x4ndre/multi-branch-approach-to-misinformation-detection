import json
import os
import random
import shutil
from pathlib import Path
from loguru import logger
from tqdm import tqdm

from corpus_truth_manipulation.config import (
    COSMOS_FORMATTED,
    XFACTA_FORMATTED,
    COSMOS_XFACTA_FORMATTED,
)

def main():
    logger.info("Creating COSMOS-XFACTA training dataset...")

    # ====================================================================
    # Step 1: Setup directories for the training set
    # ====================================================================
    train_dir = COSMOS_XFACTA_FORMATTED / "train"
    os.makedirs(train_dir / "source", exist_ok=True)

    # ====================================================================
    # Step 2: Load and merge datasets
    # ====================================================================
    cosmos_json_path = COSMOS_FORMATTED / "test" / "source" / "cosmos.json"
    xfacta_json_path = XFACTA_FORMATTED / "test" / "source" / "xfacta.json"

    with open(cosmos_json_path, 'r', encoding='utf-8') as f:
        cosmos_data = json.load(f)
        for item in cosmos_data:
            item['origin_dataset'] = 'cosmos'

    with open(xfacta_json_path, 'r', encoding='utf-8') as f:
        xfacta_data = json.load(f)
        for item in xfacta_data:
            item['origin_dataset'] = 'xfacta'

    merged_data = cosmos_data + xfacta_data
    random.shuffle(merged_data)

    logger.info(f"Total merged samples for training: {len(merged_data)}")

    # ====================================================================
    # Step 3: Process data and copy images to the training directory
    # ====================================================================
    final_train_json_data = []

    for item in tqdm(merged_data, desc="Copying images for training set"):
        image_path = item['image_path']
        
        if item['origin_dataset'] == 'cosmos':
            # image_path is 'test/0.jpg'. Source is COSMOS_FORMATTED/test/
            src_image_path = COSMOS_FORMATTED / "test" / image_path
        elif item['origin_dataset'] == 'xfacta':
            # image_path can be '/real_sample/...'
            image_path = image_path.lstrip('/')
            src_image_path = XFACTA_FORMATTED / "test" / image_path
        else:
            logger.warning(f"Unknown origin dataset: {item['origin_dataset']}")
            continue

        if not src_image_path.exists():
            logger.warning(f"Image not found, skipping: {src_image_path}")
            continue

        # The destination path should be relative to the train directory
        dest_image_path = train_dir / image_path
        os.makedirs(dest_image_path.parent, exist_ok=True)
        shutil.copy(src_image_path, dest_image_path)
        
        new_item = item.copy()
        # The image_path in the json should be relative to the train folder
        new_item['image_path'] = str(Path(image_path))
        final_train_json_data.append(new_item)

    # ====================================================================
    # Step 4: Write the final training JSON file
    # ====================================================================
    output_json_path = train_dir / "source" / "cosmos_xfacta.json"
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_train_json_data, f, ensure_ascii=False, indent=4)

    logger.success("COSMOS-XFACTA training dataset created successfully.")

if __name__ == '__main__':
    main()