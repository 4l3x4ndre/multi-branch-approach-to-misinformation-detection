import os
import json
import random
import shutil
from tqdm import tqdm
from loguru import logger
from pathlib import Path

def split_cosmos_dataset():
    """
    Splits the processed COSMOS dataset from a single 'test' directory into
    'train' and 'test_split' directories, partitioning the graph files, images,
    and metadata with a simple random split.
    """
    project_root = Path(__file__).parent.parent.parent
    logger.info(f"Project root detected at: {project_root}")

    # Configuration
    graphs_dir = project_root / "data" / "processed" / "cosmos_graphs"
    formatted_dir = project_root / "data" / "processed" / "cosmos_formatted"
    original_split = "test_original"
    train_split_name = "train"
    test_split_name = "test"
    train_ratio = 0.85

    # Paths
    original_graphs_path = graphs_dir / original_split
    original_json_path = graphs_dir / f"{original_split}_dbpedia.json"

    # Create new directories
    train_graphs_path = graphs_dir / train_split_name
    test_graphs_path = graphs_dir / test_split_name
    train_images_path = formatted_dir / train_split_name / "images"
    test_images_path = formatted_dir / test_split_name / "images"

    logger.info("Creating destination directories...")
    train_graphs_path.mkdir(parents=True, exist_ok=True)
    test_graphs_path.mkdir(parents=True, exist_ok=True)
    train_images_path.mkdir(parents=True, exist_ok=True)
    test_images_path.mkdir(parents=True, exist_ok=True)

    # Load original metadata
    logger.info(f"Loading original metadata from {original_json_path}")
    with open(original_json_path, 'r', encoding='utf-8') as f:
        metadata_list = json.load(f)

    num_samples = len(metadata_list)
    logger.info(f"Found {num_samples} total samples.")

    # Log original 'overall_truth' distribution
    original_truth_counts = {"True": 0, "False": 0}
    for meta in metadata_list:
        if meta.get("overall_truth", False):
            original_truth_counts["True"] += 1
        else:
            original_truth_counts["False"] += 1
    logger.info(f"Original 'overall_truth' distribution: {original_truth_counts}")

    # Shuffle and split indices
    indices = list(range(num_samples))
    random.shuffle(indices)
    split_point = int(num_samples * train_ratio)
    train_indices = set(indices[:split_point])
    test_indices = set(indices[split_point:])

    logger.info(f"Splitting into {len(train_indices)} training samples and {len(test_indices)} test samples.")

    # Process and copy2 files
    train_metadata = []
    test_metadata = []

    for i in tqdm(range(num_samples), desc="Processing and copying files"):
        original_metadata = metadata_list[i]

        # Graph file handling
        original_graph_filename = f"graph_{i}.pt"
        original_graph_path = original_graphs_path / original_graph_filename

        if not original_graph_path.exists():
            logger.warning(f"Graph file not found, skipping: {original_graph_path}")
            continue

        # Image file handling
        original_image_path_str = original_metadata.get("full_image_path", "")
        if not original_image_path_str:
             img_rel_path = original_metadata.get("image_path", "")
             original_image_path_str = str(formatted_dir / original_split / img_rel_path)
        
        original_image_path = Path(original_image_path_str)

        if not original_image_path.exists():
            logger.warning(f"Image file not found, skipping: {original_image_path}")
            continue

        image_filename = original_image_path.name
        
        # Determine new split and paths
        if i in train_indices:
            new_graphs_path = train_graphs_path
            new_images_path = train_images_path
            new_metadata_list = train_metadata
        elif i in test_indices:
            new_graphs_path = test_graphs_path
            new_images_path = test_images_path
            new_metadata_list = test_metadata
        else:
            continue # Should not happen in a simple split

        # Copy graph file
        new_graph_path = new_graphs_path / original_graph_filename
        shutil.copy2(str(original_graph_path), str(new_graph_path))

        # Copy image file
        new_image_path = new_images_path / image_filename
        shutil.copy2(str(original_image_path), str(new_image_path))

        # Update metadata
        updated_metadata = original_metadata.copy()
        updated_metadata["full_image_path"] = str(new_image_path.resolve())
        updated_metadata["image_path"] = f"images/{image_filename}"
        new_metadata_list.append(updated_metadata)

    # Log new 'overall_truth' distributions
    train_truth_counts = {"True": 0, "False": 0}
    for meta in train_metadata:
        if meta.get("overall_truth", False):
            train_truth_counts["True"] += 1
        else:
            train_truth_counts["False"] += 1
    logger.info(f"Train set 'overall_truth' distribution: {train_truth_counts}")

    test_truth_counts = {"True": 0, "False": 0}
    for meta in test_metadata:
        if meta.get("overall_truth", False):
            test_truth_counts["True"] += 1
        else:
            test_truth_counts["False"] += 1
    logger.info(f"Test set 'overall_truth' distribution: {test_truth_counts}")

    # Save new JSON files
    train_json_path = graphs_dir / f"{train_split_name}_dbpedia_split.json"
    test_json_path = graphs_dir / f"{test_split_name}_dbpedia_split.json"

    logger.info(f"Saving new training metadata to {train_json_path}")
    with open(train_json_path, 'w', encoding='utf-8') as f:
        json.dump(train_metadata, f, indent=4, ensure_ascii=False)

    logger.info(f"Saving new test metadata to {test_json_path}")
    with open(test_json_path, 'w', encoding='utf-8') as f:
        json.dump(test_metadata, f, indent=4, ensure_ascii=False)

    logger.success("Dataset splitting complete.")
    logger.info(f"Train graphs: {train_graphs_path}")
    logger.info(f"Test graphs: {test_graphs_path}")
    logger.info(f"Train images: {train_images_path}")
    logger.info(f"Test images: {test_images_path}")

if __name__ == "__main__":
    split_cosmos_dataset()
