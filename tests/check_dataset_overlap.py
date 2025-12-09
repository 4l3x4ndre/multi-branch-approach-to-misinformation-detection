"""
This script checks for overlapping samples between the training and testing sets.
The tested set is hardcoded, here "MMFakeBenchOriginal_graphs/test_original", 
which contains the original MMFakeBench-Test graphs (original meaning the 
original Test-Val splits).
"""

import os
import torch
from pathlib import Path
from tqdm import tqdm
from corpus_truth_manipulation.config import PROCESSED_DATA_DIR

def get_sample_identifiers(dataset_path: Path) -> set:
    """
    Loads graph files from the given dataset path and extracts a unique identifier
    for each sample.
    """
    identifiers = set()
    if not dataset_path.exists():
        print(f"Warning: Dataset path does not exist: {dataset_path}")
        return identifiers

    print(f"Loading sample identifiers from: {dataset_path}")
    files = list(dataset_path.glob("*.pt"))
    for file_path in tqdm(files, desc=f"Processing {dataset_path.name}"):
        try:
            # Load only the necessary metadata to reduce memory footprint
            data = torch.load(file_path, map_location='cpu', weights_only=False)
            if "metadata" in data and "image_path" in data["metadata"]:
                # Use image_path as a unique identifier
                identifiers.add(data["metadata"]["image_path"])
            else:
                print(f"Warning: 'metadata' or 'image_path' not found in {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    return identifiers

def main():
    # Define the base directory for the original MMFakeBench graphs
    mmfakebench_original_base = PROCESSED_DATA_DIR / "MMFakeBenchOriginal_graphs" / "test_original"

    # Define the specific train and test split directories
    train_dir = mmfakebench_original_base / "train"
    test_dir = mmfakebench_original_base / "test"

    print("Starting dataset overlap check...")

    # Get identifiers for the training set
    train_identifiers = get_sample_identifiers(train_dir)
    print(f"Found {len(train_identifiers)} unique samples in the training set.")

    # Get identifiers for the testing set
    test_identifiers = get_sample_identifiers(test_dir)
    print(f"Found {len(test_identifiers)} unique samples in the testing set.")

    # Find the intersection (overlap)
    overlap_identifiers = train_identifiers.intersection(test_identifiers)

    print(f"\n--- Overlap Analysis ---")
    if overlap_identifiers:
        print(f"🚨 ALERT: {len(overlap_identifiers)} samples found in both train and test sets!")
        print("Examples of overlapping samples (image_path):")
        for i, identifier in enumerate(list(overlap_identifiers)[:5]): # Show up to 5 examples
            print(f"- {identifier}")
        if len(overlap_identifiers) > 5:
            print(f"(and {len(overlap_identifiers) - 5} more...)")
    else:
        print("✅ No overlap found between the training and testing sets. Good job!")

if __name__ == "__main__":
    main()
