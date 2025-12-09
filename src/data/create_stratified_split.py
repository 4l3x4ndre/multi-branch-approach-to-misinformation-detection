"""
This script splits the train dataset into train and validation sets
with stratification on the target variables.
The test set is expected to exist already.
"""

import os
from shutil import copy2
from tqdm import tqdm
from loguru import logger
from typer import Typer
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import load as torch_load
from pathlib import Path

from corpus_truth_manipulation.config import MMFAKEBENCH_GRAPHS, PROJ_ROOT

app = Typer()


@app.command()
def main(
    path_to_train_json: str = '',
    val_split: float = 0.2,
    random_seed: int = 42,
):
    if path_to_train_json == '':
        path_to_train_json = os.path.join(MMFAKEBENCH_GRAPHS, 'train_dbpedia.json')

    output_dir = PROJ_ROOT / "data" / "processed" / "MMFakeBench_graphs_stratified"
    output_dir.mkdir(exist_ok=True)
    
    train_dir = output_dir / "train"
    train_dir.mkdir(exist_ok=True)
    val_dir = output_dir / "val"
    val_dir.mkdir(exist_ok=True)

    logger.info(f"Output directory: {output_dir}")

    df = pd.read_json(path_to_train_json)
    df['strata'] = df['image_real'].astype(str) + "_" + df['claim_real'].astype(str) + "_" + df['mismatch'].astype(str) + "_" + df['overall_truth'].astype(str)

    logger.info("Balancing dataset to make 'overall_truth' 50/50...")

    df_true = df[df['overall_truth'] == True]
    df_false = df[df['overall_truth'] == False]

    min_size = min(len(df_true), len(df_false))
    if min_size == 0:
        raise ValueError("One of the 'overall_truth' classes is empty. Cannot balance.")

    logger.info(f"Size of smaller 'overall_truth' class: {min_size}")

    df_true_sampled = df_true.sample(n=min_size, random_state=random_seed)
    df_false_sampled = df_false.sample(n=min_size, random_state=random_seed)

    balanced_df = pd.concat([df_true_sampled, df_false_sampled])
    balanced_df = balanced_df.sample(frac=1, random_state=random_seed)

    logger.info(f"New balanced dataset size: {len(balanced_df)}")

    logger.info("Distribution of target variables in the new balanced dataset (proportions):")
    logger.info("\nimage_real:\n" + str(balanced_df['image_real'].value_counts(normalize=True)))
    logger.info("\nclaim_real:\n" + str(balanced_df['claim_real'].value_counts(normalize=True)))
    logger.info("\nmismatch:\n" + str(balanced_df['mismatch'].value_counts(normalize=True)))
    logger.info("\noverall_truth:\n" + str(balanced_df['overall_truth'].value_counts(normalize=True)))

    # Split into train and validation sets
    train_df, val_df = train_test_split(balanced_df, test_size=val_split, random_state=random_seed, stratify=balanced_df['strata'])

    train_new = train_df.drop(columns=['strata'])
    val_new = val_df.drop(columns=['strata'])

    train_output_path = output_dir / 'train_dbpedia_split.json'
    val_output_path = output_dir / 'val_dbpedia_split.json'

    train_new.to_json(train_output_path, orient='records', indent=2)
    val_new.to_json(val_output_path, orient='records', indent=2)

    logger.info(f"Train split saved to {train_output_path} with {len(train_new)} records.")
    logger.info(f"Validation split saved to {val_output_path} with {len(val_new)} records.")

    source_dirs = [
        MMFAKEBENCH_GRAPHS / "train_original",
        MMFAKEBENCH_GRAPHS / "val_original",
    ]

    # Copy pt files
    for source_dir in source_dirs:
        if not source_dir.exists():
            logger.warning(f"Source directory {source_dir} does not exist. Skipping.")
            continue
        for graph_file in tqdm(os.listdir(source_dir), desc=f"Copying graph files from {source_dir.name}"):
            if graph_file.endswith('.pt'):
                src_path = source_dir / graph_file
                obj = torch_load(src_path, weights_only=False)
                json_idx = obj['json_idx']

                if json_idx in train_new.index:
                    dst_path = train_dir / graph_file
                    copy2(src_path, dst_path)
                elif json_idx in val_new.index:
                    dst_path = val_dir / graph_file
                    copy2(src_path, dst_path)

    logger.info(f"Copied {len(os.listdir(train_dir))} files to the training folder.")
    logger.info(f"Copied {len(os.listdir(val_dir))} files to the validation folder.")


if __name__ == "__main__":
    app()
