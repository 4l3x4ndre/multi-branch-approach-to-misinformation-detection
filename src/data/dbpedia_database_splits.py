"""
This script splits the train dataset into train and validation sets.
"""

import os
from shutil import move, copy2
from tqdm import tqdm
from loguru import logger
from typer import Typer 
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import load as torch_load

from corpus_truth_manipulation.config import MMFAKEBENCH_GRAPHS_TRAIN, MMFAKEBENCH_GRAPHS_VAL, MMFAKEBENCH_GRAPHS

app = Typer()


@app.command()
def main(
    path_to_train_json:str='',
    train_split: float = 0.9,
    random_seed: int = 42,
):
    if path_to_train_json == '':
        path_to_train_json = os.path.join(MMFAKEBENCH_GRAPHS, 'train_dbpedia.json')

    logger.info(f"At start, there were {len(os.listdir(MMFAKEBENCH_GRAPHS_TRAIN))} files in the training folder.")
    logger.info(f"At start, there were {len(os.listdir(MMFAKEBENCH_GRAPHS_VAL))} files in the validation folder.")

    df = pd.read_json(path_to_train_json)
    df['strata'] = df['text_source'].astype(str) + "_" + df['image_source'].astype(str) + "_" + df['fake_cls'].astype(str)

    train_df, val_df = train_test_split(df, train_size=train_split, random_state=random_seed, stratify=df['strata'])

    train_new = train_df.drop(columns=['strata'])
    val_new = val_df.drop(columns=['strata'])

    train_output_path = path_to_train_json.replace('.json', '_split_train.json')
    val_output_path = path_to_train_json.replace('.json', '_split_val.json')
    train_new.to_json(train_output_path, orient='records', lines=False)
    val_new.to_json(val_output_path, orient='records', lines=False)

    logger.info(f"Train split saved to {train_output_path} with {len(train_df)} records.")
    logger.info(f"Validation split saved to {val_output_path} with {len(val_df)} records.")

    # Load every pt file and move it to val folder
    for graph_file in tqdm(os.listdir(MMFAKEBENCH_GRAPHS_TRAIN), desc="Moving validation graphs"):
        if graph_file.endswith('.pt'):
            obj = torch_load(os.path.join(MMFAKEBENCH_GRAPHS_TRAIN, graph_file), weights_only=False)
            if obj['json_idx'] in val_new.index:
                src_path = os.path.join(MMFAKEBENCH_GRAPHS_TRAIN, graph_file)
                dst_path = os.path.join(MMFAKEBENCH_GRAPHS_VAL, graph_file)
                move(src_path, dst_path)
                # logger.info(f"Moved {graph_file} to validation folder.")

    logger.info(f"There are now {len(os.listdir(MMFAKEBENCH_GRAPHS_TRAIN))} files in the training folder.")
    logger.info(f"There are now {len(os.listdir(MMFAKEBENCH_GRAPHS_VAL))} files in the validation folder.")


    # Change filename of test split:
    test_json_path = os.path.join(MMFAKEBENCH_GRAPHS, 'test_dbpedia.json')
    test_new_path = os.path.join(MMFAKEBENCH_GRAPHS, 'test_dbpedia_split.json')
    if os.path.exists(test_json_path):
        copy2(test_json_path, test_new_path)
        logger.info(f"Copied test split to {test_new_path}.")
    else:
        logger.warning(f"Test split file {test_json_path} does not exist. Skipping copy.")





if __name__ == "__main__":
    app()
