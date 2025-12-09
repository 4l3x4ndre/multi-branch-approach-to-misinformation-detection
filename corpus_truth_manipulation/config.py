from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

import random

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

ENTITIES_DATA_DIR = EXTERNAL_DATA_DIR / "entities"
MMFAKEBENCH = EXTERNAL_DATA_DIR / "MMFakeBench"
MMFAKEBENCH_GRAPHS = PROCESSED_DATA_DIR / "MMFakeBench_graphs"
MMFAKEBENCH_GRAPHS_ORIGINAL_TEST = PROCESSED_DATA_DIR / "MMFakeBenchOriginal_graphs" / "test_original"
MMFAKEBENCH_GRAPHS_ORIGINAL_VAL = PROCESSED_DATA_DIR / "MMFakeBenchOriginal_graphs" / "val_original"
MMFAKEBENCH_GRAPHS_TEST = MMFAKEBENCH_GRAPHS / "test"
MMFAKEBENCH_GRAPHS_TRAIN = MMFAKEBENCH_GRAPHS / "train"
MMFAKEBENCH_GRAPHS_VAL = MMFAKEBENCH_GRAPHS / "val"
MMFAKEBENCH_GRAPHS_STRATIFIED = PROCESSED_DATA_DIR / "MMFakeBench_graphs_stratified"

XFACTA = EXTERNAL_DATA_DIR / "xfacta"
XFACTA_FORMATTED = PROCESSED_DATA_DIR / "xfacta_formatted"
XFACTA_GRAPHS = PROCESSED_DATA_DIR / "xfacta_graphs"

COSMOS = EXTERNAL_DATA_DIR / "cosmos"
COSMOS_FORMATTED = PROCESSED_DATA_DIR / "cosmos_formatted"
COSMOS_GRAPHS = PROCESSED_DATA_DIR / "cosmos_graphs"

COSMOS_XFACTA = PROCESSED_DATA_DIR / "cosmos_xfacta"
COSMOS_XFACTA_FORMATTED = COSMOS_XFACTA / "formatted"
COSMOS_XFACTA_GRAPHS = COSMOS_XFACTA / "graphs"
COSMOS_XFACTA_GRAPHS_TRAIN = COSMOS_XFACTA_GRAPHS / "train"
COSMOS_XFACTA_GRAPHS_VAL = COSMOS_XFACTA_GRAPHS / "val"
COSMOS_XFACTA_GRAPHS_TEST = COSMOS_XFACTA_GRAPHS / "test"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
    logger.add("log_optuna_p2.log", colorize=False, mode='w')
except ModuleNotFoundError:
    pass


def load_config(config_filename:str='config'):
    with initialize(config_path="../conf", version_base=None):
        cfg = compose(config_name=config_filename)
        OmegaConf.set_struct(cfg, False)
    cfg.PROJ_ROOT = PROJ_ROOT
    cfg.MMFAKEBENCH = MMFAKEBENCH

    return cfg


CONFIG = load_config()
random.seed(CONFIG.seed)


