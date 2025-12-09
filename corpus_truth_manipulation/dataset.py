from pathlib import Path
from typing import Iterator, Tuple, Optional, Iterable, Union, Dict, List, Any, Callable
import numpy as np
from PIL import Image, UnidentifiedImageError

import os
import json
import glob
import random
import torch
from torchvision import transforms
from torch_geometric.data import Batch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer

from loguru import logger
from tqdm import tqdm
import typer
from omegaconf import DictConfig

from corpus_truth_manipulation.config import CONFIG, \
    MMFAKEBENCH_GRAPHS_TRAIN, MMFAKEBENCH_GRAPHS, MMFAKEBENCH

app = typer.Typer()


@app.command()
def main():
    logger.info("Processing dataset...")

    # loader = create_claim_kg_loader(shuffle=False)
    _ = GlobalDataset(
        MMFAKEBENCH_GRAPHS,
        split="val",
        device="cpu",
        max_samples=10
    )

    logger.success("Processing dataset complete.")


class MMFakeBenchDataset(Dataset):
    """
    PyTorch Dataset for Multi-Modal Misinformation Detection.
    
    This dataset handles text-image pairs from MMFakeBench format.
    text-image inconsistencies in social media posts and news articles.
    
    Args:
        data_root (str): Root directory containing the MMFakeBench dataset
        config (DictConfig): Configuration object, pass if different from default
        info_on_startup (bool): Whether to print dataset info on initialization
        split (str): Dataset split ('train', 'val', 'test')
        transform (callable, optional): Transform to apply to images
        shuffle_data (bool): Whether to shuffle the data on initialization
        max_samples (int, optional): Maximum number of samples to load (for debugging)
    """
    
    def __init__(
        self,
        data_root: str,
        config: DictConfig = CONFIG,
        info_on_startup: bool = True,
        split: str = 'train',
        transform: Optional[callable] = None,
        shuffle_data: bool = True,
        max_samples: Optional[int] = None,
        include_img: bool = True,
    ):
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.shuffle_data = shuffle_data
        self.max_samples = max_samples
        self.include_img = include_img
        self.config = config
        
        # Initialize CLIP preprocessing
        self.device = torch.device(config.device)
        
        # Load and prepare data
        self.data = self._load_data()
        self._prepare_data()


        if info_on_startup:
            logger.info(f"Dataset size: {self.__len__()}")
            distribution, img_sources = self.get_class_distribution()
            logger.info(f"Class distribution: {distribution}")
            logger.info(f"Image sources: {img_sources}")
            sample = self.__getitem__(0)
            logger.info(f"Sample keys: {sample.keys()}")
            logger.info(f"Image shape: {np.array(sample['image']).shape}")
            logger.info(f"Text shape: {len(sample['text_str'])}")
        
    def _load_data(self) -> List[Dict[str, Any]]:
        """
        Load data from JSON file.
        Raise error if no files found, shuffle if requested, and limit samples if specified.
        """

        json_pattern = os.path.join(self.data_root, self.split, "source", "*.json")
        json_files = glob.glob(json_pattern)

        if not json_files:
            raise FileNotFoundError(f"No JSON files found at {json_pattern}")
        logger.debug(f"Loading data from {json_files[0]}")

        with open(json_files[0], 'r') as f:
            data = json.load(f)

        # Shuffle data if requested
        if self.shuffle_data:
            random.shuffle(data)

        # Limit samples if specified 
        if self.max_samples is not None:
            data = data[:self.max_samples]

        return data
    
    def _prepare_data(self):
        """Prepare and validate data paths."""
        valid_data = []
        
        for item in self.data:
            # Fix image path (remove leading "/")
            if item['image_path'].startswith('/'):
                item['image_path'] = item['image_path'][1:]
            
            # Construct full image path
            img_path = os.path.join(self.data_root, self.split, item['image_path'])
            
            # Validate that image exists
            if os.path.exists(img_path):
                item['full_image_path'] = img_path
                valid_data.append(item)
            else:
                # the base filename might be incorrect (xfact),
                # try to find the first image with the same extension:
                img_root_path, ext = os.path.splitext(img_path)
                search_pattern = os.path.join(
                    os.path.dirname(img_root_path),
                    f"*{ext}"
                )
                found_images = glob.glob(search_pattern, recursive=True)
                if found_images:
                    item['full_image_path'] = found_images[0]
                    valid_data.append(item)
                else:
                    logger.warning(f"Warning: Image not found at {img_path}")
        
        self.data = valid_data
        print(f"Loaded {len(self.data)} valid samples from {self.split} split")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Sample index
            
        Returns:
            dict: Dictionary containing:
                - 'image': Preprocessed image tensor
                - 'text': Tokenized text tensor
                - 'label': Label tensor (if available)
                - 'metadata': Additional metadata
        """
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data)}")
        
        item = self.data[idx]
        
        # Load and preprocess image
        if self.include_img:
            try:
                pil_image = Image.open(item['full_image_path']).convert("RGB")
                
                if self.transform is not None:
                    image_tensor = self.transform(pil_image)
                else:
                    image_tensor = transforms.ToTensor()(pil_image)
                    
            except Exception as e:
                print(f"Error loading image {item['full_image_path']}: {e}")
                # Return a black image as fallback
                image_tensor = torch.zeros(3, 224, 224)
        
        # Process text with CLIP tokenization
        text = item.get('text', '')

        # Check if image is real of fake (edited or generated)
        item['image_real'] = item['fake_cls'] in CONFIG.data.real_image_fake_cls or item['image_source'] in CONFIG.data.real_image_sources

        # Claim is real or fake
        item['claim_real'] = item['fake_cls'] in CONFIG.data.real_claim_fake_cls
        item['mismatch'] = item['fake_cls'] in CONFIG.data.mismatch_fake_cls

        # Prepare output dictionary
        output = {
            'image': image_tensor if self.include_img else [None],
            'text_str': text,
            'metadata': {
                'image_path': item['image_path'],
                'full_image_path': item['full_image_path'],
                'text_raw': text,
                'idx': idx,
                'image_source': item['image_source'],
                'image_real': item['image_real'],
                'claim_real': item['claim_real'],
                'mismatch': item['mismatch'],
                'overall_truth': item['fake_cls'] == 'original',
                "text_source": item["text_source"],
                "gt_answsers": item["gt_answers"],
                "fake_cls": item["fake_cls"],
            },
            "gt_answers": item["gt_answers"],
            "fake_cls": item["fake_cls"]
        }
        
        return output
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get the distribution of classes in the dataset."""
        if not hasattr(self, 'data') or not self.data:
            logger.warning("Dataset is empty, cannot compute class distribution.")
            raise ValueError("Dataset is empty.")
        
        distribution = {}
        for item in self.data:
            distribution[str(item['fake_cls'])] = distribution.get(str(item['fake_cls']), 0) + 1
            if item['fake_cls'] == '' and not 'visualnews_mani' in item['image_path']:
                raise ValueError(f"No class found: {item['image_path']}")

        img_sources = {}
        for item in self.data:
            img_sources[str(item['image_source'])] = img_sources.get(str(item['image_source']), 0) + 1
        
        return distribution, img_sources
    
    def get_sample_metadata(self, idx: int) -> Dict[str, Any]:
        """Get metadata for a specific sample without loading the actual data."""
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range")
        
        return self.data[idx]


def create_data_loaders(
    data_root: str,
    split:str,
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle: bool = True
) -> torch.utils.data.DataLoader:
    """
    Create data loaders for train, validation, and test splits.
    
    Args:
        data_root (str): Root directory of the dataset
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of worker processes
        shuffle_train (bool): Whether to shuffle training data
    
    Returns:
        tuple: DataLoaders for train, val, test splits
    """
    
    # Create datasets
    dataset = MMFakeBenchDataset(data_root, split=split, shuffle_data=shuffle)
    
    # Create data loaders
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
       
    return loader


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for handling variable-length sequences.
    
    Args:
        batch: List of sample dictionaries
    
    Returns:
        dict: Batched data, images are PIL images, texts are list of strings
    """
    # Stack images and texts
    images = [item['image'] for item in batch]
    texts = [item['text_str'] for item in batch]

    
    output = {
        'image': images,
        'text_str': texts,
        'metadata': [item['metadata'] for item in batch],
        'gt_answers': [item['gt_answers'] for item in batch],
        'fake_cls': [item['fake_cls'] for item in batch],
    }
    
    return output


# ------------------
# Claim & KG dataset
# ------------------
class ClaimKGDataset(Dataset):
    def __init__(self, graphs_dir, device="cpu", max_samples:Optional[int] = None):
        """
        Args:
            graphs_dir (str | Path): directory with precomputed .pt graph files
            device (str): device string ("cpu" / "cuda")
            max_sample (Optional[int]): max sample in data set (use for test/debug)
        """
        self.graphs_dir = graphs_dir
        self.device = device

        self.txt_encoder_model = SentenceTransformer("all-MiniLM-L6-v2", device=device, token=True, local_files_only=True)
        self.txt_encoder_model.eval()

        # Load valid graph indices
        self.data = []
        files = os.listdir(graphs_dir)
        if max_samples:
            files = files[:max_samples]
        for file in tqdm(files, desc='Loading graphs...'):
            if file.endswith(".pt"):
                idx = int(file.split("_")[1].split(".")[0])
                g_k = torch.load(os.path.join(graphs_dir, file), weights_only=False)
                graph, kg, txt, claim = g_k["claim"], g_k["kg"], g_k["text"], g_k["claim_real"]

                # Graphs were directly saved as pyg
                graph_pyg = graph 
                kg_pyg = kg

                if (graph_pyg is torch.nan or kg_pyg is torch.nan 
                    or graph_pyg.num_nodes == 0 or kg_pyg.num_nodes == 0):
                    logger.warning(f"Precomputed graph for idx {idx} is empty. Skipping.")
                    continue

                self.data.append((graph_pyg.to(device), kg_pyg.to(device), txt, claim))
        logger.success(f"Data set ready ✅ Number of sample: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def collate_claim_kg(batch):
    """
    batch: list of tuples (claim_graph: Data, kg_graph: Data, texts, claim_texts)
    Returns: (claim_batch: Batch, kg_batch: Batch, texts: list, claim_texts: list)
    """
    claims, kgs, texts, claim_texts = zip(*batch)  # tuples length = batch_size

    # Convert lists of Data -> Batched Batch objects
    claim_batch = Batch.from_data_list(list(claims))
    kg_batch    = Batch.from_data_list(list(kgs))

    # Keep texts as lists
    return claim_batch, kg_batch, list(texts), list(claim_texts)

def create_claim_kg_loader(
        device="cpu", 
        batch_size=16, 
        num_workers=4, 
        shuffle=True,
        input_folder=MMFAKEBENCH_GRAPHS_TRAIN,
        max_samples:Optional[int]=None):
    # Instantiate dataset
    dataset = ClaimKGDataset(
        graphs_dir=input_folder,
        device=device,
        max_samples=max_samples,
    )

    # Dataloader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_claim_kg,
        num_workers=num_workers,
    )
    return loader


def retrieve_images(corpus_path: Union[str, Path],
                    valid_extensions: Optional[Iterable[str]] = None,
                    recursive: bool = True,
                    follow_symlinks: bool = False
                   ) -> Iterator[Tuple[str, np.ndarray]]:
    """
    Yield (full_path, numpy_array) for each image found under `corpus_path`.

    Parameters
    ----------
    corpus_path : str | Path
        Root directory to scan.
    valid_extensions : iterable[str] | None
        Image extensions to accept (without dot). Case-insensitive.
        Default: {'png', 'jpg', 'jpeg'}
    recursive : bool
        If True (default), recursively scan subdirectories.
    follow_symlinks : bool
        Whether to follow symbolic links when scanning.

    Yields
    ------
    (full_path: str, image_array: numpy.ndarray)
        full_path is an absolute path (string). image_array is the result of
        `numpy.array(Image.open(...))`.
    """
    root = Path(corpus_path)
    if valid_extensions is None:
        valid_exts = {"png", "jpg", "jpeg"}
    else:
        valid_exts = {ext.lower().lstrip('.') for ext in valid_extensions}

    if not root.exists():
        # nothing to yield
        return

    # choose iterator: rglob for recursive, iterdir for non-recursive
    if recursive:
        iterator = root.rglob("*")
    else:
        iterator = root.iterdir()

    for p in iterator:
        try:
            # skip directories, non-files, and symlinks depending on follow_symlinks
            if not p.is_file():
                continue
            if p.is_symlink() and not follow_symlinks:
                continue

            # extension check (handles filenames without dot)
            ext = p.suffix.lower().lstrip('.')  # '' if no suffix
            if ext not in valid_exts:
                continue

            # open image safely and yield array
            try:
                with Image.open(p) as im:
                    arr = np.array(im)
                yield (str(p.resolve()), arr)
            except Exception:
                # skip files that PIL cannot open / are corrupted
                continue

        except PermissionError:
            # skip unreadable files/directories
            continue

# ------------------
# Global Dataset
# ------------------
class GlobalDataset(Dataset):
    def __init__(
        self,
        graphs_dir:Path,
        split:str,
        root_to_image:Path = MMFAKEBENCH,
        device:str="cpu",
        max_samples:int = 0,
        config:DictConfig = CONFIG,
        image_transform:Optional[Callable]=None,
    ):
        self.graphs_dir = graphs_dir
        self.root_to_image = root_to_image
        self.split=split
        self.device = device
        self.max_samples = max_samples
        self.config = config

        self.image_transform = image_transform
        self.data = self._load_data()

        logger.info(f"Dataset path: {graphs_dir}")
        logger.info(f"GlobalDataset initialized with {len(self.data)} samples from split '{self.split}'")

    def _load_data(self):
        data = []

        split_path = os.path.join(self.graphs_dir, self.split)
        files = os.listdir(split_path)
        if self.max_samples > 0:
            files = files[:self.max_samples]
        for file in tqdm(files, desc=f'Loading {self.split} graphs...'):
            if file.endswith(".pt"):
                g_k = torch.load(os.path.join(split_path, file), weights_only=False)
                graph, kg, txt, claim = g_k["claim"], g_k["kg"], g_k["text"], g_k["claim_real"]
                image_path = g_k["metadata"]["image_path"]

                        
                split_imagename = 'train' if self.split in ['train', 'val'] else self.split
                img_path = os.path.join(self.root_to_image, split_imagename, image_path)

                # dedicated process for xfacta
                if 'xfact' in img_path.lower():
                    found_candidate = False
                    try:
                        _ = Image.open(img_path).convert("RGB")
                        found_candidate = True
                    except Exception as e:
                        # for xfacta, this might happen if another image is present in the same folder
                        # => try to load the first image with the same extension
                        if 'xfact' in img_path.lower():
                            img_root_path, ext = os.path.splitext(img_path)
                            search_pattern = os.path.join(
                                os.path.dirname(img_root_path),
                                f"*{ext}"
                            )
                            found_images = glob.glob(search_pattern, recursive=True)
                            # select another image if found
                            found_candidate = False
                            for img_candidate in found_images:
                                try:
                                    logger.warning(f"Warning: Original image {img_path} could not be opened. Using {img_candidate} instead.")
                                    _ = Image.open(img_candidate).convert("RGB")
                                    found_candidate = True
                                    image_path = os.path.relpath(img_candidate, os.path.join(self.root_to_image, split_imagename))
                                    logger.debug(f"Using image path: {image_path}")
                                    break
                                except UnidentifiedImageError:
                                    continue
                            if not found_candidate:
                                continue
                        else:
                            raise e
                    if not found_candidate:
                        continue


                data.append({
                    "claim": graph,
                    "kg": kg,
                    "text": txt,
                    "claim_real": claim,
                    "image_path": image_path,
                    "metadata": g_k["metadata"]
                })

        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]

        # Try to use the absolute path from metadata first
        if 'full_image_path' in item['metadata'] and os.path.exists(item['metadata']['full_image_path']):
            img_path = item['metadata']['full_image_path']
        else: # Fallback to the old logic
            split_imagename = 'train' if self.split in ['train', 'val'] else self.split
            img_path = os.path.join(self.root_to_image, split_imagename, item['image_path'])

        if not os.path.exists(img_path):
            # the base filename might be incorrect (xfact),
            # try to find the first image with the same extension:
            img_root_path, ext = os.path.splitext(img_path)
            search_pattern = os.path.join(
                os.path.dirname(img_root_path),
                f"*{ext}"
            )
            found_images = glob.glob(search_pattern, recursive=True)
            if found_images:
                img_path = found_images[0]

        pil_image = Image.open(img_path).convert("RGB")
        if self.image_transform is not None:
            image_transformed = self.image_transform(pil_image)
        else:
            image_transformed = transforms.ToTensor()(pil_image)

        img_real = 1 if item['metadata']["image_real"] else 0
        claim_real = 1 if item['metadata']["claim_real"] else 0
        mismatch = 1 if item['metadata']["mismatch"] else 0
        overall_truth = 1 if item['metadata']["overall_truth"] else 0

        target_tensor = torch.tensor(
            [img_real, claim_real, mismatch, overall_truth], 
            dtype=torch.float32)

        out = (
            item["claim"].to(self.device),
            item["kg"].to(self.device),
            image_transformed,
            item["text"],
            target_tensor.to(self.device),
        )
        return out

def collate_globaldataset(batch):
    claims, kgs, images, texts, targets = zip(*batch)  # tuples length = batch_size

    # Convert lists of Data -> Batched Batch objects
    claim_batch = Batch.from_data_list(list(claims))
    kg_batch    = Batch.from_data_list(list(kgs))

    # Keep texts as lists
    return claim_batch, kg_batch, list(images), list(texts), torch.stack(list(targets))

def create_globaldataset_loader(
        device="cpu", 
        batch_size=16, 
        num_workers=4, 
        shuffle=True,
        input_folder=MMFAKEBENCH_GRAPHS,
        root_to_image=MMFAKEBENCH,
        split="train",
        max_samples:Optional[int]=None,
        image_transform:Optional[Callable]=None,
):
    dataset = GlobalDataset(
        graphs_dir=input_folder,
        split=split,
        device=device,
        max_samples=max_samples if max_samples else 0,
        image_transform=image_transform,
        root_to_image=root_to_image,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_globaldataset,
    )
    return loader



if __name__ == "__main__":
    app()
