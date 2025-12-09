"""
Script to pre-train the encyclopedia claim verifier model on MMFakeBench dataset.
Not used in the manuscript.
"""

from corpus_truth_manipulation.config import CONFIG, MODELS_DIR, MMFAKEBENCH_GRAPHS_TEST

from typer import Typer
from loguru import logger
import wandb
import os
from torch.optim import SGD, Adam
from torch.nn import BCELoss
from torch import stack as torch_stack, from_numpy as torch_from_numpy, save as torch_save
from torch.cuda import is_available as torch_cuda_is_available, \
    device_count as torch_cuda_device_count, \
    get_device_name as torch_cuda_get_device_name
import numpy as np
from omegaconf import OmegaConf

from corpus_truth_manipulation.dataset import create_claim_kg_loader
from src.demo_EGMMG import ClaimVerifier


app = Typer()

@app.command()
def main(device:str='', wandblog:bool=True):
    """
    If device is set, it will override the device in config. Use 'cpu' or 'gpu' (first available).
    """

    if wandblog:
        api_key = os.getenv("WANDB_API_KEY")
        wandb.login(key=api_key)
        wandb.init(project=os.getenv("WANDB_PROJECT"), config=OmegaConf.to_container(CONFIG, resolve=True))

    # --- Torch config ---
    if device != '':
        cuda_available = torch_cuda_is_available()
        num_gpus = torch_cuda_device_count()
        gpu_names = [torch_cuda_get_device_name(i) for i in range(num_gpus)]

        logger.info(f"CUDA available: {cuda_available}")
        logger.info(f"Number of GPUs: {num_gpus}")
        logger.info(f"GPU names: {gpu_names}")

        if cuda_available:
            CONFIG.device = "cuda:0" if device.lower() == 'gpu' else 'cpu'
            _device_name = torch_cuda_get_device_name(0) if device.lower() == 'gpu' else 'CPU'
        else:
            CONFIG.device = 'cpu'
            _device_name = 'CPU'
            logger.warning("CUDA is not available. Using CPU.")
    # --------------------
    logger.info(f"🧠 Using device {CONFIG.device} (device name: {_device_name})")

    loader = create_claim_kg_loader(
        device=CONFIG.device,
        batch_size=CONFIG.training.batch_size,
        num_workers=CONFIG.training.num_workers,
        shuffle=CONFIG.training.shuffle,
        input_folder=MMFAKEBENCH_GRAPHS_TEST,
    )

    model = ClaimVerifier(773).to(CONFIG.device)

    if CONFIG.training.optimizer == 'SGD':
        optimizer = SGD(
            model.parameters(), 
            lr=CONFIG.training.learning_rate, 
            momentum=CONFIG.training.momentum)
    elif CONFIG.training.optimizer == 'Adam':
        optimizer = Adam(
            model.parameters(), 
            lr=CONFIG.training.learning_rate)
    else:
        raise NotImplementedError(f"Optimizer {CONFIG.training.optimizer} not implemented.")

    if CONFIG.training.criterion == 'BCE':
        criterion = BCELoss()
    else:
        raise NotImplementedError(f"Criterion {CONFIG.training.criterion} not implemented.")

    train(model, loader, criterion, optimizer, wandblog)


def train(model, loader, criterion, optimizer, wandblog):
    step = 0
    min_loss = float('inf')
    logger.info("Training started.")
    for epoch in range(CONFIG.training.epochs):
        model.train()
        for graphs, kgs, txts, claims in loader:
            
            outputs = model(claim_data=graphs, evidence_data=kgs)

            batch_loss = compute_loss(outputs, claims, criterion)

            if batch_loss:
                batch_loss.backward()   # computes gradients for model parameters
                optimizer.step()        # updates model parameters
                optimizer.zero_grad()   # reset gradients for next batch
                bloss_item = batch_loss.detach().cpu().item()
            else:
                bloss_item = None

            if step % CONFIG.logging.frequency_log == 0:
                try:
                    logger.info(f'Text: {txts[0]}. Graph: {graphs[0]}, kg: {kgs[0]}')
                    logger.info(f'Model output: {outputs[0]} Ground truth: {claims[0]}')
                except Exception as e:
                    logger.error(f'Could not retrieve model output: {e}')
                if wandblog:
                    wandb.log({
                        "train/loss": bloss_item,
                        "train/epoch": epoch,
                        "train/step": step,
                    }, step=step)
            if step % CONFIG.logging.frequency_save == 0 and bloss_item is not None and bloss_item < min_loss:
                min_loss = batch_loss.detach().cpu().item()
                checkpoint = {
                    "epoch": epoch,
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": batch_loss.detach().cpu().item(),
                }
                model_name = f"encyclopedia_classifier_cv_{CONFIG.training.optimizer}_best.pth"
                model_filename = os.path.join(MODELS_DIR, model_name)
                torch_save(checkpoint, model_filename)

            step += 1

        print(f"Epoch {epoch+1}/{CONFIG.training.epochs} completed.")

    
def compute_loss(outputs, targets, criterion):
    if len(targets) == 0:
        raise ValueError(f"No elements in targets. {outputs}, {targets}")

    targets = torch_from_numpy(np.stack(targets)).float().to(CONFIG.device)
    if len(outputs.shape) == 0 and len(targets.shape) == 1:
        outputs = outputs.unsqueeze(-1)
    return criterion(outputs, targets)


if __name__ == "__main__":
    app()
