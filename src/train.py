from typer import Typer
from loguru import logger
import wandb
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import SGD
from torch.nn import BCELoss
from torch import stack as torch_stack, from_numpy as torch_from_numpy
import numpy as np
from omegaconf import OmegaConf, DictConfig

from corpus_truth_manipulation.dataset import GlobalDataset, collate_fn
from corpus_truth_manipulation.config import CONFIG

from src.models import MultimodalMisinformationDetector

app = Typer()

@app.command()
def main():


    api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=api_key)
    wandb.init(project=os.getenv("WANDB_PROJECT"), config=OmegaConf.to_container(CONFIG, resolve=True))


    image_transform = transforms.Compose([
        transforms.Resize((CONFIG.data.image_size, CONFIG.data.image_size)),
    ])

    dataset = MMFakeBenchDataset(
        data_root=CONFIG.MMFAKEBENCH,
        split='test',
        # max_samples=100,
        transform=image_transform
    )

    test_loader = DataLoader(
        dataset,
        batch_size=CONFIG.training.batch_size,
        shuffle=CONFIG.training.shuffle,
        num_workers=CONFIG.training.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    model = MultimodalMisinformationDetector()
    if CONFIG.training.optimizer == 'SGD':
        optimizer = SGD(
            model.parameters(), 
            lr=CONFIG.training.learning_rate, 
            momentum=CONFIG.training.momentum)
    else:
        raise NotImplementedError(f"Optimizer {CONFIG.training.optimizer} not implemented.")

    if CONFIG.training.criterion == 'BCE':
        criterion = BCELoss()
    else:
        raise NotImplementedError(f"Criterion {CONFIG.training.criterion} not implemented.")

    train(model, test_loader, criterion, optimizer)


def train(model, loader, criterion, optimizer):
    step = 0
    for epoch in range(CONFIG.training.epochs):
        for batch in loader:
            outputs = model(batch['image'], batch['text_str'])

            batch_loss = compute_loss(outputs, batch['metadata'], criterion)

            batch_loss.backward()   # computes gradients for model parameters
            optimizer.step()        # updates model parameters
            optimizer.zero_grad()   # reset gradients for next batch


            if step % CONFIG.logging.frequency_log == 0:
                logger.info(f'Model outputs: {outputs}')
                logger.info(f'Ground truth: {batch["gt_answers"]}')
                logger.info(f'Class: {batch["fake_cls"]}')
                logger.info(f'Image paths: {[item["image_path"] for item in batch["metadata"]]}')
                logger.info(f'Image real: {[item["image_real"] for item in batch["metadata"]]}')
                wandb.log({
                    "train/loss": batch_loss.detach().cpu().item(),
                    "epoch": epoch,
                    "step": step,
                }, step=step)

            step += 1

        print(f"Epoch {epoch+1}/{CONFIG.training.epochs} completed.")

    
def compute_loss(outputs, items, criterion):
    targets = torch_stack([
        torch_from_numpy(np.stack([item['image_real'] for item in items])).float(), 
        torch_from_numpy(np.stack([item['claim_real'] for item in items])).float(), 
        torch_from_numpy(np.stack([item['mismatch'] for item in items])).float(), 
        torch_from_numpy(np.stack([item['overall_truth'] for item in items])).float()
    ], dim=1)  # shape [B, 4]


    return criterion(outputs, targets)


if __name__ == "__main__":
    app()
