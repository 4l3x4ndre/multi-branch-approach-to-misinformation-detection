"""
Script to pre-train the encyclopedia claim verifier model on MMFakeBench dataset.
Not used in the manuscript.
"""

from corpus_truth_manipulation.config import load_config, MODELS_DIR, MMFAKEBENCH_GRAPHS_TRAIN, MMFAKEBENCH_GRAPHS_VAL 
from typer import Typer
from loguru import logger
import wandb
import os
from datetime import datetime
import torch
from torch.optim import SGD, Adam
from torch.nn import BCELoss
from torch import from_numpy as torch_from_numpy, save as torch_save
import numpy as np
from omegaconf import OmegaConf
import optuna 

from corpus_truth_manipulation.dataset import create_claim_kg_loader
from src.demo_EGMMG import ClaimVerifier
from src.utils.metrics import AverageMeter


app = Typer()

def validate(model, loader, criterion, device):
    """
    Computes the loss on the validation set.
    """
    model.eval()
    total_loss = 0
    num_samples = 0
    with torch.no_grad():
        for graphs, kgs, _, claims in loader:
            outputs = model(claim_data=graphs, evidence_data=kgs)
            
            try:
                # We weight the batch loss by the number of samples in it
                batch_loss = compute_loss(outputs, claims, criterion, device)
                if batch_loss is not None:
                    total_loss += batch_loss.item() * len(claims)
                    num_samples += len(claims)
            except ValueError as e:
                logger.warning(f"Skipping batch in validation due to error: {e}")
                continue

    model.train() # Set model back to training mode
    
    # Return average loss
    if num_samples == 0:
        logger.warning("Validation loader was empty or all batches failed.")
        return float('inf')
        
    return total_loss / num_samples


@app.command()
def main(config_filename = "config_encyclopedia",
         device:str='', wandblog:bool=True, n_trials:int=50, study_name:str="encyclopedia_tuning"):
    """
    If device is set, it will override the device in config. Use 'cpu' or 'gpu' (first available).
    """
    date = datetime.today().strftime('%Y-%m-%d-%H-%M')
    study_name += "_" + date

    CONFIG = load_config(config_filename)

    # --- Torch config ---
    if device != '':
        cuda_available = torch.cuda.is_available()
        num_gpus = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
        logger.info(f"CUDA available: {cuda_available}")
        logger.info(f"Number of GPUs: {num_gpus}")
        logger.info(f"GPU names: {gpu_names}")

        if cuda_available:
            CONFIG.device = "cuda:0" if device.lower() == 'gpu' else 'cpu'
            _device_name = torch.cuda.get_device_name(0) if device.lower() == 'gpu' else 'CPU'
        else:
            CONFIG.device = 'cpu'
            _device_name = 'CPU'
            logger.warning("CUDA is not available. Using CPU.")
    else:
        _device_name = torch.cuda.get_device_name(0) if 'cuda' in CONFIG.device else 'CPU'
    # --------------------
    logger.info(f"🧠 Using device {CONFIG.device} (device name: {_device_name})")
    torch.manual_seed(CONFIG.seed)
    
    if wandblog:
        api_key = os.getenv("WANDB_API_KEY")
        wandb.login(key=api_key)

    def objective(trial: optuna.trial.Trial) -> float:
        """
        Optuna objective function to find the best hyperparameters.
        """
        # --- WandB Initialization for each trial ---
        if wandblog:
            run = wandb.init(
                project=os.getenv("WANDB_PROJECT"), 
                config=OmegaConf.to_container(CONFIG, resolve=True),
                group=study_name,
                name=f"trial-{trial.number}",
                reinit=True
            )
        
        # --- Hyperparameter Suggestion ---
        # We wrap existing config values and suggest new ones for Optuna to tune.
        cfg = CONFIG.training
        cfg.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        cfg.batch_size = trial.suggest_categorical("batch_size", [2, 4, 8, 16, 32, 64, 128])
        
        # Model-specific hyperparameters
        model_int_dims = trial.suggest_categorical("int_dims", [128, 256, 512])
        model_hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])

        if wandblog:
            wandb.config.update({
                "learning_rate": cfg.learning_rate,
                "batch_size": cfg.batch_size,
                "int_dims": model_int_dims,
                "hidden_dim": model_hidden_dim,
            })

        logger.info(f"Trial {trial.number}: lr={cfg.learning_rate:.2e}, batch_size={cfg.batch_size}, int_dims={model_int_dims}, hidden_dim={model_hidden_dim}")

        # --- DataLoaders ---
        train_loader = create_claim_kg_loader(
            device=CONFIG.device,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            shuffle=cfg.shuffle,
            input_folder=MMFAKEBENCH_GRAPHS_TRAIN,
        )
        val_loader = create_claim_kg_loader(
            device=CONFIG.device,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            shuffle=False, # No need to shuffle validation data
            input_folder=MMFAKEBENCH_GRAPHS_VAL,
        )

        # --- Model, Optimizer, Criterion ---
        model = ClaimVerifier(in_channels=773, int_dims=model_int_dims, hidden_dim=model_hidden_dim).to(CONFIG.device)
        
        if cfg.optimizer == 'SGD':
            optimizer = SGD(model.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum)
        elif cfg.optimizer == 'Adam':
            optimizer = Adam(model.parameters(), lr=cfg.learning_rate)
        else:
            raise NotImplementedError(f"Optimizer {cfg.optimizer} not implemented.")

        criterion = BCELoss() if cfg.criterion == 'BCE' else NotImplementedError(f"Criterion {cfg.criterion} not implemented.")

        # --- Training & Validation Loop ---
        step = 0
        best_val_loss = float('inf')
        running_loss = AverageMeter()
        
        for epoch in range(cfg.epochs):
            model.train()
            epoch_train_loss = 0
            num_train_samples = 0
            
            for graphs, kgs, _, claims in train_loader:
                outputs = model(claim_data=graphs, evidence_data=kgs)
                batch_loss = compute_loss(outputs, claims, criterion, CONFIG.device)

                if batch_loss is not None:
                    batch_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    bloss_item = batch_loss.detach().cpu().item()
                    epoch_train_loss += bloss_item * len(claims)
                    num_train_samples += len(claims)

                    running_loss.update(batch_loss.detach().cpu().item(), n=len(claims))
                    
                    if wandblog and step % CONFIG.logging.frequency_log == 0:
                        wandb.log({
                            "train/batch_loss": bloss_item,
                            "train/step": step,
                            f"train/running_loss": running_loss.get_average()
                        }, step=step)

                        running_loss.reset()
                
                step += 1
            
            # --- Epoch End: Validation and Checkpointing ---
            avg_epoch_train_loss = epoch_train_loss / num_train_samples if num_train_samples > 0 else float('inf')
            val_loss = validate(model, val_loader, criterion, CONFIG.device)
            
            logger.info(f"Epoch {epoch+1}/{cfg.epochs} | Train Loss: {avg_epoch_train_loss:.4f} | Val Loss: {val_loss:.4f}")

            if wandblog:
                wandb.log({
                    "train/epoch_loss": avg_epoch_train_loss,
                    "val/loss": val_loss,
                    "train/epoch": epoch
                }, step=step)

            # Save the model if validation loss improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    "epoch": epoch,
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_val_loss,
                    "hyperparameters": trial.params
                }
                model_name = f"{study_name}_trial_{trial.number}_best.pth"
                model_filename = os.path.join(MODELS_DIR, model_name)
                torch_save(checkpoint, model_filename)
                logger.success(f"New best model saved with val_loss: {best_val_loss:.4f}")

            # --- Optuna Pruning ---
            trial.report(val_loss, epoch)
            if trial.should_prune():
                if wandblog: wandb.finish()
                raise optuna.exceptions.TrialPruned()

        if wandblog:
            run.finish()
            
        return best_val_loss

    # --- Run Optuna Study ---
    storage_name = f"sqlite:///reports/training_optuna/{study_name}.db"
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        pruner=optuna.pruners.NopPruner(),
        storage=storage_name,
        load_if_exists=False
    )
    study.optimize(objective, n_trials=n_trials)

    logger.info(f"Study finished. Best trial: {study.best_trial.number}")
    logger.info(f"  Value (min val_loss): {study.best_trial.value:.4f}")
    logger.info("  Params: ")
    for key, value in study.best_trial.params.items():
        logger.info(f"    {key}: {value}")


def compute_loss(outputs, targets, criterion, device):
    if len(targets) == 0:
        return None

    targets = torch_from_numpy(np.stack(targets)).float().to(device)
    if len(outputs.shape) == 0 and len(targets.shape) == 1:
        outputs = outputs.unsqueeze(-1)
    
    try:
        loss = criterion(outputs, targets)
        return loss
    except Exception as e:
        logger.error(f"Error computing loss: {e}. Outputs: {outputs.shape}, Targets: {targets.shape}")
        return None


if __name__ == "__main__":
    app()
