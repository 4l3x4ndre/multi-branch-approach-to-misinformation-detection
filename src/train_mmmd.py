"""
Train script for the MultiModal Misinformation Detector model with Optuna hyperparameter optimization and ablation studies.
Used to train models for the manuscript.
"""


from typer import Typer, Option
from loguru import logger
import wandb
import os
from datetime import datetime
from torchvision import transforms
from torch.optim import SGD, Adam
from torch.nn import BCELoss, BCEWithLogitsLoss
import torch
from omegaconf import OmegaConf
import optuna

from corpus_truth_manipulation.dataset import create_globaldataset_loader
from corpus_truth_manipulation.config import CONFIG, MODELS_DIR, MMFAKEBENCH_GRAPHS, MMFAKEBENCH_GRAPHS_STRATIFIED, \
    MMFAKEBENCH, XFACTA_GRAPHS, XFACTA_FORMATTED, COSMOS_GRAPHS, COSMOS_FORMATTED, COSMOS_XFACTA_GRAPHS, COSMOS_XFACTA_FORMATTED,  \
    MMFAKEBENCH_GRAPHS_ORIGINAL_VAL, MMFAKEBENCH_GRAPHS_ORIGINAL_TEST
from src.models import MultimodalMisinformationDetector
from src.utils.metrics import RunningLoss

app = Typer()


def validate(model, loader, criterion, only_neuron_nb_X: int = 0):
    """
    Computes the loss on the validation set.
    """
    model.set_mode(train=False)
    total_loss = 0
    num_samples = 0
    with torch.no_grad():
        for graphs, kgs, images, texts, targets in loader:
            outputs = model(
                image=images,
                text_str=texts,
                claim_data=graphs, evidence_data=kgs
                )
            try:
                _targets = targets
                if only_neuron_nb_X > 0:
                    _targets = targets[:, only_neuron_nb_X - 1].unsqueeze(1)
                batch_loss = compute_loss(outputs, _targets, criterion)
                if batch_loss is not None:
                    total_loss += batch_loss.item() * len(graphs)
                    num_samples += len(graphs)
            except ValueError as e:
                logger.warning(f"Skipping batch in validation due to error: {e}")
                continue

    model.set_mode(train=True) # Set model back to training mode
    
    # Return average loss
    if num_samples == 0:
        logger.warning("Validation loader was empty or all batches failed.")
        return float('inf')
        
    return total_loss / num_samples

def get_dataset_folders(dataset:str):
    stratified_dataset = False
    if 'MMFAKEBENCH' in dataset:
        root_to_image = MMFAKEBENCH
        if dataset == 'MMFAKEBENCH':
            data_root = MMFAKEBENCH_GRAPHS
        elif dataset == 'MMFAKEBENCH_STRATIFIED':
            data_root = MMFAKEBENCH_GRAPHS_STRATIFIED
        elif dataset == 'MMFAKEBENCH_test_original':
            data_root = MMFAKEBENCH_GRAPHS_ORIGINAL_TEST
        elif dataset == 'MMFAKEBENCH_val_original':
            data_root = MMFAKEBENCH_GRAPHS_ORIGINAL_VAL
        else:
            raise ValueError(f'Unknown dataset folder <{dataset}>.')
    elif dataset == 'XFACTA':
        data_root = XFACTA_GRAPHS
        root_to_image = XFACTA_FORMATTED
    elif dataset == 'COSMOS':
        data_root = COSMOS_GRAPHS
        root_to_image = COSMOS_FORMATTED
    elif dataset == 'COSMOS-XFACTA':
        data_root = COSMOS_XFACTA_GRAPHS
        root_to_image = COSMOS_XFACTA_FORMATTED
        stratified_dataset = True
    else:
        raise ValueError(f'Unknown dataset folder <{dataset}>.')

    return root_to_image, data_root, stratified_dataset


@app.command()
def main(
        pretrained_encyclopedia:str='',
        device:str='', split:str='train',
        wandblog:bool=True, n_trials:int=50,
        force_study_name:bool=False,
        study_name:str="ablation",
        jobid:str='',
        use_optuna_params:bool=True,
        use_deepfake_detector:bool=True,
        clip_use_image_features:bool=True,
        clip_use_text_features:bool=True,
        use_encyclopedic_knowledge:bool=True,
        finetune:bool=False,
        only_neuron_nb_X: int = Option(0, help="If set to a value from 1 to 4, train only on that specific output neuron. 0 means train on all 4.", min=0, max=4),
        dataset:str=Option('MMFAKEBENCH', help="Dataset to use: 'MMFAKEBENCH' or others."),
        use_validation_set: bool = Option(False, help="Whether to use a validation set (MMFakeBench val) during training.")
):    
    root_to_image, data_root, stratified_dataset = get_dataset_folders(dataset)


    if not force_study_name:
        date = datetime.today().strftime('%Y-%m-%d-%H-%M')
        study_name += "_" + date
    if finetune:
        study_name += "_finetune"
    if only_neuron_nb_X > 0:
        study_name += f"_neuron{only_neuron_nb_X}"
    study_name += f"_train{dataset}"

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

    image_transform = transforms.Compose([
        transforms.Resize((CONFIG.data.image_size, CONFIG.data.image_size)),
    ])

    if pretrained_encyclopedia != '':
        pretrained_obj = torch.load(pretrained_encyclopedia, map_location=CONFIG.device, weights_only=False)
        pretrained_weights = pretrained_obj['model_state_dict']
    else:
        pretrained_weights = None
        pretrained_obj = {'hyperparameters': {
            'int_dims': CONFIG.model.claimverifier_int_dims,
            'hidden_dim': CONFIG.model.claimverifier_hidden_dims
        }}

    def objective(trial: optuna.trial.Trial) -> float:

        # --- Hyperparameter Suggestion ---
        cfg = CONFIG.training
        cfg.use_deepfake_detector = use_deepfake_detector
        cfg.clip_use_image_features = clip_use_image_features
        cfg.clip_use_text_features = clip_use_text_features
        cfg.use_encyclopedic_knowledge = use_encyclopedic_knowledge
        cfg.use_clip = cfg.clip_use_image_features or cfg.clip_use_text_features

        cfg, global_layer_sizes, dropout, classifier_input_dim = suggest_ablation_and_hparams(cfg, trial, use_optuna_params, only_neuron_nb_X=only_neuron_nb_X)

        if wandblog:
            wandb_name = f"trial-{trial.number}_{jobid}"
            if use_deepfake_detector:
                wandb_name += "_deepfake"
            if cfg.clip_use_image_features:
                wandb_name += "_clipimg"
            if cfg.clip_use_text_features:
                wandb_name += "_cliptext"
            if use_encyclopedic_knowledge:
                wandb_name += "_encyclo"
            if finetune:
                wandb_name += "_finetune"

            tags = ['ablation_fixed_nopre', f'dataset_{dataset}']
            if only_neuron_nb_X > 0:
                tags.append(f"neuron_{only_neuron_nb_X}")
                wandb_name += f"_neuron{only_neuron_nb_X}"
            if stratified_dataset: 
                wandb_name += "_stratified"
                tags.append('dataset_stratified')
            if finetune:
                tags.append("finetune")
            run = wandb.init(
                project=os.getenv("WANDB_PROJECT"),
                config=OmegaConf.to_container(CONFIG, resolve=True),
                group=study_name,
                name=wandb_name,
                reinit=True,
                tags=tags
            )
            wandb.config.update({
                "learning_rate": cfg.learning_rate,
                "batch_size": cfg.batch_size,
                "int_dims": pretrained_obj['hyperparameters']['int_dims'],
                "hidden_dim": pretrained_obj['hyperparameters']['hidden_dim'],
                "global_layer_sizes": global_layer_sizes,
                "dropout": dropout,
                "use_deepfake_detector": cfg.use_deepfake_detector,
                "use_clip": cfg.use_clip,
                "use_clip_image_features": cfg.clip_use_image_features,
                "use_clip_text_features": cfg.clip_use_text_features,
                "use_encyclopedic_knowledge": cfg.use_encyclopedic_knowledge,
                "finetune": finetune,
            })

        logger.info(f"Trial {trial.number}: lr={cfg.learning_rate:.2e}, batch_size={cfg.batch_size}, global_layer_sizes={global_layer_sizes}", dropout={dropout})

        train_loader = create_globaldataset_loader(
            device=CONFIG.device,
            split=split,
            image_transform=image_transform,
            shuffle=CONFIG.training.shuffle,
            max_samples=None,
            num_workers=cfg.num_workers,
            batch_size=cfg.batch_size,
            input_folder=data_root,
            root_to_image=root_to_image
        )

        val_loader = None
        if use_validation_set:
            val_loader = create_globaldataset_loader(
                device=CONFIG.device,
                split='val',
                image_transform=image_transform,
                shuffle=False, # No need to shuffle validation data
                max_samples=None,
                num_workers=cfg.num_workers,
                batch_size=cfg.batch_size,
                input_folder=MMFAKEBENCH_GRAPHS, # Always use MMFakeBench for validation
                root_to_image=MMFAKEBENCH # Always use MMFakeBench for validation
            )

        model = MultimodalMisinformationDetector(
            classifier_input_dim=classifier_input_dim,
            use_deepfake_detector=cfg.use_deepfake_detector,
            use_clip=cfg.use_clip,
            use_encyclopedic_knowledge=cfg.use_encyclopedic_knowledge,
            clip_use_image_features=cfg.clip_use_image_features,
            clip_use_text_features=cfg.clip_use_text_features,
            ency_int_dims=pretrained_obj['hyperparameters']['int_dims'],
            ency_hidden_dim=pretrained_obj['hyperparameters']['hidden_dim'],
            layer_sizes=global_layer_sizes,
            dropout=dropout,
            activation_sigmoid=False,
            finetune_deepfake_detector=finetune,
        ).to(CONFIG.device)
        if pretrained_encyclopedia != '':
            model.load_encycopedic_knowledge_weights(pretrained_weights, strict=False)
         
        if finetune:
            finetune_params = []
            if model.use_deepfake_detector:
                finetune_params.extend([p for p in model.deepfake_detector.parameters() if p.requires_grad])

            base_params = [p for p in model.classifier.parameters() if p.requires_grad]
            if model.use_encyclopedic_knowledge:
                base_params.extend([p for p in model.encyclopedic_knowledge.parameters() if p.requires_grad])

            # check for overlapping parameters
            finetune_param_ids = {id(p) for p in finetune_params}
            base_param_ids = {id(p) for p in base_params}
            assert len(finetune_param_ids.intersection(base_param_ids)) == 0, "Found overlapping parameters between finetune and base groups."

            param_groups = [
                {'params': finetune_params, 'lr': cfg.learning_rate * 0.1},
                {'params': base_params} # will use default LR from optimizer
            ]
            if cfg.optimizer == 'SGD':
                optimizer = SGD(param_groups, lr=cfg.learning_rate, momentum=cfg.momentum)
            elif cfg.optimizer == 'Adam':
                optimizer = Adam(param_groups, lr=cfg.learning_rate)
            else:
                raise NotImplementedError(f"Optimizer {cfg.optimizer} not implemented.")
        else:
            if cfg.optimizer == 'SGD':
                optimizer = SGD(model.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum)
            elif cfg.optimizer == 'Adam':
                optimizer = Adam(model.parameters(), lr=cfg.learning_rate)
            else:
                raise NotImplementedError(f"Optimizer {cfg.optimizer} not implemented.")

        if cfg.criterion == 'BCE':
            criterion = BCELoss()
        elif cfg.criterion == 'BCEWithLogits':
            criterion = BCEWithLogitsLoss()
        else:
            raise NotImplementedError(f"Criterion {cfg.criterion} not implemented.")


        # --- Training & Validation Loop ---
        step = 0
        best_val_loss = float('inf')
        ema_loss = RunningLoss(mode='ema', ema_alpha=0.98)
        sma_loss = RunningLoss(mode='sma', window_size=50)
        cum_loss = RunningLoss(mode='cumulative')
        
        for epoch in range(cfg.epochs):
            model.set_mode(train=True)
            epoch_train_loss = 0
            num_train_samples = 0
            
            cum_loss.reset()
            ema_loss.reset()
            sma_loss.reset()

            for graphs, kgs, images, texts, targets in train_loader:
                outputs = model(
                    image=images,
                    text_str=texts,
                    claim_data=graphs, evidence_data=kgs
                )
                if torch.isnan(outputs).any():
                    logger.error("NaN in outputs!")
                if torch.isnan(targets).any():
                    logger.error("NaN in targets!")


                _targets = targets
                if only_neuron_nb_X > 0:
                    _targets = targets[:, only_neuron_nb_X - 1].unsqueeze(1)
                batch_loss = compute_loss(outputs, _targets, criterion)

                if batch_loss is not None:
                    optimizer.zero_grad()
                    batch_loss.backward()
                    if cfg.gradient_clipping > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    
                    bloss_item = batch_loss.detach().cpu().item()
                    epoch_train_loss += bloss_item * len(graphs)
                    num_train_samples += len(graphs)

                    ema_val = ema_loss.update(bloss_item)
                    sma_val = sma_loss.update(bloss_item)
                    cum_val = cum_loss.update(bloss_item, n=len(graphs)) 

                    if wandblog and step % CONFIG.logging.frequency_log == 0:
                        wandb.log({
                            "train/batch_loss": bloss_item,
                            "train/ema_loss": ema_val,
                            "train/sma_loss": sma_val,
                            "train/cum_loss": cum_val,
                            "train/step": step,
                            "train/epoch": epoch
                        }, step=step)

                del outputs, batch_loss
                
                step += 1
            
            # --- Epoch End: Validation and Checkpointing ---
            avg_epoch_train_loss = epoch_train_loss / num_train_samples if num_train_samples > 0 else float('inf')
            val_loss = avg_epoch_train_loss # Default to train loss if no validation set

            if use_validation_set and val_loader is not None:
                val_loss = validate(model, val_loader, criterion, only_neuron_nb_X=only_neuron_nb_X)
                logger.info(f"Epoch {epoch+1}/{cfg.epochs} | Train Loss: {avg_epoch_train_loss:.4f} | Val Loss: {val_loss:.4f}")
            else:
                logger.info(f"Epoch {epoch+1}/{cfg.epochs} | Train Loss: {avg_epoch_train_loss:.4f} (No validation set used)")

            if wandblog:
                log_dict = {
                        "train/epoch_loss": avg_epoch_train_loss,
                        "train/epoch": epoch
                }
                if use_validation_set:
                    log_dict["val/loss"] = val_loss
                wandb.log(log_dict, step=step)

            # Save the model if validation loss improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    "epoch": epoch,
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_val_loss,
                    "hyperparameters": {
                        **trial.params,
                        "use_deepfake_detector": cfg.use_deepfake_detector,
                        "use_clip": cfg.use_clip,
                        "use_encyclopedic_knowledge": cfg.use_encyclopedic_knowledge,
                        "clip_use_image_features": cfg.clip_use_image_features,
                        "clip_use_text_features": cfg.clip_use_text_features,
                        "finetune": finetune,
                        "only_neuron_nb_X": only_neuron_nb_X,
                        "stratified_dataset": stratified_dataset,
                        'trained_dataset':dataset
                    }
                }
                model_name = f"{study_name}_neuron{only_neuron_nb_X}.pt"
                os.makedirs(os.path.join(MODELS_DIR, f'train_{dataset}'), exist_ok=True)
                model_filename = os.path.join(MODELS_DIR, f'train_{dataset}', model_name)
                torch.save(checkpoint, model_filename)
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
        pruner=optuna.pruners.MedianPruner(),
        storage=storage_name,
        load_if_exists=force_study_name
    )
    study.optimize(objective, n_trials=n_trials)

    logger.info(f"Study finished. Best trial: {study.best_trial.number}")
    logger.info(f"  Value (min val_loss): {study.best_trial.value:.4f}")
    logger.info("  Params: ")
    for key, value in study.best_trial.params.items():
        logger.info(f"    {key}: {value}")

    
def compute_loss(outputs, targets, criterion):
    return criterion(outputs, targets)


def suggest_ablation_and_hparams(cfg, trial, use_optuna_params: bool = True, only_neuron_nb_X: int = 0):
    # Hyperparameters
    if use_optuna_params:
        cfg.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        cfg.batch_size = trial.suggest_categorical("batch_size", [2, 4, 8])
    else:
        # Use fixed parameters from CONFIG if not using Optuna
        cfg.learning_rate = CONFIG.training.learning_rate
        cfg.batch_size = CONFIG.training.batch_size

    # --- Ablation flags ---
    if use_optuna_params:
        # Clip features
        clip_image = trial.suggest_categorical("clip_use_image_features", [True, False])
        clip_text = trial.suggest_categorical("clip_use_text_features", [True, False])

        # DeepFake detector and ClaimVerifier
        deepfake = trial.suggest_categorical("use_deepfake_detector", [True, False])
        claimverifier = trial.suggest_categorical("use_encyclopedic_knowledge", [True, False])

        # Ensure at least one CLIP feature is True if clip is enabled
        if not (clip_image or clip_text):
            # force one of them to be True randomly
            if trial.suggest_float("force_clip_choice", 0, 1) < 0.5:
                clip_image = True
            else:
                clip_text = True

        # Ensure at least one feature extractor is enabled
        if not (deepfake or claimverifier or clip_image or clip_text):
            # randomly enable one
            options = ["deepfake", "claimverifier", "clip_image", "clip_text"]
            choice = trial.suggest_categorical("force_one_feature", options)
            if choice == "deepfake":
                deepfake = True
            elif choice == "claimverifier":
                claimverifier = True
            elif choice == "clip_image":
                clip_image = True
            else:
                clip_text = True

        # Set to config
        cfg.use_deepfake_detector = deepfake
        cfg.use_encyclopedic_knowledge = claimverifier
        cfg.use_clip = clip_image or clip_text
        cfg.clip_use_image_features = clip_image
        cfg.clip_use_text_features = clip_text

        dropout = trial.suggest_float("dropout", 0.0, 0.5)

        options = [
            "input-1024-512-256-4",
            "input-512-256-4",
            "input-256-4",
            "input-128-4"
        ]
        layer_choice = trial.suggest_categorical("global_layer_sizes", options)

    else:
        # Use fixed ablation flags and dropout from CONFIG
        deepfake = CONFIG.training.use_deepfake_detector
        claimverifier = CONFIG.training.use_encyclopedic_knowledge
        clip_image = CONFIG.training.clip_use_image_features
        clip_text = CONFIG.training.clip_use_text_features

        cfg.use_deepfake_detector = deepfake
        cfg.use_encyclopedic_knowledge = claimverifier
        cfg.use_clip = clip_image or clip_text
        cfg.clip_use_image_features = clip_image
        cfg.clip_use_text_features = clip_text

        dropout = CONFIG.training.dropout
        layer_choice = CONFIG.training.global_layer_sizes

    # compute classifier input dim based on ablations
    classifier_input_dim = 0
    if cfg.use_deepfake_detector:
        classifier_input_dim += CONFIG.model.fakedetector_feature_dim
    if cfg.use_clip:
        if cfg.clip_use_image_features:
            classifier_input_dim += CONFIG.model.clip_image_dim
        if cfg.clip_use_text_features:
            classifier_input_dim += CONFIG.model.clip_text_dim
    if cfg.use_encyclopedic_knowledge:
        classifier_input_dim += CONFIG.model.claimverifier_hidden_dims*2 + CONFIG.model.claimverifier_hidden_dims // 2
    
    global_layer_sizes = [int(x) if x != "input" else classifier_input_dim for x in layer_choice.split("-")]

    if only_neuron_nb_X > 0:
        if global_layer_sizes[-1] == 4:
            global_layer_sizes[-1] = 1

    return cfg, global_layer_sizes, dropout, classifier_input_dim


if __name__ == "__main__":
    app()
