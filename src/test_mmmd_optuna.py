import os
import torch
import wandb
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from loguru import logger
from torchvision import transforms
import optuna
import numpy as np
from omegaconf import OmegaConf

from corpus_truth_manipulation.dataset import create_globaldataset_loader
from corpus_truth_manipulation.config import CONFIG, MODELS_DIR, MMFAKEBENCH_GRAPHS, XFACTA_GRAPHS, XFACTA_FORMATTED
from src.models import MultimodalMisinformationDetector

def compute_metrics(y_true, y_pred, y_prob):
    metrics = {}
    for i in range(y_true.shape[1]):
        metrics[f'output_{i+1}'] = {
            'accuracy': accuracy_score(y_true[:, i], y_pred[:, i]),
            'precision': precision_score(y_true[:, i], y_pred[:, i], zero_division=0),
            'recall': recall_score(y_true[:, i], y_pred[:, i], zero_division=0),
            'f1': f1_score(y_true[:, i], y_pred[:, i], zero_division=0),
            'roc_auc': roc_auc_score(y_true[:, i], y_prob[:, i]) if len(set(y_true[:, i])) > 1 else None
        }
    return metrics

def get_model_config_from_trial(params, config_model):
    clip_image = params.get("clip_use_image_features", False)
    clip_text = params.get("clip_use_text_features", False)
    deepfake = params.get("use_deepfake_detector", False)
    claimverifier = params.get("use_encyclopedic_knowledge", False)

    # This logic is in train_mmmd.py. If clip_use_image_features and clip_use_text_features are both false
    # one is forced to be true based on "force_clip_choice"
    if not (clip_image or clip_text):
        # param might not exist in old studies, default to 0.0 so clip_image=True
        if params.get("force_clip_choice", 0.0) < 0.5:
            clip_image = True
        else:
            clip_text = True
    
    # This is in train_mmmd.py, ensure at least one feature extractor is enabled
    if not (deepfake or claimverifier or clip_image or clip_text):
        choice = params.get("force_one_feature")
        if choice == "deepfake":
            deepfake = True
        elif choice == "claimverifier":
            claimverifier = True
        elif choice == "clip_image":
            clip_image = True
        else:
            clip_text = True
            
    if "use_clip" in params:
        use_clip = params["use_clip"]
    else:
        use_clip = clip_image or clip_text

    classifier_input_dim = 0
    if deepfake:
        classifier_input_dim += config_model.fakedetector_feature_dim
    if use_clip:
        if clip_image:
            classifier_input_dim += config_model.clip_image_dim
        if clip_text:
            classifier_input_dim += config_model.clip_text_dim
    if claimverifier:
        classifier_input_dim += config_model.claimverifier_output_dim

    dropout = params["dropout"]
    
    global_layer_sizes_str = params["global_layer_sizes"]
    global_layer_sizes = [int(x) if x != "input" else classifier_input_dim for x in global_layer_sizes_str.split("-")]

    model_params = {
        "classifier_input_dim": classifier_input_dim,
        "use_deepfake_detector": deepfake,
        "use_clip": use_clip,
        "use_encyclopedic_knowledge": claimverifier,
        "clip_use_image_features": clip_image,
        "clip_use_text_features": clip_text,
        "layer_sizes": global_layer_sizes,
        "dropout": dropout,
    }
    return model_params

def test_trial(trial, pretrained_obj, study_name, dataset_folder, device, test_loader, job_id):
    hp = trial.params
    model_checkpoint_path = os.path.join(MODELS_DIR, f"{study_name}_trial_{trial.number}_best.pth")
    if not os.path.exists(model_checkpoint_path):
        logger.warning(f"Checkpoint for trial {trial.number} not found at {model_checkpoint_path}, skipping.")
        return

    api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=api_key)

    run = wandb.init(
        project=os.getenv("WANDB_PROJECT"), 
        config=OmegaConf.to_container(CONFIG, resolve=True),
        group=f"{study_name}_test",
        name=f"trial-{trial.number}-{job_id}",
        reinit=True
    )
    wandb.config.update(hp)
    wandb.config.update({"trial_number": trial.number})

    pretrained_hyperparameters = pretrained_obj['hyperparameters']

    model_constructor_params = get_model_config_from_trial(hp, CONFIG.model)

    model = MultimodalMisinformationDetector(
        **model_constructor_params,
        ency_int_dims=pretrained_hyperparameters['int_dims'],
        ency_hidden_dim=pretrained_hyperparameters['hidden_dim'],
        activation_sigmoid=False
    ).to(device)

    checkpoint = torch.load(model_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_targets = []
    all_outputs = []
    all_probs = []

    with torch.no_grad():
        for l_item in tqdm(test_loader, desc=f"Testing trial {trial.number}", leave=False):
            graphs, kgs, images, texts, targets = l_item
            outputs = model(
                image=images,
                text_str=texts,
                claim_data=graphs, evidence_data=kgs
            )

            if torch.isnan(outputs).any():
                logger.error("NaN in outputs!")
            if torch.isnan(targets).any():
                logger.error("NaN in targets!")

            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_targets.append(targets.cpu().numpy())
            all_outputs.append(preds)
            all_probs.append(probs)

    y_true = np.concatenate(all_targets, axis=0)
    y_pred = np.concatenate(all_outputs, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)

    metrics = compute_metrics(y_true, y_pred, y_prob)
    for out, vals in metrics.items():
        logger.info(f"Trial {trial.number} - {out}: {vals}")

    wandb.log({
        'test_metrics': metrics,
        'dataset': str(dataset_folder),
    })
    run.finish()

def main(study_name, pretrained_encyc, dataset_folder, device, job_id):
    if dataset_folder == 'MMFAKEBENCH':
        dataset_folder_path = MMFAKEBENCH_GRAPHS
        root_to_image = CONFIG.MMFAKEBENCH
    elif dataset_folder == 'XFACTA':
        dataset_folder_path = XFACTA_GRAPHS
        root_to_image = XFACTA_FORMATTED
    else:
        raise ValueError(f'Unknown dataset folder <{dataset_folder}>.')

    # --- Torch config ---
    cuda_available = torch.cuda.is_available()
    num_gpus = torch.cuda.device_count()
    gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
    logger.info(f"CUDA available: {cuda_available}")
    logger.info(f"Number of GPUs: {num_gpus}")
    logger.info(f"GPU names: {gpu_names}")

    if cuda_available:
        CONFIG.device = "cuda:0" if device.lower() != 'cpu' else 'cpu'
        _device_name = torch.cuda.get_device_name(0) if device.lower() != 'cpu' else 'CPU'
    else:
        CONFIG.device = 'cpu'
        _device_name = 'CPU'
        logger.warning("CUDA is not available. Using CPU.")
    logger.info(f"🧠 Using device {CONFIG.device} (device name: {_device_name})")
    torch.manual_seed(CONFIG.seed)
    # --------------------

    storage_name = f"sqlite:///reports/training_optuna/{study_name}.db"
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_name)
    except (KeyError, ValueError):
        logger.error(f"Study '{study_name}' not found in '{storage_name}'. Please check the study name.")
        return

    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    logger.info(f"Found {len(trials)} completed trials in study '{study_name}'.")

    pretrained_obj = torch.load(pretrained_encyc, map_location=device, weights_only=False)
    
    image_transform = transforms.Compose([
        transforms.Resize((CONFIG.data.image_size, CONFIG.data.image_size)),
    ])
    test_loader = create_globaldataset_loader(
        device=device,
        split='test',
        image_transform=image_transform,
        root_to_image=root_to_image,
        input_folder=dataset_folder_path,
        shuffle=False,
        max_samples=None,
        num_workers=CONFIG.training.num_workers,
        batch_size=CONFIG.training.batch_size,
    )

    for trial in tqdm(trials, desc="Testing trials"):
        test_trial(
            trial=trial,
            pretrained_obj=pretrained_obj,
            study_name=study_name,
            dataset_folder=dataset_folder,
            device=device,
            test_loader=test_loader,
            job_id=job_id
        )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--studyname', type=str, required=True)
    parser.add_argument('--pretrainedency', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--dataset_folder', type=str, default='MMFAKEBENCH', help="Dataset folder to use: 'MMFAKEBENCH' or 'XFACTA'")
    parser.add_argument('--jobid',required=True, type=str, help="Job identifier for logging purposes")
    args = parser.parse_args()
    
    main(
        study_name=args.studyname,
        pretrained_encyc=args.pretrainedency,
        dataset_folder=args.dataset_folder,
        device=args.device,
        job_id=args.jobid
    )
