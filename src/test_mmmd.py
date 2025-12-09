import os
import json
import numpy as np
import torch
import wandb
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from loguru import logger
from torchvision import transforms
from corpus_truth_manipulation.dataset import create_globaldataset_loader
from corpus_truth_manipulation.config import CONFIG, MODELS_DIR, MMFAKEBENCH_GRAPHS, XFACTA_GRAPHS, XFACTA_FORMATTED, MMFAKEBENCH, COSMOS_GRAPHS, COSMOS_FORMATTED, MMFAKEBENCH_GRAPHS_STRATIFIED
from src.models import MultimodalMisinformationDetector
from omegaconf import OmegaConf
from src.train_mmmd import get_dataset_folders

def compute_metrics(y_true, y_pred, y_prob, only_neuron_nb_X=0):
    metrics = {}
    for i in range(y_true.shape[1]):
        output_name = f'output_{i+1}'
        if y_true.shape[1] == 1 and only_neuron_nb_X > 0:
            output_name = f'output_{only_neuron_nb_X}'
        
        metrics[output_name] = {
            'accuracy': accuracy_score(y_true[:, i], y_pred[:, i]),
            'precision': precision_score(y_true[:, i], y_pred[:, i], zero_division=0),
            'recall': recall_score(y_true[:, i], y_pred[:, i], zero_division=0),
            'f1': f1_score(y_true[:, i], y_pred[:, i], zero_division=0),
            'roc_auc': roc_auc_score(y_true[:, i], y_prob[:, i]) if len(set(y_true[:, i])) > 1 else None
        }
    return metrics

def test(model_checkpoint, study_name, ablation_name, dataset_folder, jobid, device='cpu', only_neuron_nb_X:int=0):
    root_to_image, data_root, stratified_dataset = get_dataset_folders(dataset_folder)

    if only_neuron_nb_X > 0:
        model_checkpoint += f"_neuron{only_neuron_nb_X}"
    model_checkpoint += f"_train{dataset_folder}"
    model_checkpoint += f"_neuron{only_neuron_nb_X}"
    model_checkpoint += '.pt'

    study_name += dataset_folder
    if only_neuron_nb_X > 0:
        study_name += f"_neuron{only_neuron_nb_X}"
    logger.info(f"Testing model from checkpoint: train_{dataset_folder}/{model_checkpoint} on dataset: {data_root}")
    logger.info(f"Using ablation name: {ablation_name}, jobid: {jobid}, study name: {study_name}")

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

    api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=api_key)

    # Load pretrained encyclopedia
    pretrained_hyperparameters = {
        'int_dims': CONFIG.model.claimverifier_int_dims,
        'hidden_dim': CONFIG.model.claimverifier_hidden_dims
    }

    # Load checkpoint
    checkpoint = torch.load(os.path.join("models", f'train_{dataset_folder}', model_checkpoint), map_location=CONFIG.device)
    logger.info(f"Loaded checkpoint from {model_checkpoint}, with keys: {list(checkpoint.keys())}")
    hp = checkpoint['hyperparameters']
    logger.debug(f"Loaded checkpoint hyperparameters: {hp}")
    
    global_layer_sizes = get_classifier_size(hp, only_neuron_nb_X)
    model = MultimodalMisinformationDetector(
        ency_int_dims=pretrained_hyperparameters['int_dims'],
        ency_hidden_dim=pretrained_hyperparameters['hidden_dim'],
        layer_sizes=global_layer_sizes,
        dropout=CONFIG.training.dropout,
        activation_sigmoid=False,
        use_encyclopedic_knowledge=hp['use_encyclopedic_knowledge'],
        clip_use_image_features=hp['clip_use_image_features'],
        clip_use_text_features=hp['clip_use_text_features'],
        use_clip=hp['use_clip'],
        use_deepfake_detector=hp['use_deepfake_detector'],
    ).to(CONFIG.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    image_transform = transforms.Compose([
        transforms.Resize((CONFIG.data.image_size, CONFIG.data.image_size)),
    ])

    test_loader = create_globaldataset_loader(
        device=CONFIG.device,
        split='test',
        image_transform=image_transform,
        root_to_image=root_to_image,
        input_folder=data_root,
        shuffle=False,
        max_samples=None,
        num_workers=CONFIG.training.num_workers,
        batch_size=CONFIG.training.batch_size,
    )


    wandb_name = f"{ablation_name}_{jobid}"
    if hp['use_encyclopedic_knowledge']:
        wandb_name += '_encyclo'
    if hp['use_deepfake_detector']:
        wandb_name += '_deepfake'
    if hp['use_clip']:
        if hp['clip_use_image_features']:
            wandb_name += '_clipimg'
        if hp['clip_use_text_features']:
            wandb_name += '_cliptext'
    if hp['finetune']:
        wandb_name += '_finetune'
    if only_neuron_nb_X > 0:
        wandb_name += f'_neuron{only_neuron_nb_X}'
    wandb_name += '_test'
    tags = ['test', 'sample_log', f'dataset_{dataset_folder}']
    if only_neuron_nb_X > 0:
        tags.append(f'only_neuron_{only_neuron_nb_X}')
    if 'trained_dataset' in hp:
        tags.append(f"train_{hp['trained_dataset']}")
        study_name += f"train_{hp['trained_dataset']}"
    run = wandb.init(
        project=os.getenv("WANDB_PROJECT"), 
        config=OmegaConf.to_container(CONFIG, resolve=True),
        group=study_name,
        name=wandb_name,
        reinit=True,
        tags=tags
    )
    cfg = CONFIG.training
    wandb.config.update({
        "learning_rate": cfg.learning_rate,
        "batch_size": cfg.batch_size,
        "int_dims": CONFIG.model.claimverifier_int_dims,
        "hidden_dim": CONFIG.model.claimverifier_hidden_dims,
        "global_layer_sizes": global_layer_sizes,
        "dropout": cfg.dropout,
        "use_deepfake_detector": hp['use_deepfake_detector'],
        "use_clip": hp['use_clip'],
        "use_clip_image_features": hp['clip_use_image_features'],
        "use_clip_text_features": hp['clip_use_text_features'],
        "use_encyclopedic_knowledge": hp['use_encyclopedic_knowledge'],
        "finetune": hp['finetune'],
        "only_neuron_nb_X": only_neuron_nb_X,
    })
    all_targets = []
    all_outputs = []
    all_probs = []

    with torch.no_grad():
        for l_item in tqdm(test_loader, desc="Testing"):
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
            
            _targets = targets.cpu().numpy()
            if only_neuron_nb_X > 0:
                _targets = _targets[:, only_neuron_nb_X - 1].reshape(-1, 1)

            all_targets.append(_targets)
            all_outputs.append(preds)
            all_probs.append(probs)

    y_true = np.concatenate(all_targets, axis=0)
    y_pred = np.concatenate(all_outputs, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)


    # ---------------------------------------------------------------
    #           Save per-sample results with metadata
    # ---------------------------------------------------------------
    dataset = test_loader.dataset
    all_metadatas = [sample['metadata'] for sample in dataset.data]

    if len(y_true) != len(all_metadatas):
        logger.warning(f"Mismatch in number of samples and metadata: {len(y_true)} vs {len(all_metadatas)}")
        min_len = min(len(y_true), len(all_metadatas))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        y_prob = y_prob[:min_len]
        all_metadatas = all_metadatas[:min_len]

    results_per_sample = []
    for i in range(len(y_true)):
        serializable_metadata = {k: str(v) for k, v in all_metadatas[i].items()}
        results_per_sample.append({
            'metadata': serializable_metadata,
            'y_true': y_true[i].tolist(),
            'y_pred': y_pred[i].tolist(),
            'y_prob': y_prob[i].tolist(),
        })
    
    results_filename = f"test_results_{jobid}.json"
    with open(results_filename, 'w') as f:
        json.dump(results_per_sample, f, indent=4)
    
    wandb.save(results_filename)

    # Create a wandb.Table for detailed analysis
    if only_neuron_nb_X > 0:
        columns = [
            "image_path", "image_source", "fake_cls", "text_raw", "text_source",
            f"y_true_output_{only_neuron_nb_X}", f"y_pred_output_{only_neuron_nb_X}", f"y_prob_output_{only_neuron_nb_X}"
        ]
        table_data = []
        for i in range(len(y_true)):
            metadata = all_metadatas[i]
            row = [
                metadata.get('image_path', 'N/A'),
                metadata.get('image_source', 'N/A'),
                metadata.get('fake_cls', 'N/A'),
                metadata.get('text_raw', 'N/A'),
                metadata.get('text_source', 'N/A'),
                y_true[i][0], y_pred[i][0], y_prob[i][0]
            ]
            table_data.append(row)
    else:
        columns = [
            "image_path", "image_source", "fake_cls", "text_raw", "text_source",
            "y_true_img_real", "y_true_claim_real", "y_true_mismatch", "y_true_overall",
            "y_pred_img_real", "y_pred_claim_real", "y_pred_mismatch", "y_pred_overall",
            "y_prob_img_real", "y_prob_claim_real", "y_prob_mismatch", "y_prob_overall"
        ]
        table_data = []
        for i in range(len(y_true)):
            metadata = all_metadatas[i]
            row = [
                metadata.get('image_path', 'N/A'),
                metadata.get('image_source', 'N/A'),
                metadata.get('fake_cls', 'N/A'),
                metadata.get('text_raw', 'N/A'),
                metadata.get('text_source', 'N/A'),
                y_true[i][0], y_true[i][1], y_true[i][2], y_true[i][3],
                y_pred[i][0], y_pred[i][1], y_pred[i][2], y_pred[i][3],
                y_prob[i][0], y_prob[i][1], y_prob[i][2], y_prob[i][3],
            ]
            table_data.append(row)
    
    results_table = wandb.Table(data=table_data, columns=columns)
    
    metrics = compute_metrics(y_true, y_pred, y_prob, only_neuron_nb_X=only_neuron_nb_X)
    for out, vals in metrics.items():
        logger.info(f"{out}: {vals}")

    wandb.log({
        'test_metrics': metrics,
        'test_predictions': results_table,
        'dataset': str(data_root),
    })
    run.finish()


def get_classifier_size(hp:dict, only_neuron_nb_X: int = 0):
    """
    Dtermines the input size of the global classifier based on hyperparameters,
    and returns the full list of layer sizes.
    """
    layers = hp.get('global_layer_sizes', CONFIG.training.global_layer_sizes)
    classifier_input_dim = 0
    if hp['use_deepfake_detector']:
        classifier_input_dim += CONFIG.model.fakedetector_feature_dim
    if hp['use_clip']:
        if hp['clip_use_image_features']:
            classifier_input_dim += CONFIG.model.clip_image_dim
        if hp['clip_use_text_features']:
            classifier_input_dim += CONFIG.model.clip_text_dim
    if hp['use_encyclopedic_knowledge']:
        classifier_input_dim += CONFIG.model.claimverifier_hidden_dims*2 + CONFIG.model.claimverifier_hidden_dims // 2

    global_layer_sizes = [int(x) if x != "input" else classifier_input_dim for x in layers.split("-")]
    
    if only_neuron_nb_X > 0:
        if global_layer_sizes[-1] == 4:
            global_layer_sizes[-1] = 1

    return global_layer_sizes

if __name__ == "__main__":
    import argparse
    import numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--studyname', type=str, required=True)
    parser.add_argument('--ablationname', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--dataset_folder', type=str, default='MMFAKEBENCH', help="Dataset folder to use: 'MMFAKEBENCH' or 'XFACTA' or 'COSMOS' or MMFAKEBENCH_STRATIFIED")
    parser.add_argument('--jobid',required=True, type=str, help="Job identifier for logging purposes")
    parser.add_argument('--only_neuron_nb_X', type=int, default=0, help="If set to a value from 1 to 4, test only on that specific output neuron. 0 means test on all 4.")
    args = parser.parse_args()
    test(
        args.checkpoint, 
        dataset_folder=args.dataset_folder,
        study_name=args.studyname,
        ablation_name=args.ablationname,
        jobid=args.jobid,
        device=args.device,
        only_neuron_nb_X=args.only_neuron_nb_X
    )

