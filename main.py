import sys
import os
import argparse
import json
import logging

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import ContrastiveEEGDataset, SupervisedEEGDataset
from utils import load_eeg_data, validate_config, set_seed, setup_logging, setup_tensorboard, get_tensorboard_logger, close_tensorboard
from models import SimpleSleepNet, SleepStageClassifier
from training import train_contrastive_model, train_classifier
from evaluation import LatentSpaceEvaluator, get_predictions, ResultsSaver
from augmentations import load_augmentations_from_config
from experiment_tracker import start_training_timer, end_training_timer, set_phase, save_experiment  # ← AGGIUNTO

def suppress_warnings():
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

suppress_warnings()

NUM_CLASSES = 5

def parse_args():
    parser = argparse.ArgumentParser(description='Sleep Stage Classification')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default/config.json',
        help='Path to the config file. Example: configs/experiment1/config1.json'
    )
    parser.add_argument(
        '--list-configs',
        action='store_true',
        help='List all available configuration files and exit.'
    )
    return parser.parse_args()

def load_config(config_path):
    if not os.path.isfile(config_path):
        logging.error(f"Configuration file '{config_path}' not found.")
        sys.exit(1)
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from the config file: {e}")
        sys.exit(1)

def list_available_configs(configs_dir='configs'):
    print("Available configuration files:")
    for root, dirs, files in os.walk(configs_dir):
        for file in files:
            if file.endswith('.json'):
                config_path = os.path.join(root, file)
                print(config_path)

def setup_environment(config):
    setup_logging(log_level=logging.INFO, log_file=f'logs/experiment_{config["experiment_num"]}.log')
    logger = logging.getLogger(__name__)
    logger.info("Starting the EEG Project")
    
    set_seed(config["seed"])
    logger.info(f"Random seed set to {config['seed']}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    setup_tensorboard(log_dir=f'runs/experiment_{config["experiment_num"]}')
    tensorboard_logger = get_tensorboard_logger()
    logger.info(f"TensorBoard logging initialized at: runs/experiment_{config['experiment_num']}")
    
    return logger, device, tensorboard_logger

def prepare_datasets(config, logger):
    BATCH_SIZE = config["pretraining_params"]["batch_size"]
    NUM_WORKERS = config["num_workers"]
    eeg_data = load_eeg_data(dataset_path=config['dataset']['dset_path'], num_files_to_process=config['dataset']['max_files'])
    logger.info("Loaded train and test sets of EEG data")

    train_dataset = SupervisedEEGDataset(eeg_data['train'])
    test_dataset = SupervisedEEGDataset(eeg_data['test'])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    logger.info("Supervised datasets and dataloaders created.")
    return eeg_data, train_loader, test_loader

def pretrain_contrastive_model(config, eeg_data, device, logger, tensorboard_logger):
    BATCH_SIZE = config["pretraining_params"]["batch_size"]
    LATENT_DIM = config["pretraining_params"]["latent_dim"]
    DROP_PROB = config["pretraining_params"]["dropout_rate"]
    NUM_WORKERS = config["num_workers"]
    TEMP = config["pretraining_params"]["temperature"]
    augmentations = load_augmentations_from_config(config=config)

    train_contrastive_dataset = ContrastiveEEGDataset(eeg_data['train'], augmentations=augmentations)
    train_contrastive_loader = DataLoader(train_contrastive_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    logger.info(f"Contrastive train dataset created with {len(train_contrastive_dataset)} samples")

    val_contrastive_dataset = ContrastiveEEGDataset(eeg_data['test'], augmentations=augmentations)
    val_contrastive_loader = DataLoader(val_contrastive_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    logger.info(f"Contrastive test dataset created with {len(val_contrastive_dataset)} samples")

    encoder = SimpleSleepNet(latent_dim=LATENT_DIM, dropout=DROP_PROB).to(device)
    
    sample_input = torch.zeros(1, 1, 3000).to(device)
    tensorboard_logger.add_graph(encoder, sample_input)
    logger.info(f"Model created with {sum(p.numel() for p in encoder.parameters() if p.requires_grad)} trainable parameters")

    contrastive_optimizer = optim.Adam(encoder.parameters(), lr=config["pretraining_params"]["learning_rate"])
    best_encoder_pth = f"{config['pretraining_params']['best_model_pth']}{config['experiment_num']}.pth"

    set_phase("contrastive")  # ← AGGIUNTO: avvia timer contrastive

    train_contrastive_model(
        model=encoder,
        dataloader=train_contrastive_loader,
        optimizer=contrastive_optimizer,
        device=device,
        num_epochs=config["pretraining_params"]["max_epochs"],
        temperature=TEMP,
        val_dataloader=val_contrastive_loader,
        check_interval=config["pretraining_params"]["check_interval"],
        min_improvement=config["pretraining_params"]["min_improvement"],
        best_model_path=best_encoder_pth
    )
    logger.info("Contrastive training complete")

    try:
        encoder.load_state_dict(torch.load(best_encoder_pth))
        logger.info("Loaded best encoder from %s", best_encoder_pth)
    except Exception as e:
        logger.error("Error loading best encoder: %s", str(e))
        raise

    return encoder

def evaluate_latent_space(config, encoder, eeg_data, device, logger):
    BATCH_SIZE = config["pretraining_params"]["batch_size"]
    NUM_WORKERS = config["num_workers"]
    visualization_dataset = ContrastiveEEGDataset(eeg_signals=eeg_data['test'], augmentations=[], return_labels=True)
    visualization_loader = DataLoader(visualization_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    evaluator = LatentSpaceEvaluator(
        model=encoder,
        dataloader=visualization_loader,
        device=device,
        umap_enabled=config["latent_space_params"]["umap_enabled"],
        pca_enabled=config["latent_space_params"]["pca_enabled"],
        tsne_enabled=config["latent_space_params"]["tsne_enabled"],
        visualize=config["latent_space_params"]["visualize"],
        compute_metrics=config["latent_space_params"]["compute_metrics"],
        n_clusters=config["latent_space_params"]["n_clusters"],
        output_image_dir=config["latent_space_params"]["output_image_dir"],
        output_metrics_dir=config["latent_space_params"]["output_metrics_dir"],
        experiment_num=config["experiment_num"],
        visualization_fraction=config["latent_space_params"]["visualization_fraction"]
    )
    evaluator.run()
    logger.info("Latent space evaluation complete")

def train_supervised_classifier(config, encoder, train_loader, test_loader, device, logger, tensorboard_logger):
    LATENT_DIM = config["pretraining_params"]["latent_dim"]
    DROP_PROB = config["sup_training_params"]["dropout_rate"]
    classifier = SleepStageClassifier(input_dim=LATENT_DIM, num_classes=NUM_CLASSES, dropout_probs=DROP_PROB).to(device)
    
    sample_input = torch.zeros(1, LATENT_DIM).to(device)
    tensorboard_logger.add_graph(classifier, sample_input)
    
    criterion = nn.CrossEntropyLoss()
    supervised_optimizer = optim.Adam(classifier.parameters(), lr=config["sup_training_params"]["learning_rate"])
    logger.info(f"Classifier created with {sum(p.numel() for p in classifier.parameters() if p.requires_grad)} trainable parameters")

    for param in encoder.parameters():
        param.requires_grad = False
    logger.info("Encoder frozen")

    best_classifier_pth = config["sup_training_params"]["best_model_pth"] + str(config["experiment_num"]) + ".pth"

    set_phase("classifier")  # ← AGGIUNTO: ferma timer contrastive, avvia timer classifier

    train_classifier(
        encoder=encoder,
        classifier=classifier,
        train_loader=train_loader,
        val_loader=test_loader,
        criterion=criterion,
        optimizer=supervised_optimizer,
        num_epochs=config["sup_training_params"]["max_epochs"],
        device=device,
        save_path=best_classifier_pth,
        check_interval=config["sup_training_params"]["check_interval"],
        min_improvement=config["sup_training_params"]["min_improvement"]
    )
    logger.info("Classifier training complete")
    return classifier, best_classifier_pth

def test_and_save_results(config, encoder, classifier, test_loader, device, logger):
    best_classifier_pth = f"{config['sup_training_params']['best_model_pth']}{config['experiment_num']}.pth"
    classifier.load_state_dict(torch.load(best_classifier_pth))
    
    predictions, true_labels = get_predictions(encoder, classifier, test_loader, device=device)
    
    results_saver = ResultsSaver(
        results_folder=config["results_folder"],
        experiment_num=config["experiment_num"]
    )
    results_saver.save_classification_results(
        predictions=predictions,
        true_labels=true_labels,
        num_classes=NUM_CLASSES
    )
    logger.info("Classification results saved")

def main():
    args = parse_args()

    if args.list_configs:
        list_available_configs()
        sys.exit(0)

    config = load_config(args.config)
    validate_config(config)

    logger, device, tensorboard_logger = setup_environment(config)
    eeg_data, train_loader, test_loader = prepare_datasets(config, logger)

    start_training_timer()  # ← AGGIUNTO: avvia timer totale

    encoder = pretrain_contrastive_model(config, eeg_data, device, logger, tensorboard_logger)
    evaluate_latent_space(config, encoder, eeg_data, device, logger)
    classifier, _ = train_supervised_classifier(config, encoder, train_loader, test_loader, device, logger, tensorboard_logger)
    test_and_save_results(config, encoder, classifier, test_loader, device, logger)

    set_phase("end")                          # ← AGGIUNTO: ferma timer classifier
    train_duration = end_training_timer()     # ← AGGIUNTO: ferma timer totale
    save_experiment(train_duration)           # ← AGGIUNTO: salva tutto su Drive

    logger.info("Experiment complete")
    close_tensorboard()

if __name__ == "__main__":
    main()
