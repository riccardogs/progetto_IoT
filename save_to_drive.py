"""
Script per salvare i risultati degli esperimenti su Google Drive.
Usare DOPO aver completato un esperimento.

Usage:
    from google.colab import drive
    drive.mount('/content/drive')
    
    !python save_to_drive.py --experiment-num auto
"""

import os
import shutil
import argparse
import json
import re
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Save experiment results to Google Drive')
    parser.add_argument(
        '--experiment-num',
        type=str,
        default='auto',
        help='Experiment number (e.g., 10001). Use "auto" to read from config.json'
    )
    parser.add_argument(
        '--source-folder',
        type=str,
        default='.',
        help='Source folder containing results/, checkpoints/, runs/, logs/ (default: current directory)'
    )
    parser.add_argument(
        '--drive-root',
        type=str,
        default='/content/drive/MyDrive/eeg_sleep_experiments',
        help='Root folder on Google Drive for storing experiments'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force overwrite existing experiment folder on Drive'
    )
    parser.add_argument(
        '--no-notes',
        action='store_true',
        help='Skip asking for experiment notes (use default empty notes)'
    )
    return parser.parse_args()

def get_experiment_num_from_config(source_folder):
    """Legge il numero esperimento dal file config.json"""
    config_path = os.path.join(source_folder, 'configs', 'default', 'config.json')
    
    alt_paths = [
        os.path.join(source_folder, 'config.json'),
        os.path.join(source_folder, 'configs', 'config.json')
    ]
    
    for path in [config_path] + alt_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    config = json.load(f)
                    return str(config.get('experiment_num', 'unknown'))
            except:
                continue
    
    # Se non trova il config, cerca nei nomi dei file
    results_folder = os.path.join(source_folder, 'results')
    if os.path.exists(results_folder):
        for file in os.listdir(results_folder):
            if file.startswith('overall_') and file.endswith('.csv'):
                num = file.replace('overall_', '').replace('.csv', '')
                if num.isdigit():
                    return num
            elif file.startswith('execution_times_') and file.endswith('.txt'):
                num = file.replace('execution_times_', '').replace('.txt', '')
                if num.isdigit():
                    return num
    
    return None

def get_config_file(source_folder, experiment_num):
    """
    Cerca il file di configurazione usato per l'esperimento.
    Restituisce il contenuto del config e il percorso.
    """
    config_paths = [
        os.path.join(source_folder, 'configs', 'default', 'config.json'),
        os.path.join(source_folder, 'config.json'),
        os.path.join(source_folder, 'configs', 'config.json'),
        os.path.join(source_folder, f'config_experiment_{experiment_num}.json'),
    ]
    
    for path in config_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    config_content = json.load(f)
                return config_content, path
            except:
                continue
    
    return None, None

def ask_for_experiment_notes(experiment_num, default_notes=""):
    """Chiede all'utente di descrivere l'esperimento"""
    print("\n" + "=" * 60)
    print(f"📝 EXPERIMENT NOTES for experiment {experiment_num}")
    print("=" * 60)
    print("Please describe what changed in this experiment:")
    print("  - What parameters were modified?")
    print("  - What was the goal?")
    print("  - Any observations?")
    print("  - (Press Enter twice to finish, or Ctrl+C to skip)")
    print("-" * 60)
    
    lines = []
    print("Enter your notes (empty line to finish):")
    
    try:
        while True:
            line = input()
            if line == "":
                if len(lines) > 0 and lines[-1] == "":
                    lines.pop()
                    break
                elif len(lines) == 0:
                    break
            lines.append(line)
        
        notes = "\n".join(lines).strip()
        
        if not notes:
            print("No notes provided. Using empty notes.")
            return default_notes
        
        print("\n✅ Notes recorded!")
        return notes
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Skipped notes. Using empty notes.")
        return default_notes

def save_experiment_info(dest_folder, experiment_num, config_content, config_path, notes, timings=None):
    """
    Salva le informazioni dell'esperimento (config, note, metadata)
    Mostra TUTTI i parametri delle augmentations.
    """
    saved_files = []
    
    # 1. Salva il file di configurazione completo
    if config_content:
        config_dest = os.path.join(dest_folder, f'config_experiment_{experiment_num}.json')
        with open(config_dest, 'w') as f:
            json.dump(config_content, f, indent=2)
        saved_files.append('config_experiment_{experiment_num}.json')
        print(f"   ✅ Saved config: config_experiment_{experiment_num}.json")
    
    # 2. Salva le note dell'esperimento
    notes_dest = os.path.join(dest_folder, f'experiment_notes_{experiment_num}.txt')
    with open(notes_dest, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write(f"EXPERIMENT NOTES - Experiment {experiment_num}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("DESCRIPTION:\n")
        f.write("-" * 40 + "\n")
        f.write(notes if notes else "No description provided.\n")
        
        if config_content:
            f.write("\n\n" + "=" * 70 + "\n")
            f.write("KEY CONFIGURATION PARAMETERS:\n")
            f.write("-" * 40 + "\n")
            
            # Pretraining parameters
            if 'pretraining_params' in config_content:
                f.write("\n[PRETRAINING PARAMETERS]\n")
                exclude_keys = ['best_model_pth']
                for key, value in config_content['pretraining_params'].items():
                    if key not in exclude_keys:
                        f.write(f"  {key}: {value}\n")
            
            # Supervised training parameters
            if 'sup_training_params' in config_content:
                f.write("\n[SUPERVISED TRAINING PARAMETERS]\n")
                exclude_keys = ['best_model_pth']
                for key, value in config_content['sup_training_params'].items():
                    if key not in exclude_keys:
                        f.write(f"  {key}: {value}\n")
            
            # Latent space parameters
            if 'latent_space_params' in config_content:
                f.write("\n[LATENT SPACE PARAMETERS]\n")
                exclude_keys = ['output_image_dir', 'output_metrics_dir']
                for key, value in config_content['latent_space_params'].items():
                    if key not in exclude_keys:
                        f.write(f"  {key}: {value}\n")
            
            # Dataset parameters
            if 'dataset' in config_content:
                f.write("\n[DATASET]\n")
                for key, value in config_content['dataset'].items():
                    f.write(f"  {key}: {value}\n")
            
            # AUGMENTATIONS - Gestisce sia dizionario che lista
            if 'augmentations' in config_content:
            # AUGMENTATIONS - Gestisce il formato dizionario con parametro 'p'
            if 'augmentations' in config_content:
                f.write("\n[AUGMENTATIONS]\n")
                f.write("-" * 40 + "\n")
                
                augs = config_content['augmentations']
                
                if isinstance(augs, dict):
                    for name, params in augs.items():
                        f.write(f"\n  {name}:\n")
                        if isinstance(params, dict):
                            for key, value in params.items():
                                # Mostra 'p' come 'probability' per chiarezza
                                if key == 'p':
                                    f.write(f"    probability: {value}\n")
                                else:
                                    f.write(f"    {key}: {value}\n")
                        else:
                            f.write(f"    value: {params}\n")
                elif isinstance(augs, list):
                    for aug in augs:
                        if isinstance(aug, dict):
                            name = aug.get('name', 'unknown')
                            f.write(f"\n  {name}:\n")
                            for key, value in aug.items():
                                if key != 'name':
                                    if key == 'p':
                                        f.write(f"    probability: {value}\n")
                                    else:
                                        f.write(f"    {key}: {value}\n")
                        elif isinstance(aug, str):
                            f.write(f"\n  {aug}:\n")
                            f.write(f"    (no parameters)\n")
                
                f.write("\n")