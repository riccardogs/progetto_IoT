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

def get_folder_size(folder_path):
    """Calcola la dimensione della cartella in MB"""
    total_size = 0
    if os.path.exists(folder_path):
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.exists(fp):
                    total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)

def copy_filtered_results(src, dst, experiment_num, folder_name):
    """
    Copia solo i file che appartengono all'esperimento corrente.
    
    Regole di filtraggio:
    - results/: file con pattern *_EXPERIMENT_NUM.*
    - checkpoints/encoder/: encoder_EXPERIMENT_NUM.pth
    - checkpoints/classifier/: classifier_EXPERIMENT_NUM.pth
    - runs/: solo cartella experiment_EXPERIMENT_NUM
    - logs/: solo file experiment_EXPERIMENT_NUM.log
    """
    if not os.path.exists(src):
        return False, f"Source folder '{src}' not found", 0, 0
    
    os.makedirs(dst, exist_ok=True)
    
    files_copied = 0
    files_skipped = 0
    
    if folder_name == 'results':
        # Copia solo file che contengono il numero esperimento nel nome
        for file in os.listdir(src):
            src_file = os.path.join(src, file)
            if os.path.isfile(src_file):
                # Pattern: qualsiasi cosa_10002.csv o .txt
                if f'_{experiment_num}.' in file or file.startswith(f'execution_times_{experiment_num}'):
                    dst_file = os.path.join(dst, file)
                    if os.path.exists(dst_file):
                        files_skipped += 1
                    else:
                        shutil.copy2(src_file, dst_file)
                        files_copied += 1
                        print(f"      Copied: {file}")
                else:
                    files_skipped += 1
    
    elif folder_name == 'checkpoints':
        # Copia solo i checkpoint specifici dell'esperimento
        # Struttura: checkpoints/encoder/encoder_10002.pth
        for subfolder in ['encoder', 'classifier']:
            src_sub = os.path.join(src, subfolder)
            dst_sub = os.path.join(dst, subfolder)
            
            if os.path.exists(src_sub):
                os.makedirs(dst_sub, exist_ok=True)
                
                for file in os.listdir(src_sub):
                    if file.endswith('.pth'):
                        # Pattern: encoder_10002.pth o classifier_10002.pth
                        if f'_{experiment_num}.pth' in file:
                            src_file = os.path.join(src_sub, file)
                            dst_file = os.path.join(dst_sub, file)
                            
                            if os.path.exists(dst_file):
                                files_skipped += 1
                            else:
                                shutil.copy2(src_file, dst_file)
                                files_copied += 1
                                print(f"      Copied: {subfolder}/{file}")
                        else:
                            files_skipped += 1
    
    elif folder_name == 'runs':
        # Copia solo la cartella experiment_{experiment_num}
        target_folder = f'experiment_{experiment_num}'
        src_run = os.path.join(src, target_folder)
        
        if os.path.exists(src_run):
            dst_run = os.path.join(dst, target_folder)
            
            # Conta i file prima di copiare
            for root, dirs, files in os.walk(src_run):
                for file in files:
                    dst_file = os.path.join(dst_run, os.path.relpath(os.path.join(root, file), src_run))
                    if os.path.exists(dst_file):
                        files_skipped += 1
                    else:
                        files_copied += 1
            
            # Copia l'intera cartella
            if not os.path.exists(dst_run):
                shutil.copytree(src_run, dst_run)
                print(f"      Copied entire folder: {target_folder}/")
            else:
                # Copia solo file mancanti
                for root, dirs, files in os.walk(src_run):
                    rel_path = os.path.relpath(root, src_run)
                    if rel_path == '.':
                        target_dir = dst_run
                    else:
                        target_dir = os.path.join(dst_run, rel_path)
                    
                    os.makedirs(target_dir, exist_ok=True)
                    
                    for file in files:
                        src_file = os.path.join(root, file)
                        dst_file = os.path.join(target_dir, file)
                        if not os.path.exists(dst_file):
                            shutil.copy2(src_file, dst_file)
        else:
            print(f"      Warning: {target_folder}/ not found in runs/")
    
    elif folder_name == 'logs':
        # Copia solo il file log specifico
        log_file = f'experiment_{experiment_num}.log'
        src_log = os.path.join(src, log_file)
        
        if os.path.exists(src_log):
            dst_log = os.path.join(dst, log_file)
            if os.path.exists(dst_log):
                files_skipped += 1
            else:
                shutil.copy2(src_log, dst_log)
                files_copied += 1
                print(f"      Copied: {log_file}")
        else:
            print(f"      Warning: {log_file} not found in logs/")
    
    message = f"Copied: {files_copied}, Skipped: {files_skipped}"
    return True, message, files_copied, files_skipped

def check_existing_experiment(drive_path, experiment_num):
    """Controlla se l'esperimento esiste già su Drive"""
    experiment_folder = os.path.join(drive_path, f"experiment_{experiment_num}")
    
    if os.path.exists(experiment_folder):
        size_mb = get_folder_size(experiment_folder)
        
        num_files = 0
        for root, dirs, files in os.walk(experiment_folder):
            num_files += len(files)
        
        return True, experiment_folder, size_mb, num_files
    return False, None, 0, 0

def ask_for_new_number(existing_num):
    """Chiede all'utente un nuovo numero per l'esperimento"""
    print(f"\n⚠️  Experiment {existing_num} already exists on Drive!")
    print("What would you like to do?")
    print("  1. Save with a different number")
    print("  2. Overwrite (use --force flag)")
    print("  3. Cancel and exit")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == '1':
        new_num = input(f"Enter new experiment number (current: {existing_num}): ").strip()
        if not new_num:
            print("No number provided. Cancelling.")
            return None
        return new_num
    elif choice == '2':
        print("Use --force flag to overwrite existing experiment.")
        return 'force_required'
    else:
        print("Cancelled.")
        return None

def save_to_drive():
    """Funzione principale"""
    args = parse_args()
    
    # Verifica che siamo in Colab con Drive montato
    try:
        from google.colab import drive
        drive_available = True
    except ImportError:
        print("⚠️  Not running in Google Colab or google.colab not available")
        print("Attempting to save to local filesystem instead...")
        drive_available = False
    
    # Determina il numero dell'esperimento
    experiment_num = args.experiment_num
    
    if experiment_num == 'auto':
        print("🔍 Auto-detecting experiment number...")
        experiment_num = get_experiment_num_from_config(args.source_folder)
        if experiment_num:
            print(f"   Detected experiment number: {experiment_num}")
        else:
            print("   Could not auto-detect. Please provide --experiment-num")
            return
    
    print(f"\n📌 EXPERIMENT NUMBER: {experiment_num}")
    print(f"📁 Source folder: {args.source_folder}")
    
    # Prepara i percorsi
    source_base = args.source_folder
    drive_base = args.drive_root if drive_available else './drive_backup'
    
    folders_to_copy = ['results', 'checkpoints', 'runs', 'logs']
    
    # Verifica che almeno una cartella esista
    has_data = False
    for folder in folders_to_copy:
        src_path = os.path.join(source_base, folder)
        if os.path.exists(src_path):
            has_data = True
            size_mb = get_folder_size(src_path)
            print(f"📁 Found {folder}/ ({size_mb:.2f} MB)")
    
    if not has_data:
        print("❌ No data found! Make sure you have results/, checkpoints/, runs/, or logs/ folders.")
        print(f"   Looking in: {source_base}")
        return
    
    # Controlla se l'esperimento esiste già su Drive
    if drive_available:
        exists, existing_path, size_mb, num_files = check_existing_experiment(drive_base, experiment_num)
        
        if exists:
            print(f"\n📂 Existing experiment found at:")
            print(f"   {existing_path}")
            print(f"   Size: {size_mb:.2f} MB, Files: {num_files}")
            
            if not args.force:
                new_num = ask_for_new_number(experiment_num)
                if new_num is None:
                    return
                elif new_num == 'force_required':
                    print("Please re-run with --force flag to overwrite.")
                    return
                else:
                    experiment_num = new_num
                    print(f"✅ Will save as experiment_{experiment_num}")
            else:
                print(f"⚠️  Force mode enabled. Will overwrite existing experiment_{experiment_num}")
    
    # Crea la cartella di destinazione
    if drive_available:
        dest_root = os.path.join(drive_base, f"experiment_{experiment_num}")
    else:
        dest_root = os.path.join(drive_base, f"experiment_{experiment_num}")
    
    os.makedirs(dest_root, exist_ok=True)
    
    # Copia le cartelle con filtraggio
    print("\n" + "=" * 60)
    print(f"📦 Saving experiment {experiment_num} to: {dest_root}")
    print("=" * 60)
    
    total_copied = 0
    total_skipped = 0
    
    for folder in folders_to_copy:
        src_path = os.path.join(source_base, folder)
        dst_path = os.path.join(dest_root, folder)
        
        print(f"\n📋 Processing {folder}/...")
        
        if os.path.exists(src_path):
            success, message, copied, skipped = copy_filtered_results(
                src_path, dst_path, experiment_num, folder
            )
            print(f"   {message}")
            total_copied += copied
            total_skipped += skipped
        else:
            print(f"   ⚠️  {folder}/ not found, skipping")
    
    # Salva un file di metadati con informazioni su cosa è stato copiato
    metadata = {
        'experiment_num': experiment_num,
        'saved_date': datetime.now().isoformat(),
        'source_folder': os.path.abspath(source_base),
        'folders_saved': [f for f in folders_to_copy if os.path.exists(os.path.join(source_base, f))],
        'total_files_copied': total_copied,
        'total_files_skipped': total_skipped,
        'filtering_applied': True,
        'note': 'Only files specific to this experiment number were copied'
    }
    
    metadata_path = os.path.join(dest_root, 'saved_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Riepilogo finale
    print("\n" + "=" * 60)
    print("✅ SAVE COMPLETE!")
    print("=" * 60)
    print(f"📁 Location: {dest_root}")
    print(f"📊 Files copied (only experiment {experiment_num}): {total_copied}")
    print(f"⏭️  Files skipped (other experiments or already existed): {total_skipped}")
    print(f"📝 Metadata saved: saved_metadata.json")
    
    # Mostra dimensione totale
    total_size_mb = get_folder_size(dest_root)
    print(f"💾 Total size on Drive: {total_size_mb:.2f} MB")
    
    if drive_available:
        print("\n🔗 You can access your files at:")
        print(f"   https://drive.google.com/drive/search?q=experiment_{experiment_num}")

if __name__ == "__main__":
    save_to_drive()
