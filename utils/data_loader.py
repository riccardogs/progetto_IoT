import os
import numpy as np
import glob
import logging
from typing import Dict, Optional
import random

logger = logging.getLogger(__name__)

def load_eeg_data(dataset_path: str, num_files_to_process: Optional[int] = None) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Loads and organizes EEG data from .npz files.
    """
    eeg_data = {
        'train': {label: [] for label in range(5)},
        'test': {label: [] for label in range(5)},
    }

    try:
        npz_files = sorted(glob.glob(os.path.join(dataset_path, '*.npz')))
        if not npz_files:
            raise FileNotFoundError(f"No .npz files found in {dataset_path}.")
        if num_files_to_process is not None:
            npz_files = npz_files[:num_files_to_process]
        logger.info(f"Processing {len(npz_files)} npz files from {dataset_path}.")

        # Extract subject indices from filenames
        subject_indices = []
        for npz_file in npz_files:
            basename = os.path.basename(npz_file)
            subject_idx = int(basename[3:5])
            subject_indices.append(subject_idx)
        unique_subject_indices = list(set(subject_indices))

        # Shuffle the subject indices
        random.shuffle(unique_subject_indices)
        
        # Compute split sizes
        total_subjects = len(unique_subject_indices)
        train_size = int(total_subjects * 0.85)
        test_size = total_subjects - train_size

        train_subjects = unique_subject_indices[:train_size]
        test_subjects = unique_subject_indices[train_size:]

        logger.info(f"Subjects split into train ({len(train_subjects)}), test ({len(test_subjects)}).")

        # Process files
        for idx, npz_file in enumerate(npz_files, 1):
            try:
                basename = os.path.basename(npz_file)
                subject_idx = int(basename[3:5])

                if subject_idx in train_subjects:
                    set_name = 'train'
                elif subject_idx in test_subjects:
                    set_name = 'test'
                else:
                    continue

                with np.load(npz_file) as data:
                    eeg_epochs, labels = data['x'], data['y']
                    for label in range(5):
                        eeg_data[set_name][label].extend(eeg_epochs[labels == label])
            except Exception as e:
                logger.error(f"Error processing {npz_file}: {e}")
            if idx % 10 == 0 or idx == len(npz_files):
                logger.info(f"Processed {idx}/{len(npz_files)} files.")

        # Convert lists to numpy arrays
        for set_name in eeg_data.keys():
            for label in eeg_data[set_name].keys():
                eeg_data[set_name][label] = np.array(eeg_data[set_name][label])
        
        # OVERSAMPLING MENO AGRESSIVO (target_ratio=0.15 invece di 0.3)
        eeg_data = oversample_minority_classes(eeg_data, target_ratio=0.15)

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

    return eeg_data

def oversample_minority_classes(eeg_data, target_ratio=0.15, random_seed=42):
    """
    Oversampling per N1 (label=1) e N3 (label=3) - VERSIONE MENO AGRESSIVA.
    
    Parameters:
    - target_ratio (float): Rapporto desiderato rispetto a W (es. 0.15 = 15% di W)
    """
    np.random.seed(random_seed)
    
    class_names = ['W', 'N1', 'N2', 'N3', 'REM']
    minority_classes = [1, 3]  # N1 e N3
    
    for set_name in ['train']:
        logger.info(f"=" * 50)
        logger.info(f"OVERSAMPLING (target_ratio={target_ratio}) per {set_name} set:")
        
        counts = {}
        for label in range(5):
            counts[label] = len(eeg_data[set_name][label])
        
        max_count = counts[0]  # W è la classe più grande
        target_count = int(max_count * target_ratio)
        
        logger.info(f"  Classe W: {counts[0]} campioni")
        logger.info(f"  Target per N1/N3: {target_count} campioni ({target_ratio*100}% di W)")
        
        for label in minority_classes:
            current_count = counts[label]
            class_name = class_names[label]
            
            if current_count < target_count:
                n_to_add = target_count - current_count
                existing_samples = eeg_data[set_name][label]
                
                n_repeats = n_to_add // current_count if current_count > 0 else 0
                n_remainder = n_to_add % current_count if current_count > 0 else 0
                
                new_samples = []
                for _ in range(n_repeats):
                    new_samples.extend(existing_samples)
                
                if n_remainder > 0 and current_count > 0:
                    random_indices = np.random.choice(current_count, n_remainder, replace=False)
                    for idx in random_indices:
                        new_samples.append(existing_samples[idx])
                
                if len(new_samples) > 0:
                    new_samples = np.array(new_samples)
                    eeg_data[set_name][label] = np.concatenate([existing_samples, new_samples])
                    logger.info(f"  {class_name}: {current_count} → {len(eeg_data[set_name][label])} (+{len(new_samples)})")
            else:
                logger.info(f"  {class_name}: {current_count} (già sufficiente)")
        
        logger.info(f"\n  BILANCIAMENTO FINALE {set_name}:")
        for label in range(5):
            logger.info(f"    {class_names[label]}: {len(eeg_data[set_name][label])}")
        logger.info("=" * 50)
    
    return eeg_data
