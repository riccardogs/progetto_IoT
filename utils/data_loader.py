import os
import numpy as np
import glob
import logging
from typing import Dict, Optional
import random

logger = logging.getLogger(__name__)

def load_eeg_data(dataset_path: str, num_files_to_process: Optional[int] = None) -> Dict[str, Dict[int, np.ndarray]]:
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

        subject_indices = []
        for npz_file in npz_files:
            basename = os.path.basename(npz_file)
            subject_idx = int(basename[3:5])
            subject_indices.append(subject_idx)
        unique_subject_indices = list(set(subject_indices))

        random.shuffle(unique_subject_indices)
        
        total_subjects = len(unique_subject_indices)
        train_size = int(total_subjects * 0.85)
        test_size = total_subjects - train_size

        train_subjects = unique_subject_indices[:train_size]
        test_subjects = unique_subject_indices[train_size:]

        logger.info(f"Subjects split into train ({len(train_subjects)}), test ({len(test_subjects)}).")

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

        for set_name in eeg_data.keys():
            for label in eeg_data[set_name].keys():
                eeg_data[set_name][label] = np.array(eeg_data[set_name][label])
        
        # OVERSAMPLING MIRATO: N1=5500, N2=8500, N3=6800, REM=8500
        eeg_data = oversample_minority_classes(eeg_data, target_n1=5500, target_n2=8500, target_n3=6800, target_rem=8500)

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

    return eeg_data

def oversample_minority_classes(eeg_data, target_n1=5500, target_n2=8500, target_n3=6800, target_rem=8500, random_seed=42):
    """
    Oversampling differenziato per classe:
    - N1: ridotto per diminuire falsi positivi
    - N2 e REM: aumentati per migliorare performance
    - N3: invariato
    """
    np.random.seed(random_seed)
    
    class_names = ['W', 'N1', 'N2', 'N3', 'REM']
    targets = {1: target_n1, 2: target_n2, 3: target_n3, 4: target_rem}
    
    for set_name in ['train']:
        logger.info(f"=" * 50)
        logger.info(f"OVERSAMPLING MIRATO:")
        logger.info(f"  N1 target: {target_n1}")
        logger.info(f"  N2 target: {target_n2}")
        logger.info(f"  N3 target: {target_n3}")
        logger.info(f"  REM target: {target_rem}")
        
        counts = {}
        for label in range(5):
            counts[label] = len(eeg_data[set_name][label])
        
        logger.info(f"  W: {counts[0]} (lasciato invariato)")
        
        for label, target_count in targets.items():
            current_count = counts[label]
            class_name = class_names[label]
            
            if current_count < target_count:
                n_to_add = target_count - current_count
                existing_samples = eeg_data[set_name][label]
                
                if current_count > 0:
                    n_repeats = n_to_add // current_count
                    n_remainder = n_to_add % current_count
                    
                    new_samples = []
                    for _ in range(n_repeats):
                        new_samples.extend(existing_samples)
                    
                    if n_remainder > 0:
                        random_indices = np.random.choice(current_count, n_remainder, replace=False)
                        for idx in random_indices:
                            new_samples.append(existing_samples[idx])
                    
                    if len(new_samples) > 0:
                        new_samples = np.array(new_samples)
                        eeg_data[set_name][label] = np.concatenate([existing_samples, new_samples])
                        logger.info(f"  {class_name}: {current_count} → {len(eeg_data[set_name][label])} (+{len(new_samples)})")
            else:
                logger.info(f"  {class_name}: {current_count} (già sopra target)")
        
        logger.info(f"\n  BILANCIAMENTO FINALE:")
        for label in range(5):
            logger.info(f"    {class_names[label]}: {len(eeg_data[set_name][label])}")
        logger.info("=" * 50)
    
    return eeg_data
