import os
import json
import time
import re
from datetime import datetime

# Configurazione
PROJECT_DIR = "/content/progetto_IoT"
RESULTS_BASE = "/content/drive/MyDrive/progetto_IoT_risultati"
CONFIG_FILE = "configs/default/config.json"

# Variabili globali per i tempi
_training_start = None
_contrastive_start = None
_classifier_start = None
_contrastive_duration = 0
_classifier_duration = 0

def get_experiment_num_from_config():
    """Legge experiment_num dal file di configurazione"""
    config_path = f"{PROJECT_DIR}/{CONFIG_FILE}"
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
            return config_data.get('experiment_num')
    except:
        return None

def init_experiment_tracker():
    """Inizializza il tracker degli esperimenti"""
    os.system(f"mkdir -p {RESULTS_BASE}")
    
    counter_file = f"{RESULTS_BASE}/experiment_counter.json"
    
    # Leggi il numero dal config
    current_exp_num = get_experiment_num_from_config()
    
    if not os.path.exists(counter_file):
        # Crea il file con il numero dal config
        counter_data = {
            'next_exp_num': current_exp_num if current_exp_num else 1001,
            'start_from': current_exp_num if current_exp_num else 1001,
            'saved_experiments': []
        }
        with open(counter_file, 'w') as f:
            json.dump(counter_data, f, indent=4)
        print(f"Tracker inizializzato. Esperimento corrente: {counter_data['next_exp_num']}")
    else:
        with open(counter_file, 'r') as f:
            counter_data = json.load(f)
        
        # Se il config ha un numero diverso, aggiorna il tracker
        if current_exp_num and current_exp_num != counter_data.get('next_exp_num'):
            print(f"⚠️ Attenzione: Config ha experiment_num={current_exp_num}, tracker aveva {counter_data.get('next_exp_num')}")
            print(f"Aggiorno il tracker a {current_exp_num}")
            counter_data['next_exp_num'] = current_exp_num
            with open(counter_file, 'w') as f:
                json.dump(counter_data, f, indent=4)
        
        print(f"Esperimento corrente: {counter_data.get('next_exp_num')}")
    
    return counter_data.get('next_exp_num')

def start_training_timer():
    """Avvia il timer per il training totale"""
    global _training_start
    _training_start = time.time()
    print(f"Training TOTALE iniziato alle: {datetime.now().strftime('%H:%M:%S')}")

def end_training_timer():
    """Ferma il timer per il training totale"""
    global _training_start
    if _training_start is not None:
        duration = time.time() - _training_start
        _training_start = None
        return duration
    return 0

def set_phase(phase):
    """Imposta la fase corrente e avvia/ferma i timer"""
    global _contrastive_start, _classifier_start, _contrastive_duration, _classifier_duration
    
    if phase == "contrastive":
        _contrastive_start = time.time()
        print(f"[FASE] Contrastive training INIZIATO alle: {datetime.now().strftime('%H:%M:%S')}")
    
    elif phase == "classifier":
        # Ferma il contrastive
        if _contrastive_start is not None:
            _contrastive_duration = time.time() - _contrastive_start
            minutes = int(_contrastive_duration // 60)
            seconds = int(_contrastive_duration % 60)
            print(f"[FASE] Contrastive training FINITO - Durata: {minutes}m {seconds}s")
            _contrastive_start = None
        # Avvia il classifier
        _classifier_start = time.time()
        print(f"[FASE] Classifier training INIZIATO alle: {datetime.now().strftime('%H:%M:%S')}")
    
    elif phase == "end":
        # Ferma il classifier
        if _classifier_start is not None:
            _classifier_duration = time.time() - _classifier_start
            minutes = int(_classifier_duration // 60)
            seconds = int(_classifier_duration % 60)
            print(f"[FASE] Classifier training FINITO - Durata: {minutes}m {seconds}s")
            _classifier_start = None

def get_contrastive_duration():
    """Restituisce la durata del contrastive training"""
    global _contrastive_duration
    return _contrastive_duration

def get_classifier_duration():
    """Restituisce la durata del classifier training"""
    global _classifier_duration
    return _classifier_duration

def save_experiment(train_duration=None):
    """Salva l'esperimento corrente con controllo duplicati"""
    global _contrastive_duration, _classifier_duration
    
    os.system(f"mkdir -p {RESULTS_BASE}")
    
    counter_file = f"{RESULTS_BASE}/experiment_counter.json"
    
    if not os.path.exists(counter_file):
        init_experiment_tracker()
    
    with open(counter_file, 'r') as f:
        counter_data = json.load(f)
        saved_experiments = counter_data.get('saved_experiments', [])
    
    # Leggi il numero esperimento dal file di configurazione
    exp_num = get_experiment_num_from_config()
    
    if exp_num is None:
        print("❌ ERRORE: Impossibile leggere experiment_num dal config!")
        return None
    
    # Verifica se questo esperimento è già stato salvato
    if exp_num in saved_experiments:
        print(f"⚠️ ATTENZIONE: Esperimento {exp_num} è già stato salvato!")
        print(f"   Non verrà salvato di nuovo per evitare duplicati.")
        return None
    
    exp_name = f"exp_{exp_num:04d}"
    exp_path = f"{RESULTS_BASE}/{exp_name}"
    
    print(f"Salvataggio {exp_name} (experiment_num: {exp_num})...")
    os.system(f"mkdir -p {exp_path}")
    
    # Salva le cartelle
    if os.path.exists(f"{PROJECT_DIR}/results"):
        os.system(f"cp -r {PROJECT_DIR}/results {exp_path}/")
        print(f"results/ salvati")
    if os.path.exists(f"{PROJECT_DIR}/runs"):
        os.system(f"cp -r {PROJECT_DIR}/runs {exp_path}/")
        print(f"runs/ salvati")
    if os.path.exists(f"{PROJECT_DIR}/logs"):
        os.system(f"cp -r {PROJECT_DIR}/logs {exp_path}/")
        print(f"logs/ salvati")
    if os.path.exists(f"{PROJECT_DIR}/checkpoints"):
        os.system(f"cp -r {PROJECT_DIR}/checkpoints {exp_path}/")
        print(f"checkpoints/ salvati")
    
    # Salva config
    config_path = f"{PROJECT_DIR}/{CONFIG_FILE}"
    if os.path.exists(config_path):
        os.system(f"cp {config_path} {exp_path}/config_used.txt")
        print(f"config_used.txt salvato")
    
    # Leggi accuracy dai log
    accuracy = None
    macro_f1 = None
    import glob
    log_files = glob.glob(f"{PROJECT_DIR}/logs/experiment_*.log")
    if log_files:
        log_file = max(log_files, key=os.path.getctime)
        with open(log_file, 'r') as f:
            content = f.read()
            match = re.search(r'Validation Accuracy: ([\d.]+)', content)
            if match:
                accuracy = float(match.group(1))
            match = re.search(r'Macro F1 Score: ([\d.]+)', content)
            if match:
                macro_f1 = float(match.group(1))
    
    # Salva file tempi
    times_file = f"{exp_path}/tempi_esecuzione.txt"
    with open(times_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write(f"ESPERIMENTO {exp_name}\n")
        f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Experiment num: {exp_num}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("TEMPI DI ESECUZIONE:\n")
        f.write("-" * 40 + "\n")
        
        if train_duration and train_duration > 0:
            minutes = int(train_duration // 60)
            seconds = int(train_duration % 60)
            f.write(f"Training totale: {minutes} minuti e {seconds} secondi\n")
        
        if _contrastive_duration > 0:
            minutes = int(_contrastive_duration // 60)
            seconds = int(_contrastive_duration % 60)
            f.write(f"Contrastive training: {minutes} minuti e {seconds} secondi\n")
        
        if _classifier_duration > 0:
            minutes = int(_classifier_duration // 60)
            seconds = int(_classifier_duration % 60)
            f.write(f"Classifier training: {minutes} minuti e {seconds} secondi\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("RISULTATI:\n")
        f.write("-" * 40 + "\n")
        if accuracy:
            f.write(f"Accuracy finale: {accuracy*100:.2f}%\n")
        if macro_f1:
            f.write(f"Macro F1 Score: {macro_f1:.4f}\n")
    
    # Aggiorna contatore
    saved_experiments.append(exp_num)
    counter_data['saved_experiments'] = saved_experiments
    counter_data['next_exp_num'] = exp_num + 1
    with open(counter_file, 'w') as f:
        json.dump(counter_data, f, indent=4)
    
    print(f"\n Esperimento {exp_name} salvato con successo!")
    print(f"Percorso: {exp_path}")
    return exp_num

def show_status():
    """Mostra lo stato degli esperimenti"""
    counter_file = f"{RESULTS_BASE}/experiment_counter.json"
    current_exp = get_experiment_num_from_config()
    
    if os.path.exists(counter_file):
        with open(counter_file, 'r') as f:
            data = json.load(f)
            print(f"Experiment num dal config: {current_exp}")
            print(f"Prossimo esperimento: exp_{data.get('next_exp_num', 1001):04d}")
            print(f"Esperimenti salvati: {data.get('saved_experiments', [])}")
    else:
        print("Nessun esperimento ancora avviato.")
