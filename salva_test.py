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

# Flag per sapere se siamo in una fase specifica
_current_phase = None

def init_experiment_tracker():
    """Inizializza il tracker degli esperimenti"""
    os.system(f"mkdir -p {RESULTS_BASE}")
    
    counter_file = f"{RESULTS_BASE}/experiment_counter.json"
    
    if not os.path.exists(counter_file):
        counter_data = {'next_exp_num': 1001, 'start_from': 1001}
        with open(counter_file, 'w') as f:
            json.dump(counter_data, f, indent=4)
        print(f"📊 Tracker inizializzato. Prossimo esperimento: 1001")
    else:
        with open(counter_file, 'r') as f:
            counter_data = json.load(f)
        print(f"📊 Prossimo esperimento: {counter_data.get('next_exp_num', 1001)}")
    
    return counter_data.get('next_exp_num', 1001)

def start_training_timer():
    """Avvia il timer per il training totale"""
    global _training_start
    _training_start = time.time()
    print(f"⏱️ Training TOTALE iniziato alle: {datetime.now().strftime('%H:%M:%S')}")

def end_training_timer():
    """Ferma il timer per il training totale"""
    global _training_start
    if _training_start is not None:
        duration = time.time() - _training_start
        _training_start = None
        return duration
    return 0

def set_phase(phase):
    """Imposta la fase corrente e avvia il timer appropriato"""
    global _current_phase, _contrastive_start, _classifier_start
    
    if phase == "contrastive":
        _contrastive_start = time.time()
        print(f"⏱️ [FASE] Contrastive training INIZIATO alle: {datetime.now().strftime('%H:%M:%S')}")
    elif phase == "classifier":
        # Ferma il contrastive se era in corso
        if _contrastive_start is not None:
            contrastive_time = time.time() - _contrastive_start
            print(f"⏱️ [FASE] Contrastive training FINITO - Durata: {int(contrastive_time//60)}m {int(contrastive_time%60)}s")
            _contrastive_start = None
        # Avvia il classifier
        _classifier_start = time.time()
        print(f"⏱️ [FASE] Classifier training INIZIATO alle: {datetime.now().strftime('%H:%M:%S')}")
    elif phase == "end":
        if _classifier_start is not None:
            classifier_time = time.time() - _classifier_start
            print(f"⏱️ [FASE] Classifier training FINITO - Durata: {int(classifier_time//60)}m {int(classifier_time%60)}s")
            _classifier_start = None

def get_contrastive_duration():
    """Restituisce la durata del contrastive training"""
    global _contrastive_start
    if _contrastive_start is None:
        return 0
    return time.time() - _contrastive_start

def get_classifier_duration():
    """Restituisce la durata del classifier training"""
    global _classifier_start
    if _classifier_start is None:
        return 0
    return time.time() - _classifier_start

def save_experiment(config_file=CONFIG_FILE, train_duration=None, contrastive_duration=None, classifier_duration=None, log_file=None):
    """Salva l'esperimento corrente"""
    os.system(f"mkdir -p {RESULTS_BASE}")
    
    counter_file = f"{RESULTS_BASE}/experiment_counter.json"
    
    if not os.path.exists(counter_file):
        init_experiment_tracker()
    
    with open(counter_file, 'r') as f:
        counter_data = json.load(f)
        exp_num = counter_data.get('next_exp_num', 1001)
    
    exp_name = f"exp_{exp_num:04d}"
    exp_path = f"{RESULTS_BASE}/{exp_name}"
    
    print(f"💾 Salvataggio {exp_name}...")
    os.system(f"mkdir -p {exp_path}")
    
    # Salva le cartelle
    if os.path.exists(f"{PROJECT_DIR}/results"):
        os.system(f"cp -r {PROJECT_DIR}/results {exp_path}/")
    if os.path.exists(f"{PROJECT_DIR}/runs"):
        os.system(f"cp -r {PROJECT_DIR}/runs {exp_path}/")
    if os.path.exists(f"{PROJECT_DIR}/logs"):
        os.system(f"cp -r {PROJECT_DIR}/logs {exp_path}/")
    if os.path.exists(f"{PROJECT_DIR}/checkpoints"):
        os.system(f"cp -r {PROJECT_DIR}/checkpoints {exp_path}/")
    
    # Salva config
    config_path = f"{PROJECT_DIR}/{config_file}"
    if os.path.exists(config_path):
        os.system(f"cp {config_path} {exp_path}/config_used.txt")
    
    # Leggi accuracy dai log
    accuracy = None
    macro_f1 = None
    
    if log_file is None:
        import glob
        log_files = glob.glob(f"{PROJECT_DIR}/logs/experiment_*.log")
        if log_files:
            log_file = max(log_files, key=os.path.getctime)
    
    if log_file and os.path.exists(log_file):
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
        f.write("=" * 60 + "\n\n")
        
        f.write("TEMPI DI ESECUZIONE:\n")
        f.write("-" * 40 + "\n")
        
        if train_duration and train_duration > 0:
            minutes = int(train_duration // 60)
            seconds = int(train_duration % 60)
            f.write(f"Training totale: {minutes} minuti e {seconds} secondi\n")
        
        if contrastive_duration and contrastive_duration > 0:
            minutes = int(contrastive_duration // 60)
            seconds = int(contrastive_duration % 60)
            f.write(f"Contrastive training: {minutes} minuti e {seconds} secondi\n")
        
        if classifier_duration and classifier_duration > 0:
            minutes = int(classifier_duration // 60)
            seconds = int(classifier_duration % 60)
            f.write(f"Classifier training: {minutes} minuti e {seconds} secondi\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("RISULTATI:\n")
        f.write("-" * 40 + "\n")
        if accuracy:
            f.write(f"Accuracy finale: {accuracy*100:.2f}%\n")
        if macro_f1:
            f.write(f"Macro F1 Score: {macro_f1:.4f}\n")
    
    # Aggiorna contatore
    counter_data = {
        'next_exp_num': exp_num + 1, 
        'last_exp': exp_num, 
        'last_save': str(datetime.now())
    }
    with open(counter_file, 'w') as f:
        json.dump(counter_data, f, indent=4)
    
    print(f"✅ Esperimento {exp_name} salvato!")
    return exp_num

def show_status():
    counter_file = f"{RESULTS_BASE}/experiment_counter.json"
    if os.path.exists(counter_file):
        with open(counter_file, 'r') as f:
            data = json.load(f)
            print(f"📊 Prossimo esperimento: exp_{data.get('next_exp_num', 1001):04d}")
    else:
        print("Nessun esperimento ancora avviato.")
