import os
import json
import time
import re
from datetime import datetime

# Configurazione
PROJECT_DIR = "/content/progetto_IoT"
RESULTS_BASE = "/content/drive/MyDrive/progetto_IoT_risultati"
CONFIG_FILE = "configs/default/config.json"

# Variabili globali
_train_start = None
_contrastive_start = None
_classifier_start = None

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
    global _train_start
    _train_start = time.time()
    print(f"⏱️ Training iniziato alle: {datetime.now().strftime('%H:%M:%S')}")

def end_training_timer():
    """Ferma il timer per il training totale e restituisce la durata"""
    global _train_start
    if _train_start is not None:
        duration = time.time() - _train_start
        _train_start = None
        return duration
    return 0

def start_contrastive_timer():
    """Avvia il timer per il contrastive training"""
    global _contrastive_start
    _contrastive_start = time.time()
    print(f"⏱️ Contrastive training iniziato alle: {datetime.now().strftime('%H:%M:%S')}")

def end_contrastive_timer():
    """Ferma il timer per il contrastive training"""
    global _contrastive_start
    if _contrastive_start is not None:
        duration = time.time() - _contrastive_start
        _contrastive_start = None
        return duration
    return 0

def start_classifier_timer():
    """Avvia il timer per il classifier training"""
    global _classifier_start
    _classifier_start = time.time()
    print(f"⏱️ Classifier training iniziato alle: {datetime.now().strftime('%H:%M:%S')}")

def end_classifier_timer():
    """Ferma il timer per il classifier training"""
    global _classifier_start
    if _classifier_start is not None:
        duration = time.time() - _classifier_start
        _classifier_start = None
        return duration
    return 0

def save_experiment(config_file=CONFIG_FILE, train_duration=None, contrastive_duration=None, classifier_duration=None, log_file=None):
    """Salva l'esperimento corrente con i tempi di training e risultati"""
    # Assicurati che la cartella base esista
    os.system(f"mkdir -p {RESULTS_BASE}")
    
    counter_file = f"{RESULTS_BASE}/experiment_counter.json"
    
    # Se il file non esiste, crealo
    if not os.path.exists(counter_file):
        print("📁 Tracker non trovato. Lo inizializzo...")
        init_experiment_tracker()
    
    # Leggi il contatore
    with open(counter_file, 'r') as f:
        counter_data = json.load(f)
        exp_num = counter_data.get('next_exp_num', 1001)
    
    # Nome della cartella per questo esperimento
    exp_name = f"exp_{exp_num:04d}"
    exp_path = f"{RESULTS_BASE}/{exp_name}"
    
    print(f"💾 Salvataggio {exp_name}...")
    
    # Crea la cartella
    os.system(f"mkdir -p {exp_path}")
    
    # Salva i risultati
    if os.path.exists(f"{PROJECT_DIR}/results"):
        os.system(f"cp -r {PROJECT_DIR}/results {exp_path}/")
        print(f"  ✅ results/ salvati")
    
    if os.path.exists(f"{PROJECT_DIR}/runs"):
        os.system(f"cp -r {PROJECT_DIR}/runs {exp_path}/")
        print(f"  ✅ runs/ salvati")
    
    if os.path.exists(f"{PROJECT_DIR}/logs"):
        os.system(f"cp -r {PROJECT_DIR}/logs {exp_path}/")
        print(f"  ✅ logs/ salvati")
    
    if os.path.exists(f"{PROJECT_DIR}/checkpoints"):
        os.system(f"cp -r {PROJECT_DIR}/checkpoints {exp_path}/")
        print(f"  ✅ checkpoints/ salvati")
    
    # Salva il config
    config_path = f"{PROJECT_DIR}/{config_file}"
    if os.path.exists(config_path):
        os.system(f"cp {config_path} {exp_path}/config_used.txt")
        print(f"  ✅ config_used.txt salvato")
    
    # Salva anche i file di configurazione completi
    if os.path.exists(f"{PROJECT_DIR}/configs"):
        os.system(f"cp -r {PROJECT_DIR}/configs {exp_path}/")
        print(f"  ✅ configs/ salvati")
    
    # ===== LEGGI I RISULTATI DAI LOG =====
    accuracy = None
    macro_f1 = None
    
    # Cerca il file di log
    if log_file is None:
        import glob
        log_files = glob.glob(f"{PROJECT_DIR}/logs/experiment_*.log")
        if log_files:
            log_file = max(log_files, key=os.path.getctime)
    
    if log_file and os.path.exists(log_file):
        with open(log_file, 'r') as f:
            log_content = f.read()
        
        # Accuracy finale
        match = re.search(r'Validation Accuracy: ([\d.]+)', log_content)
        if match:
            accuracy = float(match.group(1))
        
        # Macro F1 Score
        match = re.search(r'Macro F1 Score: ([\d.]+)', log_content)
        if match:
            macro_f1 = float(match.group(1))
    
    # ===== SALVA FILE CON I TEMPI =====
    times_file = f"{exp_path}/tempi_esecuzione.txt"
    with open(times_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write(f"ESPERIMENTO {exp_name}\n")
        f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        
        # TEMPI
        f.write("TEMPI DI ESECUZIONE:\n")
        f.write("-" * 40 + "\n")
        
        if train_duration is not None and train_duration > 0:
            minutes = int(train_duration // 60)
            seconds = int(train_duration % 60)
            f.write(f"Training totale: {minutes} minuti e {seconds} secondi ({train_duration:.2f} secondi)\n")
        
        if contrastive_duration is not None and contrastive_duration > 0:
            minutes = int(contrastive_duration // 60)
            seconds = int(contrastive_duration % 60)
            f.write(f"  - Contrastive training: {minutes} minuti e {seconds} secondi ({contrastive_duration:.2f} secondi)\n")
        
        if classifier_duration is not None and classifier_duration > 0:
            minutes = int(classifier_duration // 60)
            seconds = int(classifier_duration % 60)
            f.write(f"  - Classifier training: {minutes} minuti e {seconds} secondi ({classifier_duration:.2f} secondi)\n")
        
        f.write("\n" + "=" * 60 + "\n")
        
        # RISULTATI
        f.write("RISULTATI:\n")
        f.write("-" * 40 + "\n")
        
        if accuracy is not None:
            f.write(f"Accuracy finale: {accuracy*100:.2f}%\n")
        
        if macro_f1 is not None:
            f.write(f"Macro F1 Score: {macro_f1:.4f}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        
        # DETTAGLI CONFIG
        f.write("DETTAGLI ESPERIMENTO:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Configurazione usata: {config_file}\n")
    
    print(f"  ✅ tempi_esecuzione.txt salvato")
    
    # Aggiorna il contatore
    counter_data = {
        'next_exp_num': exp_num + 1, 
        'last_exp': exp_num, 
        'last_save': str(datetime.now()),
        'start_from': 1001
    }
    with open(counter_file, 'w') as f:
        json.dump(counter_data, f, indent=4)
    
    print(f"\n✅ Esperimento {exp_name} salvato con successo!")
    print(f"📁 Percorso: {exp_path}")
    
    return exp_num

def show_status():
    """Mostra lo stato degli esperimenti"""
    counter_file = f"{RESULTS_BASE}/experiment_counter.json"
    if os.path.exists(counter_file):
        with open(counter_file, 'r') as f:
            data = json.load(f)
            last_exp = data.get('last_exp', 'N/A')
            if last_exp != 'N/A':
                print(f"📊 Ultimo esperimento: exp_{last_exp:04d}")
            else:
                print(f"📊 Ultimo esperimento: Nessuno")
            print(f"🔢 Prossimo esperimento: exp_{data.get('next_exp_num', 1001):04d}")
            print(f"📅 Ultimo salvataggio: {data.get('last_save', 'Mai')}")
    else:
        print("Nessun esperimento ancora avviato. Prossimo sarà exp_1001")
