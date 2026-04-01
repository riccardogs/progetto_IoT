import os
import json
from datetime import datetime

# Configurazione
PROJECT_DIR = "/content/progetto_IoT"
RESULTS_BASE = "/content/drive/MyDrive/progetto_IoT_risultati"
CONFIG_FILE = "configs/default/config.json"

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

def save_experiment(config_file=CONFIG_FILE):
    """Salva l'esperimento corrente"""
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
    
    # Aggiorna il contatore
    counter_data = {
        'next_exp_num': exp_num + 1, 
        'last_exp': exp_num, 
        'last_save': str(datetime.now()),
        'start_from': 1001
    }
    with open(counter_file, 'w') as f:
        json.dump(counter_data, f, indent=4)
    
    # Crea un file info
    info = {
        'experiment_number': exp_num,
        'date': str(datetime.now()),
        'config_used': config_file
    }
    with open(f"{exp_path}/experiment_info.json", 'w') as f:
        json.dump(info, f, indent=4)
    
    print(f"✅ Esperimento {exp_name} salvato con successo!")
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
