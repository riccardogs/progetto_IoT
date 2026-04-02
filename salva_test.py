import os
import json
import time
from datetime import datetime

# Configurazione
PROJECT_DIR = "/content/progetto_IoT"
RESULTS_BASE = "/content/drive/MyDrive/progetto_IoT_risultati"
CONFIG_FILE = "configs/default/config.json"

# Variabili globali per tenere traccia dei tempi
_train_start = None
_test_start = None

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
    """Avvia il timer per il training"""
    global _train_start
    _train_start = time.time()
    print(f"⏱️ Training iniziato alle: {datetime.now().strftime('%H:%M:%S')}")

def end_training_timer():
    """Ferma il timer per il training e restituisce la durata"""
    global _train_start
    if _train_start is not None:
        duration = time.time() - _train_start
        _train_start = None
        return duration
    return 0

def start_test_timer():
    """Avvia il timer per il test"""
    global _test_start
    _test_start = time.time()
    print(f"⏱️ Test iniziato alle: {datetime.now().strftime('%H:%M:%S')}")

def end_test_timer():
    """Ferma il timer per il test e restituisce la durata"""
    global _test_start
    if _test_start is not None:
        duration = time.time() - _test_start
        _test_start = None
        return duration
    return 0

def save_experiment(config_file=CONFIG_FILE, train_duration=None, test_duration=None):
    """Salva l'esperimento corrente con i tempi di training e test"""
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
    
    # ===== SALVA FILE CON I TEMPI =====
    times_file = f"{exp_path}/tempi_esecuzione.txt"
    with open(times_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write(f"ESPERIMENTO {exp_name}\n")
        f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("TEMPI DI ESECUZIONE:\n")
        f.write("-" * 40 + "\n")
        
        if train_duration is not None and train_duration > 0:
            minutes = int(train_duration // 60)
            seconds = int(train_duration % 60)
            f.write(f"Training: {minutes} minuti e {seconds} secondi ({train_duration:.2f} secondi)\n")
        else:
            f.write("Training: tempo non registrato\n")
        
        if test_duration is not None and test_duration > 0:
            minutes = int(test_duration // 60)
            seconds = int(test_duration % 60)
            f.write(f"Test: {minutes} minuti e {seconds} secondi ({test_duration:.2f} secondi)\n")
        else:
            f.write("Test: tempo non registrato\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("DETTAGLI ESPERIMENTO:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Configurazione usata: {config_file}\n")
        
        # Cerca di leggere il numero di epoche dal config
        try:
            with open(config_path, 'r') as cfg:
                config_data = json.load(cfg)
                if 'num_epochs' in config_data:
                    f.write(f"Epoche: {config_data['num_epochs']}\n")
                if 'batch_size' in config_data:
                    f.write(f"Batch size: {config_data['batch_size']}\n")
                if 'learning_rate' in config_data:
                    f.write(f"Learning rate: {config_data['learning_rate']}\n")
        except:
            pass
    
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
    
    # Crea un file info aggiornato
    info = {
        'experiment_number': exp_num,
        'date': str(datetime.now()),
        'config_used': config_file,
        'training_duration_sec': train_duration if train_duration else 0,
        'test_duration_sec': test_duration if test_duration else 0
    }
    with open(f"{exp_path}/experiment_info.json", 'w') as f:
        json.dump(info, f, indent=4)
    
    print(f"\n✅ Esperimento {exp_name} salvato con successo!")
    print(f"📁 Percorso: {exp_path}")
    print(f"⏱️ Tempi salvati in: tempi_esecuzione.txt")
    
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
