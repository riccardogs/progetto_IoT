import os
import json
import time
from datetime import datetime

# Percorsi
PROJECT_DIR = "/content/progetto_IoT"
RESULTS_BASE = "/content/drive/MyDrive/progetto_IoT_risultati"
CONFIG_FILE = "configs/default/config.json"

# Variabili globali per i tempi
_training_start = None
_contrastive_start = None
_classifier_start = None
_contrastive_duration = 0
_classifier_duration = 0


# ---------------------------------------------------------------------------
# Lettura config
# ---------------------------------------------------------------------------

def get_experiment_num():
    """Legge experiment_num dal config.json"""
    config_path = f"{PROJECT_DIR}/{CONFIG_FILE}"
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        exp_num = config_data.get('experiment_num')
        if exp_num is None:
            raise ValueError("'experiment_num' non trovato nel config.json")
        return exp_num
    except FileNotFoundError:
        raise FileNotFoundError(f"Config non trovato: {config_path}")


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------

def start_training_timer():
    """Avvia il timer per il training totale"""
    global _training_start
    _training_start = time.time()
    print(f"[TIMER] Training totale iniziato alle: {datetime.now().strftime('%H:%M:%S')}")


def end_training_timer():
    """Ferma il timer per il training totale e restituisce la durata in secondi"""
    global _training_start
    if _training_start is not None:
        duration = time.time() - _training_start
        _training_start = None
        return duration
    return 0


def set_phase(phase):
    """
    Gestisce i timer delle due fasi.
    Chiamare con:
        set_phase("contrastive")  -> all'inizio del contrastive training
        set_phase("classifier")   -> all'inizio del classifier training
        set_phase("end")          -> alla fine del classifier training
    """
    global _contrastive_start, _classifier_start
    global _contrastive_duration, _classifier_duration

    if phase == "contrastive":
        _contrastive_start = time.time()
        print(f"[FASE] Contrastive training iniziato alle: {datetime.now().strftime('%H:%M:%S')}")

    elif phase == "classifier":
        if _contrastive_start is not None:
            _contrastive_duration = time.time() - _contrastive_start
            _contrastive_start = None
            m, s = divmod(int(_contrastive_duration), 60)
            print(f"[FASE] Contrastive training finito — durata: {m}m {s}s")
        _classifier_start = time.time()
        print(f"[FASE] Classifier training iniziato alle: {datetime.now().strftime('%H:%M:%S')}")

    elif phase == "end":
        if _classifier_start is not None:
            _classifier_duration = time.time() - _classifier_start
            _classifier_start = None
            m, s = divmod(int(_classifier_duration), 60)
            print(f"[FASE] Classifier training finito — durata: {m}m {s}s")


# ---------------------------------------------------------------------------
# Salvataggio esperimento
# ---------------------------------------------------------------------------

def save_experiment(train_duration=None):
    """
    Salva i risultati dell'esperimento su Google Drive.

    - Legge experiment_num dal config.json
    - Se la cartella esiste già chiede conferma prima di sovrascrivere
    - Copia results/, runs/, logs/, checkpoints/
    - Scrive tempi_esecuzione.txt con durate e risultati
    """
    global _contrastive_duration, _classifier_duration

    # Leggi numero esperimento
    exp_num = get_experiment_num()
    exp_name = f"exp_{exp_num:04d}"
    exp_path = f"{RESULTS_BASE}/{exp_name}"

    os.makedirs(RESULTS_BASE, exist_ok=True)

    # Controlla se esiste già
    if os.path.exists(exp_path):
        print(f"\n⚠️  La cartella '{exp_name}' esiste già su Drive.")
        risposta = input("Vuoi sovrascriverla? [s/N]: ").strip().lower()
        if risposta != 's':
            print("Salvataggio annullato.")
            return None
        print(f"Sovrascrittura confermata, procedo...")

    os.makedirs(exp_path, exist_ok=True)

    # Copia cartelle risultati
    cartelle = ["results", "runs", "logs", "checkpoints"]
    for cartella in cartelle:
        src = f"{PROJECT_DIR}/{cartella}"
        if os.path.exists(src):
            os.system(f"cp -r {src} {exp_path}/")
            print(f"  ✓ {cartella}/ copiato")
        else:
            print(f"  — {cartella}/ non trovato, saltato")

    # Copia config
    config_src = f"{PROJECT_DIR}/{CONFIG_FILE}"
    if os.path.exists(config_src):
        os.system(f"cp {config_src} {exp_path}/config_used.json")
        print(f"  ✓ config_used.json salvato")

    # Scrivi file tempi
    _write_tempi_file(exp_path, exp_name, exp_num, train_duration)

    print(f"\n✅ Esperimento {exp_name} salvato in: {exp_path}")
    return exp_num


def _write_tempi_file(exp_path, exp_name, exp_num, train_duration):
    """Scrive tempi_esecuzione.txt nella cartella dell'esperimento"""
    global _contrastive_duration, _classifier_duration

    filepath = f"{exp_path}/tempi_esecuzione.txt"

    def fmt(seconds):
        m, s = divmod(int(seconds), 60)
        return f"{m} min {s} sec"

    with open(filepath, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write(f"Esperimento : {exp_name}\n")
        f.write(f"Data        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Exp num     : {exp_num}\n")
        f.write("=" * 50 + "\n\n")

        f.write("TEMPI DI ESECUZIONE\n")
        f.write("-" * 30 + "\n")

        if train_duration and train_duration > 0:
            f.write(f"Training totale      : {fmt(train_duration)}\n")
        if _contrastive_duration > 0:
            f.write(f"Contrastive training : {fmt(_contrastive_duration)}\n")
        if _classifier_duration > 0:
            f.write(f"Classifier training  : {fmt(_classifier_duration)}\n")

        f.write("\n")

    print(f"  ✓ tempi_esecuzione.txt scritto")
