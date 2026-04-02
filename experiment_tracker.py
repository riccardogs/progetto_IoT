import os
import json
from datetime import datetime


# Percorsi
PROJECT_DIR = "/content/progetto_IoT"
RESULTS_BASE = "/content/drive/MyDrive/progetto_IoT_risultati"
CONFIG_FILE = "configs/default/config.json"


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
# Salvataggio esperimento
# ---------------------------------------------------------------------------

def save_experiment(train_duration=0, contrastive_duration=0, classifier_duration=0):
    """
    Salva i risultati dell'esperimento su Google Drive.

    Parametri
    ---------
    train_duration        : durata totale del training in secondi
    contrastive_duration  : durata del contrastive training in secondi
    classifier_duration   : durata del classifier training in secondi
    """

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
        print("Sovrascrittura confermata, procedo...")

    os.makedirs(exp_path, exist_ok=True)

    # Copia cartelle risultati
    for cartella in ["results", "runs", "logs", "checkpoints"]:
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
    _write_tempi_file(exp_path, exp_name, exp_num, train_duration, contrastive_duration, classifier_duration)

    print(f"\n✅ Esperimento {exp_name} salvato in: {exp_path}")
    return exp_num


def _write_tempi_file(exp_path, exp_name, exp_num, train_duration, contrastive_duration, classifier_duration):

    def fmt(seconds):
        m, s = divmod(int(seconds), 60)
        return f"{m} min {s} sec"

    filepath = f"{exp_path}/tempi_esecuzione.txt"
    with open(filepath, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write(f"Esperimento : {exp_name}\n")
        f.write(f"Data        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Exp num     : {exp_num}\n")
        f.write("=" * 50 + "\n\n")
        f.write("TEMPI DI ESECUZIONE\n")
        f.write("-" * 30 + "\n")
        if train_duration > 0:
            f.write(f"Training totale      : {fmt(train_duration)}\n")
        if contrastive_duration > 0:
            f.write(f"Contrastive training : {fmt(contrastive_duration)}\n")
        if classifier_duration > 0:
            f.write(f"Classifier training  : {fmt(classifier_duration)}\n")
        f.write("\n")

    print(f"  ✓ tempi_esecuzione.txt scritto")
