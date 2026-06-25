import sys
import subprocess
from pathlib import Path
from utils.plot import plot
from model.plot_history_grad import plot_optimization

ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "model"

OPT_SCRIPT_PATH = MODEL_DIR / "calculate_opt_total.py"

X_STAR_FILE = MODEL_DIR / "x_star.txt"
HISTORY_FILE = MODEL_DIR / "optimization_history.txt"

def main():
    print("====================================================")
    print(" AVVIO PIPELINE DI OTTIMIZZAZIONE")
    print("====================================================\n")
    
    if not OPT_SCRIPT_PATH.exists():
        print(f"[ERRORE] Non trovo lo script: {OPT_SCRIPT_PATH}")
        print("Assicurati che il file si chiami 'calculate_opt_total.py' e sia in 'model/'")
        sys.exit(1)

    print(f"[INFO] Esecuzione di {OPT_SCRIPT_PATH.name} in corso...")
    try:
        subprocess.run([sys.executable, str(OPT_SCRIPT_PATH)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERRORE] L'ottimizzazione si è interrotta con un errore: {e}")
        sys.exit(1)

    print("\n====================================================")
    print(" RISULTATI DELL'OTTIMIZZAZIONE")
    print("====================================================\n")

    if HISTORY_FILE.exists():
        plot_optimization(HISTORY_FILE)
    else:
        print(f"File history non trovato in {HISTORY_FILE}")

    print("-" * 52)

    if X_STAR_FILE.exists():
        if sys.platform == "win32":
            exe_name = "mobd.exe"
        elif sys.platform == "darwin":
            exe_name = "mobd_macos"
        elif sys.platform.startswith("linux"):
            exe_name = "mobd_linux"
        else:
            raise OSError(f"Sistema operativo non supportato: {sys.platform}")
        exe_path = Path("executables") / exe_name
        print("Valore funzione obiettivo calcolato con eseguibile fornito:")
        subprocess.run([str(exe_path), X_STAR_FILE, '-t'])
        print("Grafico dei punti:")
        plot(X_STAR_FILE)

    else:
        print(f"File soluzione non trovato in {X_STAR_FILE}")

    print("\nEsecuzione completata con successo.")

if __name__ == "__main__":
    main()
