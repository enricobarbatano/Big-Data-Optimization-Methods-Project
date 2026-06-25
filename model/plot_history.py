import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
HISTORY_FILE = CURRENT_DIR / "optimization_history.txt"
PLOT_OUTPUT = CURRENT_DIR / "optimization_plot.png"

def plot_optimization(filename):
    HISTORY_FILE = filename

    print(f"Caricamento dati da {HISTORY_FILE}...")
    history = np.loadtxt(HISTORY_FILE)

    plt.figure(figsize=(10, 6))

    if history.ndim == 1 or (history.ndim == 2 and history.shape[1] == 1):
        plt.plot(history, color='#1f77b4', marker='o', markersize=3, label='Funzione Obiettivo')
        plt.ylabel("f(x)", fontsize=12)
        plt.legend(loc='upper right')
        
    else:
        print(f"Trovati dati multidimensionali di forma {history.shape}.")
        print("Calcolo la norma della differenza tra le iterazioni per valutare la convergenza.")
        
        step_sizes = np.linalg.norm(np.diff(history, axis=0), axis=1)
        
        plt.plot(range(1, len(history)), step_sizes, color='#d62728', marker='o', markersize=3, label='Variazione Parametri')
        plt.ylabel("Ampiezza del passo (Norma L2)", fontsize=12)
        plt.legend(loc='upper right')

    plt.title("Valutazione della Convergenza (GDM)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Iterazione", fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    
    plt.tight_layout()

    plt.savefig(PLOT_OUTPUT, dpi=300)
    print(f"Grafico salvato con successo in: {PLOT_OUTPUT}")
    
    plt.show()

if __name__ == "__main__":
    plot_optimization(HISTORY_FILE)