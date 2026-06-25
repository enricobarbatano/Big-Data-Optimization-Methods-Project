import numpy as np
import matplotlib.pyplot as plt

def plot(filename):

    # 1. Carica i dati dal file x.txt
    data = np.loadtxt(filename)
    points = data.reshape(-1, 2)

    x = points[:, 0]
    y = points[:, 1]

    # 2. Crea la figura e l'oggetto "asse" (ax)
    fig, ax = plt.subplots(figsize=(6, 6)) # Impostiamo una figura di base quadrata (6x6 pollici)

    # 3. Disegna i punti
    ax.scatter(x, y, color='blue', edgecolors='k', alpha=0.7, s=50, label='Punti')

    # 4. Trova il minimo e il massimo assoluto tra TUTTI i dati (sia X che Y)
    # Questo serve per definire un'area di visualizzazione perfettamente quadrata
    lim_min = min(x.min(), y.min())
    lim_max = max(x.max(), y.max())

    # Aggiungiamo un piccolo margine del 5% per non far toccare i punti ai bordi del grafico
    margin = (lim_max - lim_min) * 0.05
    griglia_min = lim_min - margin
    griglia_max = lim_max + margin

    # 5. FORZA gli assi ad avere gli stessi identici limiti
    ax.set_xlim(griglia_min, griglia_max)
    ax.set_ylim(griglia_min, griglia_max)

    # 6. FORZA il rapporto di forma (aspect ratio) a 1:1
    # 'adjustable=box' costringe il riquadro del grafico a ridimensionarsi pur di mantenere le scale uguali
    ax.set_aspect('equal', adjustable='box')

    # 7. Personalizzazione del grafico
    ax.set_xlabel('Coordinata x1')
    ax.set_ylabel('Coordinata x2')
    ax.set_title('Grafico con Scala 1:1')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    # Salva il grafico
    plt.savefig(r'.\points_opt\grafico_scala_perfetta.png', bbox_inches='tight')

    # Mostra a schermo (se eseguito localmente)
    plt.show()