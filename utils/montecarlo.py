import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import re

# ==========================================
# IMPOSTAZIONI
# ==========================================
FILE_INPUT = "base_esplorativa_proxy.txt"

NUM_TENTATIVI = 500  
DEV_STD_RUMORE = 20.0  


def valuta_con_mobd(nome_file):
    """
    Chiama l'eseguibile mobd.exe per farsi calcolare il valore reale della Black-Box.
    """
    try:
        comando = ["mobd.exe", nome_file, "-b"]

        # Se sei su Linux o Mac, commenta la riga sopra e decommenta quella sotto:
        # comando = ["./mobd", nome_file, "-b"]

        risultato = subprocess.run(comando, capture_output=True, text=True, check=True)
        output = risultato.stdout

        numeri = re.findall(r"[-+]?\d*\.\d+|\d+", output)
        if numeri:
            return float(numeri[-1])
        else:
            return float('inf')
    except FileNotFoundError:
        print("Eseguibile 'mobd.exe' non trovato!")
        print("Assicurati di aver incollato mobd.exe nella stessa cartella di questo script Python.")
        return float('inf')
    except Exception as e:
        print(f"Errore di esecuzione con mobd.exe: {e}")
        return float('inf')


def esegui_ricerca_montecarlo():
    if not os.path.exists(FILE_INPUT):
        print(f"'{FILE_INPUT}' non trovato! Esegui prima l'Espansione Termica.")
        return

    # 1. Carichiamo il seme di base
    P_base = np.loadtxt(FILE_INPUT).reshape(-1, 2)

    # Variabili per tenere traccia del vincitore
    miglior_P = None
    miglior_punteggio = float('inf')  # Partiamo da infinito, cerchiamo il minimo
    peggior_punteggio = 0

    # Liste per plottare il grafico dei punteggi a fine esecuzione
    storico_punteggi = []
    tentativi_validi = []

    print(f"Avvio Random Search: Valutazione di {NUM_TENTATIVI} configurazioni tramite mobd.exe...")

    file_temporaneo = "temp_candidato.txt"

    # ==========================================
    # FASE DI RICERCA (CICLO MONTE CARLO)
    # ==========================================
    for i in range(NUM_TENTATIVI):
        # Stampa il contatore sovrascrivendo la stessa riga del terminale (grazie a \r)
        print(f"Calcolo e valutazione configurazione {i + 1}/{NUM_TENTATIVI}...", end='\r', flush=True)

        # Generiamo la perturbazione gaussiana per tutti i moduli
        rumore = np.random.normal(loc=0.0, scale=DEV_STD_RUMORE, size=(1000, 2))
        P_candidato = P_base + rumore

        # Salviamo su disco il candidato per darlo in pasto all'eseguibile
        np.savetxt(file_temporaneo, P_candidato.flatten(), fmt='%.6f')

        # Interroghiamo la vera Black-Box
        punteggio = valuta_con_mobd(file_temporaneo)

        if punteggio == float('inf'):
            continue  # Salta questo tentativo se mobd.exe ha fallito

        # Registriamo il punteggio per il grafico finale
        storico_punteggi.append(punteggio)
        tentativi_validi.append(i + 1)

        # Aggiorniamo le statistiche
        if punteggio > peggior_punteggio:
            peggior_punteggio = punteggio

        # Se questo candidato è il migliore visto finora, lo salviamo!
        if punteggio < miglior_punteggio:
            miglior_punteggio = punteggio
            miglior_P = P_candidato.copy()
            # Usiamo \n all'inizio per andare a capo e non sovrascrivere il contatore
            print(f"\n Nuovo record al tentativo {i + 1}! Valore abbassato a: {miglior_punteggio:.2f}")

    # Pulizia: eliminiamo il file temporaneo usato per le valutazioni
    if os.path.exists(file_temporaneo):
        os.remove(file_temporaneo)

    # Aggiungiamo un a capo \n extra prima del sommario finale
    print("\n" + "-" * 50)
    print(f"RISULTATI DELLA RICERCA:")
    print(f"   - Peggior configurazione generata: {peggior_punteggio:.2f} (Tante collisioni)")
    print(f"   - Miglior configurazione trovata:  {miglior_punteggio:.2f} (Reticolo ottimizzato)")
    print("-" * 50)

    # Salviamo la configurazione vincitrice
    nome_out = "migliore_perturbazione_montecarlo.txt"
    np.savetxt(nome_out, miglior_P.flatten(), fmt='%.6f')
    print(f"La variante vincitrice è stata salvata in '{nome_out}'")

    # ==========================================
    # PLOT DEL RISULTATO (Confronto Base vs Vincitore + Andamento Punteggi)
    # ==========================================

    # --- FIGURA 1: Mappa delle coordinate ---
    plt.figure(figsize=(8, 8))
    plt.scatter(P_base[:, 0], P_base[:, 1],
                c='lightgray', s=15, alpha=0.8, edgecolor='black', linewidth=0.3,
                label="Seme di Partenza")

    plt.scatter(miglior_P[:, 0], miglior_P[:, 1],
                c='gold', s=15, alpha=0.9, edgecolor='black', linewidth=0.4,
                label=f"Vincitore (Miglior f4)")

    plt.title(f"Disposizione Spaziale (Rumore: {DEV_STD_RUMORE} km)", fontsize=13, fontweight='bold')
    plt.xlabel("Coordinata X (km)")
    plt.ylabel("Coordinata Y (km)")
    plt.axis('equal')
    plt.xlim(-250, 250)
    plt.ylim(-250, 250)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # --- FIGURA 2: Grafico dei punteggi f4 ---
    plt.figure(figsize=(10, 6))
    if storico_punteggi:
        plt.plot(tentativi_validi, storico_punteggi, color='dodgerblue', alpha=0.3, zorder=1)
        plt.scatter(tentativi_validi, storico_punteggi, color='royalblue', s=25, zorder=2, label="Tentativi Valutati")

        indice_migliore = storico_punteggi.index(miglior_punteggio)
        iter_migliore = tentativi_validi[indice_migliore]

        plt.scatter([iter_migliore], [miglior_punteggio],
                    color='red', edgecolor='black', s=100, zorder=3,
                    label=f"Migliore Assoluto: {miglior_punteggio:.2f}\n(Tentativo #{iter_migliore})")

        plt.title("Evoluzione Punteggi Black-Box (mobd.exe)", fontsize=13, fontweight='bold')
        plt.xlabel("Numero del Tentativo")
        plt.ylabel("Valore Restituito dalla Black-Box")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
    else:
        plt.text(0.5, 0.5, "Nessun punteggio valido registrato", ha='center', va='center')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    esegui_ricerca_montecarlo()