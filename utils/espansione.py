import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# IMPOSTAZIONI
# ==========================================
FILE_INPUT = r".\points_opt\f1f2f3minimo.txt"

# --- PARAMETRI DELL'ESPANSIONE TERMICA ---
FATTORE_DILATAZIONE = 3.0 
RUMORE_TERMICO = 30.0 

# --- PARAMETRI DI RAFFREDDAMENTO (Micro-rilassamento) ---
PASSI_RILASSAMENTO = 40
MAX_STEP_RILASSAMENTO = 2.0 


def gradiente_repulsivo_spaziale(P):
    """Forza repulsiva per il raffreddamento (raggio d'azione ridotto a 10km)"""
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)

    exp_term = np.exp(-dist_sq / (10.0 ** 2))
    np.fill_diagonal(exp_term, 0) 

    forze = exp_term[:, :, np.newaxis] * diff
    return np.sum(forze, axis=1)


def esegui_espansione_termica():
    if not os.path.exists(FILE_INPUT):
        print(f"Errore: '{FILE_INPUT}' non trovato")
        return

    P_base = np.loadtxt(FILE_INPUT).reshape(-1, 2)
    P_originale = P_base.copy()

    print("Avvio Espansione Termica Stocastica...")

    P_dilatato = P_base * FATTORE_DILATAZIONE

    rumore = np.random.normal(loc=0.0, scale=RUMORE_TERMICO, size=(1000, 2))
    P_nuovo = P_dilatato + rumore

    for step in range(PASSI_RILASSAMENTO):
        spinta = gradiente_repulsivo_spaziale(P_nuovo)

        # Gradient Clipping
        norme = np.linalg.norm(spinta, axis=1, keepdims=True)
        spinta_clippata = spinta * np.where(norme > MAX_STEP_RILASSAMENTO, MAX_STEP_RILASSAMENTO / (norme + 1e-8), 1.0)

        P_nuovo = P_nuovo + spinta_clippata

    # Salvataggio
    nome_out = "base_esplorativa_proxy.txt"
    np.savetxt(nome_out, P_nuovo.flatten(), fmt='%.6f')
    print(f"Espansione conclusa! Seme salvato in '{nome_out}'")


if __name__ == "__main__":
    esegui_espansione_termica()