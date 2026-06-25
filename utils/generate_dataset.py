import os
import subprocess
import numpy as np

# --- CONFIGURAZIONE ---
INPUT_FILE = 'dataset_unito.txt'
OUTPUT_DATASET = 'dataset_pytorch.csv'
TEMP_X_FILE = 'x.txt'
SEPARATOR = '==='

ORACOLO_STAMPA_A_SCHERMO = True           
TEMP_Y_FILE = 'output.txt'

def process_dataset():
    print(f"Lettura del file '{INPUT_FILE}'...")
    with open(INPUT_FILE, 'r') as f:
        content = f.read()

    campioni_testo = [b.strip() for b in content.split(f'\n{SEPARATOR}\n') if b.strip()]
    num_campioni = len(campioni_testo)
    print(f"Trovati {num_campioni} campioni validi da processare.")

    start_idx = 0

    # 1. RIPRESA DEL LAVORO
    if os.path.exists(OUTPUT_DATASET) and os.path.getsize(OUTPUT_DATASET) > 0:
        print(f"Rilevato dataset parziale '{OUTPUT_DATASET}'. Calcolo da dove ci siamo fermati...")
        try:
            with open(OUTPUT_DATASET, 'r') as f:
                start_idx = sum(1 for _ in f)
            print(f"Ripristinati {start_idx} campioni già salvati. Riprendo dal campione {start_idx + 1}...")
        except Exception as e:
            print(f"Errore nella lettura del file esistente ({e}). Riparto da zero.")
            start_idx = 0

    # 2. CICLO DI ELABORAZIONE
    for i in range(start_idx, num_campioni):
        blocco = campioni_testo[i]
        
        try:
            x_features = [float(val) for val in blocco.split('\n') if val.strip()]
        except ValueError:
            print(f"ATTENZIONE: Trovati dati non numerici al campione {i + 1}. Salto il campione.")
            continue
        
        with open(TEMP_X_FILE, 'w') as f:
            f.write(blocco)

        # Esecuzione oracolo
        try:
            if ORACOLO_STAMPA_A_SCHERMO:
                risultato = subprocess.run([r'.\executables\mobd.exe', TEMP_X_FILE, '-b'], capture_output=True, text=True, check=True)
                y_label = float(risultato.stdout.strip())
            else:
                subprocess.run([r'.\executables\mobd.exe', TEMP_X_FILE, '-b'], check=True)
                with open(TEMP_Y_FILE, 'r') as f:
                    y_label = float(f.read().strip())
                    
        except subprocess.CalledProcessError as e:
            print(f"Errore oracolo al campione {i + 1}: {e.stderr if ORACOLO_STAMPA_A_SCHERMO else 'Errore'}")
            break
        except ValueError:
            print(f"L'oracolo ha restituito un valore non valido al campione {i + 1}.")
            break

        # 3. SALVATAGGIO INCREMENTALE
        riga_dataset = x_features + [y_label]
        
        with open(OUTPUT_DATASET, 'a') as f_out:
            riga_str = ','.join([f"{val:.15f}" for val in riga_dataset])
            f_out.write(riga_str + '\n')
        
        if (i + 1) % 100 == 0:
            print(f"Processati e salvati {i + 1}/{num_campioni} campioni...")
        print(f"Campione {i + 1} processato e salvato.")

    print("Elaborazione terminata.")
    
    # Pulizia file temporanei
    if os.path.exists(TEMP_X_FILE):
        os.remove(TEMP_X_FILE)
    if not ORACOLO_STAMPA_A_SCHERMO and os.path.exists(TEMP_Y_FILE):
        os.remove(TEMP_Y_FILE)

if __name__ == "__main__":
    process_dataset()