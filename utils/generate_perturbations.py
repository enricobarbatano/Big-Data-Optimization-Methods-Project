import numpy as np
import os

# --- CONFIGURAZIONE ---
INPUT_FILE = '.\points_opt\migliore_perturbazione_montecarlo.txt'
OUTPUT_DIR = 'dataset_perturbato'  
NUM_SAMPLES = 15000                 
STD_DEV_1 = 1.0                    
STD_DEV_2 = 2.0                   

def load_data(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            cleaned_line = line.split(']')[-1].strip()
            if not cleaned_line:
                continue
            try:
                data.append(float(cleaned_line))
            except ValueError:
                pass
    return np.array(data)

def generate_and_save_dataset(base_vector, num_samples, std_dev_1, std_dev_2, output_dir):
    """
    Genera le perturbazioni usando due STD_DEV diverse e le salva nei file.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    half_1 = num_samples // 100 * 40
    half_2 = num_samples - half_1
    
    print(f"Generazione in corso: {half_1} campioni con STD={std_dev_1}, {half_2} campioni con STD={std_dev_2}...")
    
    noise_1 = np.random.normal(loc=0.0, scale=std_dev_1, size=(half_1, len(base_vector)))
    noise_2 = np.random.normal(loc=0.0, scale=std_dev_2, size=(half_2, len(base_vector)))
    
    noise_completo = np.vstack((noise_1, noise_2))
    
    # Applica il rumore al vettore originale tramite broadcasting
    perturbed_matrix = base_vector + noise_completo
    
    print(f"Salvataggio di {num_samples} file nella cartella '{output_dir}'...")
    
    # Salva ogni campione come file di testo indipendente
    for i, sample in enumerate(perturbed_matrix):
        filename = os.path.join(output_dir, f'x_perturbato_{i:04d}.txt')
        
        testo_formattato = '\n'.join([f"{val:.15f}" for val in sample])
        
        with open(filename, 'w') as f:
            f.write(testo_formattato)

if __name__ == "__main__":
    print(f"Lettura del vettore di base da: {INPUT_FILE}...")
    base_vector = load_data(INPUT_FILE)
    
    if len(base_vector) == 0:
        print("Errore: Nessun dato numerico trovato nel file.")
    else:
        print(f"Vettore caricato con successo. Dimensione: {len(base_vector)}")
        generate_and_save_dataset(base_vector, NUM_SAMPLES, STD_DEV_1, STD_DEV_2, OUTPUT_DIR)
        print("Operazione completata!")