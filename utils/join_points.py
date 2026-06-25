import os
import glob

# --- CONFIGURAZIONE ---
INPUT_DIR = 'dataset_perturbato'      
OUTPUT_FILE = 'dataset_unito.txt'     
SEPARATOR = '==='                     

def unisci_file(input_dir, output_file, separator):
    pattern = os.path.join(input_dir, 'x_perturbato_*.txt')
    file_list = sorted(glob.glob(pattern))
    
    if not file_list:
        print(f"Nessun file trovato nella cartella '{input_dir}'.")
        return

    print(f"Trovati {len(file_list)} file. Inizio l'unione...")
    
    with open(output_file, 'w') as outfile:
        for i, filepath in enumerate(file_list):
            with open(filepath, 'r') as infile:
                content = infile.read()
                outfile.write(content)
                
                if i < len(file_list) - 1:
                    outfile.write(f'\n{separator}\n')

    print(f"File uniti con successo in: '{output_file}'")

if __name__ == "__main__":
    unisci_file(INPUT_DIR, OUTPUT_FILE, SEPARATOR)