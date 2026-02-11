import torch
import torch.nn.functional as F
import numpy as np
from rdkit import Chem # <--- El cerebro quimico
from model_architecture import EcoSieveVAE
from data_loader import ChemicalDataset
import sys

# --- CONFIGURACION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "ecosieve_vae.pth"
CSV_PATH = "data/molecules.csv"
NUM_WANTED = 10   # Cuantas moleculas validas queremos al final
TEMPERATURE = 0.6 # Bajamos un poco la temperatura para ser mas precisos
MAX_LEN = 80

def generate_molecules():
    print(f"--- Generador Eco-Sieve con RDKit (Temp: {TEMPERATURE}) ---")
    
    # 1. Cargar Vocabulario
    try:
        dataset = ChemicalDataset(CSV_PATH, max_len=MAX_LEN, limit=10000)
    except FileNotFoundError:
        print("Error: No encuentro data/molecules.csv")
        return

    inv_vocab = dataset.inv_vocab
    vocab_size = len(dataset.vocab)
    sos_idx = dataset.vocab['<sos>']
    eos_idx = dataset.vocab['<eos>']
    
    # 2. Cargar Modelo
    model = EcoSieveVAE(vocab_size=vocab_size).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except:
        print("Error cargando el modelo.")
        return
    model.eval()

    print(f"\nBuscando {NUM_WANTED} moléculas válidas...")
    
    valid_molecules = []
    attempts = 0
    
    # Bucle infinito hasta encontrar las deseadas
    while len(valid_molecules) < NUM_WANTED:
        attempts += 1
        
        # Generar Lotes de 10 en 10 para ir rapido
        batch_size = 10
        with torch.no_grad():
            z = torch.randn(batch_size, 64).to(DEVICE)
            d_hidden = model.decoder_input(z).unsqueeze(0)
            current_token = torch.tensor([[sos_idx]] * batch_size).to(DEVICE)
            
            all_tokens = []
            
            for _ in range(MAX_LEN):
                embedded = model.embedding(current_token)
                output, d_hidden = model.decoder_rnn(embedded, d_hidden)
                logits = model.fc_out(output).squeeze(1)
                
                probs = F.softmax(logits / TEMPERATURE, dim=1)
                next_token = torch.multinomial(probs, 1)
                
                all_tokens.append(next_token.cpu().numpy())
                current_token = next_token
        
        # Decodificar y Validar
        token_matrix = np.array(all_tokens).squeeze(2).T
        
        for i in range(batch_size):
            row = token_matrix[i]
            mol_str = ""
            for idx in row:
                if idx == eos_idx: break
                char = inv_vocab.get(idx, '')
                if char not in ['<pad>', '<sos>', '<eos>']:
                    mol_str += char
            
            # --- EL FILTRO DE RDKIT ---
            if len(mol_str) > 1: # Ignorar cosas muy cortas
                mol = Chem.MolFromSmiles(mol_str)
                if mol is not None:
                    # Es valida quimicamente!
                    valid_molecules.append(mol_str)
                    print(f"✅ ¡Encontrada! {mol_str}")
                    
                    if len(valid_molecules) >= NUM_WANTED:
                        break
        
        # Evitar bucle infinito si el modelo es muy malo
        if attempts > 500: # 5000 intentos totales
            print("\nEl modelo está sufriendo para generar moléculas válidas.")
            break
            
        sys.stdout.write(f"\rIntentos: {attempts*10} | Válidas: {len(valid_molecules)}")
        sys.stdout.flush()

    print("\n\n--- Colección Final ---")
    for idx, m in enumerate(valid_molecules):
        print(f"{idx+1}. {m}")

if __name__ == "__main__":
    generate_molecules()