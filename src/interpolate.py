import torch
import numpy as np
from model_architecture import EcoSieveVAE
from data_loader import ChemicalDataset

# --- CONFIGURACION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "ecosieve_vae.pth"
STEPS = 10  # Pasos intermedios para la metamorfosis

# Datos (Vocabulario)
raw_data = [
    "C=CC(=O)O", "c1ccccc1", "C=CC#N", "O=C(O)c1ccccc1C(=O)O",
    "NC(=O)C=C", "C1COCCO1", "C(=O)(O)c1ccc(cc1)C(=O)O",
    "OCCO", "C1=CC=C(C=C1)N", "ClC=C"
]

def interpolate():
    print("--- Iniciando Interpolación Química ---")
    
    # 1. Cargar Datos y Modelo
    dataset = ChemicalDataset(raw_data, max_len=30)
    inv_vocab = dataset.inv_vocab
    vocab_size = len(dataset.vocab)
    
    model = EcoSieveVAE(vocab_size=vocab_size).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 2. Definir Inicio y Fin (Metamorfosis)
    # Vamos a transformar Benceno (Anillo) en Acrilamida (Cadena lineal)
    start_smile = "c1ccccc1" 
    end_smile = "NC(=O)C=C"
    
    print(f"Inicio: {start_smile}")
    print(f"Fin:    {end_smile}")
    print("-" * 30)

    # 3. Convertir a Tensores (Codificar)
    with torch.no_grad():
        # Tokenizar
        t_start = dataset.tokenize(start_smile).unsqueeze(0).to(DEVICE)
        t_end = dataset.tokenize(end_smile).unsqueeze(0).to(DEVICE)
        
        # Pasar por Encoder para obtener sus vectores latentes (Mu)
        # No usamos reparameterize porque queremos una transicion limpia, no aleatoria
        _, hidden_start = model.encoder_rnn(model.embedding(t_start))
        _, hidden_end = model.encoder_rnn(model.embedding(t_end))
        
        # Proyectar al espacio latente
        z_start = model.fc_mu(hidden_start.squeeze(0))
        z_end = model.fc_mu(hidden_end.squeeze(0))

        # 4. El Bucle de Interpolación
        # Caminamos en línea recta desde el vector A al vector B
        for i in range(STEPS + 1):
            alpha = i / STEPS  # 0.0 -> 0.1 -> ... -> 1.0
            
            # Formula de interpolacion lineal: Z = (1-a)*Start + a*End
            z_inter = (1 - alpha) * z_start + alpha * z_end
            
            # --- DECODIFICAR EL PUNTO INTERMEDIO ---
            d_hidden = model.decoder_input(z_inter).unsqueeze(0)
            
            # Iniciar secuencia
            current_token = torch.tensor([[dataset.vocab['<sos>']]]).to(DEVICE)
            
            mol_str = ""
            
            # Generar atomo por atomo
            for _ in range(30):
                embedded = model.embedding(current_token)
                output, d_hidden = model.decoder_rnn(embedded, d_hidden)
                prediction = model.fc_out(output)
                current_token = torch.argmax(prediction, dim=2)
                
                token_idx = current_token.item()
                if token_idx == dataset.vocab['<eos>']:
                    break
                
                char = inv_vocab.get(token_idx, '')
                if char not in ['<pad>', '<sos>', '<eos>']:
                    mol_str += char
            
            print(f"Paso {i}/{STEPS} (Mezcla {int(alpha*100)}%): {mol_str}")

if __name__ == "__main__":
    interpolate()