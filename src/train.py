import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from rdkit import Chem
from rdkit import RDLogger
import numpy as np
import time
import os

# Importamos tus modulos (asegurate que sigan ahi)
from model_architecture import EcoSieveVAE
from data_loader import ChemicalDataset
import torch.nn.functional as F

# --- SILENCIAR RDKIT ---
# Para que no llene la consola de errores mientras probamos validez
RDLogger.DisableLog('rdApp.*')

# --- CONFIGURACION PROFESIONAL ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSV_PATH = "data/molecules.csv"
SAVE_PATH = "ecosieve_pro.pth"

# Hiperparámetros "Heavy Duty"
EPOCHS = 20              # Menos epocas, pero mas intensas
BATCH_SIZE = 128         # Batch mas grande para estabilidad (si tienes poca RAM, baja a 64)
LEARNING_RATE = 0.0005   # Mas lento y preciso
HIDDEN_DIM = 512         # Doble de capacidad neuronal
LATENT_DIM = 128         # Espacio latente mas rico
MAX_LEN = 85             # Un poco mas de margen

# Configuración de KL Annealing (La clave del exito)
KLD_MAX = 0.005          # El peso maximo que alcanzara
KLD_CYCLES = 4           # Cuantas veces subimos y bajamos la presion

def get_kld_weight(step, total_steps, cycles, max_val):
    """Calcula el peso ciclico para KLD"""
    cycle_len = total_steps // cycles
    pos = step % cycle_len
    ratio = pos / (cycle_len * 0.5) # Sube en la primera mitad
    return min(max_val, max_val * ratio)

def validate_model(model, dataset, device, num_samples=50):
    """Prueba de fuego: Generar y verificar validez real"""
    model.eval()
    valid_count = 0
    unique_set = set()
    
    sos_idx = dataset.vocab['<sos>']
    eos_idx = dataset.vocab['<eos>']
    inv_vocab = dataset.inv_vocab
    
    with torch.no_grad():
        # Generamos latentes
        z = torch.randn(num_samples, LATENT_DIM).to(device)
        d_hidden = model.decoder_input(z).unsqueeze(0)
        curr = torch.tensor([[sos_idx]] * num_samples).to(device)
        
        all_tokens = []
        for _ in range(MAX_LEN):
            emb = model.embedding(curr)
            out, d_hidden = model.decoder_rnn(emb, d_hidden)
            logits = model.fc_out(out).squeeze(1)
            # Greedy decoding para validacion (mas rapido y determinista)
            next_token = torch.argmax(logits, dim=1).unsqueeze(1)
            all_tokens.append(next_token.cpu().numpy())
            curr = next_token

    # Reconstruir strings
    token_mat = np.array(all_tokens).squeeze(2).T
    for i in range(num_samples):
        s = ""
        for idx in token_mat[i]:
            if idx == eos_idx: break
            char = inv_vocab.get(idx, '')
            if char not in ['<pad>', '<sos>', '<eos>']: s += char
        
        # Filtro de calidad
        if len(s) > 3: # Ignorar moleculas enanas
            mol = Chem.MolFromSmiles(s)
            if mol:
                valid_count += 1
                unique_set.add(s)
                
    valid_pct = (valid_count / num_samples) * 100
    unique_pct = (len(unique_set) / num_samples) * 100
    model.train() # Volver a modo entrenamiento
    return valid_pct, unique_pct

def train():
    print(f"--- ⚗️ INICIANDO ENTRENAMIENTO PROFESIONAL EN {DEVICE} ---")
    
    # 1. Carga de Datos (FULL)
    # Quitamos el limite. Si tu PC explota, pon limit=50000
    dataset = ChemicalDataset(CSV_PATH, max_len=MAX_LEN, limit=None) 
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    vocab_size = len(dataset.vocab)
    pad_idx = dataset.vocab['<pad>']
    
    print(f"Dataset: {len(dataset)} moléculas cargadas.")
    print(f"Vocabulario: {vocab_size} tokens.")

    # 2. Inicializar Modelo (Arquitectura mejorada)
    # Nota: Tendras que ajustar model_architecture.py para aceptar hidden_dim y latent_dim variables
    # Si no quieres editar el otro archivo, usa los valores por defecto, pero lo ideal es subirlos.
    model = EcoSieveVAE(vocab_size=vocab_size, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    total_steps = len(dataloader) * EPOCHS
    global_step = 0
    best_validity = 0.0

    print(f"\nComenzando ciclo de {EPOCHS} épocas...")
    start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_loss = 0
        epoch_recon = 0
        epoch_kld = 0
        
        for batch_idx, data in enumerate(dataloader):
            data = data.to(DEVICE)
            
            # --- KL Annealing Dinamico ---
            kld_weight = get_kld_weight(global_step, total_steps, KLD_CYCLES, KLD_MAX)
            
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            
            # Calculo de Loss
            shift_logits = recon_batch[:, :-1, :].contiguous().view(-1, vocab_size)
            shift_targets = data[:, 1:].contiguous().view(-1)
            
            BCE = F.cross_entropy(shift_logits, shift_targets, ignore_index=pad_idx, reduction='mean')
            KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            
            loss = BCE + (KLD * kld_weight)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Evitar explosiones
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_recon += BCE.item()
            epoch_kld += KLD.item()
            global_step += 1
            
            # Barra de progreso simple
            if batch_idx % 50 == 0:
                print(f"\rEpoca {epoch+1} [{batch_idx}/{len(dataloader)}] "
                      f"Loss: {loss.item():.3f} (KL-W: {kld_weight:.5f})", end="")

        # --- REPORTE AL FINAL DE LA EPOCA ---
        avg_loss = epoch_loss / len(dataloader)
        
        # Validacion Real
        print(f"\nValidando calidad química...", end="")
        valid_pct, unique_pct = validate_model(model, dataset, DEVICE)
        
        time_elapsed = (time.time() - start_time) / 60
        print(f"\n>>> RESUMEN EPOCA {epoch+1} ({time_elapsed:.1f} min)")
        print(f"    Loss Promedio: {avg_loss:.4f}")
        print(f"    Reconstrucción: {epoch_recon/len(dataloader):.4f} | KLD Real: {epoch_kld/len(dataloader):.4f}")
        print(f"    🧪 VALIDEZ QUÍMICA: {valid_pct:.1f}% | 🦄 UNICIDAD: {unique_pct:.1f}%")
        
        # Guardar el mejor modelo basado en VALIDÉZ, no en loss
        if valid_pct > best_validity:
            best_validity = valid_pct
            torch.save(model.state_dict(), SAVE_PATH)
            print("    💾 ¡Nuevo Récord de Calidad! Modelo guardado.")
        
        # Guardar checkpoint regular cada 5 epocas
        if (epoch+1) % 5 == 0:
             torch.save(model.state_dict(), f"checkpoint_ep{epoch+1}.pth")

    print("\nEntrenamiento Profesional Finalizado.")

if __name__ == "__main__":
    train()