import torch
import csv
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
from model_architecture import EcoSieveVAE
from data_loader import ChemicalDataset

# --- CONFIGURACIÓN ---
DEVICE = torch.device("cpu")
MODEL_PATH = "ecosieve_pro.pth"
CSV_PATH = "data/molecules.csv"
NUM_GENERATE = 500   # Generar masivamente
MIN_WEIGHT = 100     # Ignorar cosas muy chicas como 'C'
MAX_WEIGHT = 600     # Ignorar cosas gigantes

def get_properties(mol):
    """Calcula propiedades fisicoquímicas clave"""
    try:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        qed_score = QED.qed(mol) # 0.0 (Malo) a 1.0 (Droga Perfecta)
        
        # Regla de Lipinski:
        # PM < 500, LogP < 5, Donantes H < 5, Aceptores H < 10
        es_lipinski = (mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10)
        
        return {
            "MW": round(mw, 2),
            "LogP": round(logp, 2),
            "QED": round(qed_score, 3),
            "Lipinski": es_lipinski
        }
    except:
        return None

def main():
    print(f" INICIANDO PIPELINE DE DESCUBRIMIENTO")
    
    # 1. Cargar Modelo (con el fix de vocabulario)
    dataset = ChemicalDataset(CSV_PATH, limit=None, max_len=85)
    vocab_size = len(dataset.vocab)
    
    model = EcoSieveVAE(vocab_size, hidden_dim=512, latent_dim=128).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    # 2. Generación Masiva
    print(f"⚗️  Generando {NUM_GENERATE} candidatos brutos...")
    
    sos_idx = dataset.vocab['<sos>']
    eos_idx = dataset.vocab['<eos>']
    inv_vocab = dataset.inv_vocab
    
    candidates = []
    
    with torch.no_grad():
        z = torch.randn(NUM_GENERATE, 128).to(DEVICE)
        d_hidden = model.decoder_input(z).unsqueeze(0)
        curr = torch.tensor([[sos_idx]] * NUM_GENERATE).to(DEVICE)
        
        finished = [False] * NUM_GENERATE
        sequences = [[] for _ in range(NUM_GENERATE)]
        
        for _ in range(85):
            emb = model.embedding(curr)
            out, d_hidden = model.decoder_rnn(emb, d_hidden)
            logits = model.fc_out(out).squeeze(1)
            
            # Usamos un poco mas de temperatura para variedad
            probs = torch.softmax(logits / 1.0, dim=1)
            next_token = torch.multinomial(probs, 1)
            curr = next_token
            
            tokens = next_token.cpu().numpy().flatten()
            for i, t in enumerate(tokens):
                if not finished[i]:
                    if t == eos_idx: finished[i] = True
                    else: sequences[i].append(t)

    # 3. Filtrado y Análisis
    print("Analizando propiedades moleculares")
    
    valid_data = []
    seen_smiles = set()
    
    for seq in sequences:
        s = ""
        for token in seq:
            char = inv_vocab.get(token, '')
            if char not in ['<pad>', '<sos>', '<eos>']: s += char
            
        if s not in seen_smiles and len(s) > 1:
            mol = Chem.MolFromSmiles(s)
            if mol:
                props = get_properties(mol)
                if props and props["MW"] >= MIN_WEIGHT: # Filtro de peso mínimo
                    row = {
                        "SMILES": s,
                        **props
                    }
                    valid_data.append(row)
                    seen_smiles.add(s)

    # 4. Ranking (Las mejores drogas primero)
    # Ordenamos por QED (Quantitative Estimation of Drug-likeness)
    valid_data.sort(key=lambda x: x["QED"], reverse=True)
    
    top_candidates = valid_data[:20] # Top 20
    
    # 5. Reporte
    print(f"RESUMEN:")
    print(f"   Generadas: {NUM_GENERATE}")
    print(f"   Válidas Únicas: {len(valid_data)}")
    print(f"   Candidatos Viables (Top 20 guardados):")
    
    filename = "CANDIDATOS_TOP.csv"
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["SMILES", "MW", "LogP", "QED", "Lipinski"])
        writer.writeheader()
        
        print(f"\n{'RANK':<4} | {'QED':<6} | {'PESO':<8} | {'LOGP':<6} | {'SMILES'}")
        print("-" * 60)
        
        for i, row in enumerate(top_candidates):
            writer.writerow(row)
            print(f"#{i+1:<3} | {row['QED']:<6} | {row['MW']:<8} | {row['LogP']:<6} | {row['SMILES']}")

    print(f"Archivo guardado: {filename}")
    print("¡Listo para enviar al laboratorio virtual! 🚀")

if __name__ == "__main__":
    main()