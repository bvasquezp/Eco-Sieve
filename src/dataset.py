import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class ChemicalDataset(Dataset):
    def __init__(self, csv_file, max_len=100, smiles_col='smiles', limit=None):
        """
        Args:
            csv_file: Ruta al archivo .csv
            max_len: Longitud máxima de la cadena SMILES (cortaremos las más largas)
            smiles_col: Nombre de la columna en el CSV que tiene los datos
            limit: (Opcional) Cargar solo N filas para pruebas rápidas
        """
        self.max_len = max_len
        print(f"--- Cargando Dataset: {csv_file} ---")
        
        # 1. Cargar CSV
        # Usamos engine='python' para evitar errores con caracteres extraños
        if limit:
            df = pd.read_csv(csv_file, nrows=limit)
            print(f"Modo prueba: Cargadas solo {limit} moléculas.")
        else:
            df = pd.read_csv(csv_file)
            print(f"Cargadas {len(df)} filas crudas.")

        # 2. Limpieza y Filtrado
        # Quitamos filas vacías y nos aseguramos que sean strings
        df = df.dropna(subset=[smiles_col])
        df[smiles_col] = df[smiles_col].astype(str)
        
        # Filtramos moléculas demasiado largas (ahorran memoria y tiempo)
        # Esto elimina polímeros gigantes que complicarían el entrenamiento inicial
        mask = df[smiles_col].str.len() <= (max_len - 2) # -2 por <sos> y <eos>
        self.smiles_list = df.loc[mask, smiles_col].tolist()
        
        print(f"Moléculas válidas (longitud <= {max_len}): {len(self.smiles_list)}")

        # 3. Construir Vocabulario (El Diccionario de la IA)
        chars = set()
        for s in self.smiles_list:
            chars.update(set(s))
            
        self.chars = sorted(list(chars))
        # Mapeo Caracter -> Indice
        self.vocab = {c: i for i, c in enumerate(self.chars)}
        
        # Agregar tokens especiales al final
        self.vocab['<pad>'] = len(self.vocab)
        self.vocab['<sos>'] = len(self.vocab)
        self.vocab['<eos>'] = len(self.vocab)
        
        # Mapeo Inverso (Indice -> Caracter)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        
        print(f"Vocabulario construido: {len(self.vocab)} tokens únicos.")
        
    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smile = self.smiles_list[idx]
        
        # Tokenizar
        tokenized = [self.vocab['<sos>']] + \
                    [self.vocab[c] for c in smile] + \
                    [self.vocab['<eos>']]
        
        # Padding (Relleno)
        pad_len = self.max_len - len(tokenized)
        tokenized += [self.vocab['<pad>']] * pad_len
        
        return torch.tensor(tokenized, dtype=torch.long)