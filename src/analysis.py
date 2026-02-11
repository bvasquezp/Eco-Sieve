import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA

# --- IMPORTS DE RDKIT CORREGIDOS ---
from rdkit import Chem
from rdkit.Chem import Descriptors  # <--- ¡Importante! Esto faltaba
from rdkit.Chem import AllChem, DataStructs
from rdkit import RDLogger

# --- CONFIGURACIÓN DE RUTAS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

CSV_TRAIN = os.path.join(PROJECT_ROOT, 'data', 'molecules.csv')
# Intenta buscar el archivo con mayúsculas, si no, lo buscaremos en check_files
CSV_GEN_DEFAULT = os.path.join(PROJECT_ROOT, 'results', 'CANDIDATOS_TOP.csv') 

SAMPLE_SIZE = 1000

# Desactivar logs
RDLogger.DisableLog('rdApp.*')

def check_files():
    print(f" Directorio del proyecto: {PROJECT_ROOT}")
    
    if not os.path.exists(CSV_TRAIN):
        raise FileNotFoundError(f" No encuentro el dataset en: {CSV_TRAIN}")
    else:
        print(f"Dataset encontrado.")

    # Lógica para encontrar el archivo generado (Mayúsculas o Minúsculas)
    if os.path.exists(CSV_GEN_DEFAULT):
        print(f"Candidatos encontrados.")
        return CSV_GEN_DEFAULT
    
    # Intento alternativo (minúsculas)
    alt_path = os.path.join(PROJECT_ROOT, 'results', 'candidates_top.csv')
    if os.path.exists(alt_path):
        print(f"Candidatos encontrados (versión minúscula).")
        return alt_path 
        
    raise FileNotFoundError(f"No encuentro el archivo generado en results/")

def load_data(gen_path):
    print("Cargando datos...")
    df_train = pd.read_csv(CSV_TRAIN).sample(SAMPLE_SIZE)
    df_gen = pd.read_csv(gen_path)
    return df_train, df_gen

def plot_distributions(df_train, df_gen):
    print("Generando gráfico de distribuciones...")
    
    data = []
    
    # Datos de entrenamiento
    for s in df_train.iloc[:, 0]:
        mol = Chem.MolFromSmiles(s)
        if mol: 
            data.append({
                "MW": Descriptors.MolWt(mol),      # <--- Corregido
                "LogP": Descriptors.MolLogP(mol),  # <--- Corregido
                "Source": "Entrenamiento"
            })
            
    # Datos generados
    for s in df_gen["SMILES"]:
        mol = Chem.MolFromSmiles(s)
        if mol: 
            data.append({
                "MW": Descriptors.MolWt(mol),      # <--- Corregido
                "LogP": Descriptors.MolLogP(mol),  # <--- Corregido
                "Source": "IA Generada"
            })
    
    df_final = pd.DataFrame(data)
    
    # Graficar
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.kdeplot(data=df_final, x="MW", hue="Source", fill=True, ax=ax[0], palette=["grey", "blue"])
    ax[0].set_title("Distribución de Peso Molecular")
    
    sns.kdeplot(data=df_final, x="LogP", hue="Source", fill=True, ax=ax[1], palette=["grey", "blue"])
    ax[1].set_title("Distribución de LogP")
    
    # Guardar
    out_dir = os.path.join(PROJECT_ROOT, 'results', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'dist_comparison.png')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f" Gráfico guardado en: {out_path}")

def plot_pca(df_train, df_gen):
    print(" Calculando PCA del Espacio Químico...")
    
    def get_fp_arr(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                arr = np.zeros((0,), dtype=np.int8)
                DataStructs.ConvertToNumpyArray(fp, arr)
                return arr
        except: pass
        return None

    # Obtener fingerprints
    fps_train = [get_fp_arr(s) for s in df_train.iloc[:, 0] if get_fp_arr(s) is not None]
    fps_gen = [get_fp_arr(s) for s in df_gen["SMILES"] if get_fp_arr(s) is not None]
    
    if not fps_train or not fps_gen:
        print("No hay suficientes datos para PCA.")
        return

    X = np.concatenate([fps_train, fps_gen])
    y = ["Entrenamiento"] * len(fps_train) + ["IA Generada"] * len(fps_gen)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y, alpha=0.6, palette=['lightgrey', 'blue'])
    plt.title("Espacio Químico (PCA)")
    
    out_dir = os.path.join(PROJECT_ROOT, 'results', 'figures')
    os.makedirs(out_dir, exist_ok=True) # Asegurar que existe la carpeta
    out_path = os.path.join(out_dir, 'chemical_space_pca.png')
    
    plt.savefig(out_path, dpi=300)
    print(f"PCA guardado en: {out_path}")

if __name__ == "__main__":
    try:
        real_gen_path = check_files()
        df_t, df_g = load_data(real_gen_path)
        plot_distributions(df_t, df_g)
        plot_pca(df_t, df_g)
        print("¡Análisis completo finalizado!")
    except Exception as e:
        print(f"Error: {e}")