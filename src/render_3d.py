import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

# Tu molécula ganadora (La copié de tu tabla anterior)
# SMILES: C[C@H](NC(=O)c1ccco1)C(F)F
smiles = "C[C@H](NC(=O)c1ccco1)C(F)F" 
mol_name = "EcoSieve_Lead_01"

print(f" Procesando: {mol_name}")

# 1. Crear objeto molécula
mol = Chem.MolFromSmiles(smiles)
mol = Chem.AddHs(mol) # Agregar Hidrógenos (importante para 3D)

# 2. Generar coordenadas 3D (Embedding)
# Usamos un algoritmo aleatorio para proponer una forma inicial
res = AllChem.EmbedMolecule(mol, randomSeed=42)

if res == 0:
    # 3. Optimización Geométrica (MMFF94)
    # Esto usa física real (campos de fuerza) para "relajar" la molécula
    # y encontrar su forma más estable en la naturaleza.
    try:
        AllChem.MMFFOptimizeMolecule(mol)
        print("Geometría optimizada con Fuerza MMFF94")
        
        # 4. Guardar archivo .MOL (Formato estándar de la industria)
        # Este archivo se puede abrir en PyMOL, VMD o Chimera.
        print(f"💾 Guardando {mol_name}.mol ...")
        print(Chem.MolToMolBlock(mol), file=open(f"{mol_name}.mol", "w"))
        
        # 5. Generar una imagen 2D limpia también para referencia
        img = Draw.MolToImage(mol, size=(600, 600))
        img.save(f"{mol_name}_2d.png")
        print("Imagen 2D guardada.")
        
    except Exception as e:
        print(f"Error en optimización: {e}")
else:
    print("No se pudo generar la estructura 3D inicial.")