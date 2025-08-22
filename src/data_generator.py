import numpy as np
import torch
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial.transform import Rotation as R
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_3d_coords(mol_input):
    try:
        mol = Chem.MolFromSmiles(mol_input)
        if mol is None:
            logging.warning(f"Failed to parse SMILES: {mol_input}")
            return None
        mol = Chem.AddHs(mol)
        # Use single conformation attempt
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        conf = mol.GetConformer()
        coords = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
        return coords
    except Exception as e:
        logging.error(f"Error processing {mol_input}: {e}")
        return None

def apply_random_rotation(coords):
    rotation = R.random().as_matrix()
    return coords @ rotation.T

def generate_molecular_data(csv_path='qm9.csv', n_samples=1000, max_atoms=15, noise=0.1):
    logging.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path, header=0).head(n_samples)
    coords_list = []
    labels = []
    valid_samples = 0

    for i, row in df.iterrows():
        smiles = row.get('smiles', None)
        if smiles is None or pd.isna(smiles):
            logging.warning(f"Missing smiles at row {i}")
            continue

        homo = row.get('homo', row.get('HOMO', 0.0))
        lumo = row.get('lumo', row.get('LUMO', 0.0))
        if pd.isna(homo) or pd.isna(lumo):
            logging.warning(f"Missing HOMO/LUMO at row {i}")
            continue

        coords = get_3d_coords(smiles)
        if coords is None or coords.shape[0] > max_atoms:
            logging.warning(f"Invalid coords for {smiles} at row {i}")
            continue

        coords = apply_random_rotation(coords)
        coords += np.random.normal(0, noise, coords.shape)
        padded = np.zeros((max_atoms, 3))
        padded[:coords.shape[0]] = coords[:max_atoms]
        coords_list.append(padded.flatten())  # Up to 45 features
        gap = lumo - homo
        labels.append(gap)
        valid_samples += 1
        if valid_samples >= n_samples:
            break

    if not coords_list:
        logging.error("No valid samples generated. Check CSV format and RDKit setup.")
        return torch.tensor([]), torch.tensor([])

    data = np.array(coords_list)
    labels = np.array(labels)
    logging.info(f"Generated {len(coords_list)} samples with shape {data.shape}")
    return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

if __name__ == "__main__":
    train_data, train_labels = generate_molecular_data(n_samples=800)
    test_data, test_labels = generate_molecular_data(n_samples=200)
    if train_data.numel() == 0 or test_data.numel() == 0:
        logging.error("Data generation failed. Check logs for details.")
    else:
        torch.save((train_data, train_labels, test_data, test_labels), 'data.pt')