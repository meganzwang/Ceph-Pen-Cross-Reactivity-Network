"""
Data preparation pipeline for beta-lactam cross-reactivity prediction
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import os
import pickle


# Drug list from Northwestern Medicine chart
DRUGS = {
    'penicillins': [
        'Penicillin G/V', 'Oxacillin', 'Amoxicillin', 'Ampicillin', 'Piperacillin'
    ],
    '1st_gen_ceph': ['Cefadroxil', 'Cephalexin', 'Cefazolin'],
    '2nd_gen_ceph': ['Cefaclor', 'Cefoxitin', 'Cefprozil', 'Cefuroxime'],
    '3rd_gen_ceph': [
        'Cefdinir', 'Cefditoren', 'Cefixime', 'Cefotaxime',
        'Cefpodoxime', 'Ceftazidime', 'Ceftibuten', 'Ceftriaxone'
    ],
    '4th_gen_ceph': ['Cefepime'],
    '5th_gen_ceph': ['Ceftaroline', 'Ceftolozane'],
    'carbapenems': ['Ertapenem', 'Meropenem'],
    'monobactams': ['Aztreonam']
}

# Flatten drug list
ALL_DRUGS = []
for category, drugs in DRUGS.items():
    ALL_DRUGS.extend(drugs)


def load_smiles_from_csv(csv_path='data/drug_smiles.csv'):
    """
    Load SMILES from pre-collected CSV file.

    Args:
        csv_path: Path to CSV with columns: drug_name, category, smiles

    Returns:
        Dictionary mapping drug names to SMILES strings
    """
    import os

    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found!")
        print(f"\nRun: python collect_smiles.py")
        print("Or manually create CSV with format:")
        print("  drug_name,category,smiles")
        print("  Amoxicillin,penicillins,CC1(C)SC2C(...)C(=O)O")
        return {}

    df = pd.read_csv(csv_path)
    smiles_dict = {}

    for _, row in df.iterrows():
        if pd.notna(row['smiles']):
            smiles_dict[row['drug_name']] = row['smiles']

    print(f"✓ Loaded SMILES for {len(smiles_dict)} drugs from {csv_path}")
    return smiles_dict


def smiles_to_graph(smiles):
    """
    Convert SMILES string to PyTorch Geometric graph object.

    Args:
        smiles: SMILES string

    Returns:
        torch_geometric.data.Data object
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Atom features
    atom_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),  # Atomic number
            atom.GetTotalDegree(),  # Degree
            atom.GetFormalCharge(),  # Formal charge
            atom.GetHybridization().real,  # Hybridization (sp, sp2, sp3)
            atom.GetIsAromatic(),  # Is aromatic
            atom.GetTotalNumHs(),  # Number of hydrogens
        ]
        atom_features.append(features)

    x = torch.tensor(atom_features, dtype=torch.float)

    # Edge indices and features
    edge_indices = []
    edge_features = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        # Add both directions for undirected graph
        edge_indices.append([i, j])
        edge_indices.append([j, i])

        bond_feature = [
            bond.GetBondTypeAsDouble(),  # Bond order (1, 2, 3, 1.5 for aromatic)
            bond.IsInRing(),  # Is in ring
        ]

        edge_features.append(bond_feature)
        edge_features.append(bond_feature)

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def extract_labels_from_chart(csv_path='data/cross_reactivity_labels.csv'):
    """
    Extract cross-reactivity labels from CSV file.

    Args:
        csv_path: Path to CSV with columns: drug1, drug2, label

    Returns:
        DataFrame with columns: drug1, drug2, label (0=SUGGEST, 1=CAUTION, 2=AVOID)
    """
    import os

    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found!")
        print("\nYou need to manually create this file from the Northwestern chart.")
        print("Format:")
        print("  drug1,drug2,label")
        print("  Amoxicillin,Cefadroxil,2")
        print("  Penicillin G/V,Cefadroxil,1")
        print("\nLabel encoding:")
        print("  2 = AVOID (X mark)")
        print("  1 = CAUTION (triangle)")
        print("  0 = SUGGEST (empty) - optional, can be inferred")
        return pd.DataFrame(columns=['drug1', 'drug2', 'label'])

    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} labeled pairs from {csv_path}")
    return df


def create_drug_embeddings(smiles_dict):
    """
    Create molecular graph embeddings for all drugs.

    Args:
        smiles_dict: Dictionary mapping drug names to SMILES strings

    Returns:
        Dictionary mapping drug names to PyG Data objects
    """
    drug_graphs = {}

    for drug_name, smiles in smiles_dict.items():
        graph = smiles_to_graph(smiles)
        if graph is not None:
            drug_graphs[drug_name] = graph
            print(f"✓ Created graph for {drug_name}: {graph.num_nodes} atoms, {graph.num_edges} bonds")
        else:
            print(f"✗ Failed to create graph for {drug_name}")

    return drug_graphs


def create_dataset(labels_df, drug_graphs, structural_features=None):
    """
    Create training dataset from drug pairs.

    Args:
        labels_df: DataFrame with drug1, drug2, label columns
        drug_graphs: Dictionary mapping drug names to PyG graphs

    Returns:
        List of (graph1, graph2, label) tuples
    """
    dataset = []

    for _, row in labels_df.iterrows():
        drug1, drug2, label = row['drug1'], row['drug2'], row['label']

        if drug1 in drug_graphs and drug2 in drug_graphs:
            if structural_features is not None and (drug1, drug2) in structural_features:
                feats = structural_features[(drug1, drug2)]
                # Convert numpy array to torch tensor if needed
                if not isinstance(feats, torch.Tensor):
                    feats = torch.tensor(feats, dtype=torch.float)

                dataset.append((
                    drug_graphs[drug1],
                    drug_graphs[drug2],
                    torch.tensor([label], dtype=torch.long),
                    feats
                ))
            else:
                dataset.append((
                    drug_graphs[drug1],
                    drug_graphs[drug2],
                    torch.tensor([label], dtype=torch.long)
                ))

    return dataset


def split_data(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Split dataset into train/val/test sets with stratification.

    Args:
        dataset: List of (graph1, graph2, label) tuples
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        random_state: Random seed

    Returns:
        train_set, val_set, test_set
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    # Extract labels for stratification
    labels = [item[2].item() for item in dataset]

    # First split: train vs (val + test)
    train_set, temp_set = train_test_split(
        dataset,
        train_size=train_ratio,
        stratify=labels,
        random_state=random_state
    )

    # Second split: val vs test
    temp_labels = [item[2].item() for item in temp_set]
    val_set, test_set = train_test_split(
        temp_set,
        train_size=val_ratio / (val_ratio + test_ratio),
        stratify=temp_labels,
        random_state=random_state
    )

    return train_set, val_set, test_set


def get_class_weights(dataset):
    """
    Calculate class weights for imbalanced dataset.

    Args:
        dataset: List of (graph1, graph2, label) tuples

    Returns:
        torch.Tensor of class weights
    """
    labels = [item[2].item() for item in dataset]
    unique, counts = np.unique(labels, return_counts=True)

    # Inverse frequency weighting
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(unique)  # Normalize

    return torch.tensor(weights, dtype=torch.float)


def main():
    """
    Main data preparation pipeline.
    """
    print("=" * 50)
    print("Beta-Lactam Cross-Reactivity Data Preparation")
    print("=" * 50)

    # Step 1: Load SMILES from CSV
    print("\n[Step 1] Loading SMILES from CSV...")
    smiles_dict = load_smiles_from_csv('data/drug_smiles.csv')

    if len(smiles_dict) == 0:
        print("\n⚠ No SMILES loaded. Run 'python collect_smiles.py' first.")
        return

    # Step 2: Convert to molecular graphs
    print("\n[Step 2] Converting SMILES to molecular graphs...")
    drug_graphs = create_drug_embeddings(smiles_dict)

    # Step 3: Extract labels from chart
    print("\n[Step 3] Loading cross-reactivity labels...")
    labels_df = extract_labels_from_chart()

    if len(labels_df) > 0:
        print(f"Loaded {len(labels_df)} drug pairs")
        print(f"Class distribution:\n{labels_df['label'].value_counts()}")

        # Step 4: Create dataset
        print("\n[Step 4] Creating dataset...")

        # Try to load precomputed structural features (optional)
        structural_features = None
        sf_path = 'data/structural_features.pkl'
        if os.path.exists(sf_path):
            with open(sf_path, 'rb') as f:
                structural_features = pickle.load(f)
            print(f"✓ Loaded structural features from {sf_path} ({len(structural_features)//2} unique pairs approx)")

        dataset = create_dataset(labels_df, drug_graphs, structural_features)

        # Step 5: Split data
        print("\n[Step 5] Splitting data...")
        train_set, val_set, test_set = split_data(dataset)

        print(f"Train: {len(train_set)} pairs")
        print(f"Val: {len(val_set)} pairs")
        print(f"Test: {len(test_set)} pairs")

        # Step 6: Calculate class weights
        class_weights = get_class_weights(train_set)
        print(f"\nClass weights: {class_weights}")

        # Save processed data
        save_dict = {
            'train': train_set,
            'val': val_set,
            'test': test_set,
            'class_weights': class_weights,
            'drug_graphs': drug_graphs
        }

        # Also save structural features if available so downstream code can access them
        if structural_features is not None:
            save_dict['structural_features'] = structural_features

        torch.save(save_dict, 'data/processed_data.pt')

        print("\n✓ Data saved to data/processed_data.pt")
    else:
        print("\n⚠ No labels loaded. Please create labels CSV first.")


if __name__ == '__main__':
    main()
