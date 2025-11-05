import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors


def load_drug_metadata():
    """Load drug class information from drug_smiles.csv"""
    df = pd.read_csv('data/drug_smiles.csv')
    metadata = {}
    for _, row in df.iterrows():
        metadata[row['drug_name']] = {
            'category': row['category'],
            'class_code': row['class_code'],
            'smiles': row['smiles']
        }
    return metadata


def extract_structural_features(drug1, drug2, metadata):
    """
    Extract 10 structural features from existing data.
    No DrugBank needed!
      
    Returns:
        np.array of shape (10,)
    """
    info1 = metadata[drug1]
    info2 = metadata[drug2]

    features = []

    # === Drug Class Features (6 features) ===

    # 1. Same drug class (binary)
    same_class = int(info1['category'] == info2['category'])
    features.append(same_class)

    # 2. Both penicillins (binary)
    both_pen = int('penicillin' in info1['category'].lower() and
                   'penicillin' in info2['category'].lower())
    features.append(both_pen)

    # 3. Both cephalosporins (binary)
    both_ceph = int('ceph' in info1['category'].lower() and
                    'ceph' in info2['category'].lower())
    features.append(both_ceph)

    # 4. Penicillin + Cephalosporin pair (HIGH RISK) (binary)
    pen_ceph_pair = int(
        ('penicillin' in info1['category'].lower() and 'ceph' in info2['category'].lower()) or
        ('penicillin' in info2['category'].lower() and 'ceph' in info1['category'].lower())
    )
    features.append(pen_ceph_pair)

    # 5. Both carbapenems (binary)
    both_carb = int('carbapenem' in info1['category'].lower() and
                    'carbapenem' in info2['category'].lower())
    features.append(both_carb)

    # 6. Generation distance (for cephalosporins)
    # Extract generation number from class_code (1G, 2G, 3G, etc.)
    gen1 = extract_generation(info1['class_code'])
    gen2 = extract_generation(info2['class_code'])
    gen_distance = abs(gen1 - gen2) if (gen1 > 0 and gen2 > 0) else 0
    features.append(gen_distance / 5.0)  # Normalize by max distance (5)

    # === Molecular Property Differences (4 features) ===

    mol1 = Chem.MolFromSmiles(info1['smiles'])
    mol2 = Chem.MolFromSmiles(info2['smiles'])

    # 7. Molecular weight difference (normalized)
    mw_diff = abs(Descriptors.MolWt(mol1) - Descriptors.MolWt(mol2)) / 100.0
    features.append(min(mw_diff, 1.0))  # Cap at 1.0

    # 8. LogP difference (hydrophobicity)
    logp_diff = abs(Descriptors.MolLogP(mol1) - Descriptors.MolLogP(mol2))
    features.append(min(logp_diff / 5.0, 1.0))  # Normalize

    # 9. Number of aromatic rings difference
    aromatic_diff = abs(Descriptors.NumAromaticRings(mol1) -
                        Descriptors.NumAromaticRings(mol2))
    features.append(aromatic_diff / 3.0)  # Normalize

    # 10. Number of H-bond donors difference
    hbd_diff = abs(Descriptors.NumHDonors(mol1) - Descriptors.NumHDonors(mol2))
    features.append(hbd_diff / 5.0)  # Normalize

    return np.array(features, dtype=np.float32)


def extract_generation(class_code):
    """Extract generation number from class code"""
    if class_code in ['1G']:
        return 1
    elif class_code in ['2G']:
        return 2
    elif class_code in ['3G']:
        return 3
    elif class_code in ['4G']:
        return 4
    elif class_code in ['5G', '5Gpip']:
        return 5
    else:
        return 0  # Not a cephalosporin


def create_feature_dict_for_all_pairs():
    """Create features for all drug pairs and save"""
    import pickle

    metadata = load_drug_metadata()
    drug_names = list(metadata.keys())

    features_dict = {}

    for i, drug1 in enumerate(drug_names):
        for j, drug2 in enumerate(drug_names):
            if i >= j:
                continue

            feats = extract_structural_features(drug1, drug2, metadata)
            features_dict[(drug1, drug2)] = feats
            features_dict[(drug2, drug1)] = feats  # Symmetric

    # Save
    with open('data/structural_features.pkl', 'wb') as f:
        pickle.dump(features_dict, f)

    print(f"✓ Created structural features for {len(features_dict)} drug pairs")
    print(f"✓ Saved to data/structural_features.pkl")


if __name__ == '__main__':
    create_feature_dict_for_all_pairs()
