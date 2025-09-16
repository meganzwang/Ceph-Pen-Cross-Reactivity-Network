import time
from typing import Optional
try:
    import pubchempy as pcp
except ImportError:
    pcp = None  # optional; only needed by make_dataset

from rdkit import Chem
import torch
from torch_geometric.data import Data
from rdkit.Chem.rdchem import HybridizationType as Hyb

# Try to import rdMolStandardize (optional)
try:
    from rdkit.Chem import rdMolStandardize
    HAS_STD = True
except Exception:
    rdMolStandardize = None
    HAS_STD = False

# Atom/bond vocab
ATOM_LIST = ['C','N','O','S','F','Cl','Br','I','P','H']
HYB_LIST = [Hyb.SP, Hyb.SP2, Hyb.SP3, Hyb.SP3D, Hyb.SP3D2]
IN_DIM = len(ATOM_LIST) + 4 + len(HYB_LIST)  # 19

def fetch_smiles_pubchem(name: str, pause: float = 0.2) -> Optional[str]:
    if pcp is None:
        raise ImportError("pubchempy not installed. Install with 'pip install pubchempy' or 'conda install -c conda-forge pubchempy'.")
    
    # Common name mappings for drugs that might have issues
    name_mappings = {
        'Penicillin G': 'Benzylpenicillin',
        'Penicillin V': 'Phenoxymethylpenicillin',
        'Ceftolozane': 'Ceftolozane/tazobactam',  # Common combination drug
    }
    
    # Try the mapped name if it exists, otherwise try the original name
    names_to_try = [name_mappings.get(name, name)]
    
    # For combination drugs, also try individual components
    if '/' in name:
        names_to_try.extend([n.strip() for n in name.split('/')])
    
    for name_variant in names_to_try:
        try:
            # First try exact name match
            res = pcp.get_compounds(name_variant, 'name', listkey_count=3)
            time.sleep(pause)
            
            # If no results, try a more flexible search
            if not res:
                res = pcp.get_compounds(name_variant, 'name', searchtype='similarity')
                time.sleep(pause)
            
            if res:
                # Prefer exact matches if available
                exact_matches = [c for c in res if c.iupac_name and name_variant.lower() in c.iupac_name.lower()]
                if exact_matches:
                    return exact_matches[0].isomeric_smiles or exact_matches[0].canonical_smiles
                # Otherwise return the first result
                return res[0].isomeric_smiles or res[0].canonical_smiles
                
        except Exception as e:
            print(f"Error searching for {name_variant}: {str(e)}")
            time.sleep(pause)  # Be nice to PubChem's servers
            continue
            
    return None

def standardize_smiles(smiles: str) -> Optional[str]:
    # Training does not depend on this; it’s used by make_dataset only.
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        if HAS_STD:
            rdMolStandardize.Cleanup(mol)
            uncharger = rdMolStandardize.Uncharger()
            mol = uncharger.uncharge(mol)
        Chem.SanitizeMol(mol)
        try:
            Chem.Kekulize(mol, clearAromaticFlags=True)
        except Exception:
            pass
        return Chem.MolToSmiles(mol, isomeric=True, canonical=True)
    except Exception:
        return None

def atom_features(a: Chem.Atom) -> torch.Tensor:
    return torch.tensor(
        [int(a.GetSymbol()==el) for el in ATOM_LIST] +
        [a.GetTotalDegree(),
         a.GetFormalCharge(),
         int(a.GetIsAromatic()),
         int(a.IsInRing())] +
        [int(a.GetHybridization()==hyb) for hyb in HYB_LIST],
        dtype=torch.float
    )

def bond_features(b: Chem.Bond) -> torch.Tensor:
    bt = b.GetBondType()
    return torch.tensor([
        int(bt == Chem.BondType.SINGLE),
        int(bt == Chem.BondType.DOUBLE),
        int(bt == Chem.BondType.TRIPLE),
        int(bt == Chem.BondType.AROMATIC),
        int(b.GetIsConjugated()),
        int(b.IsInRing())
    ], dtype=torch.float)

def smiles_to_graph(smi: str) -> Data:
    mol = Chem.MolFromSmiles(smi)
    atoms = mol.GetAtoms()
    x = torch.stack([atom_features(a) for a in atoms], dim=0)
    e_idx, e_attr = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bf = bond_features(b)
        e_idx += [[i, j], [j, i]]
        e_attr += [bf, bf]
    edge_index = torch.tensor(e_idx, dtype=torch.long).t().contiguous()
    edge_attr = torch.stack(e_attr, dim=0) if e_attr else torch.zeros((0,6), dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)