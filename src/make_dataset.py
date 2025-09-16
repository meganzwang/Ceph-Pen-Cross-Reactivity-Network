import pandas as pd
from pathlib import Path
from src.chem_utils import fetch_smiles_pubchem, standardize_smiles

DATA_DIR = Path("data")

def main():
    df = pd.read_csv(DATA_DIR / "drugs.csv")
    out = []
    for _, row in df.iterrows():
        name = row["name"]
        cls = row["class"]
        smi = fetch_smiles_pubchem(name)
        std = standardize_smiles(smi) if smi else None
        out.append({"name": name, "class": cls, "smiles": std})
        print(f"{name:20s} -> {std if std else 'NOT FOUND'}")
    out_df = pd.DataFrame(out)
    missing = out_df[out_df["smiles"].isna()]
    if not missing.empty:
        print("\nWARNING: Missing SMILES for:")
        print(missing[["name","class"]].to_string(index=False))
    out_df.to_csv(DATA_DIR / "drugs_with_smiles.csv", index=False)
    print(f"\nSaved: {DATA_DIR/'drugs_with_smiles.csv'}")

if __name__ == "__main__":
    main()