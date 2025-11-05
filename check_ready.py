"""
Pre-flight check - verify you have all required data before training
"""

import os
import pandas as pd


def check_ready():
    """Check if all required files are present and valid."""

    print("=" * 60)
    print("PRE-FLIGHT CHECK - Are you ready to train?")
    print("=" * 60)

    ready = True

    # Check 1: SMILES file
    print("\n[1/3] Checking SMILES data (data/drug_smiles.csv)...")
    if os.path.exists('data/drug_smiles.csv'):
        df = pd.read_csv('data/drug_smiles.csv')
        n_drugs = len(df)
        n_with_smiles = df['smiles'].notna().sum()

        print(f"    ✓ Found drug_smiles.csv")
        print(f"    ✓ Contains {n_drugs} drugs")
        print(f"    ✓ {n_with_smiles} have SMILES ({n_drugs - n_with_smiles} missing)")

        if n_with_smiles < 20:
            print(f"    ⚠ WARNING: Only {n_with_smiles} drugs have SMILES")
            print(f"              You need at least 20-30 for meaningful training")
            ready = False

        if n_with_smiles < n_drugs:
            missing = df[df['smiles'].isna()]['drug_name'].tolist()
            print(f"    ⚠ Missing SMILES for: {', '.join(missing[:5])}")
            if len(missing) > 5:
                print(f"      ... and {len(missing) - 5} more")
    else:
        print("    ✗ data/drug_smiles.csv NOT FOUND")
        print("      → Run: python organize_data.py")
        ready = False

    # Check 2: Labels file
    print("\n[2/3] Checking cross-reactivity labels (data/cross_reactivity_labels.csv)...")
    if os.path.exists('data/cross_reactivity_labels.csv'):
        df = pd.read_csv('data/cross_reactivity_labels.csv')
        n_pairs = len(df)

        print(f"    ✓ Found cross_reactivity_labels.csv")
        print(f"    ✓ Contains {n_pairs} labeled pairs")

        if n_pairs < 30:
            print(f"    ⚠ WARNING: Only {n_pairs} labeled pairs")
            print(f"              You need at least 50-100 for good training")
            print(f"              Current split: ~{int(n_pairs*0.7)} train / ~{int(n_pairs*0.15)} val / ~{int(n_pairs*0.15)} test")
            ready = False

        # Check class distribution
        class_counts = df['label'].value_counts().sort_index()
        print(f"    Class distribution:")
        for label, count in class_counts.items():
            label_name = ['SUGGEST', 'CAUTION', 'AVOID'][int(label)]
            print(f"      {label_name:8s} (label={int(label)}): {count:3d} pairs")

        # Check for severe imbalance
        if len(class_counts) < 3:
            print(f"    ⚠ WARNING: Only {len(class_counts)} classes present")
            print(f"              You need examples of all 3 classes")
            ready = False
        else:
            ratio = class_counts.max() / class_counts.min()
            if ratio > 10:
                print(f"    ⚠ WARNING: Severe class imbalance (ratio {ratio:.1f}:1)")
                print(f"              Model may struggle with minority class")
    else:
        print("    ✗ data/cross_reactivity_labels.csv NOT FOUND")
        print("      → Run: python organize_data.py")
        print("      This will convert your existing data/labels.csv to the correct format")
        ready = False

    # Check 3: Dependencies
    print("\n[3/3] Checking Python dependencies...")
    required = [
        'torch', 'torch_geometric', 'rdkit', 'pandas',
        'numpy', 'sklearn', 'matplotlib', 'tqdm'
    ]

    missing = []
    for package in required:
        try:
            if package == 'rdkit':
                from rdkit import Chem
            elif package == 'torch_geometric':
                import torch_geometric
            elif package == 'sklearn':
                import sklearn
            else:
                __import__(package)
            print(f"    ✓ {package}")
        except ImportError:
            print(f"    ✗ {package} NOT INSTALLED")
            missing.append(package)

    if missing:
        print(f"\n    ⚠ Missing packages: {', '.join(missing)}")
        print(f"      → Run: pip install -r requirements.txt")
        ready = False

    # Final verdict
    print("\n" + "=" * 60)
    if ready:
        print("✓ ALL CHECKS PASSED - You're ready to train!")
        print("\nNext steps:")
        print("  1. python data_preparation.py  (prepare dataset)")
        print("  2. python train.py             (train model)")
        print("  3. python visualize.py         (generate plots)")
    else:
        print("✗ NOT READY - Please fix the issues above")
        print("\nQuick fix guide:")
        print("  • Missing SMILES → run: python collect_smiles.py")
        print("  • Missing labels → manually create cross_reactivity_labels.csv from chart")
        print("  • Missing packages → run: pip install -r requirements.txt")
    print("=" * 60)

    return ready


if __name__ == '__main__':
    check_ready()
