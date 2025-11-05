"""
Organize existing data files and convert to new format
"""

import pandas as pd
import os

# Create data directory if needed
os.makedirs('data', exist_ok=True)

print("=" * 60)
print("ORGANIZING DATA FILES")
print("=" * 60)

# ============================================================
# 1. Convert drugs_with_smiles.csv to correct format
# ============================================================
print("\n[1/3] Processing SMILES data...")

# Read existing file
smiles_df = pd.read_csv('data/drugs_with_smiles.csv')

# Add category column based on class codes
category_map = {
    'PEN': 'penicillins',
    '1G': '1st_gen_ceph',
    '2G': '2nd_gen_ceph',
    '3G': '3rd_gen_ceph',
    '4G': '4th_gen_ceph',
    '5G': '5th_gen_ceph',
    '5Gpip': '5th_gen_ceph',
    'CARB': 'carbapenems',
    'MONO': 'monobactams'
}

# Rename columns to match new code
smiles_df = smiles_df.rename(columns={'name': 'drug_name', 'class': 'class_code'})
smiles_df['category'] = smiles_df['class_code'].map(category_map)

# Reorder columns
smiles_df = smiles_df[['drug_name', 'category', 'class_code', 'smiles']]

# Handle Penicillin G/V separately (chart shows them combined)
# We'll use Penicillin G as the representative
pen_g_row = smiles_df[smiles_df['drug_name'] == 'Penicillin G'].copy()
pen_g_row['drug_name'] = 'Penicillin G/V'
smiles_df = pd.concat([smiles_df, pen_g_row], ignore_index=True)

# Save
smiles_df.to_csv('data/drug_smiles.csv', index=False)
print(f"✓ Saved {len(smiles_df)} drugs to data/drug_smiles.csv")
print(f"  Columns: {list(smiles_df.columns)}")

# ============================================================
# 2. Convert labels.csv to 3-class format
# ============================================================
print("\n[2/3] Converting labels to 3-class format...")

# Read existing labels
labels_df = pd.read_csv('data/labels.csv')

# Current format: pen, ceph, label (0, 0.5, 1)
# New format: drug1, drug2, label (0, 1, 2)

# Convert labels:
# 0 → 0 (SUGGEST)
# 0.5 → 1 (CAUTION)
# 1 → 2 (AVOID)
labels_df['label_new'] = labels_df['label'].map({
    0.0: 0,    # SUGGEST
    0.5: 1,    # CAUTION
    1.0: 2     # AVOID
})

# Drop old label column and rename
labels_df = labels_df.drop(columns=['label'])
labels_df = labels_df.rename(columns={
    'pen': 'drug1',
    'ceph': 'drug2',
    'label_new': 'label'
})

# Reorder columns
labels_df = labels_df[['drug1', 'drug2', 'label']]

# Save
labels_df.to_csv('data/cross_reactivity_labels.csv', index=False)
print(f"✓ Saved {len(labels_df)} labeled pairs to data/cross_reactivity_labels.csv")

# Show distribution
print(f"\n  Class distribution:")
for label, count in labels_df['label'].value_counts().sort_index().items():
    label_name = ['SUGGEST', 'CAUTION', 'AVOID'][int(label)]
    print(f"    {label_name:8s} (label={int(label)}): {count:3d} pairs")

# ============================================================
# 3. Create summary
# ============================================================
print("\n[3/3] Summary...")

print(f"\n✓ All data files organized in data/ directory:")
print(f"  • data/drug_smiles.csv              - {len(smiles_df)} drugs with SMILES")
print(f"  • data/cross_reactivity_labels.csv  - {len(labels_df)} labeled pairs")

print(f"\n✓ Ready to run:")
print(f"  python data_preparation.py")

print("\n" + "=" * 60)
