"""
Regenerate cross_reactivity_labels.csv from ReferenceTableLabels.xlsx
"""

import pandas as pd
import numpy as np

print("=" * 60)
print("REGENERATING CROSS_REACTIVITY_LABELS.CSV FROM EXCEL")
print("=" * 60)

# Read Excel file
print("\nReading ReferenceTableLabels.xlsx...")
df = pd.read_excel('data/ReferenceTableLabels.xlsx', index_col=0)

print(f"Excel shape: {df.shape}")
print(f"Drugs: {list(df.index)}")

# Create list of all pairs
pairs = []

for i, drug1 in enumerate(df.index):
    for j, drug2 in enumerate(df.columns):
        if drug1 == drug2:
            continue  # Skip self-pairs

        label = df.iloc[i, j]

        # Labels are already: 0 = SUGGEST, 1 = CAUTION, 2 = AVOID
        if pd.notna(label):
            label = int(label)

            pairs.append({
                'drug1': drug1,
                'drug2': drug2,
                'label': label
            })

# Create DataFrame
labels_df = pd.DataFrame(pairs)

# Save
labels_df.to_csv('data/cross_reactivity_labels.csv', index=False)

print(f"\n✓ Generated {len(labels_df)} pairs")
print(f"\nClass distribution:")
print(labels_df['label'].value_counts().sort_index())

print(f"\n✓ Saved to data/cross_reactivity_labels.csv")
print("=" * 60)
