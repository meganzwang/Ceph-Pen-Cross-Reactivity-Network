from pathlib import Path
import numpy as np
import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from torch_geometric.data import Batch
from src.chem_utils import smiles_to_graph, IN_DIM
from src.train_siamese import Siamese

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

def pyg_batch(graphs):
    return Batch.from_data_list(graphs)

def main():
    df = pd.read_csv(DATA_DIR / "drugs_with_smiles.csv").dropna(subset=["smiles"])
    pen = df[df["class"]=="PEN"].copy()
    ceph = df[df["class"].isin(["1G","2G","3G","4G","5G"])].copy()

    model = Siamese(in_dim=IN_DIM)
    model.load_state_dict(torch.load(MODELS_DIR / "siamese.pt", map_location="cpu"))
    model.eval()

    pen_names = list(pen["name"])
    ceph_names = list(ceph["name"])
    score_mat = np.zeros((len(ceph_names), len(pen_names)))

    with torch.no_grad():
        for i, (_, prow) in enumerate(pen.iterrows()):
            for j, (_, crow) in enumerate(ceph.iterrows()):
                g1 = smiles_to_graph(prow["smiles"])
                g2 = smiles_to_graph(crow["smiles"])
                s = model(pyg_batch([g1]), pyg_batch([g2])).item()
                score_mat[j, i] = s

    # Save score matrix for evaluation
    score_df = pd.DataFrame(score_mat, index=ceph_names, columns=pen_names)
    score_df.to_csv(PLOTS_DIR / "score_mat.csv")
    
    # Generate and save heatmap visualization
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(score_mat, xticklabels=pen_names, yticklabels=ceph_names,
                     cmap="RdYlGn_r", vmin=0, vmax=1, annot=True, fmt=".2f",
                     linewidths=0.5, linecolor='gray')
    ax.set_title("Predicted Cross-Reactivity (Structure-Based)", fontsize=14, pad=20)
    ax.set_xlabel("Penicillins", fontsize=12, labelpad=10)
    ax.set_ylabel("Cephalosporins", fontsize=12, labelpad=10)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    out = PLOTS_DIR / "heatmap.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Saved heatmap to {out}")

if __name__ == "__main__":
    main()