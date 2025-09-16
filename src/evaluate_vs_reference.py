from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import seaborn as sns
import matplotlib.pyplot as plt

DATA_DIR = Path("data")
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

def main():
    df = pd.read_csv(DATA_DIR / "drugs_with_smiles.csv").dropna(subset=["smiles"])
    pen = df[df["class"]=="PEN"].copy()
    ceph = df[df["class"].isin(["1G","2G","3G","4G","5G"])].copy()
    pen_names = list(pen["name"])
    ceph_names = list(ceph["name"])

    # Load predicted matrix from file generated in predict_heatmap, or recompute here.
    # For simplicity, recompute using saved CSV of predictions. If not available, run predict_heatmap first.
    # Here we assume you will load predictions from a CSV; modify predict_heatmap to save score_mat.csv if desired.
    # Fallback: rebuild from labels only for demonstration (will not be model predictions).
    try:
        score_df = pd.read_csv(PLOTS_DIR / "score_mat.csv", index_col=0)
        score_mat = score_df.values
        print("Loaded predictions from plots/score_mat.csv")
    except Exception:
        print("No score_mat.csv found. Reusing labels as dummy predictions (not ideal).")
        score_mat = np.zeros((len(ceph_names), len(pen_names)))

    lab = pd.read_csv(DATA_DIR / "labels.csv")
    ref = np.zeros_like(score_mat, dtype=float)

    pen_idx = {n:i for i,n in enumerate(pen_names)}
    ceph_idx = {n:j for j,n in enumerate(ceph_names)}
    for _, r in lab.iterrows():
        p, c, v = r["pen"], r["ceph"], float(r["label"])
        if p in pen_idx and c in ceph_idx:
            ref[ceph_idx[c], pen_idx[p]] = v

    flat_pred = score_mat.flatten()
    flat_ref = ref.flatten()
    rho, pval = spearmanr(flat_pred, flat_ref)
    print(f"Spearman correlation vs reference: rho={rho:.3f}, p={pval:.2e}")

    # Overlay plot: show thresholds for 1 and 0.5
    plt.figure(figsize=(10,8))
    ax = sns.heatmap(score_mat, xticklabels=pen_names, yticklabels=ceph_names,
                     cmap="RdYlGn_r", vmin=0, vmax=1)
    ax.set_title(f"Predicted vs Reference (Spearman={rho:.2f})")
    ax.set_xlabel("Penicillins")
    ax.set_ylabel("Cephalosporins")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "overlay.png", dpi=200)
    print(f"Saved overlay to {PLOTS_DIR/'overlay.png'}")

if __name__ == "__main__":
    main()