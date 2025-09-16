import os
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Batch
from src.chem_utils import smiles_to_graph, IN_DIM

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

class PairDataset(Dataset):
    def __init__(self, drugs_csv: Path, labels_csv: Path):
        df = pd.read_csv(drugs_csv).dropna(subset=["smiles"])
        self.pen = df[df["class"]=="PEN"].copy()
        self.ceph = df[df["class"].isin(["1G","2G","3G","4G","5G"])].copy()
        lab = pd.read_csv(labels_csv)
        # Create label dict
        self.labels = {(r["pen"], r["ceph"]): float(r["label"]) for _, r in lab.iterrows()}
        # Build all pairs
        self.items = []
        for _, prow in self.pen.iterrows():
            for _, crow in self.ceph.iterrows():
                y = self.labels.get((prow["name"], crow["name"]), 0.0)
                self.items.append((prow["name"], prow["smiles"], crow["name"], crow["smiles"], y))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        pn, ps, cn, cs, y = self.items[idx]
        g1 = smiles_to_graph(ps); g1.name = pn
        g2 = smiles_to_graph(cs); g2.name = cn
        return g1, g2, torch.tensor([y], dtype=torch.float)

def collate(batch):
    g1s, g2s, ys = zip(*batch)
    return list(g1s), list(g2s), torch.cat(ys, dim=0)

class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden=64, out_dim=128):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.lin = nn.Linear(hidden, out_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        batch = getattr(data, 'batch', torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        x = global_mean_pool(x, batch)
        return self.lin(x)

class Siamese(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.enc = GCNEncoder(in_dim=in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, g1_batch, g2_batch):
        z1 = self.enc(g1_batch)
        z2 = self.enc(g2_batch)
        z = torch.cat([z1, z2], dim=-1)
        return torch.sigmoid(self.mlp(z))

def pyg_batch(graphs):
    return Batch.from_data_list(graphs)

def main():
    dataset = PairDataset(DATA_DIR / "drugs_with_smiles.csv", DATA_DIR / "labels.csv")
    loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate)
    model = Siamese(in_dim=IN_DIM)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCELoss()

    epochs = 50
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for g1s, g2s, y in loader:
            g1b, g2b = pyg_batch(g1s).to(device), pyg_batch(g2s).to(device)
            y = y.to(device)
            opt.zero_grad()
            pred = model(g1b, g2b).squeeze(1)
            loss = bce(pred, y)
            loss.backward()
            opt.step()
            total += loss.item() * y.size(0)
        print(f"Epoch {epoch+1:02d}/{epochs} - loss: {total/len(dataset):.4f}")

    torch.save(model.state_dict(), MODELS_DIR / "siamese.pt")
    print(f"Saved model to {MODELS_DIR/'siamese.pt'}")

if __name__ == "__main__":
    main()