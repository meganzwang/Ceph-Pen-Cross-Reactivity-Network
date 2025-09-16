import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from rdkit import Chem

# -----------------------
# 1. Helper: SMILES → Graph
# -----------------------
def smiles_to_graph(smiles, label):
    mol = Chem.MolFromSmiles(smiles)
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()

    # Node features: atom type (one-hot for simplicity)
    atom_types = [a.GetSymbol() for a in atoms]
    unique_atoms = ["C", "O", "N", "S", "H"]  # keep it simple
    x = []
    for at in atom_types:
        one_hot = [1 if at == u else 0 for u in unique_atoms]
        x.append(one_hot)
    x = torch.tensor(x, dtype=torch.float)

    # Edge index
    edge_index = []
    for b in bonds:
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long))

# -----------------------
# 2. Toy Dataset
# -----------------------
# Example SMILES (short list!)
penicillins = [
    ("CC1(C)SCC(N1C(=O)C(CO)NC(=O)C)C(=O)O", 1),  # Penicillin G (cross-reactive)
    ("CC1(C)SCC(N1C(=O)C(CO)NC(=O)C)C(=O)O", 1)   # duplicate just for demo
]

cephalosporins = [
    ("CC1=C(N2C(=O)CC2=CC(=O)O1)C(=O)O", 1),  # Cephalothin (cross-reactive)
    ("CC1=C(N2C(=O)CC2=CC(=O)O1)C", 0)        # Fake non-cross-reactive example
]

dataset = [smiles_to_graph(smiles, label) for smiles, label in penicillins + cephalosporins]
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# -----------------------
# 3. GCN Model
# -----------------------
class GCN(nn.Module):
    def __init__(self, hidden_dim=32, num_classes=2):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(5, hidden_dim)   # 5 input features (C,O,N,S,H)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)  # aggregate graph-level representation
        return self.fc(x)

model = GCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# -----------------------
# 4. Training Loop
# -----------------------
for epoch in range(20):
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = F.cross_entropy(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# -----------------------
# 5. Test Prediction
# -----------------------
test_smiles = "CC1=C(N2C(=O)CC2=CC(=O)O1)C(=O)O"  # Cephalothin
test_graph = smiles_to_graph(test_smiles, 1)
test_graph.batch = torch.zeros(test_graph.num_nodes, dtype=torch.long)  # single graph
with torch.no_grad():
    pred = model(test_graph.x, test_graph.edge_index, test_graph.batch)
    probs = F.softmax(pred, dim=-1)
    print("Prediction:", probs)

