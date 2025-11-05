"""
Enhanced model: Molecular graphs + DDI network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GCNConv, global_mean_pool


class MolecularEncoder(nn.Module):
    """
    Encodes molecular structure (atoms + bonds) → drug embedding
    Same as before
    """
    def __init__(self, node_feat_dim=6, hidden_dim=128, embedding_dim=256, num_layers=4):
        super().__init__()
        self.node_encoder = nn.Linear(node_feat_dim, hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp))

        self.output = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, data):
        x = self.node_encoder(data.x)

        for conv in self.convs:
            x = F.relu(conv(x, data.edge_index))

        x = global_mean_pool(x, data.batch)
        return self.output(x)


class DDINetworkEncoder(nn.Module):
    """
    NEW: Encodes drug-drug interaction network

    Input: Full DDI graph (all drugs as nodes, cross-sensitivities as edges)
    Output: Drug embedding from network perspective
    """
    def __init__(self, node_feat_dim=10, hidden_dim=64, embedding_dim=128, num_layers=2):
        super().__init__()

        self.node_encoder = nn.Linear(node_feat_dim, hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.output = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, ddi_graph, drug_indices):
        """
        Args:
            ddi_graph: Full DDI network (all drugs)
            drug_indices: Indices of drugs we want embeddings for [batch_size]

        Returns:
            Drug embeddings from network perspective [batch_size, embedding_dim]
        """
        x = self.node_encoder(ddi_graph.x)

        # Message passing on DDI network
        for conv in self.convs:
            x = F.relu(conv(x, ddi_graph.edge_index, ddi_graph.edge_attr))

        x = self.output(x)

        # Extract embeddings for specific drugs
        return x[drug_indices]


class DualGraphCrossReactivityPredictor(nn.Module):
    """
    Full model: Combines molecular structure + DDI network

    Architecture:
        1. Molecular encoder: Drug structure → embedding
        2. DDI network encoder: Drug in network context → embedding
        3. Fusion: Combine both perspectives
        4. Prediction: MLP → cross-reactivity class
    """

    def __init__(
        self,
        molecular_encoder,
        ddi_encoder,
        molecular_emb_dim=256,
        ddi_emb_dim=128,
        hidden_dim=512,
        num_classes=3,
        dropout=0.3
    ):
        super().__init__()

        self.molecular_encoder = molecular_encoder
        self.ddi_encoder = ddi_encoder

        # Combined dimension
        # [mol1, mol2, |mol1-mol2|, mol1*mol2, ddi1, ddi2, |ddi1-ddi2|, ddi1*ddi2]
        combined_dim = molecular_emb_dim * 4 + ddi_emb_dim * 4

        self.predictor = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, drug1_mol_graph, drug2_mol_graph, ddi_graph, drug1_idx, drug2_idx):
        """
        Args:
            drug1_mol_graph: Molecular graph of drug 1
            drug2_mol_graph: Molecular graph of drug 2
            ddi_graph: Full DDI network (all drugs)
            drug1_idx: Index of drug1 in DDI network
            drug2_idx: Index of drug2 in DDI network

        Returns:
            Logits [batch_size, num_classes]
        """
        # 1. Encode molecular structure
        h1_mol = self.molecular_encoder(drug1_mol_graph)  # [batch, 256]
        h2_mol = self.molecular_encoder(drug2_mol_graph)  # [batch, 256]

        # 2. Encode from DDI network
        drug_indices = torch.cat([drug1_idx, drug2_idx])  # [2*batch]
        ddi_embeddings = self.ddi_encoder(ddi_graph, drug_indices)  # [2*batch, 128]

        batch_size = h1_mol.shape[0]
        h1_ddi = ddi_embeddings[:batch_size]   # [batch, 128]
        h2_ddi = ddi_embeddings[batch_size:]   # [batch, 128]

        # 3. Combine both perspectives
        combined = torch.cat([
            # Molecular perspective
            h1_mol,
            h2_mol,
            torch.abs(h1_mol - h2_mol),
            h1_mol * h2_mol,

            # DDI network perspective
            h1_ddi,
            h2_ddi,
            torch.abs(h1_ddi - h2_ddi),
            h1_ddi * h2_ddi
        ], dim=-1)

        # 4. Predict cross-reactivity
        logits = self.predictor(combined)
        return logits


# Example usage
if __name__ == '__main__':
    # Build encoders
    mol_encoder = MolecularEncoder(
        node_feat_dim=6,
        embedding_dim=256,
        num_layers=4
    )

    ddi_encoder = DDINetworkEncoder(
        node_feat_dim=10,  # [class, generation, DrugBank features]
        embedding_dim=128,
        num_layers=2
    )

    # Build full model
    model = DualGraphCrossReactivityPredictor(
        molecular_encoder=mol_encoder,
        ddi_encoder=ddi_encoder
    )

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("\nModel combines:")
    print("  • Molecular structure (atoms, bonds, side chains)")
    print("  • DDI network (DrugBank cross-sensitivities, clinical evidence)")
