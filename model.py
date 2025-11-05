"""
GNN model for beta-lactam cross-reactivity prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GATConv, GCNConv, global_mean_pool, global_add_pool
from torch_geometric.data import Batch


class MolecularEncoder(nn.Module):
    """
    Graph Neural Network to encode molecular structures into fixed-size embeddings.

    Supports multiple GNN architectures: GIN, GAT, GCN
    """

    def __init__(
        self,
        node_feat_dim=6,  # Number of atom features
        edge_feat_dim=2,  # Number of bond features
        hidden_dim=128,
        embedding_dim=256,
        num_layers=4,
        dropout=0.2,
        gnn_type='GIN',
        pooling='mean'
    ):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.gnn_type = gnn_type
        self.pooling = pooling

        # Input projection
        self.node_encoder = nn.Linear(node_feat_dim, hidden_dim)

        # GNN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            if gnn_type == 'GIN':
                # Graph Isomorphism Network
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
                conv = GINConv(mlp, train_eps=True)

            elif gnn_type == 'GAT':
                # Graph Attention Network
                conv = GATConv(
                    hidden_dim,
                    hidden_dim // 8,  # 8 attention heads
                    heads=8,
                    dropout=dropout,
                    concat=True
                )

            elif gnn_type == 'GCN':
                # Graph Convolutional Network
                conv = GCNConv(hidden_dim, hidden_dim)

            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")

            self.convs.append(conv)
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Output projection to embedding space
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, data):
        """
        Args:
            data: torch_geometric.data.Data or Batch object

        Returns:
            Graph embedding of shape [batch_size, embedding_dim]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Encode node features
        x = self.node_encoder(x)

        # Apply GNN layers with residual connections
        for i in range(self.num_layers):
            x_prev = x
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            # Residual connection (if dimensions match)
            if x.shape == x_prev.shape:
                x = x + x_prev

        # Graph-level pooling
        if self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'add':
            x = global_add_pool(x, batch)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        # Project to embedding space
        x = self.output_projection(x)

        return x


class CrossReactivityPredictor(nn.Module):
    """
    Full model: Encodes two drugs and predicts cross-reactivity.

    Architecture:
        1. Encode each drug with MolecularEncoder
        2. Combine embeddings (concatenate + difference + product)
        3. MLP to predict 3-class cross-reactivity
    """

    def __init__(
        self,
        molecular_encoder,
        embedding_dim=256,
        hidden_dim=512,
        num_classes=3,
        dropout=0.3
    ):
        super().__init__()

        self.molecular_encoder = molecular_encoder
        self.embedding_dim = embedding_dim

        # Combined feature dimension: [h1 || h2 || |h1-h2| || h1*h2]
        combined_dim = embedding_dim * 4

        # Prediction MLP
        self.predictor = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, drug1_graph, drug2_graph):
        """
        Args:
            drug1_graph: torch_geometric.data.Data or Batch for drug 1
            drug2_graph: torch_geometric.data.Data or Batch for drug 2

        Returns:
            Logits of shape [batch_size, num_classes]
        """
        # Encode both drugs
        h1 = self.molecular_encoder(drug1_graph)
        h2 = self.molecular_encoder(drug2_graph)

        # Combine embeddings
        combined = torch.cat([
            h1,                     # Drug 1 embedding
            h2,                     # Drug 2 embedding
            torch.abs(h1 - h2),     # Absolute difference
            h1 * h2                 # Element-wise product
        ], dim=-1)

        # Predict cross-reactivity
        logits = self.predictor(combined)

        return logits

    def predict_proba(self, drug1_graph, drug2_graph):
        """
        Predict class probabilities.

        Returns:
            Probabilities of shape [batch_size, num_classes]
        """
        logits = self.forward(drug1_graph, drug2_graph)
        return F.softmax(logits, dim=-1)

    def predict(self, drug1_graph, drug2_graph):
        """
        Predict class labels.

        Returns:
            Class labels of shape [batch_size]
        """
        logits = self.forward(drug1_graph, drug2_graph)
        return torch.argmax(logits, dim=-1)


def build_model(config):
    """
    Build model from configuration dictionary.

    Args:
        config: Dictionary with model hyperparameters

    Returns:
        CrossReactivityPredictor model
    """
    encoder = MolecularEncoder(
        node_feat_dim=config.get('node_feat_dim', 6),
        edge_feat_dim=config.get('edge_feat_dim', 2),
        hidden_dim=config.get('hidden_dim', 128),
        embedding_dim=config.get('embedding_dim', 256),
        num_layers=config.get('num_layers', 4),
        dropout=config.get('dropout', 0.2),
        gnn_type=config.get('gnn_type', 'GIN'),
        pooling=config.get('pooling', 'mean')
    )

    model = CrossReactivityPredictor(
        molecular_encoder=encoder,
        embedding_dim=config.get('embedding_dim', 256),
        hidden_dim=config.get('predictor_hidden_dim', 512),
        num_classes=config.get('num_classes', 3),
        dropout=config.get('predictor_dropout', 0.3)
    )

    return model


if __name__ == '__main__':
    # Test model
    from torch_geometric.data import Data

    # Create dummy molecular graphs
    graph1 = Data(
        x=torch.randn(10, 6),  # 10 atoms, 6 features each
        edge_index=torch.randint(0, 10, (2, 20)),  # 20 bonds
        batch=torch.zeros(10, dtype=torch.long)
    )

    graph2 = Data(
        x=torch.randn(15, 6),  # 15 atoms
        edge_index=torch.randint(0, 15, (2, 30)),  # 30 bonds
        batch=torch.zeros(15, dtype=torch.long)
    )

    # Build model
    config = {
        'hidden_dim': 128,
        'embedding_dim': 256,
        'num_layers': 4,
        'gnn_type': 'GIN'
    }

    model = build_model(config)

    # Forward pass
    logits = model(graph1, graph2)
    probs = model.predict_proba(graph1, graph2)
    pred = model.predict(graph1, graph2)

    print("Model test:")
    print(f"Logits shape: {logits.shape}")  # [1, 3]
    print(f"Probabilities: {probs}")
    print(f"Prediction: {pred}")  # Class 0, 1, or 2
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
