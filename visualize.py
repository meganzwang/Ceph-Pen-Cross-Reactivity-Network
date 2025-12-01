"""
Visualization tools for cross-reactivity predictions.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay
import json

from model import build_model
from torch_geometric.data import Batch


def load_model(checkpoint_path):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    model = build_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, config


# def visualize_embeddings(model, drug_graphs, drug_names, drug_categories, save_path='embeddings_tsne.png'):
#     """
#     Visualize drug embeddings using t-SNE.

#     Args:
#         model: Trained model
#         drug_graphs: Dict of {drug_name: graph}
#         drug_names: List of drug names
#         drug_categories: Dict of {drug_name: category}
#         save_path: Path to save figure
#     """
#     # Get embeddings
#     embeddings = []
#     valid_drugs = []
#     categories = []

#     with torch.no_grad():
#         for drug in drug_names:
#             if drug in drug_graphs:
#                 graph = drug_graphs[drug]
#                 # Add batch attribute
#                 graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long)

#                 # Get embedding
#                 emb = model.molecular_encoder(graph)
#                 embeddings.append(emb.cpu().numpy())
#                 valid_drugs.append(drug)
#                 categories.append(drug_categories.get(drug, 'Unknown'))

#     embeddings = np.vstack(embeddings)

#     # Apply t-SNE
#     print("Running t-SNE...")
#     tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
#     embeddings_2d = tsne.fit_transform(embeddings)

#     # Plot
#     fig, ax = plt.subplots(figsize=(12, 8))

#     # Color by category
#     category_names = list(set(categories))
#     colors = plt.cm.tab10(np.linspace(0, 1, len(category_names)))
#     category_to_color = {cat: colors[i] for i, cat in enumerate(category_names)}

#     for i, drug in enumerate(valid_drugs):
#         cat = categories[i]
#         ax.scatter(
#             embeddings_2d[i, 0],
#             embeddings_2d[i, 1],
#             c=[category_to_color[cat]],
#             s=100,
#             alpha=0.7,
#             edgecolors='black',
#             linewidths=1
#         )
#         ax.annotate(
#             drug,
#             (embeddings_2d[i, 0], embeddings_2d[i, 1]),
#             fontsize=8,
#             alpha=0.8
#         )

#     # Legend
#     handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=category_to_color[cat],
#                           markersize=10, label=cat) for cat in category_names]
#     ax.legend(handles=handles, loc='best', framealpha=0.9)

#     ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
#     ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
#     ax.set_title('Drug Embeddings (t-SNE)', fontsize=14, fontweight='bold')
#     ax.grid(alpha=0.3)

#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     print(f"✓ Saved embedding visualization to {save_path}")
#     plt.close()


def predict_all_pairs(model, drug_graphs, drug_names):
    """
    Predict cross-reactivity for all drug pairs.

    Returns:
        DataFrame with columns: drug1, drug2, predicted_class, prob_suggest, prob_caution, prob_avoid
    """
    predictions = []

    with torch.no_grad():
        for i, drug1 in enumerate(drug_names):
            if drug1 not in drug_graphs:
                continue

            for j, drug2 in enumerate(drug_names):
                if j <= i or drug2 not in drug_graphs:
                    continue

                # Get graphs
                graph1 = drug_graphs[drug1]
                graph2 = drug_graphs[drug2]

                # Add batch attributes
                graph1.batch = torch.zeros(graph1.num_nodes, dtype=torch.long)
                graph2.batch = torch.zeros(graph2.num_nodes, dtype=torch.long)

                # Predict
                probs = model.predict_proba(graph1, graph2)
                pred_class = torch.argmax(probs, dim=-1).item()

                predictions.append({
                    'drug1': drug1,
                    'drug2': drug2,
                    'predicted_class': pred_class,
                    'prob_suggest': probs[0, 0].item(),
                    'prob_caution': probs[0, 1].item(),
                    'prob_avoid': probs[0, 2].item()
                })

    return pd.DataFrame(predictions)


# def create_cross_reactivity_heatmap(predictions_df, drug_names, save_path='heatmap_predictions.png'):
#     """
#     Create heatmap of predicted cross-reactivity.

#     Args:
#         predictions_df: DataFrame from predict_all_pairs()
#         drug_names: List of all drug names
#         save_path: Path to save figure
#     """
#     # Create matrix
#     n = len(drug_names)
#     matrix = np.full((n, n), -1.0)  # -1 for diagonal (not applicable)

#     drug_to_idx = {drug: i for i, drug in enumerate(drug_names)}

#     for _, row in predictions_df.iterrows():
#         i = drug_to_idx.get(row['drug1'])
#         j = drug_to_idx.get(row['drug2'])

#         if i is not None and j is not None:
#             # Use predicted class: 0=SUGGEST (green), 1=CAUTION (yellow), 2=AVOID (red)
#             matrix[i, j] = row['predicted_class']
#             matrix[j, i] = row['predicted_class']

#     # Plot
#     fig, ax = plt.subplots(figsize=(14, 12))

#     # Custom colormap: green (0) -> yellow (1) -> red (2), gray (-1) for diagonal
#     cmap = plt.cm.colors.ListedColormap(['lightgreen', 'yellow', 'red', 'lightgray'])
#     bounds = [-1.5, -0.5, 0.5, 1.5, 2.5]
#     norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

#     im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect='auto')

#     # Set ticks
#     ax.set_xticks(np.arange(n))
#     ax.set_yticks(np.arange(n))
#     ax.set_xticklabels(drug_names, rotation=90, ha='right', fontsize=8)
#     ax.set_yticklabels(drug_names, fontsize=8)

#     # Colorbar
#     cbar = plt.colorbar(im, ax=ax, ticks=[-1, 0, 1, 2])
#     cbar.ax.set_yticklabels(['N/A', 'SUGGEST', 'CAUTION', 'AVOID'])

#     ax.set_title('Predicted Cross-Reactivity Heatmap', fontsize=14, fontweight='bold')

#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     print(f"✓ Saved heatmap to {save_path}")
#     plt.close()


def plot_training_history(results_path='results/results.json', save_path='plots/training_history.png'):
    """
    Plot training history (loss, accuracy, F1).
    """
    with open(results_path, 'r') as f:
        results = json.load(f)

    history = results['history']

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Validation', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Accuracy
    axes[0, 1].plot(history['val_accuracy'], linewidth=2, color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Validation Accuracy')
    axes[0, 1].grid(alpha=0.3)

    # F1
    axes[1, 0].plot(history['val_f1'], linewidth=2, color='orange')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 (macro)')
    axes[1, 0].set_title('Validation F1 Score')
    axes[1, 0].grid(alpha=0.3)

    # Kappa
    axes[1, 1].plot(history['val_kappa'], linewidth=2, color='purple')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel("Cohen's Kappa")
    axes[1, 1].set_title('Validation Kappa')
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved training history to {save_path}")
    plt.close()


def plot_confusion_matrix(results_path='results/results.json', save_path='plots/confusion_matrix.png'):
    """
    Plot confusion matrix from test results.
    """
    with open(results_path, 'r') as f:
        results = json.load(f)

    cm = np.array(results['test_metrics']['confusion_matrix'])
    class_names = ['SUGGEST', 'CAUTION', 'AVOID']

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    # Add counts
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=14)

    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_title('Confusion Matrix (Test Set)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved confusion matrix to {save_path}")
    plt.close()


if __name__ == '__main__':
    print("Generating visualizations...")

    # Load model and data
    model, config = load_model('models/best_model.pt')
    data = torch.load('data/processed_data.pt', weights_only=False)

    # Drug names (you'll need to get these from your data)
    # For now, placeholder
    drug_names = [
        'Penicillin G/V', 'Oxacillin', 'Amoxicillin', 'Ampicillin', 'Piperacillin',
        'Cefadroxil', 'Cephalexin', 'Cefazolin',
        'Cefaclor', 'Cefoxitin', 'Cefprozil', 'Cefuroxime'
        # ... add all drugs
    ]

    drug_categories = {
        # Map drug names to categories
    }

    drug_graphs = data['drug_graphs']

    # 1. Training history
    plot_training_history()

    # 2. Confusion matrix
    plot_confusion_matrix()

    # 3. Embeddings
    # visualize_embeddings(model, drug_graphs, drug_names, drug_categories)

    # 4. Predictions heatmap
    # predictions_df = predict_all_pairs(model, drug_graphs, drug_names)
    # create_cross_reactivity_heatmap(predictions_df, drug_names)

    print("\n✓ All visualizations generated!")
