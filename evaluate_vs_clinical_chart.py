"""
Evaluate model predictions against Northwestern Medicine clinical chart

Generates:
1. Side-by-side heatmap (Clinical Chart vs Model Predictions)
2. Precision/Recall/F1 for pairs in clinical chart
3. Agreement analysis (Cohen's Kappa)
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score, confusion_matrix
import os

from model import build_model


# Drug order matching Northwestern chart
DRUGS_IN_ORDER = [
    # Penicillins
    'Penicillin G/V', 'Oxacillin', 'Amoxicillin', 'Ampicillin', 'Piperacillin',
    # 1st gen
    'Cefadroxil', 'Cephalexin', 'Cefazolin',
    # 2nd gen
    'Cefaclor', 'Cefoxitin', 'Cefprozil', 'Cefuroxime',
    # 3rd gen
    'Cefdinir', 'Cefditoren', 'Cefixime', 'Cefotaxime',
    'Cefpodoxime', 'Ceftazidime', 'Ceftibuten', 'Ceftriaxone',
    # 4th gen
    'Cefepime',
    # 5th gen
    'Ceftaroline', 'Ceftolozane',
    # Carbapenems
    'Ertapenem', 'Meropenem',
    # Monobactams
    'Aztreonam'
]


def load_model_and_data():
    """Load trained model and processed data."""
    print("Loading model and data...")

    # Load model
    checkpoint = torch.load('models/best_model.pt', map_location='cpu', weights_only=False)
    config = checkpoint['config']
    model = build_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load data
    data = torch.load('data/processed_data.pt', map_location='cpu', weights_only=False)
    drug_graphs = data['drug_graphs']

    print(f"✓ Model loaded ({sum(p.numel() for p in model.parameters()):,} parameters)")
    print(f"✓ Data loaded ({len(drug_graphs)} drugs)")

    return model, drug_graphs


def predict_all_pairs(model, drug_graphs, drug_list):
    """
    Predict cross-reactivity for all drug pairs.

    Returns:
        N x N matrix of predictions (0, 1, 2)
    """
    print(f"\nPredicting cross-reactivity for all {len(drug_list)}x{len(drug_list)} pairs...")

    n = len(drug_list)
    predictions = np.full((n, n), -1, dtype=int)  # -1 for diagonal/not applicable

    with torch.no_grad():
        for i, drug1 in enumerate(drug_list):
            if drug1 not in drug_graphs:
                continue

            for j, drug2 in enumerate(drug_list):
                if j <= i or drug2 not in drug_graphs:
                    continue

                # Get graphs
                graph1 = drug_graphs[drug1]
                graph2 = drug_graphs[drug2]

                # Add batch attribute
                graph1.batch = torch.zeros(graph1.num_nodes, dtype=torch.long)
                graph2.batch = torch.zeros(graph2.num_nodes, dtype=torch.long)

                # Predict
                logits = model(graph1, graph2)
                pred_class = torch.argmax(logits, dim=-1).item()

                # Fill symmetric matrix
                predictions[i, j] = pred_class
                predictions[j, i] = pred_class

    print(f"✓ Generated {np.sum(predictions >= 0)} predictions")
    return predictions


def load_clinical_labels():
    """
    Load clinical labels from data/cross_reactivity_labels.csv

    Returns:
        N x N matrix matching Northwestern chart
    """
    print("\nLoading clinical reference labels...")

    labels_df = pd.read_csv('data/cross_reactivity_labels.csv')

    # Create matrix
    n = len(DRUGS_IN_ORDER)
    clinical_matrix = np.full((n, n), -1, dtype=int)  # -1 for unlabeled

    drug_to_idx = {drug: i for i, drug in enumerate(DRUGS_IN_ORDER)}

    for _, row in labels_df.iterrows():
        drug1 = row['drug1']
        drug2 = row['drug2']
        label = int(row['label'])

        i = drug_to_idx.get(drug1)
        j = drug_to_idx.get(drug2)

        if i is not None and j is not None:
            clinical_matrix[i, j] = label
            clinical_matrix[j, i] = label

    n_labeled = np.sum(clinical_matrix >= 0)
    print(f"✓ Loaded {n_labeled} labeled pairs from clinical chart")

    return clinical_matrix


def evaluate_against_clinical(predictions, clinical_matrix):
    """
    Evaluate model predictions against clinical reference.

    Returns:
        Dictionary of metrics
    """
    print("\nEvaluating against clinical reference...")

    # Extract only pairs that have clinical labels
    mask = clinical_matrix >= 0
    y_true = clinical_matrix[mask]
    y_pred = predictions[mask]

    # Overall metrics
    accuracy = np.mean(y_true == y_pred)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )

    # Cohen's Kappa
    kappa = cohen_kappa_score(y_true, y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    metrics = {
        'accuracy': accuracy,
        'precision_per_class': precision,
        'recall_per_class': recall,
        'f1_per_class': f1,
        'support_per_class': support,
        'kappa': kappa,
        'confusion_matrix': cm,
        'n_pairs_evaluated': len(y_true)
    }

    return metrics


def plot_side_by_side_heatmaps(clinical_matrix, predictions, save_path='plots/clinical_vs_model_heatmap.png'):
    """
    Create side-by-side heatmap: Clinical Chart vs Model Predictions
    """
    print("\nGenerating side-by-side heatmap...")

    os.makedirs('plots', exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))

    # Color map: SUGGEST (green), CAUTION (yellow), AVOID (red), N/A (gray)
    cmap = plt.cm.colors.ListedColormap(['lightgreen', 'yellow', 'red', 'lightgray'])
    bounds = [-1.5, -0.5, 0.5, 1.5, 2.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    # Plot 1: Clinical Chart
    im1 = ax1.imshow(clinical_matrix, cmap=cmap, norm=norm, aspect='auto')
    ax1.set_title('Northwestern Medicine Clinical Chart', fontsize=16, fontweight='bold')
    ax1.set_xticks(np.arange(len(DRUGS_IN_ORDER)))
    ax1.set_yticks(np.arange(len(DRUGS_IN_ORDER)))
    ax1.set_xticklabels(DRUGS_IN_ORDER, rotation=90, ha='right', fontsize=8)
    ax1.set_yticklabels(DRUGS_IN_ORDER, fontsize=8)

    # Plot 2: Model Predictions
    im2 = ax2.imshow(predictions, cmap=cmap, norm=norm, aspect='auto')
    ax2.set_title('GNN Model Predictions', fontsize=16, fontweight='bold')
    ax2.set_xticks(np.arange(len(DRUGS_IN_ORDER)))
    ax2.set_yticks(np.arange(len(DRUGS_IN_ORDER)))
    ax2.set_xticklabels(DRUGS_IN_ORDER, rotation=90, ha='right', fontsize=8)
    ax2.set_yticklabels(DRUGS_IN_ORDER, fontsize=8)

    # Shared colorbar - moved further to the right
    plt.subplots_adjust(right=0.92)  # Make room for colorbar on the right
    cbar = fig.colorbar(im2, ax=[ax1, ax2], ticks=[-1, 0, 1, 2], fraction=0.03, pad=0.02)
    cbar.ax.set_yticklabels(['N/A', 'SUGGEST', 'CAUTION', 'AVOID'])

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved heatmap to {save_path}")
    plt.close()


def plot_confusion_matrix(cm, save_path='plots/clinical_confusion_matrix.png'):
    """Plot confusion matrix for clinical evaluation."""
    print("\nGenerating confusion matrix...")

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    class_names = ['SUGGEST', 'CAUTION', 'AVOID']
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

    ax.set_ylabel('Clinical Chart (True)', fontsize=12)
    ax.set_xlabel('Model Prediction', fontsize=12)
    ax.set_title('Confusion Matrix: Model vs Clinical Chart', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved confusion matrix to {save_path}")
    plt.close()


def print_detailed_results(metrics):
    """Print detailed evaluation results."""
    print("\n" + "=" * 70)
    print("EVALUATION AGAINST NORTHWESTERN MEDICINE CLINICAL CHART")
    print("=" * 70)

    print(f"\nEvaluated {metrics['n_pairs_evaluated']} drug pairs with clinical labels\n")

    print(f"Overall Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Cohen's Kappa: {metrics['kappa']:.4f}", end="")

    # Interpret Kappa
    kappa = metrics['kappa']
    if kappa > 0.80:
        interpretation = "Almost Perfect Agreement"
    elif kappa > 0.60:
        interpretation = "Substantial Agreement ✓"
    elif kappa > 0.40:
        interpretation = "Moderate Agreement"
    elif kappa > 0.20:
        interpretation = "Fair Agreement"
    else:
        interpretation = "Poor Agreement"
    print(f" ({interpretation})")

    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-" * 70)

    class_names = ['SUGGEST', 'CAUTION', 'AVOID']
    for i, class_name in enumerate(class_names):
        p = metrics['precision_per_class'][i]
        r = metrics['recall_per_class'][i]
        f = metrics['f1_per_class'][i]
        s = metrics['support_per_class'][i]
        print(f"{class_name:<12} {p:>10.3f} {r:>10.3f} {f:>10.3f} {s:>10.0f}")

    # Highlight AVOID metrics (most important for safety)
    avoid_recall = metrics['recall_per_class'][2]
    print(f"\n⚠️  AVOID Recall: {avoid_recall:.3f} (critical for safety - ", end="")
    if avoid_recall > 0.80:
        print("✓ catching most dangerous pairs)")
    elif avoid_recall > 0.60:
        print("⚠️ missing some dangerous pairs)")
    else:
        print("✗ missing many dangerous pairs!)")

    print(f"\nConfusion Matrix:")
    print(f"             Predicted")
    print(f"           SUG  CAU  AVO")
    cm = metrics['confusion_matrix']
    for i, class_name in enumerate(['SUG', 'CAU', 'AVO']):
        print(f"True {class_name}  {cm[i, 0]:4.0f} {cm[i, 1]:4.0f} {cm[i, 2]:4.0f}")

    print("\n" + "=" * 70)


def main():
    """Main evaluation pipeline."""
    print("=" * 70)
    print("EVALUATING MODEL VS NORTHWESTERN MEDICINE CLINICAL CHART")
    print("=" * 70)

    # Load model and data
    model, drug_graphs = load_model_and_data()

    # Predict all pairs
    predictions = predict_all_pairs(model, drug_graphs, DRUGS_IN_ORDER)

    # Load clinical labels
    clinical_matrix = load_clinical_labels()

    # Evaluate
    metrics = evaluate_against_clinical(predictions, clinical_matrix)

    # Print results
    print_detailed_results(metrics)

    # Generate visualizations
    plot_side_by_side_heatmaps(clinical_matrix, predictions)
    plot_confusion_matrix(metrics['confusion_matrix'])

    # Save metrics to JSON
    import json
    os.makedirs('results', exist_ok=True)
    with open('results/clinical_evaluation.json', 'w') as f:
        # Convert numpy types to Python types for JSON
        metrics_json = {
            'accuracy': float(metrics['accuracy']),
            'kappa': float(metrics['kappa']),
            'precision_per_class': metrics['precision_per_class'].tolist(),
            'recall_per_class': metrics['recall_per_class'].tolist(),
            'f1_per_class': metrics['f1_per_class'].tolist(),
            'support_per_class': metrics['support_per_class'].tolist(),
            'confusion_matrix': metrics['confusion_matrix'].tolist(),
            'n_pairs_evaluated': int(metrics['n_pairs_evaluated'])
        }
        json.dump(metrics_json, f, indent=2)

    print(f"\n✓ Detailed metrics saved to results/clinical_evaluation.json")
    print(f"✓ Visualizations saved to plots/")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
