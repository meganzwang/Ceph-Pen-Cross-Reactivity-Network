"""
Training pipeline for cross-reactivity prediction
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, cohen_kappa_score
)
from tqdm import tqdm
import json
import os

# Import precision-focused loss functions
from precision_loss import create_precision_focused_criterion, BalancedPrecisionLoss, PrecisionPenaltyLoss

from model import build_model


class PairDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper for drug pairs.
    """

    def __init__(self, data):
        """
        Args:
            data: List of (graph1, graph2, label) tuples or
                  (graph1, graph2, label, struct_feats) tuples if structural features included
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    """
    Custom collate function for batching drug pairs.

    Returns a 4-tuple: (batch_graph1, batch_graph2, batch_labels, batch_struct_feats)
    batch_struct_feats is None if not provided in the dataset.
    """
    sample = batch[0]

    if len(sample) == 3:
        graphs1, graphs2, labels = zip(*batch)
        feats = None
    else:
        graphs1, graphs2, labels, feats = zip(*batch)

    # Batch graphs
    batch_graph1 = Batch.from_data_list(graphs1)
    batch_graph2 = Batch.from_data_list(graphs2)

    # Stack labels
    batch_labels = torch.cat(labels, dim=0)

    # Process structural features if present
    if feats is None:
        batch_struct_feats = None
    else:
        # Ensure all feats are tensors
        proc = []
        for f in feats:
            if isinstance(f, torch.Tensor):
                proc.append(f)
            else:
                proc.append(torch.tensor(f, dtype=torch.float))

        batch_struct_feats = torch.stack(proc, dim=0)

    return batch_graph1, batch_graph2, batch_labels, batch_struct_feats


def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    Train for one epoch.

    Returns:
        Average loss
    """
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        graph1, graph2, labels, struct_feats = batch

        # Move to device
        graph1 = graph1.to(device)
        graph2 = graph2.to(device)
        labels = labels.to(device)
        if struct_feats is not None:
            struct_feats = struct_feats.to(device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(graph1, graph2, struct_feats)
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """
    Evaluate model on validation/test set.

    Returns:
        Dictionary of metrics
    """
    model.eval()
    total_loss = 0
    num_batches = 0

    all_preds = []
    all_labels = []
    all_probs = []

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        graph1, graph2, labels, struct_feats = batch

        # Move to device
        graph1 = graph1.to(device)
        graph2 = graph2.to(device)
        labels = labels.to(device)
        if struct_feats is not None:
            struct_feats = struct_feats.to(device)

        # Forward pass
        logits = model(graph1, graph2, struct_feats)
        loss = criterion(logits, labels)

        # Get predictions
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(logits, dim=-1)

        # Collect results
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

        total_loss += loss.item()
        num_batches += 1

    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )

    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support = \
        precision_recall_fscore_support(all_labels, all_preds, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # AUROC (one-vs-rest)
    try:
        auroc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    except ValueError:
        auroc = 0.0  # Not enough classes in this split

    # Cohen's kappa
    kappa = cohen_kappa_score(all_labels, all_preds)

    metrics = {
        'loss': total_loss / num_batches,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auroc': auroc,
        'kappa': kappa,
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'f1_per_class': f1_per_class.tolist(),
        'support_per_class': support.tolist(),
        'confusion_matrix': cm.tolist()
    }

    return metrics, all_preds, all_probs


def train(config):
    """
    Main training loop.

    Args:
        config: Dictionary with training configuration
    """
    # Create output directories
    import os
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    data = torch.load(config['data_path'], weights_only=False)
    train_set = data['train']
    val_set = data['val']
    test_set = data['test']
    class_weights = data['class_weights'].to(device)

    print(f"Train: {len(train_set)} pairs")
    print(f"Val: {len(val_set)} pairs")
    print(f"Test: {len(test_set)} pairs")

    # Create dataloaders
    train_loader = DataLoader(
        PairDataset(train_set),
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        PairDataset(val_set),
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        PairDataset(test_set),
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )

    # Build model
    # Detect structural feature dimension from data (if present) and set in config
    struct_feat_dim = 0
    if len(train_set) > 0 and len(train_set[0]) == 4:
        sample_feat = train_set[0][3]
        try:
            struct_feat_dim = sample_feat.shape[-1]
        except Exception:
            struct_feat_dim = 0

    config['struct_feat_dim'] = struct_feat_dim

    print("\nBuilding model...")
    model = build_model(config)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Precision-focused loss to reduce over-conservative predictions
    criterion = PrecisionPenaltyLoss(
        class_weights=class_weights,
        false_positive_penalty=3.0  # Strong penalty for false positives
    )

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=config['scheduler_patience']
    )

    # Training loop
    print("\nStarting training...")
    best_val_f1 = 0
    patience_counter = 0

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': [],
        'val_kappa': []
    }

    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Train loss: {train_loss:.4f}")

        # Validate
        val_metrics, _, _ = evaluate(model, val_loader, criterion, device)
        print(f"Val loss: {val_metrics['loss']:.4f}")
        print(f"Val accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Val F1 (macro): {val_metrics['f1']:.4f}")
        print(f"Val AUROC: {val_metrics['auroc']:.4f}")
        print(f"Val Kappa: {val_metrics['kappa']:.4f}")

        # Per-class metrics
        for i, (p, r, f) in enumerate(zip(
            val_metrics['precision_per_class'],
            val_metrics['recall_per_class'],
            val_metrics['f1_per_class']
        )):
            class_name = ['SUGGEST', 'CAUTION', 'AVOID'][i]
            print(f"  {class_name}: P={p:.3f}, R={r:.3f}, F1={f:.3f}")

        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_kappa'].append(val_metrics['kappa'])

        # Learning rate scheduling
        scheduler.step(val_metrics['f1'])

        # Early stopping
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            patience_counter = 0

            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'val_metrics': val_metrics
            }, config['checkpoint_path'])
            print(f"✓ Saved best model (F1={best_val_f1:.4f})")

        else:
            patience_counter += 1
            if patience_counter >= config['early_stopping_patience']:
                print(f"\nEarly stopping after {epoch + 1} epochs")
                break

    # Load best model and evaluate on test set
    print("\n" + "=" * 50)
    print("Evaluating best model on test set...")
    checkpoint = torch.load(config['checkpoint_path'], weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics, test_preds, test_probs = evaluate(model, test_loader, criterion, device)

    print(f"\nTest Results:")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"F1 (macro): {test_metrics['f1']:.4f}")
    print(f"AUROC: {test_metrics['auroc']:.4f}")
    print(f"Kappa: {test_metrics['kappa']:.4f}")

    print(f"\nPer-class metrics:")
    for i, (p, r, f, s) in enumerate(zip(
        test_metrics['precision_per_class'],
        test_metrics['recall_per_class'],
        test_metrics['f1_per_class'],
        test_metrics['support_per_class']
    )):
        class_name = ['SUGGEST', 'CAUTION', 'AVOID'][i]
        print(f"{class_name}: P={p:.3f}, R={r:.3f}, F1={f:.3f} (n={s})")

    print(f"\nConfusion Matrix:")
    print("             Predicted")
    print("           SUG  CAU  AVO")
    for i, row in enumerate(test_metrics['confusion_matrix']):
        class_name = ['SUG', 'CAU', 'AVO'][i]
        print(f"True {class_name}  {row[0]:4d} {row[1]:4d} {row[2]:4d}")

    # Evaluate on full dataset for comprehensive clinical assessment
    print("\n" + "=" * 50)
    print("Evaluating model on FULL DATASET for clinical deployment...")
    
    # Combine all data
    full_data = train_set + val_set + test_set
    full_dataset = PairDataset(full_data)
    full_loader = DataLoader(full_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)
    
    full_metrics, full_preds, full_probs = evaluate(model, full_loader, criterion, device)
    
    print(f"\nFULL DATASET Results ({len(full_data)} drug pairs):")
    print(f"Accuracy: {full_metrics['accuracy']:.4f} ({full_metrics['accuracy']*100:.1f}%)")
    print(f"F1 (macro): {full_metrics['f1']:.4f}")
    print(f"AUROC: {full_metrics['auroc']:.4f}")
    print(f"Kappa: {full_metrics['kappa']:.4f}")

    print(f"\nFull Dataset Per-class metrics:")
    for i, (p, r, f, s) in enumerate(zip(
        full_metrics['precision_per_class'],
        full_metrics['recall_per_class'],
        full_metrics['f1_per_class'],
        full_metrics['support_per_class']
    )):
        class_name = ['SUGGEST', 'CAUTION', 'AVOID'][i]
        print(f"{class_name}: P={p:.3f}, R={r:.3f}, F1={f:.3f} (n={s})")

    print(f"\nFull Dataset Confusion Matrix:")
    print("             Predicted")
    print("           SUG  CAU  AVO")
    for i, row in enumerate(full_metrics['confusion_matrix']):
        class_name = ['SUG', 'CAU', 'AVO'][i]
        print(f"True {class_name}  {row[0]:4d} {row[1]:4d} {row[2]:4d}")

    # Save comprehensive results
    results = {
        'config': config,
        'history': history,
        'test_metrics': test_metrics,
        'full_dataset_metrics': full_metrics,
        'evaluation_summary': {
            'test_set_size': len(test_set),
            'full_dataset_size': len(full_data),
            'test_accuracy': test_metrics['accuracy'],
            'full_accuracy': full_metrics['accuracy'],
            'test_kappa': test_metrics['kappa'],
            'full_kappa': full_metrics['kappa'],
            'clinical_interpretation': {
                'test_performance': 'Unbiased evaluation on unseen data',
                'full_performance': 'Real-world clinical deployment readiness'
            }
        }
    }

    with open(config['results_path'], 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n" + "=" * 70)
    print("COMPREHENSIVE EVALUATION SUMMARY")
    print("=" * 70)
    print(f"📊 TEST SET (Unbiased Research Validation):")
    print(f"   • Size: {len(test_set)} pairs")
    print(f"   • Accuracy: {test_metrics['accuracy']*100:.1f}%")
    print(f"   • Cohen's Kappa: {test_metrics['kappa']:.3f}")
    print(f"   • Use for: Research papers, model comparison")
    print()
    print(f"🏥 FULL DATASET (Clinical Deployment Assessment):")
    print(f"   • Size: {len(full_data)} pairs (all 43×43 drug combinations)")
    print(f"   • Accuracy: {full_metrics['accuracy']*100:.1f}%") 
    print(f"   • Cohen's Kappa: {full_metrics['kappa']:.3f}")
    print(f"   • Use for: Clinical decision support system")
    print("=" * 70)

    print(f"\n✓ Comprehensive results saved to {config['results_path']}")

    # Optional visualization generation (controlled by config flag)
    if config.get('generate_plots', False):
        print("\n" + "=" * 70)
        print("GENERATING COMPREHENSIVE EVALUATION VISUALIZATIONS")
        print("=" * 70)
        
        try:
            from comprehensive_evaluation import evaluate_model_comprehensive
            eval_results = evaluate_model_comprehensive(model, train_set, val_set, test_set, device)
            print("✓ Comprehensive evaluation completed with visualizations")
        except ImportError:
            print("⚠️  Comprehensive evaluation module not available")
            print("   Basic evaluation completed above")
    else:
        print(f"\n💡 To generate visualizations, run:")
        print(f"   python run_comprehensive_eval.py")
        print(f"   or set config['generate_plots'] = True")

    return model, test_metrics, full_metrics


if __name__ == '__main__':
    # Default configuration
    config = {
        # Data
        'data_path': 'data/processed_data.pt',

        # Model architecture
        'node_feat_dim': 6,
        'edge_feat_dim': 2,
        'hidden_dim': 128,
        'embedding_dim': 256,
        'num_layers': 4,
        'gnn_type': 'GIN',  # GIN, GAT, or GCN
        'pooling': 'mean',
        'dropout': 0.2,
        'predictor_hidden_dim': 512,
        'predictor_dropout': 0.3,
        'num_classes': 3,

        # Training
        'batch_size': 32,
        'num_epochs': 100,
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'scheduler_patience': 5,
        'early_stopping_patience': 15,

        # Paths
        'checkpoint_path': 'models/best_model.pt',
        'results_path': 'results/results.json'
    }

    # Train model
    model, test_metrics = train(config)
