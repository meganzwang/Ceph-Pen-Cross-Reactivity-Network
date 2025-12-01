"""
Training pipeline for cross-reactivity prediction
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, cohen_kappa_score
)
from tqdm import tqdm
import json
import os

from model import build_model


class FocalLossWithSmoothing(nn.Module):
    """
    Enhanced Focal Loss with Label Smoothing for Kappa optimization.
    Combines:
    1. Focal loss - focuses on hard examples  
    2. Class weighting - addresses imbalance
    3. Label smoothing - reduces overconfidence
    """
    def __init__(self, alpha=None, gamma=2.0, smoothing=0.1, reduction='mean'):
        super(FocalLossWithSmoothing, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        num_classes = inputs.size(1)
        
        # Label smoothing
        if self.smoothing > 0:
            # Create smooth labels
            smooth_labels = torch.zeros_like(inputs)
            smooth_labels.fill_(self.smoothing / (num_classes - 1))
            smooth_labels.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
            
            # Use smooth labels for loss calculation
            log_pt = F.log_softmax(inputs, dim=-1)
            loss = -smooth_labels * log_pt
            
            # Get probability for focal weighting (still use hard targets)
            pt = F.softmax(inputs, dim=-1)
            pt_class = pt.gather(1, targets.unsqueeze(1)).squeeze(1)
            
            # Apply focal weight
            focal_weight = (1 - pt_class) ** self.gamma
            loss = loss.sum(dim=-1) * focal_weight
            
        else:
            # Standard focal loss
            log_pt = F.log_softmax(inputs, dim=-1)
            pt = torch.exp(log_pt)
            
            log_pt = log_pt.gather(1, targets.unsqueeze(1)).squeeze(1)  
            pt = pt.gather(1, targets.unsqueeze(1)).squeeze(1)
            
            focal_weight = (1 - pt) ** self.gamma
            loss = -focal_weight * log_pt
        
        # Apply class weights
        if self.alpha is not None:
            alpha_weight = self.alpha[targets]
            loss = alpha_weight * loss
            
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


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
    original_weights = data['class_weights'].to(device)

    print(f"Train: {len(train_set)} pairs")
    print(f"Val: {len(val_set)} pairs")
    print(f"Test: {len(test_set)} pairs")
    
    # Check class distribution and apply more aggressive rebalancing
    labels = [item[2].item() for item in train_set]
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    
    print(f"\nClass distribution in training set:")
    for cls, count in zip(unique, counts):
        class_name = ['SUGGEST', 'CAUTION', 'AVOID'][cls]
        percentage = (count/total)*100
        print(f"  {class_name}: {count} samples ({percentage:.1f}%)")
    
    # Balanced approach: Detect AVOID cases but reduce false positives
    # Target: Cohen's Kappa >= 0.6 for good clinical agreement
    
    # Calculate more balanced weights based on inverse frequency but not too extreme
    inv_freq = 1.0 / counts
    base_weights = inv_freq / inv_freq.sum() * len(unique)
    
    # GRADUATED KAPPA OPTIMIZATION STRATEGY
    # Problem: CAUTION class ignored (0% F1), AVOID low precision (12.5%)
    # Solution: Three-stage training with progressive weight increases
    
    # Stage 1: Gentle weights to establish basic minority class detection
    gentle_weights = torch.tensor([1.0, 4.0, 5.0], dtype=torch.float).to(device)
    
    # Stage 2: Moderate weights for balanced development  
    moderate_weights = torch.tensor([1.0, 6.0, 7.0], dtype=torch.float).to(device)
    
    # Stage 3: Final weights for Kappa optimization
    final_weights = torch.tensor([1.0, 8.0, 10.0], dtype=torch.float).to(device)
    
    # Start with gentle weights
    improved_weights = gentle_weights
    current_stage = 1
    
    print(f"Original class weights: {original_weights}")
    print(f"Balanced class weights: {improved_weights}")
    print(f"🎯 PRIMARY TARGET: Cohen's Kappa ≥ 0.6 (previous best: 0.363)")
    print("🔧 GRADUATED TRAINING STRATEGY: Progressive weights to optimize all 3 classes for Kappa")
    print(f"📊 Stage 1 Weights: SUGGEST={gentle_weights[0]:.1f}, CAUTION={gentle_weights[1]:.1f}, AVOID={gentle_weights[2]:.1f}")
    print("🎓 GRADUATED TRAINING: Will progress through 3 stages to optimize all classes")

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

    # Loss function: Balanced Focal Loss + Label Smoothing for Kappa optimization  
    # Strategy: Moderate parameters to balance precision/recall for better Kappa
    criterion = FocalLossWithSmoothing(
        alpha=improved_weights,  # Moderate class weighting (1.2, 8.0, 12.0)
        gamma=1.5,              # Reduced focus intensity for better precision
        smoothing=0.1,          # More smoothing to prevent overconfidence  
        reduction='mean'
    )

    # Optimizer - Lower LR for more stable Kappa optimization
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'] * 0.5,  # Reduce LR for better convergence
        weight_decay=config['weight_decay'] * 2  # Increase regularization  
    )

    # Learning rate scheduler - More aggressive for Kappa focus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # Monitor validation Kappa
        factor=0.3,  # More aggressive LR reduction
        patience=3,  # Faster adaptation
        min_lr=1e-6
    )

    # Training loop targeting Cohen's Kappa ≥ 0.6
    print(f"\nStarting training with balanced class weighting (Target: Kappa ≥ 0.6)...")
    best_val_f1 = 0
    best_kappa = 0  # Track Cohen's Kappa (primary metric)
    best_avoid_recall = 0  # Track AVOID class performance
    patience_counter = 0

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': [],
        'val_avoid_recall': [],  
        'val_kappa': []
    }

    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Train loss: {train_loss:.4f}")

        # Validate
        val_metrics, _, _ = evaluate(model, val_loader, criterion, device)
        current_kappa = val_metrics['kappa']
        
        print(f"Val loss: {val_metrics['loss']:.4f}")
        print(f"Val accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Val F1 (macro): {val_metrics['f1']:.4f}")
        print(f"Val AUROC: {val_metrics['auroc']:.4f}")
        
        # Emphasize Kappa progress toward target of 0.6
        kappa_status = ""
        if current_kappa >= 0.6:
            kappa_status = "🎯 TARGET REACHED!"
        elif current_kappa >= 0.5:
            kappa_status = "📈 VERY GOOD"  
        elif current_kappa >= 0.4:
            kappa_status = "👍 GOOD"
        elif current_kappa >= 0.3:
            kappa_status = "📊 FAIR"
        else:
            kappa_status = "⚠️ POOR"
        
        print(f"Val Kappa: {current_kappa:.4f} ({kappa_status}, target: ≥0.6)")

        # Per-class metrics with AVOID class emphasis
        avoid_recall = 0
        for i, (p, r, f) in enumerate(zip(
            val_metrics['precision_per_class'],
            val_metrics['recall_per_class'],
            val_metrics['f1_per_class']
        )):
            class_name = ['SUGGEST', 'CAUTION', 'AVOID'][i]
            if i == 2:  # AVOID class - highlight it
                avoid_recall = r
                print(f"  🚨 {class_name}: P={p:.3f}, R={r:.3f}, F1={f:.3f} (CRITICAL)")
            else:
                print(f"  {class_name}: P={p:.3f}, R={r:.3f}, F1={f:.3f}")

        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_avoid_recall'].append(avoid_recall)
        history['val_kappa'].append(val_metrics['kappa'])

        # GRADUATED WEIGHT PROGRESSION based on class performance
        caution_f1 = val_metrics['f1_per_class'][1]  # CAUTION F1
        avoid_f1 = val_metrics['f1_per_class'][2]    # AVOID F1
        
        # Stage progression logic
        stage_changed = False
        if current_stage == 1 and caution_f1 > 0.1 and avoid_f1 > 0.15:
            # Stage 2: CAUTION starting to be detected
            improved_weights = moderate_weights
            current_stage = 2
            stage_changed = True
            print(f"🎓 STAGE 2: Progressing to moderate weights - CAUTION F1={caution_f1:.3f}")
            
        elif current_stage == 2 and caution_f1 > 0.2 and avoid_f1 > 0.25 and epoch > 15:
            # Stage 3: Both minorities showing progress
            improved_weights = final_weights  
            current_stage = 3
            stage_changed = True
            print(f"🎓 STAGE 3: Final weights - CAUTION F1={caution_f1:.3f}, AVOID F1={avoid_f1:.3f}")
        
        # Update criterion if weights changed
        if stage_changed:
            criterion = FocalLossWithSmoothing(
                alpha=improved_weights,
                gamma=1.5,
                smoothing=0.1,
                reduction='mean'
            )
            print(f"📊 Updated Weights: SUGGEST={improved_weights[0]:.1f}, CAUTION={improved_weights[1]:.1f}, AVOID={improved_weights[2]:.1f}")

        # Learning rate scheduling
        # Schedule based on Kappa (primary target metric)
        scheduler.step(val_metrics['kappa'])

        # KAPPA-FIRST MODEL SAVING: Prioritize Cohen's Kappa ≥ 0.6 for clinical agreement
        model_improved = False
        save_reason = ""
        current_kappa = val_metrics['kappa']
        
        # PRIMARY: Direct Kappa improvement (most important metric)
        if current_kappa > best_kappa:
            model_improved = True
            save_reason = f"κ: {best_kappa:.3f}→{current_kappa:.3f}"
            best_kappa = current_kappa
            
        # SECONDARY: Balanced performance with decent Kappa
        elif (current_kappa >= 0.35 and val_metrics['f1'] > best_val_f1 and 
              avoid_recall >= 0.4):  # Ensure AVOID detection isn't lost
            model_improved = True
            save_reason = f"Balanced: F1↑{val_metrics['f1']:.3f}, κ={current_kappa:.3f}, AVOID={avoid_recall:.1%}"
            best_val_f1 = val_metrics['f1']
            
        if model_improved:
            best_avoid_recall = max(best_avoid_recall, avoid_recall)
            patience_counter = 0

            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'val_metrics': val_metrics,
                'class_weights_used': improved_weights.cpu()
            }, config['checkpoint_path'])
            print(f"✓ Saved best model ({save_reason}, AVOID recall={avoid_recall:.3f})")

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

    print(f"\n🎯 BALANCED MODEL TEST RESULTS (Target: Cohen's Kappa ≥ 0.6)")
    print(f"=" * 70)
    
    final_kappa = test_metrics['kappa']
    kappa_status = ""
    if final_kappa >= 0.6:
        kappa_status = "🎯 EXCELLENT - TARGET ACHIEVED!"
    elif final_kappa >= 0.5:
        kappa_status = "📈 VERY GOOD - Close to target"  
    elif final_kappa >= 0.4:
        kappa_status = "👍 GOOD - Moderate agreement"
    elif final_kappa >= 0.3:
        kappa_status = "📊 FAIR - Some agreement"
    else:
        kappa_status = "⚠️ POOR - Weak agreement"
    
    print(f"🏆 Cohen's Kappa: {final_kappa:.3f} ({kappa_status})")
    print(f"Overall Accuracy: {test_metrics['accuracy']:.1%}")
    print(f"Macro F1 Score: {test_metrics['f1']:.1%}")
    print(f"AUROC: {test_metrics['auroc']:.1%}")

    print(f"\n📊 Detailed Per-Class Performance:")
    avoid_recall = 0
    avoid_precision = 0
    for i, (p, r, f, s) in enumerate(zip(
        test_metrics['precision_per_class'],
        test_metrics['recall_per_class'],
        test_metrics['f1_per_class'],
        test_metrics['support_per_class']
    )):
        class_name = ['SUGGEST', 'CAUTION', 'AVOID'][i]
        if i == 2:  # AVOID class
            avoid_recall = r
            avoid_precision = p
            print(f"🚨 {class_name:7s}: Precision={p:.1%}, Recall={r:.1%}, F1={f:.1%} (n={s}) ← CRITICAL")
        else:
            print(f"   {class_name:7s}: Precision={p:.1%}, Recall={r:.1%}, F1={f:.1%} (n={s})")

    print(f"\n🔥 Confusion Matrix:")
    print("              Predicted")
    print("           SUG  CAU  AVO")
    for i, row in enumerate(test_metrics['confusion_matrix']):
        class_name = ['SUG', 'CAU', 'AVO'][i]
        print(f"True {class_name}  {row[0]:4d} {row[1]:4d} {row[2]:4d}")

    # Analyze AVOID class performance
    print(f"\n🎯 AVOID CLASS ANALYSIS:")
    if avoid_recall > 0:
        print(f"✅ SUCCESS: AVOID class now detects {avoid_recall:.1%} of dangerous combinations!")
        print(f"   Precision: {avoid_precision:.1%} (when model says AVOID, it's right {avoid_precision:.1%} of time)")
        if avoid_recall >= 0.5:
            print(f"🏆 EXCELLENT: Model catches majority of dangerous drug pairs!")
        elif avoid_recall >= 0.3:
            print(f"👍 GOOD: Significant improvement in safety detection!")
        else:
            print(f"📈 PROGRESS: Improvement from 0%, but could be better")
    else:
        print(f"⚠️  STILL FAILING: 0% recall - AVOID class not being detected")
        print(f"   Consider: Even more aggressive weighting or different approach")
    
    print(f"\n💡 Class Weights Used:")
    print(f"   SUGGEST: {improved_weights[0]:.1f} (reduced to allow other predictions)")
    print(f"   CAUTION: {improved_weights[1]:.1f} (moderately increased)")  
    print(f"   AVOID:   {improved_weights[2]:.1f} (heavily increased for safety)")
    
    # Clinical interpretation
    total_avoid_cases = int(test_metrics['support_per_class'][2]) if len(test_metrics['support_per_class']) > 2 else 0
    detected_avoid = int(avoid_recall * total_avoid_cases)
    print(f"\n🏥 Clinical Impact:")
    print(f"   Dangerous combinations in test set: {total_avoid_cases}")
    print(f"   Successfully detected by model: {detected_avoid}")
    print(f"   Missed dangerous combinations: {total_avoid_cases - detected_avoid}")

    # Save results
    results = {
        'config': config,
        'history': history,
        'test_metrics': test_metrics
    }

    with open(config['results_path'], 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {config['results_path']}")

    return model, test_metrics


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
