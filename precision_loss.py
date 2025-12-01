"""
Precision-Focused Loss Functions to Reduce Over-Conservative Predictions

Problem: Model is "overavoiding" - predicting too many false AVOID pairs
- AVOID Precision: 33% (67% false positives)
- 100 false AVOID predictions vs 58 correct ones

Solutions:
1. Precision-Penalty Loss: Penalize false positives more than false negatives
2. Confidence-Calibrated Loss: Only predict AVOID with high confidence
3. Asymmetric Focal Loss: Different gamma values for different error types
"""

import torch
import torch.nn.functional as F
import numpy as np


class PrecisionPenaltyLoss(torch.nn.Module):
    """
    Loss function that penalizes false positives more heavily than false negatives
    to reduce over-conservative predictions.
    """
    
    def __init__(self, class_weights=None, false_positive_penalty=2.0):
        super(PrecisionPenaltyLoss, self).__init__()
        self.class_weights = class_weights
        self.fp_penalty = false_positive_penalty
        
    def forward(self, logits, targets):
        # Base cross-entropy loss
        ce_loss = F.cross_entropy(logits, targets, weight=self.class_weights, reduction='none')
        
        # Get predictions
        predictions = torch.argmax(logits, dim=1)
        
        # Identify false positives for minority classes (CAUTION=1, AVOID=2)
        fp_mask_caution = (predictions == 1) & (targets != 1)  # Predicted CAUTION but not true CAUTION
        fp_mask_avoid = (predictions == 2) & (targets != 2)    # Predicted AVOID but not true AVOID
        
        # Apply penalty to false positives
        penalty = torch.ones_like(ce_loss)
        penalty[fp_mask_caution] = self.fp_penalty
        penalty[fp_mask_avoid] = self.fp_penalty * 1.5  # Higher penalty for AVOID false positives
        
        return (ce_loss * penalty).mean()


class ConfidenceThresholdLoss(torch.nn.Module):
    """
    Loss that requires high confidence for minority class predictions.
    Encourages model to only predict CAUTION/AVOID when very confident.
    """
    
    def __init__(self, class_weights=None, confidence_threshold=0.7):
        super(ConfidenceThresholdLoss, self).__init__()
        self.class_weights = class_weights
        self.confidence_threshold = confidence_threshold
        
    def forward(self, logits, targets):
        # Base cross-entropy
        ce_loss = F.cross_entropy(logits, targets, weight=self.class_weights, reduction='none')
        
        # Get probabilities
        probs = F.softmax(logits, dim=1)
        max_probs = torch.max(probs, dim=1)[0]
        
        # Penalty for low-confidence minority predictions
        predictions = torch.argmax(logits, dim=1)
        
        # If predicting CAUTION/AVOID with low confidence, add penalty
        low_confidence_minority = (
            ((predictions == 1) | (predictions == 2)) & 
            (max_probs < self.confidence_threshold)
        )
        
        penalty = torch.ones_like(ce_loss)
        penalty[low_confidence_minority] = 3.0  # High penalty for uncertain minority predictions
        
        return (ce_loss * penalty).mean()


class AsymmetricFocalLoss(torch.nn.Module):
    """
    Focal Loss with different gamma values for false positives vs false negatives.
    Higher gamma for false positives to reduce over-prediction of minorities.
    """
    
    def __init__(self, class_weights=None, gamma_fn=2.0, gamma_fp=3.0, alpha=1.0):
        super(AsymmetricFocalLoss, self).__init__()
        self.class_weights = class_weights
        self.gamma_fn = gamma_fn  # Gamma for false negatives
        self.gamma_fp = gamma_fp  # Gamma for false positives (higher to penalize more)
        self.alpha = alpha
        
    def forward(self, logits, targets):
        # Cross entropy loss
        ce_loss = F.cross_entropy(logits, targets, weight=self.class_weights, reduction='none')
        
        # Probabilities
        probs = F.softmax(logits, dim=1)
        
        # Get probability of true class
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Get predictions
        predictions = torch.argmax(logits, dim=1)
        
        # Determine if this is a false positive or false negative
        correct_predictions = (predictions == targets)
        
        # For incorrect predictions, determine type
        false_positives = ~correct_predictions & ((predictions == 1) | (predictions == 2))
        false_negatives = ~correct_predictions & ~false_positives
        
        # Apply different gamma values
        gamma = torch.ones_like(ce_loss) * self.gamma_fn  # Default for false negatives
        gamma[false_positives] = self.gamma_fp  # Higher for false positives
        
        # Focal loss formula: -alpha * (1-pt)^gamma * log(pt)
        focal_loss = -self.alpha * torch.pow(1 - pt, gamma) * torch.log(pt + 1e-8)
        
        return focal_loss.mean()


class BalancedPrecisionLoss(torch.nn.Module):
    """
    Balanced loss that aims to improve precision while maintaining recall.
    Combines cross-entropy with precision penalty and confidence requirements.
    """
    
    def __init__(self, class_weights=None, precision_weight=0.3, confidence_threshold=0.6):
        super(BalancedPrecisionLoss, self).__init__()
        self.class_weights = class_weights
        self.precision_weight = precision_weight
        self.confidence_threshold = confidence_threshold
        
    def forward(self, logits, targets):
        # Base cross-entropy
        ce_loss = F.cross_entropy(logits, targets, weight=self.class_weights)
        
        # Precision penalty component
        probs = F.softmax(logits, dim=1)
        predictions = torch.argmax(logits, dim=1)
        
        # Calculate precision penalty
        precision_penalty = 0.0
        
        # For CAUTION class (1)
        caution_pred_mask = (predictions == 1)
        if caution_pred_mask.sum() > 0:
            caution_correct = ((predictions == 1) & (targets == 1)).float().sum()
            caution_precision = caution_correct / caution_pred_mask.sum()
            precision_penalty += (1.0 - caution_precision) ** 2
        
        # For AVOID class (2)  
        avoid_pred_mask = (predictions == 2)
        if avoid_pred_mask.sum() > 0:
            avoid_correct = ((predictions == 2) & (targets == 2)).float().sum()
            avoid_precision = avoid_correct / avoid_pred_mask.sum()
            precision_penalty += (1.0 - avoid_precision) ** 2
        
        # Confidence penalty - penalize low-confidence minority predictions
        max_probs = torch.max(probs, dim=1)[0]
        low_conf_minority = (
            ((predictions == 1) | (predictions == 2)) & 
            (max_probs < self.confidence_threshold)
        )
        confidence_penalty = low_conf_minority.float().mean()
        
        # Combine losses
        total_loss = ce_loss + self.precision_weight * (precision_penalty + confidence_penalty)
        
        return total_loss


def get_precision_optimized_weights(current_performance):
    """
    Calculate class weights optimized for precision improvement.
    
    Args:
        current_performance: dict with precision/recall per class
    
    Returns:
        torch.tensor of optimized class weights
    """
    # Current performance from your model:
    # SUGGEST: P=98.8%, R=89.3% 
    # CAUTION: P=29.7%, R=59.5%
    # AVOID: P=33.0%, R=63.0%
    
    suggest_precision = current_performance.get('suggest_precision', 0.988)
    caution_precision = current_performance.get('caution_precision', 0.297)
    avoid_precision = current_performance.get('avoid_precision', 0.330)
    
    # Reduce weights for classes with low precision to discourage over-prediction
    suggest_weight = 1.0  # Already high precision
    caution_weight = max(0.5, caution_precision * 8.0)  # Scale based on precision
    avoid_weight = max(0.5, avoid_precision * 10.0)  # Scale based on precision
    
    return torch.tensor([suggest_weight, caution_weight, avoid_weight], dtype=torch.float)


# Example usage in training
def create_precision_focused_criterion(strategy='balanced'):
    """
    Create a precision-focused loss function.
    
    Args:
        strategy: 'penalty', 'confidence', 'asymmetric', or 'balanced'
    """
    # Your current performance (from evaluation results)
    current_perf = {
        'suggest_precision': 0.988,
        'caution_precision': 0.297, 
        'avoid_precision': 0.330
    }
    
    # Get optimized weights
    optimized_weights = get_precision_optimized_weights(current_perf)
    
    if strategy == 'penalty':
        return PrecisionPenaltyLoss(
            class_weights=optimized_weights,
            false_positive_penalty=2.5
        )
    elif strategy == 'confidence':
        return ConfidenceThresholdLoss(
            class_weights=optimized_weights,
            confidence_threshold=0.7
        )
    elif strategy == 'asymmetric':
        return AsymmetricFocalLoss(
            class_weights=optimized_weights,
            gamma_fn=2.0,  # Normal penalty for missing dangerous pairs
            gamma_fp=4.0   # Higher penalty for false alarms
        )
    elif strategy == 'balanced':
        return BalancedPrecisionLoss(
            class_weights=optimized_weights,
            precision_weight=0.4,
            confidence_threshold=0.65
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


if __name__ == "__main__":
    print("🎯 PRECISION-FOCUSED LOSS FUNCTIONS")
    print("=" * 50)
    print("Problem: Over-conservative predictions (too many false AVOID)")
    print("Solution: Loss functions that penalize false positives more heavily")
    print()
    
    # Example current performance
    print("📊 Current Performance Issues:")
    print("  AVOID Precision: 33.0% (67% false positives)")
    print("  CAUTION Precision: 29.7% (70% false positives)")
    print("  Result: Model is too conservative - overavoiding drug pairs")
    print()
    
    print("🔧 Available Loss Functions:")
    print("1. PrecisionPenaltyLoss - Direct penalty for false positives")
    print("2. ConfidenceThresholdLoss - Require high confidence for minorities") 
    print("3. AsymmetricFocalLoss - Different penalties for FP vs FN")
    print("4. BalancedPrecisionLoss - Comprehensive precision optimization")
    print()
    
    # Calculate optimized weights
    current_perf = {
        'suggest_precision': 0.988,
        'caution_precision': 0.297,
        'avoid_precision': 0.330
    }
    
    optimized_weights = get_precision_optimized_weights(current_perf)
    print("📈 Optimized Class Weights:")
    print(f"  SUGGEST: {optimized_weights[0]:.2f}")
    print(f"  CAUTION: {optimized_weights[1]:.2f}") 
    print(f"  AVOID: {optimized_weights[2]:.2f}")
    print()
    
    print("✅ Next Steps:")
    print("1. Update train.py to use BalancedPrecisionLoss")
    print("2. Train with precision-focused loss function")
    print("3. Evaluate precision improvements vs accuracy trade-offs")