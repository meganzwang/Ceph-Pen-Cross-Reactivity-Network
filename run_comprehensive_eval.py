#!/usr/bin/env python3
"""
Run comprehensive evaluation on trained model with proper unique pair handling
"""

import torch
import os
import sys

def main():
    """Run comprehensive evaluation on the best trained model."""
    
    # Configuration matching train.py
    config = {
        'data_path': 'data/processed_data.pt',
        'checkpoint_path': 'models/best_model.pt',
        'node_feat_dim': 6,
        'edge_feat_dim': 2,
        'hidden_dim': 128,
        'embedding_dim': 256,
        'num_layers': 4,
        'gnn_type': 'GIN',
        'pooling': 'mean',
        'dropout': 0.2,
        'predictor_hidden_dim': 512,
        'predictor_dropout': 0.3,
        'num_classes': 3,
    }
    
    print("🔍 COMPREHENSIVE MODEL EVALUATION")
    print("="*50)
    print("Loading trained model and data...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    if not os.path.exists(config['data_path']):
        print(f"❌ Data file not found: {config['data_path']}")
        print("Please run training first or check the data path.")
        return
    
    data = torch.load(config['data_path'], weights_only=False)
    train_set = data['train']
    val_set = data['val'] 
    test_set = data['test']
    
    print(f"✓ Loaded data:")
    print(f"  Train: {len(train_set)} pairs")
    print(f"  Validation: {len(val_set)} pairs") 
    print(f"  Test: {len(test_set)} pairs")
    
    # Load model checkpoint and configuration
    if not os.path.exists(config['checkpoint_path']):
        print(f"❌ Model checkpoint not found: {config['checkpoint_path']}")
        print("Please train a model first.")
        return
        
    checkpoint = torch.load(config['checkpoint_path'], weights_only=False)
    
    # Use the saved configuration if available
    if 'config' in checkpoint:
        saved_config = checkpoint['config']
        # Update our config with the saved parameters
        for key in ['struct_feat_dim', 'node_feat_dim', 'edge_feat_dim', 'hidden_dim', 
                   'embedding_dim', 'num_layers', 'gnn_type', 'pooling', 'dropout',
                   'predictor_hidden_dim', 'predictor_dropout', 'num_classes']:
            if key in saved_config:
                config[key] = saved_config[key]
        print(f"✓ Using saved model configuration")
    else:
        print("⚠️  No saved config found, using default parameters")
    
    from model import build_model
    model = build_model(config)
    model = model.to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Loaded model with {total_params:,} parameters")
    
    # Run comprehensive evaluation
    print(f"\n🚀 Running comprehensive evaluation...")
    
    try:
        from comprehensive_evaluation import evaluate_model_comprehensive
        results = evaluate_model_comprehensive(model, train_set, val_set, test_set, device)
        
        print(f"\n✅ EVALUATION COMPLETE!")
        print(f"📊 Results saved to: plots/comprehensive_evaluation.json")
        print(f"📈 Visualizations saved to: plots/")
        print(f"   • comprehensive_evaluation_dashboard.png")
        print(f"   • detailed_confusion_matrices.png")
        
        # Summary
        test_acc = results['test']['accuracy']
        test_kappa = results['test']['kappa']
        full_acc = results['full']['accuracy'] 
        full_kappa = results['full']['kappa']
        
        print(f"\n📋 SUMMARY:")
        print(f"   Test Set:     {test_acc:.1%} accuracy, κ={test_kappa:.3f}")
        print(f"   Full Dataset: {full_acc:.1%} accuracy, κ={full_kappa:.3f}")
        print(f"   Unique Test Pairs: {results['test']['unique_pairs']}")
        print(f"   Unique Full Pairs: {results['full']['unique_pairs']}")
        
    except ImportError as e:
        print(f"❌ Error importing comprehensive evaluation: {e}")
        print("Please ensure all required packages are installed.")
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()