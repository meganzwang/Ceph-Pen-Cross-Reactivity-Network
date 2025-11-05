# Beta-Lactam Antibiotic Cross-Reactivity Prediction

**Predicting allergic cross-reactivity between penicillins and cephalosporins using Graph Neural Networks**

**Authors:** Megan Wang, Haley Kahn
**Course:** Network Analysis in Healthcare

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Quick Start - How to Run](#quick-start)
3. [Materials](#materials)
4. [Modeling Approach](#modeling-approach)
5. [Experimental Design](#experimental-design)
6. [Preliminary Results](#preliminary-results)
7. [Problems & Solutions](#problems--solutions)
8. [Detailed Documentation](#detailed-documentation)

---

## Project Overview

### Objective

**Task Type:** Link prediction (pairwise drug cross-reactivity classification)

**Objective:** Predict allergic cross-reactivity risk between beta-lactam antibiotic pairs using molecular graph neural networks, and validate predictions against the Northwestern Medicine clinical reference chart.

**Target Outcome:** 3-class classification
- Class 0 (SUGGEST): Low cross-reactivity risk (dissimilar side chains)
- Class 1 (CAUTION): Moderate risk (similar R1/R2 side chains)
- Class 2 (AVOID): High risk (identical R1/R2 side chains)

**Population/Scope:** 28 FDA-approved beta-lactam antibiotics across 4 drug classes (penicillins, cephalosporins, carbapenems, monobactams)

**Hypothesis:** Graph neural networks can learn molecular structure patterns from atom-bond graphs to replicate expert clinical judgment on cross-reactivity risk, achieving substantial agreement (Cohen's κ > 0.60) with established clinical guidelines.

---

## Materials

### Data and Unit of Analysis

**Unit of Analysis:** Drug pairs (pairwise combinations of beta-lactam antibiotics)

**Dataset:**
- **28 drugs** across 4 beta-lactam classes:
  - 5 penicillins (Penicillin G/V, Oxacillin, Amoxicillin, Ampicillin, Piperacillin)
  - 18 cephalosporins across 5 generations (1st gen: Cefadroxil, Cephalexin, Cefazolin; 2nd gen: Cefaclor, Cefoxitin, Cefprozil, Cefuroxime; 3rd gen: 8 drugs; 4th gen: Cefepime; 5th gen: Ceftaroline, Ceftolozane)
  - 3 carbapenems (Ertapenem, Meropenem, Imipenem)
  - 2 monobactams (Aztreonam)
- **325 labeled drug pairs** extracted from Northwestern Medicine β-Lactam Cross-Reactivity Chart
- **Data sources:**
  - SMILES molecular structures: PubChem (https://pubchem.ncbi.nlm.nih.gov/)
  - Cross-reactivity labels: Northwestern Medicine clinical reference chart
  - Original reference: `data/ReferenceTableLabels.xlsx` (26×27 matrix format)

**Class Distribution:**
- Class 0 (SUGGEST): 266 pairs (82%) - Low cross-reactivity risk
- Class 1 (CAUTION): 40 pairs (12%) - Moderate risk
- Class 2 (AVOID): 19 pairs (6%) - High risk

**Note:** Severe class imbalance (82/12/6 split) is realistic for this problem - most beta-lactam pairs are safe, but identifying the dangerous pairs is critical for patient safety.

### Network Inputs

**Graph Representation:** Each drug is represented as a molecular graph where:

**Nodes (Atoms):**
- Each atom in the molecule is a node
- Example: Amoxicillin has 23 atoms → 23 nodes
- Node features (6-dimensional):
  1. **Atomic number** (e.g., 6 for carbon, 7 for nitrogen, 8 for oxygen, 16 for sulfur)
  2. **Degree** (number of bonds to neighboring atoms)
  3. **Formal charge** (ionic charge on the atom)
  4. **Hybridization** (sp, sp², sp³, sp³d, sp³d²)
  5. **Aromaticity** (binary: 1 if atom is in aromatic ring, 0 otherwise)
  6. **Total hydrogens** (number of attached hydrogen atoms)

**Edges (Bonds):**
- Each chemical bond between atoms is an edge
- Example: Amoxicillin has 48 bonds → 48 edges (96 directed edges)
- Edge features (2-dimensional):
  1. **Bond type** (1.0 for single, 2.0 for double, 3.0 for triple, 1.5 for aromatic)
  2. **Is in ring** (binary: 1 if bond is part of a ring structure, 0 otherwise)

**Graph Statistics:**
- **Number of graphs:** 28 (one per drug)
- **Graph sizes:** Variable (15-30 atoms per molecule)
- **Average degree:** ~2-3 bonds per atom (typical for organic molecules)
- **Sparsity:** Molecular graphs are highly sparse (only bonded atoms are connected)

**Pairwise Input:**
- For each drug pair (Drug A, Drug B), the model processes **two graphs simultaneously**
- Example: To predict Amoxicillin-Cefadroxil cross-reactivity, the model takes:
  - Graph 1: Amoxicillin molecular graph (23 nodes, 48 edges)
  - Graph 2: Cefadroxil molecular graph (similar structure with R1 side chain variation)

### Variables and Labels

**Input Variables:**
- **SMILES strings** (Simplified Molecular Input Line Entry System)
  - Example: Amoxicillin = `CC1(C)S[C@@H]2[C@H](NC(=O)[C@H](N)c3ccc(O)cc3)C(=O)N2[C@H]1C(=O)O`
  - Converted to molecular graphs using RDKit library
- **Drug metadata:**
  - Drug name
  - Drug class (penicillin, 1st-gen cephalosporin, 2nd-gen, etc.)
  - Class code (for categorization)

**Target Variable (Labels):**
- **3-class classification**
  - **Class 0 (SUGGEST):** Low cross-reactivity risk - structurally dissimilar side chains
  - **Class 1 (CAUTION):** Moderate risk - similar R1 or R2 side chains, use with caution
  - **Class 2 (AVOID):** High risk - identical R1 and R2 side chains, strong likelihood of cross-reaction

**Class Imbalance Handling:**
- **Class weights** computed as inverse frequency:
  - SUGGEST weight: 0.1473 (downweight majority class)
  - CAUTION weight: 0.9271 (upweight minority class)
  - AVOID weight: 1.9256 (strongly upweight rarest class)
- Applied in cross-entropy loss function to force model attention on minority classes

**Data Splitting:**
- **Stratified 70/15/15 split** (maintains class distribution in each set):
  - Training: 210 pairs (70%)
  - Validation: 45 pairs (15%)
  - Test: 45 pairs (15%)
- **Stratification ensures** each split has ~82/12/6 class distribution
- **No data leakage:** Same drug pair never appears in multiple splits

---

## Modeling Approach

### Architecture Overview

**Two-stage architecture:**

**Stage 1: Molecular Encoder (Graph Neural Network)**
- **Purpose:** Convert variable-size molecular graph → fixed-size embedding vector
- **Architecture:** Graph Isomorphism Network (GIN)
  - Input: Molecular graph with N atoms (variable size)
  - Node encoder: 6-dim atom features → 128-dim hidden representations
  - 4 message-passing layers (GINConv)
  - Global mean pooling: Aggregate all atom representations → single graph embedding
  - Output: 256-dim drug embedding vector
- **Parameters:** ~420K (encoder only)
- **Message passing intuition:**
  - Layer 1: Each atom learns from immediate neighbors (1-hop)
  - Layer 2: Information propagates 2 bonds away
  - Layer 3: Information propagates 3 bonds away
  - Layer 4: Information propagates 4 bonds away
  - After 4 layers, each atom's representation captures its local substructure (critical for identifying R1/R2 side chains that drive cross-reactivity)

**Stage 2: Pairwise Predictor (MLP Classifier)**
- **Purpose:** Combine two drug embeddings → predict cross-reactivity class
- **Input:** Two drug embeddings h₁, h₂ (each 256-dim)
- **Feature combination:**
  - Concatenate: [h₁, h₂, |h₁ - h₂|, h₁ ⊙ h₂] → 1024-dim
  - |h₁ - h₂|: Absolute difference (captures dissimilarity)
  - h₁ ⊙ h₂: Element-wise product (captures similarity)
- **Architecture:**
  - Linear(1024 → 512) + ReLU + Dropout(0.3)
  - Linear(512 → 256) + ReLU + Dropout(0.3)
  - Linear(256 → 3) → Logits for [SUGGEST, CAUTION, AVOID]
- **Parameters:** ~420K (predictor only)
- **Output:** 3 class probabilities (sum to 1.0 via softmax)

**Total Model:** ~840K parameters

### Loss Function

**Weighted Cross-Entropy Loss:**
```
Loss = -Σ w_c · y_c · log(p_c)
```
where:
- w_c = class weight (0.147 for SUGGEST, 0.927 for CAUTION, 1.926 for AVOID)
- y_c = true label (one-hot encoded)
- p_c = predicted probability

**Why weighted loss?**
- Addresses severe class imbalance (82/12/6)
- Forces model to pay attention to rare but critical AVOID class
- Without weights, model would achieve 82% accuracy by always predicting SUGGEST

### Optimization

- **Optimizer:** Adam
  - Learning rate: 1e-3
  - Weight decay: 1e-5 (L2 regularization)
- **Learning rate scheduler:** ReduceLROnPlateau
  - Monitors validation F1 score
  - Reduces LR by 0.5× if no improvement for 5 epochs
  - Prevents overshooting in later training stages
- **Early stopping:**
  - Patience: 15 epochs
  - Stops training if validation F1 doesn't improve
  - Prevents overfitting

### Alternative GNN Architectures

**GIN (Graph Isomorphism Network)** - Default
- Most expressive GNN for molecular graphs
- Provably as powerful as Weisfeiler-Lehman graph isomorphism test
- Best for distinguishing subtle structural differences (R1/R2 side chains)

**GAT (Graph Attention Network)** - Alternative
- Learns attention weights over neighboring atoms
- Provides interpretability (which atoms/bonds are important)
- Useful for understanding what drives model decisions

**GCN (Graph Convolutional Network)** - Baseline
- Simpler message passing
- Faster but less expressive than GIN

---

## Experimental Design

### Data Splitting Strategy

**Stratified 70/15/15 split:**
- **Training set:** 210 pairs (70%)
  - Used for learning model parameters
  - Model sees these examples during backpropagation
- **Validation set:** 45 pairs (15%)
  - Used for hyperparameter tuning and early stopping
  - Model never trained on these, but they guide training decisions
- **Test set:** 45 pairs (15%)
  - **Final evaluation only** - completely held out
  - Never used during training or model selection
  - Represents real-world performance on unseen drug pairs

**Why stratified?**
- Ensures each split maintains the 82/12/6 class distribution
- Prevents validation/test sets from having zero examples of AVOID class
- Critical for meaningful evaluation with imbalanced data

### Training Procedure

**Hyperparameters:**
```python
{
    'batch_size': 32,              # 32 pairs per batch
    'num_epochs': 100,             # Maximum training epochs
    'learning_rate': 1e-3,         # Initial learning rate
    'weight_decay': 1e-5,          # L2 regularization
    'dropout': 0.2,                # GNN dropout
    'predictor_dropout': 0.3,      # Classifier dropout
    'early_stopping_patience': 15, # Stop if no improvement for 15 epochs
    'scheduler_patience': 5        # Reduce LR if no improvement for 5 epochs
}
```

**Training loop:**
1. **Forward pass:** For each batch of drug pairs, compute predictions
2. **Loss computation:** Weighted cross-entropy loss
3. **Backward pass:** Compute gradients via backpropagation
4. **Parameter update:** Adam optimizer updates weights
5. **Validation:** After each epoch, evaluate on validation set
6. **Model selection:** Save model if validation F1 improves
7. **Early stopping:** Stop training if no improvement for 15 epochs

**Expected training time:**
- CPU: 10-30 minutes
- GPU: 5-10 minutes

### Evaluation Metrics

**Primary metric:** Cohen's Kappa (κ)
- Measures agreement with Northwestern clinical chart
- Accounts for chance agreement (more robust than accuracy)
- Target: κ > 0.60 (substantial agreement)
- Interpretation:
  - κ < 0.20: Poor agreement
  - κ 0.21-0.40: Fair agreement
  - κ 0.41-0.60: Moderate agreement
  - **κ 0.61-0.80: Substantial agreement** ← Our goal
  - κ 0.81-1.00: Almost perfect agreement

**Secondary metrics:**
- **Accuracy:** Overall correct predictions / total predictions
- **Precision (per class):** True positives / (True positives + False positives)
- **Recall (per class):** True positives / (True positives + False negatives)
  - **AVOID recall is critical** - measures how many dangerous pairs we catch
- **F1 score (macro):** Harmonic mean of precision and recall, averaged across classes
- **AUROC:** Area under ROC curve (measures class separability)
- **Confusion matrix:** Shows exactly where model makes mistakes

### Expected Performance

**Realistic expectations given dataset size:**

**Overall metrics:**
- Accuracy: 80-85%
- F1 (macro): 70-75%
- Cohen's Kappa: **60-70%** (substantial agreement target)
- AUROC: 80-85%

**Per-class performance:**
- **SUGGEST (266 samples):** F1 = 85-92% (well-represented, easy to learn)
- **CAUTION (40 samples):** F1 = 40-60% (limited data, moderate performance)
- **AVOID (19 samples):** F1 = 50-75% (very limited data, hardest class)
  - **AVOID recall:** 50-70% (critical safety metric - how many dangerous pairs we catch)

**Why minority class performance is limited:**
- Only 19 AVOID training examples
- Model needs more data to learn subtle structural patterns
- Trade-off: High precision (avoid false alarms) vs High recall (catch all dangerous pairs)

### Validation Against Clinical Chart

**Final step:** Compare model predictions to Northwestern Medicine chart

**Evaluation script:** `evaluate_vs_clinical_chart.py`
- Loads trained model
- Predicts cross-reactivity for all 325 labeled pairs
- Compares predictions to clinical reference labels
- Generates:
  - **Side-by-side heatmap** (clinical chart vs model predictions)
  - **Confusion matrix** with precision/recall
  - **Detailed metrics** (accuracy, kappa, per-class F1)

**Success criteria:**
- κ > 0.60: Model has substantial agreement with clinical guidelines
- AVOID recall > 60%: Model catches majority of dangerous pairs
- Qualitative analysis: Which pairs does model get wrong? Why?

---

## Preliminary Results

### Current Status

✅ **Data Collection Complete**
- 28 drugs with SMILES from PubChem
- 325 labeled pairs from Northwestern Medicine chart
- Excel reference table converted to training format

🚀 **Model Training** (in progress)
- GNN architecture: 4-layer GIN with 128-dim hidden, 256-dim embeddings
- ~840K parameters
- Training on 210 pairs, validating on 45 pairs, testing on 45 pairs
- Target: Cohen's Kappa > 0.60 (substantial agreement)

📊 **Next Steps**
- Complete training (expected: 10-30 minutes)
- Evaluate predictions vs Northwestern chart
- Generate side-by-side heatmap comparison
- Analyze which drug pairs model gets wrong
- Write final report & presentation (due Nov 10-14)

---

## Problems & Solutions

### Problem 1: Severe Class Imbalance

**Initial issue:** Original `labels.csv` had only 2 AVOID samples out of 90 pairs
- Class distribution: 83 SUGGEST, 5 CAUTION, 2 AVOID
- **Impact:** Impossible to create stratified train/val/test split
- **Error:** `ValueError: The least populated class in y has only 1 member`

**Solution:**
- User provided complete Northwestern Medicine chart in Excel format (`ReferenceTableLabels.xlsx`)
- Created `convert_excel_to_labels.py` to extract all 325 pairs from 26×27 matrix
- **New distribution:** 266 SUGGEST, 40 CAUTION, 19 AVOID
- Still imbalanced (82/12/6), but now stratification is possible
- Applied class weighting in loss function to handle remaining imbalance

**Files affected:**
- Created: `data/cross_reactivity_labels.csv` (325 pairs)
- Deleted: Old incomplete `labels.csv`

---

### Problem 2: PyTorch 2.6 Compatibility Issue

**Error:**
```
_pickle.UnpicklingError: Weights only load failed...
GLOBAL torch_geometric.data.data.DataEdgeAttr was not an allowed global
```

**Root cause:** PyTorch 2.6 changed default `weights_only` parameter from `False` to `True` for security

**Solution:**
- Added `weights_only=False` to all `torch.load()` calls:
  - `train.py` (lines 199, 328)
  - `evaluate_vs_clinical_chart.py` (lines 48, 55)
  - `visualize.py` (line 20, 291)

**Code fix:**
```python
# Before (breaks in PyTorch 2.6)
data = torch.load('data/processed_data.pt')

# After (works in all PyTorch versions)
data = torch.load('data/processed_data.pt', weights_only=False)
```

---

### Problem 3: Duplicate Column Names in Data Processing

**Error:**
```
ValueError: Grouper for 'label' not 1-dimensional
```

**Root cause:** `organize_data.py` created duplicate 'label' columns when renaming
- Old 'label' column remained after creating 'label_new'
- `value_counts()` failed because it couldn't determine which 'label' column to use

**Solution:**
- Explicitly drop old column before renaming:
```python
# Before (caused duplicate columns)
labels_df = labels_df.rename(columns={'label': 'label_new'})

# After (clean rename)
labels_df = labels_df.drop(columns=['label'])
labels_df = labels_df.rename(columns={'label_new': 'label'})
```

**Files affected:** `organize_data.py`

---

### Problem 4: ReduceLROnPlateau Compatibility

**Error:**
```
TypeError: ReduceLROnPlateau.__init__() got an unexpected keyword argument 'verbose'
```

**Root cause:** User's PyTorch version doesn't support `verbose` parameter in scheduler

**Solution:**
- Removed `verbose=True` from `ReduceLROnPlateau` initialization in `train.py`
```python
# Before
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5, verbose=True
)

# After
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5
)
```

---

### Problem 5: File Organization and Duplicates

**Issue:** User complained about too many scattered files:
- Multiple markdown files (README, QUICKSTART, FEEDBACK_RESPONSES)
- Duplicate data files (drugs.csv, drugs_with_smiles.csv, drug_smiles.csv)
- Old src/ directory with outdated code
- "Why do I have so many damn markdowns?"

**Solution:**
- **Consolidated documentation:** Single comprehensive README.md
- **Deleted unnecessary files:**
  - QUICKSTART.md (content merged into README)
  - FEEDBACK_RESPONSES.md (instructor feedback addressed in code)
  - revised_proposal.md (proposal already submitted)
  - Old src/ directory (replaced with root-level scripts)
  - Duplicate data files (kept only drug_smiles.csv and cross_reactivity_labels.csv)
- **Organized file structure:**
  - All data in `data/` directory
  - All scripts at root level
  - Clear naming: data_preparation.py, train.py, evaluate_vs_clinical_chart.py

**Result:** Clean, minimal file structure with only essential files

---

## Quick Start

---

## Quick Start

### Prerequisites
```bash
# Install dependencies
pip install -r requirements.txt
```

### 3-Step Workflow

```bash
# Step 1: Prepare dataset (convert SMILES to graphs, create splits)
python data_preparation.py

# Step 2: Train model (~10-30 minutes on CPU, ~5 minutes on GPU)
python train.py

# Step 3: Compare with Northwestern chart (generates heatmap!)
python evaluate_vs_clinical_chart.py
```

**Done!** Check:
- `plots/clinical_vs_model_heatmap.png` - Side-by-side comparison
- `plots/clinical_confusion_matrix.png` - Precision/Recall analysis
- `results/clinical_evaluation.json` - Detailed metrics

---

## File Structure

```
📁 penicillin-cephalosporin-graph-analysis/
│
├── 📂 data/                           ← All data lives here
│   ├── drugs_with_smiles.csv          Your existing drug SMILES
│   ├── labels.csv                     Your existing cross-reactivity labels
│   ├── drug_smiles.csv               ✅ Generated by organize_data.py
│   ├── cross_reactivity_labels.csv   ✅ Generated by organize_data.py
│   └── processed_data.pt             ✅ Generated by data_preparation.py
│
├── 📂 models/                         ← Trained models
│   └── best_model.pt                 ✅ Generated by train.py
│
├── 📂 results/                        ← Training results
│   └── results.json                  ✅ Generated by train.py
│
├── 📂 plots/                          ← Visualizations
│   ├── training_history.png          ✅ Generated by visualize.py
│   └── confusion_matrix.png          ✅ Generated by visualize.py
│
├── 🐍 MAIN SCRIPTS (run in order)
│   ├── data_preparation.py           [1] Convert SMILES to graphs, create splits
│   ├── train.py                      [2] Train GNN model
│   └── evaluate_vs_clinical_chart.py [3] Compare with Northwestern chart
│
├── 🐍 UTILITY SCRIPTS (optional)
│   ├── check_ready.py                Verify data files are correct
│   └── visualize.py                  Additional training plots
│
├── 🧠 MODEL CODE
│   └── model.py                      GNN architecture (don't modify unless needed)
│
└── 📄 DOCUMENTATION
    ├── README.md                     👈 You are here
    └── requirements.txt              Python dependencies
```

---

## How It Works

### Overview: Molecular Structure → Cross-Reactivity Prediction

```
Drug 1 (Amoxicillin)     Drug 2 (Cefadroxil)
        ↓                         ↓
   Molecular Graph           Molecular Graph
   (atoms + bonds)           (atoms + bonds)
        ↓                         ↓
   GNN Encoder               GNN Encoder
        ↓                         ↓
   Embedding (256-dim)       Embedding (256-dim)
        ↓_________________________↓
                    ↓
              Combine Features
        [h1, h2, |h1-h2|, h1*h2]
                    ↓
             MLP Classifier
                    ↓
    [SUGGEST, CAUTION, AVOID]
                    ↓
              Prediction: AVOID ❌
```

---

### Step 1: Data Preparation (`data_preparation.py`)

**What it does:** Converts SMILES strings to molecular graphs and creates train/val/test splits.

**Input data (already prepared):**
- `data/drug_smiles.csv` - 28 drugs with SMILES strings
- `data/cross_reactivity_labels.csv` - 325 labeled pairs from Northwestern chart
- `data/ReferenceTableLabels.xlsx` - Original Northwestern chart (matrix format)

**Converts SMILES to graphs:**
- Each drug becomes a graph with atoms as nodes and bonds as edges
- Node features (6-dim): atomic number, degree, charge, hybridization, aromaticity, hydrogens
- Edge features (2-dim): bond type, is in ring

**Creates stratified splits:**
- Train: 210 pairs (70%)
- Validation: 45 pairs (15%)
- Test: 45 pairs (15%)

**Output:**
- `data/processed_data.pt` - Ready for training

**Run:**
```bash
python data_preparation.py
```

---

### Step 2: Molecular Graph Creation (`data_preparation.py`)

**What it does:** Converts SMILES strings → graph structures that GNNs can process.

#### Example: Amoxicillin

**SMILES:**
```
CC1(C(N2C(S1)C(C2=O)NC(=O)C(C3=CC=C(C=C3)O)NC(=O)C)C(=O)O)C
```

**Becomes a graph:**
```
Nodes (atoms): [C, C, C, N, C, S, O, ...] (23 atoms total)
Edges (bonds): [(0,1), (1,2), (2,3), ...] (48 bonds total)

Node features (per atom, 6-dim):
  - Atomic number (6 for C, 7 for N, 8 for O, ...)
  - Degree (how many bonds)
  - Formal charge
  - Hybridization (sp, sp2, sp3)
  - Is aromatic?
  - Number of hydrogens

Edge features (per bond, 2-dim):
  - Bond type (1.0=single, 2.0=double, 1.5=aromatic)
  - Is in ring?
```

#### Data Splitting

Creates three sets with **stratified sampling** (maintains class balance):

| Split | Size | Purpose |
|-------|------|---------|
| Train | 70% | Model learns from these |
| Validation | 15% | Monitor during training, tune hyperparameters, early stopping |
| Test | 15% | **Final evaluation only** - never seen during training |

**Run:**
```bash
python data_preparation.py
```

**Output:** `data/processed_data.pt` containing:
- Train/val/test splits
- Class weights (for handling imbalance)
- Drug molecular graphs

---

### Step 3: GNN Model Architecture (`model.py`)

#### Stage 1: Molecular Encoder

**Purpose:** Convert variable-size molecular graph → fixed-size embedding

**Architecture:**
```
Input: Molecular graph (variable # of atoms)
  ↓
Node Encoder (6-dim → 128-dim)
  ↓
GNN Layer 1 (message passing)
  ↓
GNN Layer 2 (atoms learn from neighbors)
  ↓
GNN Layer 3 (deeper patterns)
  ↓
GNN Layer 4 (even deeper)
  ↓
Global Pooling (aggregate all atoms)
  ↓
Output: Drug embedding (256-dim vector)
```

**GNN Types Supported:**
- **GIN (Graph Isomorphism Network)** - Default, best for molecular graphs
- **GAT (Graph Attention Network)** - Provides attention weights (interpretability)
- **GCN (Graph Convolutional Network)** - Baseline

**Message Passing Intuition:**

Think of atoms "talking" to their neighbors:
```
Iteration 1: Each atom knows its immediate neighbors
Iteration 2: Each atom knows neighbors-of-neighbors
Iteration 3: Information spreads 3 bonds away
Iteration 4: Information spreads 4 bonds away
```

After 4 iterations, each atom's representation captures its **local substructure** (e.g., R1/R2 side chains that drive cross-reactivity).

#### Stage 2: Pairwise Prediction

**Purpose:** Take two drug embeddings → predict cross-reactivity class

```
Drug 1 embedding: h1 [256-dim]
Drug 2 embedding: h2 [256-dim]
              ↓
Combine features:
  - h1                 (Drug 1 structure)
  - h2                 (Drug 2 structure)
  - |h1 - h2|          (How different?)
  - h1 * h2            (How similar?)
              ↓
Combined vector [1024-dim]
              ↓
MLP (3 layers)
  Linear(1024 → 512) + ReLU + Dropout
  Linear(512 → 256) + ReLU + Dropout
  Linear(256 → 3)
              ↓
Logits: [score_SUGGEST, score_CAUTION, score_AVOID]
              ↓
Softmax
              ↓
Probabilities: [0.1, 0.3, 0.6]
              ↓
Argmax → Prediction: AVOID
```

---

### Step 4: Training (`train.py`)

#### Loss Function: Cross-Entropy with Class Weights

**Problem:** Data is imbalanced (more SUGGEST than AVOID/CAUTION)

**Solution:** Weight losses by inverse class frequency
```python
class_weights = [0.5, 1.25, 0.83]
# SUGGEST has lower weight (common)
# CAUTION has higher weight (rare)
# AVOID has medium weight
```

This forces the model to pay attention to all classes, not just the majority.

#### Training Loop

```python
for epoch in range(100):
    # 1. Train on training set
    for batch in train_loader:
        logits = model(drug1_graph, drug2_graph)
        loss = CrossEntropyLoss(logits, true_labels, weights=class_weights)
        loss.backward()
        optimizer.step()

    # 2. Evaluate on validation set
    val_metrics = evaluate(model, val_loader)

    # 3. Save best model
    if val_metrics['f1'] > best_f1:
        save_checkpoint()

    # 4. Early stopping
    if no improvement for 15 epochs:
        break
```

**Run:**
```bash
python train.py
```

**Output:**
- `models/best_model.pt` - Best model weights
- `results/results.json` - All metrics (accuracy, F1, kappa, confusion matrix)

**Expected training output:**
```
Using device: cpu  (or cuda if GPU available)

Loading data...
Train: 210 pairs
Val: 45 pairs
Test: 45 pairs

Building model...
Total parameters: 840,455

Starting training...

Epoch 1/100
Training: 100%|████████| 7/7 [00:05<00:00]
Train loss: 1.0234
Evaluating: 100%|████████| 2/2 [00:01<00:00]
Val loss: 0.9876
Val accuracy: 0.5778
Val F1 (macro): 0.4125
Val Kappa: 0.1841
  SUGGEST: P=0.900, R=0.800, F1=0.847
  CAUTION: P=0.200, R=0.333, F1=0.250
  AVOID: P=0.000, R=0.000, F1=0.000
✓ Saved best model (F1=0.4125)

...

Epoch 25/100
Train loss: 0.2841
Val accuracy: 0.8444
Val F1 (macro): 0.7234
Val Kappa: 0.6123
✓ Saved best model (F1=0.7234)

Early stopping after 35-50 epochs

==================================================
Evaluating best model on test set...

Test Results:
Accuracy: ~0.80-0.85
F1 (macro): ~0.70-0.75
AUROC: ~0.80-0.85
Kappa: ~0.60-0.70  ← TARGET: Substantial Agreement

Per-class metrics (approximate):
SUGGEST: P=0.85-0.90, R=0.90-0.95, F1=0.87-0.92 (n~37)
CAUTION: P=0.40-0.60, R=0.40-0.60, F1=0.40-0.60 (n~5)
AVOID: P=0.60-0.80, R=0.50-0.70, F1=0.55-0.75 (n~3)

Confusion Matrix (approximate):
             Predicted
           SUG  CAU  AVO
True SUG    ~35   ~2   ~0
True CAU    ~2    ~2   ~1
True AVO    ~1    ~1   ~1
```

**Note:** Exact results will vary due to random initialization. CAUTION and AVOID classes are harder to predict due to limited samples (40 and 19 training examples respectively).

---

### Step 5: Evaluation Metrics

#### 1. Accuracy
```
Accuracy = Correct predictions / Total predictions
```
Example: 11/14 = 78.6%

**Limitation:** Doesn't account for class imbalance

---

#### 2. Precision, Recall, F1 (per class)

**For AVOID class (most clinically important):**

**Precision:** When model says AVOID, how often is it correct?
- High precision = Few false alarms
- Example: P=0.857 → 85.7% of AVOID predictions are correct

**Recall:** Of all true AVOID pairs, how many did we catch?
- High recall = Few missed dangerous pairs (critical for safety!)
- Example: R=0.857 → We caught 85.7% of dangerous pairs

**F1 Score:** Harmonic mean of precision and recall
- Balances both metrics
- F1 = 2 × (P × R) / (P + R)

---

#### 3. Confusion Matrix

Shows where the model makes mistakes:

```
             Predicted
           SUG  CAU  AVO
True SUG     4    1    0
True CAU     0    2    1
True AVO     0    1    5
```

**Reading this:**
- **Diagonal** = Correct predictions
- **Off-diagonal** = Mistakes
- **Row "AVOID", Column "SUGGEST"** = 0 → Good! No dangerous false negatives

---

#### 4. AUROC (Area Under ROC Curve)

Measures how well the model distinguishes between classes.

```
0.5 = Random guessing
0.7 = Fair
0.85 = Good ← Our target
0.95+ = Excellent
```

---

#### 5. Cohen's Kappa ⭐ **Most Important**

**Measures agreement with Northwestern chart, accounting for chance.**

```
Kappa     Interpretation
------    --------------
< 0.20    Poor agreement
0.21-0.40 Fair agreement
0.41-0.60 Moderate agreement
0.61-0.80 Substantial agreement ← OUR GOAL
0.81-1.00 Almost perfect agreement
```

**Why Kappa?**
- Standard metric for inter-rater reliability
- Directly answers: "Does our model agree with clinical guidelines?"
- More robust than accuracy for imbalanced data

**Example:** κ = 0.67 → **Substantial agreement** with Northwestern chart!

---

### Step 6: Visualization (`visualize.py`)

**Run:**
```bash
python visualize.py
```

**Generates:**

1. **Training History** (`plots/training_history.png`)
   - Loss curves (train vs validation)
   - Accuracy over epochs
   - F1 score over epochs
   - Cohen's Kappa over epochs

2. **Confusion Matrix** (`plots/confusion_matrix.png`)
   - Visual breakdown of predictions vs true labels
   - Identifies systematic errors

3. **Drug Embeddings** (optional, t-SNE visualization)
   - 2D projection of drug embeddings
   - Colored by drug class
   - Shows if similar drugs cluster together

4. **Predicted Heatmap** (optional)
   - Full N×N cross-reactivity prediction matrix
   - Compare side-by-side with Northwestern chart

---

## Installation

### System Requirements
- Python 3.8+
- ~2GB RAM minimum (4GB recommended)
- GPU optional (speeds up training 3-5x)

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Main packages:**
- `torch>=2.0.0` - PyTorch (deep learning)
- `torch-geometric>=2.3.0` - Graph neural networks
- `rdkit>=2023.3.1` - Molecular parsing (SMILES → graphs)
- `pandas`, `numpy`, `scikit-learn` - Data processing
- `matplotlib`, `seaborn` - Visualization

---

## Configuration

### Model Hyperparameters

Edit `train.py` config dictionary:

```python
config = {
    # Architecture
    'hidden_dim': 128,          # GNN hidden dimension (64, 128, 256)
    'embedding_dim': 256,       # Drug embedding size (128, 256, 512)
    'num_layers': 4,            # GNN depth (3, 4, 5)
    'gnn_type': 'GIN',          # GIN, GAT, or GCN
    'dropout': 0.2,             # Dropout rate (0.1, 0.2, 0.3)

    # Training
    'batch_size': 32,           # Batch size (16, 32, 64)
    'num_epochs': 100,          # Max epochs
    'learning_rate': 1e-3,      # Learning rate (1e-4, 5e-4, 1e-3)
    'early_stopping_patience': 15,  # Stop if no improvement
}
```

### Tips for Tuning

**If overfitting** (train accuracy >> val accuracy):
- Increase dropout: `0.3`
- Decrease num_layers: `3`
- Decrease hidden_dim: `64`

**If underfitting** (both train and val accuracy low):
- Increase hidden_dim: `256`
- Increase num_layers: `5`
- Train longer: `num_epochs: 200`

**If training is slow:**
- Decrease batch_size: `16`
- Use GPU if available
- Decrease num_layers: `3`

---

## Troubleshooting

### "No SMILES loaded"
**Cause:** Missing `data/drug_smiles.csv`

**Fix:**
```bash
python organize_data.py
```

---

### "cross_reactivity_labels.csv not found"
**Cause:** Missing labels file

**Fix:**
```bash
python organize_data.py
```

---

### Low accuracy (<70%)
**Expected behavior:** With 325 pairs (266 SUGGEST, 40 CAUTION, 19 AVOID), the model may struggle with minority classes.

**Expected performance:**
- SUGGEST class: 85-90% F1 (well represented)
- CAUTION class: 40-60% F1 (limited data: 40 samples)
- AVOID class: 50-70% F1 (limited data: 19 samples)
- Overall Kappa: 0.60-0.70 (substantial agreement target)

**If results are worse:**
- Increase model capacity: `hidden_dim: 256`, `num_layers: 5`
- Train longer: `num_epochs: 200`
- Try different GNN: `gnn_type: 'GAT'`

---

### Out of memory
**Cause:** GPU/CPU memory exhausted

**Fix:**
```python
# In train.py config
'batch_size': 16,  # Reduce from 32
```

---

### Model not learning (loss not decreasing)
**Causes:**
1. Learning rate too high/low
2. Labels incorrect
3. Not enough training epochs

**Fixes:**
- Try different learning rates: `5e-4`, `1e-4`
- Verify labels are correct (2=AVOID, 1=CAUTION, 0=SUGGEST)
- Train longer: `num_epochs: 200`

---

## Research Questions Answered

### 1. Can GNN models predict cross-reactivity as accurately as clinical charts?

**Metric:** Cohen's Kappa
- κ > 0.60 → Yes, substantial agreement
- κ < 0.40 → No, model disagrees with chart

### 2. Which molecular substructures drive cross-reactivity?

**Method:**
- Use GAT (Graph Attention Network): `gnn_type: 'GAT'`
- Analyze attention weights to see which atoms/bonds model focuses on
- Expected: R1/R2 side chains have highest attention

### 3. Do graph-based embeddings outperform traditional fingerprints?

**Comparison:**
- Train baseline using Tanimoto similarity (ECFP fingerprints)
- Compare F1 scores
- Expected: GNN should outperform by 5-10%

---

## Current Status

✅ **Data Collection Complete**
- 28 drugs with SMILES from PubChem
- 325 labeled pairs from Northwestern Medicine chart
- Excel reference table converted to training format

🚀 **Model Training** (you are here)
- GNN architecture: 4-layer GIN with 128-dim hidden, 256-dim embeddings
- ~840K parameters
- Training on 210 pairs, validating on 45 pairs, testing on 45 pairs
- Target: Cohen's Kappa > 0.60 (substantial agreement)

📊 **Next Steps**
- Evaluate predictions vs Northwestern chart
- Generate side-by-side heatmap comparison
- Analyze which drug pairs model gets wrong
- Write report & presentation (due Nov 10-14)

---

## Expected Deliverables

### Written Report (due Nov 10)
1. **Introduction** - Cross-reactivity problem, clinical importance
2. **Methods** - GNN architecture, data split, training procedure
3. **Results** - Test metrics (accuracy, F1, kappa), confusion matrix, figures
4. **Analysis** - Which pairs were misclassified? Why? Model insights
5. **Discussion** - Comparison with Northwestern chart, limitations, future work

### Presentation
- Problem overview (allergic cross-reactivity importance)
- Your approach (GNN on molecular graphs)
- Results (test metrics, visualizations)
- Key findings (agreement with clinical chart, R1/R2 importance)

### Code (optional)
- This entire directory
- Include trained model (`models/best_model.pt`)
- Include results (`results/results.json`)

---

## Citations

- **Northwestern Medicine β-Lactam Cross-Reactivity Chart** - Clinical reference
- **PubChem** - SMILES data source: https://pubchem.ncbi.nlm.nih.gov/
- **PyTorch Geometric** - GNN framework: https://pytorch-geometric.readthedocs.io/
- **RDKit** - Molecular parsing: https://www.rdkit.org/

---

## Contact

**Authors:** Megan Wang, Haley Kahn
**Course:** Network Analysis in Healthcare
**Project:** Beta-Lactam Cross-Reactivity Prediction

For questions about the code or methodology, refer to this README or the comments in the source files.

---

## License

Academic project for coursework.
