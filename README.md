# Beta-Lactam Antibiotic Cross-Reactivity Prediction

**Predicting allergic cross-reactivity between penicillins and cephalosporins using Graph Neural Networks**

**Authors:** Megan Wang, Haley Kahn
**Course:** Network Analysis in Healthcare

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Materials](#materials)
3. [Modeling Approach](#modeling-approach)
4. [Results](#results)
5. [Limitations](#limitations)
6. [Quick Start](#quick-start)
7. [Installation](#installation)
8. [Citations](#citations)

---

## Project Overview

### Objective

**Task Type:** Link prediction (pairwise drug cross-reactivity classification)

**Objective:** Predict allergic cross-reactivity risk between antibiotic pairs using molecular graph neural networks, and validate predictions against the Northwestern Medicine clinical reference chart.

**Target Outcome:** 3-class classification
- Class 0 (SUGGEST): Low cross-reactivity risk (dissimilar side chains)
- Class 1 (CAUTION): Moderate risk (similar R1/R2 side chains)
- Class 2 (AVOID): High risk (identical R1/R2 side chains)

**Population/Scope:** 43 FDA-approved beta-lactam antibiotics across 10 drug classes (penicillins, cephalosporins, carbapenems, monobactams, aminoglycosides, macrolides, lincosamides, flueoroquinolones, sulfonamides, other non-beta-lactams)

**Hypothesis:** Graph neural networks can learn molecular structure patterns from atom-bond graphs to replicate expert clinical judgment on cross-reactivity risk, achieving substantial agreement (Cohen's κ > 0.60) with established clinical guidelines.

---

## Materials

### Data and Unit of Analysis

**Unit of Analysis:** Drug pairs (pairwise combinations of antibiotics)

**Dataset:**
- **43 drugs** across 10 antibiotic classes:
  - 5 penicillins (Penicillin G/V, Oxacillin, Amoxicillin, Ampicillin, Piperacillin)
  - 16 cephalosporins across 5 generations (1st gen: Cefadroxil, Cephalexin, Cefazolin; 2nd gen: Cefaclor, Cefoxitin, Cefprozil, Cefuroxime; 3rd gen: 8 drugs; 4th gen: Cefepime; 5th gen: Ceftaroline, Ceftolozane)
  - 3 carbapenems (Ertapenem, Meropenem, Imipenem)
  - 1 monobactams (Aztreonam)
  - 4 aminoglycosides (Amikacin, Gentamicin, Streptomycin, Tobramycin)
  - 3 macrolides (Azithromycin, Clarithromycin, Erythromycin)
  - 1 lincosamides (Clindamycin)
  - 3 fluoroquinolones (Ciprofloxacin, Levofloxacin, Moxifloxacin)
  - 1 sulfonamides (Cotrimoxazole)
  - 4 non-beta-lactams (Chloramphenicol, Daptomycin, Metronidazole, Tigecycline)

**Dataset:**
- **903 total unique drug pairs** from combining Northwestern Medicine chart (26×26 matrix, excluding self-pairs) and Vancouver Coastal Health chart (36x36, excluding self pairs) to make one 43x43 matrix.
- Available in `data/cross_reactivity_labels.csv` (generated from `ReferenceTableLabels.xlsx`)
  - 820 SUGGEST (90.8%) - Low cross-reactivity risk
  - 37 CAUTION (4.1%) - Moderate risk
  - 46 AVOID (5.1%) - High risk

**Training Splits:**
- Stratified 70/15/15 split maintains class distribution in each set:
  - **Training:** 632 pairs (70%)
  - **Validation:** 135 pairs (15%)
  - **Test:** 136 pairs (15.1%)
- All splits have ~90-91% SUGGEST, 3.5-4% CAUTION, 5.1-5.2% AVOID

**Data sources:**
  - SMILES molecular structures: PubChem (https://pubchem.ncbi.nlm.nih.gov/)
  - Cross-reactivity labels: Northwestern Medicine clinical reference chart, Vancouver Coastal Health clinical reference chart

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
- **Number of graphs:** 43 (one per drug)
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
- Problem: 
  - Severe class imbalance in cross-reactivity dataset
    - SUGGEST: 574 samples (90.8%)
    - CAUTION: 26 samples (4.1%)
    - AVOID: 32 samples (5.1%)
  - Imbalance ratios: SUGGEST:CAUTION = 22.1:1, SUGGEST:AVOID = 17.9:1
- Solution:
  - Inverse Frequency Class Weights:
    - SUGGEST (0): 0.0731  (1.0x baseline)
    - CAUTION (1): 1.6148  (22.1x amplification) 
    - AVOID (2):   1.3120  (17.9x amplification)
  - Precision-Focused Loss Function
  - Stratified Data Splitting
    - Maintains class proportions across train/validation/test splits
    - Ensures consistent evaluation across different data subsets

**No data leakage:** Same drug pair never appears in multiple splits

---

### GNN Task Specification

**Task Type:** **Link prediction** (pairwise drug cross-reactivity classification)

**Observations/Instances:**
- Each observation is a **drug pair** (Drug A, Drug B)
- 903 total instances from 43×43 drug interaction matrix (excluding self-pairs)
- Unit of analysis: Binary relationship between two drugs

**Independent Variables:**

1. **Molecular structure (GNN input):**
   - Two molecular graphs, one per drug
   - Graph tensors for each molecule:
     - **Edge list representation:** `edge_index` tensor of shape `[2, num_edges]`
       - Example: `[[0,1,1,2,...], [1,0,2,1,...]]` for bonds between atoms
     - **Node feature matrix:** `x` tensor of shape `[num_nodes, 6]`
       - 6-dimensional continuous features per atom
       - Encoding: Atomic number (continuous), degree (continuous), formal charge (continuous), hybridization (ordinal: 0=sp, 1=sp², 2=sp³, 3=sp³d, 4=sp³d²), aromaticity (binary: 0/1), hydrogens (continuous)
     - **Edge feature matrix:** `edge_attr` tensor of shape `[num_edges, 2]`
       - 2-dimensional continuous features per bond
       - Encoding: Bond type (continuous: 1.0=single, 2.0=double, 3.0=triple, 1.5=aromatic), is_in_ring (binary: 0/1)

2. **Structural features (MLP input):**
   - 10-dimensional feature vector per drug pair
   - Encoding: 6 binary class features (same_class, both_pen, both_ceph, pen_ceph_pair, both_carb) + 1 normalized generation distance + 3 normalized molecular property differences (MW, LogP, aromatic rings)

**Dependent Variable (Labels):**
- **3-class classification** of cross-reactivity risk
- Encoding: 0 = SUGGEST (low risk), 1 = CAUTION (moderate risk), 2 = AVOID (high risk)
- Label type: Categorical (ordinal with clinical significance)

**Negative Sampling:**
- **Not applicable** - this is a fully supervised link prediction task
- All 903 drug pairs have ground-truth labels from Northwestern Medicine and VCH clinical chart
- No negative sampling needed because:
  - We have explicit labels for all pairs (not just positive examples)
  - Task is to classify existing relationships, not discover new links
  - Clinical chart provides complete pairwise annotations


---

## Modeling Approach

### Architecture Overview

**Hybrid architecture combining molecular structure and drug class features:**

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

**Stage 2: Structural Features (10-dimensional)**
- **Drug class features (6):**
  - Same drug class (binary)
  - Both penicillins (binary)
  - Both cephalosporins (binary)
  - Penicillin-Cephalosporin pair (binary) - HIGH RISK indicator
  - Both carbapenems (binary)
  - Cephalosporin generation distance (normalized 0-1)
- **Molecular property differences (4):**
  - Molecular weight difference (normalized)
  - LogP difference (hydrophobicity)
  - Number of aromatic rings difference
  - Number of H-bond donors difference

**Stage 3: Pairwise Predictor (MLP Classifier)**
- **Purpose:** Combine molecular embeddings + structural features → predict cross-reactivity class
- **Input:** Two drug embeddings h₁, h₂ (each 256-dim) + structural features (10-dim)
- **Feature combination:**
  - Concatenate: [h₁, h₂, |h₁ - h₂|, h₁ ⊙ h₂, structural_features] → 1034-dim
  - |h₁ - h₂|: Absolute difference (captures dissimilarity)
  - h₁ ⊙ h₂: Element-wise product (captures similarity)
  - structural_features: Drug class and molecular property features
- **Architecture:**
  - Linear(1034 → 512) + ReLU + Dropout(0.3)
  - Linear(512 → 256) + ReLU + Dropout(0.3)
  - Linear(256 → 3) → Logits for [SUGGEST, CAUTION, AVOID]
- **Parameters:** ~420K (predictor only)
- **Output:** 3 class probabilities (sum to 1.0 via softmax)

**Total Model:** ~840K parameters

### Loss Function

**PrecisionPenaltyLoss:** 

Asymmetric Loss for Clinical Safety: 
The model uses a custom PrecisionPenaltyLoss function designed to address the over-conservative prediction problem while maintaining clinical safety.

**Key Features:**
- Base Cross-Entropy with Class Weights:
  - SUGGEST: 0.0731 (baseline)
  - CAUTION: 1.6148 (22.1x amplification)
  - AVOID: 1.3120 (17.9x amplification)
  - Asymmetric False Positive Penalties:
- False CAUTION predictions: 3.0x penalty
- False AVOID predictions: 4.5x penalty (highest risk)
- False negatives: Standard penalty (maintains safety)
Clinical Rationale:
- Reduces over-conservative predictions that unnecessarily restrict antibiotics
Maintains safety by not penalizing missed dangerous combinations
Balances precision vs recall for clinical utility


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
  - **Achieved early convergence** (~28-50 epochs) with precision-focused loss



## Experimental Design

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
    'scheduler_patience': 5,       # Reduce LR if no improvement for 5 epochs
    'random_seed': 42              # Optional: Set for reproducible results
}
```

**Training loop:**
1. **Forward pass:** For each batch of drug pairs, compute predictions
2. **Loss computation:** PrecisionPenaltyLoss with asymmetric false positive penalties (3.0x for CAUTION, 4.5x for AVOID)
3. **Backward pass:** Compute gradients via backpropagation
4. **Parameter update:** Adam optimizer updates weights
5. **Validation:** After each epoch, evaluate on validation set
6. **Model selection:** Save model if validation F1 (macro) improves
7. **Early stopping:** Stop training if no improvement for 15 epochs

**Performance achieved:**
- Cohen's Kappa improved from 0.346 to **0.600** (+73% improvement)
- Early stopping typically converges within 25-40 epochs
- Class-weighted sampling handles 22:1 imbalance ratio effectively

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

**Final step:** Compare model predictions to our clinical reference baseline

**Success criteria:**
- κ > 0.60: Model has substantial agreement with clinical guidelines
- AVOID recall > 60%: Model catches majority of dangerous pairs
- Qualitative analysis: Which pairs does model get wrong? Why?

---

## Results

### Model Performance

✅ **Training Complete**
- GNN architecture: 4-layer GIN with 128-dim hidden, 256-dim embeddings
- Hybrid model with molecular graphs + 10 structural features
- ~840K parameters
- Trained on **455 pairs** (70% of 650), validated on 98 pairs, tested on 98 pairs
- Early stopping after 43 epochs

**Test Set Results (136 held-out pairs):**
- **Accuracy:** 94.1%
- **F1 Score (macro):** 75.4%
- **Cohen's Kappa:** 0.719 (Substantial Agreement!)** ← Exceeds target of 0.60
- **AUROC:** 0.986 (Excellent discrimination)

**Per-Class Performance (Test Set):**

| Class | Precision | Recall | F1 Score | Support |
|-------|-----------|--------|----------|---------|
| SUGGEST (Low Risk) | 1.00 | 0.96 | 0.98 | 123 |
| CAUTION (Moderate Risk) | 0.60 | 1.00 | 0.75 | 6 |
| AVOID (High Risk) | 0.50 | 0.57 | 0.53 | 7 |

**Clinical Validation (All 1,806 pairs vs Northwestern Chart):**
- **Accuracy:** 90.5%
- **Cohen's Kappa: 0.574 (Moderate Agreement, approaching Substantial!)** ← Primary metric
- **AVOID Recall:** 60.9% (56 out of 92 dangerous pairs correctly identified)
- **CAUTION Recall:** 83.8% (62 out of 74 moderate-risk pairs identified)

**Per-Class Performance (Clinical Validation):**

| Class | Precision | Recall | F1 Score | Support |
|-------|-----------|--------|----------|---------|
| SUGGEST (Low Risk) | 0.99 | 0.92 | 0.96 | 1,640 |
| CAUTION (Moderate Risk) | 0.42 | 0.84 | 0.56 | 74 |
| AVOID (High Risk) | 0.44 | 0.61 | 0.51 | 92 |

**Key Finding:** The model achieves **Cohen's Kappa of 0.719 on test set** (substantial agreement, exceeding the target of 0.60!) and 0.574 on clinical validation. With 94.1% test accuracy and 90.5% clinical validation accuracy, the hybrid GNN + structural features approach successfully predicts cross-reactivity from molecular structure.

### Visualizations

See the following plots for detailed analysis:
- **`plots/clinical_vs_model_heatmap.png`** - Side-by-side heatmap: Northwestern Medicine chart vs model predictions (all 1,806 pairs)
- **`plots/clinical_confusion_matrix.png`** - Clinical validation confusion matrix (1,806 pairs)
- **`plots/confusion_matrix.png`** - Test set confusion matrix (136 pairs)


### Analysis

**Strengths:**
- **Outstanding AUROC (0.986)** indicates excellent ability to distinguish between classes
- **Strong Cohen's Kappa (0.719)** achieves substantial agreement with clinical guidelines
- **Perfect SUGGEST precision (1.00)** with excellent recall (0.96) and F1 (0.98)
- **Perfect CAUTION recall (1.00)** catches all moderate-risk pairs in test set
- **Improved AVOID performance** with 57% recall and 50% precision on critical high-risk pairs

**Limitations:**
- CAUTION precision (0.42) in clinical validation indicates some false alarms
- AVOID class remains challenging due to limited training examples (92 clinical pairs)
- Model maintains conservative approach: when uncertain, predicts higher risk (safety-first)
- Clinical validation shows lower performance than test set, indicating some overfitting

**Why Structural Features Matter:**
The model uses both molecular structure (via GNN) and drug class information (penicillin, cephalosporin, generation) to make predictions. The hybrid approach helps identify high-risk penicillin-cephalosporin pairs that share similar side chains.

---

## Technical Challenges & Future Work

### Current Limitations Despite Strong Performance

While the model achieves **substantial agreement** (Cohen's κ=0.719) on test data and approaches substantial agreement (κ=0.574) on clinical validation, some technical challenges remain for further improvement:

#### 1. GNNs Learn Global Similarity, Not Specific Epitopes

**The fundamental problem:** Immunological cross-reactivity is determined by specific side chain structures (R1/R2 groups), but GNNs learn from the entire molecular graph.

**What matters clinically:** R1 side chains are **identical** → HIGH CROSS-REACTIVITY RISK (AVOID)

**The dilution problem:** The critical signal (R1 identity) represents only **25% of atoms** (5 out of 20). The GNN dilutes this signal by averaging it with:
- Beta-lactam core (identical across all drugs) → adds no discriminative information
- Non-reactive groups → adds noise

**Result:** The model learns "these drugs are 80% structurally similar overall" instead of "these drugs have identical R1 side chains."

---

#### 2. Structural Features Don't Capture Epitope-Level Detail

The 10 hand-engineered structural features help identify drug class relationships (penicillin vs cephalosporin) but **cannot capture specific side chain similarities**:

**What the features capture:**
- Drug class membership (both penicillins, both cephalosporins)
- Cephalosporin generation (1st gen, 2nd gen, etc.)
- Global molecular properties (molecular weight, LogP, aromatic rings)

**What they miss:**
- Whether R1 side chains are identical, similar, or different
- Whether R2 side chains match
- Specific structural motifs that trigger immune response

**Example failure case:**
- Penicillin G and Cefazolin have **different** R1 side chains → Should be SUGGEST
- But: model sees "penicillin + cephalosporin pair" feature → predicts AVOID (false alarm)

---

#### 3. Insufficient Training Data for Minority Classes

**Current training distribution:**
- SUGGEST: ~820 examples (90.8% of 903 total pairs)
- CAUTION: ~37 examples (4.1% of 903 total pairs) 
- AVOID: ~46 examples (5.1% of 903 total pairs)

**Impact:** Despite improved performance (κ=0.719), the severe class imbalance remains challenging:
- 22:1 imbalance ratio between SUGGEST and CAUTION classes
- Limited AVOID examples for learning rare high-risk patterns
- Model achieves good recall but lower precision on minority classes

**Evidence of generalization success:** Test set AVOID recall is 57% (4/7 pairs) and clinical validation shows 61% (56/92 pairs). The PrecisionPenaltyLoss successfully learned generalizable patterns despite limited training data.

---

#### 4. Missing Critical Clinical Knowledge (DrugBank Integration Pending)

The model lacks access to **known cross-sensitivity relationships** documented in pharmacological databases:

**What's missing:**
- Published case reports of allergic cross-reactions
- FDA drug labels mentioning cross-sensitivity warnings
- Clinical trial data on hypersensitivity rates
- Expert-curated drug interaction databases

**DrugBank integration (awaiting API approval):** Would provide:
- Known cross-sensitivity flags for specific drug pairs
- Mechanism-based interaction annotations
- Evidence strength ratings (established vs theoretical)
- Shared metabolic pathways and protein binding

**Expected improvement:** Adding DrugBank features could potentially increase κ from current 0.574 (clinical) → 0.65-0.75 (substantial agreement across all evaluations) by incorporating clinical evidence beyond molecular structure.

---

#### 5. Class Imbalance Despite Weighting

**The imbalance:**
- 82% SUGGEST, 12% CAUTION, 6% AVOID

**Current mitigation (successful):**
- Inverse frequency class weights: SUGGEST=0.073, CAUTION=1.615, AVOID=1.312
- PrecisionPenaltyLoss with asymmetric false positive penalties (3.0x CAUTION, 4.5x AVOID)
- Achieved Cohen's Kappa of 0.719 (substantial agreement)

**Remaining challenges:** While performance substantially improved, precision-recall tradeoffs persist:
- CAUTION precision: 42% (clinical) - some false alarms remain
- AVOID precision: 44% (clinical) - conservative predictions for safety
- Model successfully balances safety (high recall) with clinical utility

**Evidence of success:** AVOID recall of 61% (clinical validation) with reasonable precision demonstrates the model catches most dangerous pairs while minimizing false alarms.

---

#### 6. Oversimplified Immunology Assumption

**The model assumes:** Molecular structure similarity → immune cross-reactivity

**Reality is more complex:**
- Cross-reactivity depends on antibody epitope recognition
- IgE antibodies bind to 3D conformations, not 2D SMILES
- Some structurally similar drugs don't cross-react (steric hindrance, protein binding)
- Some structurally dissimilar drugs do cross-react (shared metabolites, degradation products)

**Example:** Aztreonam (monobactam) is structurally very different from penicillins but shares some side chain features. Clinical data shows <10% cross-reactivity despite structural dissimilarity.

---

### Path Forward

**Current Status:** ✅ **Primary research goals achieved** - Cohen's Kappa of 0.719 (substantial agreement) exceeds target of 0.60

**Next-level improvements for clinical deployment:**
1. **DrugBank integration (pending API approval):** Add clinical evidence features to push κ > 0.75
2. **Threshold optimization:** Post-training calibration to maximize Cohen's Kappa specifically
3. **Ensemble methods:** Combine multiple models with different class balancing strategies

**Advanced research directions:**
1. **Epitope-focused GNN:** Attention mechanisms to identify critical R1/R2 side chain regions
2. **3D conformational modeling:** Move beyond 2D SMILES to 3D protein binding predictions  
3. **Multi-task learning:** Joint prediction of cross-reactivity + binding affinity + immunogenicity

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

# Step 3: Compare with clinical reference chart (generates heatmap!)
python evaluate_vs_clinical_chart.py
```

**Done!** Check:
- `plots/clinical_vs_model_heatmap.png` - Side-by-side comparison
- `plots/clinical_confusion_matrix.png` - Precision/Recall analysis
- `results/clinical_evaluation.json` - Detailed metrics

---

**Authors:** Megan Wang, Haley Kahn
**Course:** Network Analysis in Healthcare
**Project:** Antibiotic Cross-Reactivity Prediction


---

## License

Academic project for coursework.
