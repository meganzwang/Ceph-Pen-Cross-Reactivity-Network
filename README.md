Structure-driven β-lactam cross-reactivity prediction

Steps:
1) Create/verify drug list
   - Edit data/drugs.csv (provided with template names/classes).
2) Fetch SMILES
   - python -m src.make_dataset
   - This creates data/drugs_with_smiles.csv
3) Enter reference labels from the chart
   - Edit data/labels.csv (pen,ceph,label with 1, 0.5, or 0)
4) Train the Siamese GNN
   - python -m src.train_siamese
   - Outputs models/siamese.pt
5) Generate heat map
   - python -m src.predict_heatmap
   - Saves plots/heatmap.png
6) Evaluate vs reference
   - python -m src.evaluate_vs_reference
   - Prints Spearman correlation and saves plots/overlay.png

Notes:
- If torch-geometric installation fails via requirements.txt, see:
  https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
- You can limit classes in data/drugs.csv to PEN and cephalosporins first.