# Prediction of Protein Structure from Sequence

Comparing Transformer, 1D CNN, and BiLSTM architectures for predicting coarse-grained Cα protein structure from amino acid sequence alone.

Course project for TIF360/FYM360 Advanced Machine Learning at Chalmers University of Technology.

## Overview

Each model predicts pairwise Cα distance matrices from sequence, which are reconstructed into 3D coordinates via multidimensional scaling (MDS) and evaluated using RMSD and DME after Kabsch alignment.

## Results

| Model | RMSD mean | RMSD median | DME mean |
|---|---|---|---|
| Transformer | 10.72 Å | 11.21 Å | 3.29 Å |
| 1D CNN | 9.89 Å | 11.09 Å | 2.81 Å |
| BiLSTM | 9.34 Å | 11.21 Å | 2.60 Å |

## Usage

```
pip install -r requirements.txt
python main.py --synthetic          # test with synthetic data
python main.py                      # train on real PDB data
python main.py --arch transformer   # train single architecture
python regenerate_report.py         # regenerate plots from checkpoints
```

## Data

800 single-chain proteins from the [Protein Data Bank](https://www.rcsb.org) (X-ray, ≤2.5 Å resolution, 30-200 residues).
