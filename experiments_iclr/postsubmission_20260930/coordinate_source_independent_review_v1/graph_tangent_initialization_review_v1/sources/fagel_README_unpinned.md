# FAGEL: Fast and Accurate Graph Ensemble Learning

This repository contains the official implementation of **FAGEL**, an efficient graph ensemble learning framework for large-scale graph learning.

J. Shen, Y. Jin and X. Zhu, "Fast and Accurate Graph Ensemble Learning using Staged Training and Diversified Sampling," *European Conference on Machine Learning and Knowledge Discovery in Databases (ECML PKDD)*, 2026.
## Overview

Graph Neural Networks (GNNs) have become an effective approach for learning on large-scale graph data. However, conventional GNN ensemble methods require training multiple models independently, resulting in substantial computational and memory costs.

**FAGEL (Fast and Accurate Graph Ensemble Learning)** addresses this challenge by generating diverse GNN models within a **single training pipeline**, avoiding repetitive training from scratch.

FAGEL combines three key components:

1. **Staged Epoch Training**  
   Trains a GNN using a fixed sampling configuration until the validation performance reaches a plateau.

2. **Structured Sampling Diversification**  
   Continues training with varied neighborhood sampling configurations by changing hop depth and neighbor size, enabling the model to capture complementary multi-scale graph structures.

3. **Residual Weighted Ensemble Aggregation**  
   Selects the best-performing models during training and combines their predictions using an adaptive residual weighted ensemble strategy.

---
## Method

The overall FAGEL framework consists of two training stages:

![FAGEL Framework](FAGEL_framework.pdf)

Stage 1: The GNN is trained with mini-batch neighbor sampling under an initial sampling configuration until the validation performance reaches a plateau for t epochs. 

Stage 2: The sampler is switched to alternative configurations for c epochs to inject embedding diversity, and the best models are selected based on validation performance. 

The selected models are combined using a residual weighted ensemble to produce the final prediction.
## Requirements

#### 1. Neural network libraries for GNNs

* [pytorch](https://pytorch.org/get-started/locally/)
* [pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

Please check your cuda version first and install the above libraries matching your cuda. If possible, we recommend to install the latest versions of these libraries.

## Data preparation
* Homogeneous datasets for node classification
* Ogb datasets for node classification

These datasets include four medium-scale datasets. Please download them from pytorch geometric [pytorch-geometric-dataset](https://pytorch-geometric.readthedocs.io/en/2.5.2/modules/datasets.html#homogeneous-datasets).
You can download Ogb datasets [Ogbn](https://ogb.stanford.edu/docs/nodeprop/)

---
## Repository Structure

| File / Directory | Description |
|---|---|
| `train.py` | Main training script for FAGEL |
| `model.py` | GNN backbone implementations, including GCN and GraphSAGE |
| `dataset.py` | Dataset files and preprocessing utilities |
| `README.md` | Project documentation |

---

## Training

### 1. Train FAGEL

```bash
python train.py 
    --dataset reddit 
    --batch_size 0.1 
    --hop 2 
    --neighbors 10 
    --hop2 3 
    --neighbors2 5 
    --cycle 1
```
---

## Citation

If you find this work useful in your research, please cite:

```bibtex
@inproceedings{fagel2026ecml,
  title     = {FAGEL: Fast and Accurate Graph Ensemble Learning using Staged Training and Diversified Sampling},
  author    = {Jiajun Shen and Yufei Jin and Xingquan Zhu},
  booktitle = {European Conference on Machine Learning and Knowledge Discovery in Databases (ECML PKDD)},
  year      = {2026},
  publisher = {Springer},
  address   = {Naples, Italy}
}
```
If you encounter any issues, please feel free to reach out to me at jshen2024@fau.edu.
