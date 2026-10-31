# HGEN
Code of the paper: HGEN: Heterogeneous Graph Ensemble Networks.
The camera-ready paper for IJCAI 25 can be found at: 

## Requirements

#### 1. Neural network libraries for GNNs

* [pytorch](https://pytorch.org/get-started/locally/)
* [pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

Please check your cuda version first and install the above libraries matching your cuda. If possible, we recommend to install the latest versions of these libraries.

## Data preparation

* HGB datasets for node classification

These datasets include four medium-scale datasets. Please download them from pytorch geometric [pytorch-geometric-dataset](https://pytorch-geometric.readthedocs.io/en/2.5.3/modules/datasets.html#heterogeneous-datasets).

* Ogbn-mag

It is a large dataset from [OGB Challenge](https://ogb.stanford.edu/docs/leader_nodeprop/#ogbn-mag).

---

## Citation

If you find this work useful in your research, please cite:

```bibtex
@inproceedings{hgen2025ijcai,
  title     = {HGEN: Heterogeneous Graph Ensemble Networks},
  author    = {Shen, Jiajun and Jin, Yufei and Feng, Kaibu and He, Yi and Zhu, Xingquan},
  booktitle = {Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence, {IJCAI-25}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {James Kwok},
  pages     = {6156--6163},
  year      = {2025},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2025/685},
  url       = {https://doi.org/10.24963/ijcai.2025/685},
}
```
If you encounter any issues, please feel free to reach out to me at jshen2024@fau.edu.
