0001: # Simple and Efficient Heterogeneous Graph Neural Network (SeHGNN)
0002: 
0003: The camera-ready paper for AAAI 23 can be found at: [http://arxiv.org/abs/2207.02547](http://arxiv.org/abs/2207.02547)
0004: 
0005: ## Requirements
0006: 
0007: #### 1. Neural network libraries for GNNs
0008: 
0009: * [pytorch](https://pytorch.org/get-started/locally/)
0010: * [pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
0011: 
0012: Please check your cuda version first and install the above libraries matching your cuda. If possible, we recommend to install the latest versions of these libraries.
0013: 
0014: * [dgl](https://www.dgl.ai/pages/start.html)
0015: 
0016: If you want to generate ComplEx embeddings for ogbn-mag, we recommend to install `dgl<1.0`.
0017: 
0018: #### 2. Other dependencies
0019: 
0020: Install other requirements:
0021: 
0022: ```setup
0023: pip install -r requirements.txt
0024: git clone https://github.com/Yangxc13/sparse_tools.git --depth=1
0025: cd sparse_tools
0026: python setup.py develop
0027: cd ..
0028: ```
0029: 
0030: ## Data preparation
0031: 
0032: * HGB datasets for node classification
0033: 
0034: These datasets include four medium-scale datasets. Please download them (`DBLP.zip`, `ACM.zip`, `IMDB.zip`, `Freebase.zip`) from [HGB repository](https://github.com/THUDM/HGB) and extract content under the folder `'./data/'`.
0035: 
0036: * Ogbn-mag
0037: 
0038: It is a large dataset from [OGB challenge](https://ogb.stanford.edu/docs/leader_nodeprop/#ogbn-mag). Thus dataset will be automatically downloaded for the first time running.
0039: 
0040: ---
0041: 
0042: If you encounter any issues, please feel free to reach out to me at yangxc96@gmail.com. The previous email address, yangxiaocheng@ict.ac.cn, is no longer in use.
