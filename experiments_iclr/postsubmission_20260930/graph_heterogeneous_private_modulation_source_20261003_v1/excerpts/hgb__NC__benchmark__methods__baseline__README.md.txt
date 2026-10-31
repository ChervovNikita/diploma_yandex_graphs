0001: # Simple-HGN for HGB
0002: 
0003: For message passing with relation attention version:
0004: 
0005: ```
0006: python run_new.py --dataset DBLP
0007: python run_new.py --dataset ACM --feats-type 2
0008: python run_multi.py --dataset IMDB --feats-type 0
0009: python run_new.py --dataset Freebase
0010: ```
0011: 
0012: ## running environment
0013: 
0014: * torch 1.6.0 cuda 10.1
0015: * dgl 0.4.3 cuda 10.1
0016: * networkx 2.3
0017: * scikit-learn 0.23.2
0018: * scipy 1.5.2
