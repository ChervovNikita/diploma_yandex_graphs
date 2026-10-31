0001: # GCN and GAT for benchmark
0002: 
0003: (To be tuned)
0004: 
0005: ```
0006: python run.py --dataset DBLP --model-type gat
0007: python run.py --dataset DBLP --model-type gcn --weight-decay 1e-6 --lr 1e-3
0008: 
0009: python run.py --dataset ACM --model-type gat --feats-type 2
0010: python run.py --dataset ACM --model-type gcn --weight-decay 1e-6 --lr 1e-3 --feats-type=0
0011: 
0012: python run.py --dataset Freebase --model-type gat
0013: python run.py --dataset Freebase --model-type gcn
0014: 
0015: python run_multi.py --dataset IMDB --model-type gat --feats-type 0 --num-layers 4
0016: python run_multi.py --dataset IMDB --model-type gcn --feats-type 0 --num-layers 3
0017: ```
0018: 
0019: ## running environment
0020: 
0021: * torch 1.6.0 cuda 10.1
0022: * dgl 0.4.3 cuda 10.1
0023: * networkx 2.3
0024: * scikit-learn 0.23.2
0025: * scipy 1.5.2
