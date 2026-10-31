0001: # SeHGNN on the four medium-scale datasets
0002: 
0003: ## Training
0004: 
0005: To reproduce the results of SeHGNN on four medium-scale datasets, please run following commands.
0006: 
0007: For **DBLP**:
0008: 
0009: ```bash
0010: python main.py --epoch 200 --dataset DBLP --n-fp-layers 2 --n-task-layers 3 --num-hops 2 --num-label-hops 4 \
0011: 	--label-feats --residual --hidden 512 --embed-size 512 --dropout 0.5 --input-drop 0.5 --amp --seeds 1 2 3 4 5
0012: ```
0013: 
0014: For **ACM**:
0015: 
0016: ```bash
0017: python main.py --epoch 200 --dataset ACM --n-fp-layers 2 --n-task-layers 1 --num-hops 4 --num-label-hops 4 \
0018: 	--label-feats --hidden 512 --embed-size 512 --dropout 0.5 --input-drop 0.5 --amp --seeds 1 2 3 4 5
0019: ```
0020: 
0021: For **IMDB**:
0022: 
0023: ```bash
0024: python main.py --epoch 200 --dataset IMDB --n-fp-layers 2 --n-task-layers 4 --num-hops 4 --num-label-hops 4 \
0025: 	--label-feats --hidden 512 --embed-size 512 --dropout 0.5 --input-drop 0. --amp --seeds 1 2 3 4 5
0026: ```
0027: 
0028: For **Freebase**:
0029: 
0030: ```bash
0031: python main.py --epoch 200 --dataset Freebase --n-fp-layers 2 --n-task-layers 4 --num-hops 2 --num-label-hops 3 \
0032: 	--label-feats --residual --hidden 512 --embed-size 512 --dropout 0.5 --input-drop 0.5 \
0033: 	--lr 3e-5 --weight-decay 3e-5 --batch-size 256 --amp --patience 30 --seeds 1 2 3 4 5
0034: ```
0035: 
0036: ## Performance
0037: 
0038: ![image-sehgnn-middle](./image-sehgnn_hgb.png)
