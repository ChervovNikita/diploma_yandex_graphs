0020 
0021 For **IMDB**:
0022 
0023 ```bash
0024 python main.py --epoch 200 --dataset IMDB --n-fp-layers 2 --n-task-layers 4 --num-hops 4 --num-label-hops 4 \
0025 	--label-feats --hidden 512 --embed-size 512 --dropout 0.5 --input-drop 0. --amp --seeds 1 2 3 4 5
0026 ```
0027 
0028 For **Freebase**:
