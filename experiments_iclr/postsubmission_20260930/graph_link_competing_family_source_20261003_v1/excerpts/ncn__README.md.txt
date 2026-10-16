0001: This repository contains the official code for the paper [Neural Common Neighbor with Completion for Link Prediction](https://arxiv.org/pdf/2302.00890.pdf).
0002: 
0003: **Environment**
0004: 
0005: Tested Combination:
0006: torch 1.13.0 + pyg 2.2.0 + ogb 1.3.5
0007: 
0008: ```
0009: conda env create -f env.yaml
0010: ```
0011: 
0012: **Prepare Datasets**
0013: 
0014: ```
0015: python ogbdataset.py
0016: ```
0017: 
0018: **Reproduce Results**
0019: 
0020: We implement the following models.
0021: 
0032: 
0033: To reproduce the results, please modify the following commands as shown in the table above.
0034: 
0035: Cora
0036: ```
0037: python NeighborOverlap.py   --xdp 0.7 --tdp 0.3 --pt 0.75 --gnnedp 0.0 --preedp 0.4 --predp 0.05 --gnndp 0.05  --probscale 4.3 --proboffset 2.8 --alpha 1.0  --gnnlr 0.0043 --prelr 0.0024  --batch_size 1152  --ln --lnnn --predictor $model --dataset Cora  --epochs 100 --runs 10 --model puregcn --hiddim 256 --mplayers 1  --testbs 8192  --maskinput  --jk  --use_xlin  --tailact 
0038: ```
0039: 
0040: Citeseer
0041: ```
0042: python NeighborOverlap.py   --xdp 0.4 --tdp 0.0 --pt 0.75 --gnnedp 0.0 --preedp 0.0 --predp 0.55 --gnndp 0.75  --probscale 6.5 --proboffset 4.4 --alpha 0.4  --gnnlr 0.0085 --prelr 0.0078  --batch_size 384  --ln --lnnn --predictor $model --dataset Citeseer  --epochs 100 --runs 10 --model puregcn --hiddim 256 --mplayers 1  --testbs 4096  --maskinput  --jk  --use_xlin  --tailact  --twolayerlin
0043: ```
0044: 
0045: Pubmed
0046: ```
0047: python NeighborOverlap.py   --xdp 0.3 --tdp 0.0 --pt 0.5 --gnnedp 0.0 --preedp 0.0 --predp 0.05 --gnndp 0.1  --probscale 5.3 --proboffset 0.5 --alpha 0.3  --gnnlr 0.0097 --prelr 0.002  --batch_size 2048  --ln --lnnn --predictor $model --dataset Pubmed  --epochs 100 --runs 10 --model puregcn --hiddim 256 --mplayers 1  --testbs 8192  --maskinput  --jk  --use_xlin  --tailact 
0048: ```
0049: 
0050: collab
0051: ```
0052: python NeighborOverlap.py   --xdp 0.25 --tdp 0.05 --pt 0.1 --gnnedp 0.25 --preedp 0.0 --predp 0.3 --gnndp 0.1  --probscale 2.5 --proboffset 6.0 --alpha 1.05  --gnnlr 0.0082 --prelr 0.0037  --batch_size 65536  --ln --lnnn --predictor $model --dataset collab  --epochs 100 --runs 10 --model gcn --hiddim 64 --mplayers 1  --testbs 131072  --maskinput --use_valedges_as_input   --res  --use_xlin  --tailact 
0053: ```
0054: 
0055: ppa
0056: ```
0057: python NeighborOverlap.py  --xdp 0.0 --tdp 0.0 --gnnedp 0.1 --preedp 0.0 --predp 0.1 --gnndp 0.0 --gnnlr 0.0013 --prelr 0.0013  --batch_size 16384  --ln --lnnn --predictor $model --dataset ppa   --epochs 25 --runs 10 --model gcn --hiddim 64 --mplayers 3 --maskinput  --tailact  --res  --testbs 65536 --proboffset 8.5 --probscale 4.0 --pt 0.1 --alpha 0.9 --splitsize 131072
0058: ```
0059: 
0060: The following datasets use separate commands for NCN and NCNC. To use other models, please modify NCN's command. Note that NCNC models in these datasets initialize parameters with trained NCN models to accelerate training. Please use our pre-trained model or run NCN first.
0061: 
0062: citation2
0063: ```
0064: python NeighborOverlapCitation2.py --xdp 0.0 --tdp 0.3 --gnnedp 0.0 --preedp 0.0 --predp 0.2 --gnndp 0.2 --gnnlr 0.0088 --prelr 0.0058 --batch_size 32768 --ln --lnnn --predictor cn1 --dataset citation2 --epochs 20 --runs 10 --model puregcn --hiddim 64 --mplayers 3 --res --testbs 65536 --use_xlin --tailact --proboffset 4.7 --probscale 7.0 --pt 0.3 --trndeg 128 --tstdeg 128 --save_gemb 
0065: 
0066: 
0067: python NeighborOverlapCitation2.py --xdp 0.0 --tdp 0.3 --gnnedp 0.0 --preedp 0.0 --predp 0.2 --gnndp 0.2 --gnnlr 0.0088 --prelr 0.001 --batch_size 24576 --ln --lnnn --predictor incn1cn1 --dataset citation2 --epochs 20 --runs 10 --model none --hiddim 64 --mplayers 0 --res --testbs 65536 --use_xlin --tailact --load gemb/citation2_puregcn_cn1.pt --proboffset -0.3 --probscale 1.4 --pt 0.25 --trndeg 96 --tstdeg 96 --load gemb/citation2_puregcn_cn1.pt 
0068: ```
0069: 
0070: 
0071: ddi
0072: ```
0073: python NeighborOverlap.py  --xdp 0.05 --tdp 0.0 --gnnedp 0.0 --preedp 0.0 --predp 0.6 --gnndp 0.4 --gnnlr 0.0021 --prelr 0.0018  --batch_size 24576  --ln --lnnn --predictor cn1 --dataset ddi  --epochs 100 --runs 10 --model puresum --hiddim 224 --mplayers 1  --testbs 131072   --use_xlin  --twolayerlin  --res  --maskinput --savemod
0074: 
0075: python NeighborOverlap.py --xdp 0.05 --tdp 0.0 --gnnedp 0.0 --preedp 0.0 --predp 0.6 --gnndp 0.4 --gnnlr 0.0000000 --prelr 0.0025 --batch_size 24576 --ln --lnnn --predictor incn1cn1 --dataset ddi --proboffset 3 --probscale 10 --pt 0.1 --alpha 0.5 --epochs 2 --runs 10 --model puresum --hiddim 224 --mplayers 1 --testbs 24576 --splitsize 262144 --use_xlin --twolayerlin --res --maskinput --loadmod
0076: ```
