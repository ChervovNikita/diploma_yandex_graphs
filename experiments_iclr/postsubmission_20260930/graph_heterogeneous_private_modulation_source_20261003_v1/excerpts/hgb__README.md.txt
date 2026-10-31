0001: # Heterogeneous Graph Benchmark
0002: 
0003: Revisiting, benchmarking, and refining Heterogeneous Graph Neural Networks.
0004: 
0005: **2023.3.2 update**: We make benchmark data including test set pulic. You can download data as follows:
0006: 
0007: * Node Classification: https://drive.google.com/drive/folders/10-pf2ADCjq_kpJKFHHLHxr_czNNCJ3aX?usp=sharing
0008: * Link Prediction: https://drive.google.com/drive/folders/1RNOPAQ_jrHlNWMhfNT4wqclnW_O1PqYL?usp=sharing
0009: 
0010: Therefore, you can get your metric scores locally. Actually, when you run Simple-HGN in [NC-benchmark](NC/benchmark/methods/baseline/) and [LP-benchmark](LP/benchmark/methods/baseline/), data will be downloaded automatically.
0011: 
0012: ***Warning***: As we have opened test data, you should try not to overfit or leak data during training. For example, the order of test data is not random permuted. If you use BatchNorm, you will get a biased norm value.
0013: 
0014: ## Roadmap
0015: 
0016: We organize our repo by task, and one sub-folder per task. Currently, we have four tasks, i.e., node classification (NC), link prediction (LP), knowledge-aware recommendation (Recom) and text classification (TC).
0017: 
0018: ### Revisiting
0019: 
0020: This part refers to Section 3 and Table 1 in our paper.
0021: 
0022: * [HAN](./NC/HAN)
0023: * [GTN](./NC/GTN)
0024: * [RSHN](./NC/RSHN)
0025: * [HetGNN](./NC/HetGNN)
0026: * [MAGNN-nc](./NC/MAGNN) and [MAGNN-lp](./LP/benchmark/methods/MAGNN_ini)
0027: 
0028: ### Benchmarking and Refining
0029: 
0030: This part refers to Section 4,5,6 in our paper.
0031: 
0032: * [Node classification benchmark](./NC/benchmark), [NC-baseline](./NC/benchmark/methods/baseline)
0033: * [Link prediction benchmark](./LP/benchmark), [LP-baseline](./LP/benchmark/methods/baseline)
0034: * [Knowledge-aware recommendation benchmark](./Recom), [Recom-baseline](./Recom/baseline)
0035: 
0036: **We only make half of test labels public to prevent data leakage issues.** The public half data is to help you debug locally. If you want to obtain test scores, you need to **rename all the xxx_full to xxx in data folder** and submit your prediction to our [website](https://www.biendata.xyz/hgb/).
0037: 
0038: For node classification and link prediction tasks, you can submit online. But for recommendation task, since the prediction files are too large to submit, you have to test offline by yourself.
0039: 
0040: **If you want to show your method on our official leaderboard on HGB website, you can make an issue on this repo. Once your code or paper is verified, your method will be displayed on the official leaderboard.**
0041: 
0042: ### FAQ
0043: 
0044: 1. **Where is the code for all HGNNs in paper for HGB?**
0045: 
0046: Follow this roadmap in this repo:
0047: 
0048: ```
0049: NC or LP
0053: ```
0054: 
0055: 2. **How to take part in HGB?**
0056: 
0057: See [Get Started](https://www.biendata.xyz/hgb/#/about).
0058: 
0059: 3. **Why can't I obtain test score locally?**
0060: 
0061: To prevent data leakage issues, we randomly replace the test set labels. Therefore, you can only get test scores after you submit your prediction to biendata competitions.
0062: 
0063: 4. **What is the format of training data and submission files?**
0064: 
0065: You can read the instructions in biendata competitions in "Data" and "Evaluation" panel. What should be noticed is that, your prediction files should be on the top level of the zipped file.
0066: 
0067: For example, you should submit a zip like this:
0068: 
0069: ```
0070: submit.zip
0074: ```
0075: 
0076: Instead of
0077: 
0078: ```
0079: submit.zip
0084: ```
0085: 
0086: It is recommended that zip a file using ```zip``` command rather than right click. Because subfolder may be automatically built for some operating systems when using right click.
0087: 
0088: ### More
0089: 
0090: **This repo is actively under development.** Therefore, there are some extra experiments in this repo beyond our paper, such as graph-based text classification. For more information, see our [website](https://www.biendata.xyz/hgb/). Welcome contribute new tasks, datasets, methods to HGB!
0091: 
0092: Moreover, we also have an implementation of Simple-HGN in [cogdl](https://github.com/THUDM/cogdl/tree/master/examples/simple_hgn).
0093: 
0094: 
0095: ## Citation
0096: 
0097: * **Title:** Are we really making much progress? Revisiting, benchmarking and refining the Heterogeneous Graph Neural Networks.
0098: * **Authors:** Qingsong Lv\*, Ming Ding\*, Qiang Liu, Yuxiang Chen, Wenzheng Feng, Siming He, Chang Zhou, Jianguo Jiang, Yuxiao Dong, Jie Tang.
0099: * **In proceedings:** KDD 2021.
