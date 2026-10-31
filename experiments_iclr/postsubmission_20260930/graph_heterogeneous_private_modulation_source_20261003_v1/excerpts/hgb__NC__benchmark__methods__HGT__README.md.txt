0001: # HGT code
0002: 
0003: Adapted from [HGT-DGL](https://github.com/acbull/HGT-DGL).
0004: 
0005: ## running environment
0006: 
0007: * Python 3.7
0008: * torch 1.7.0
0009: * dgl 0.5.2
0010: 
0011: ## running procedure
0012: 
0013: * download data from [tsinghua-cloud](https://cloud.tsinghua.edu.cn/d/8b9644cfa8344f26878c/)
0014: * cd to HGT/
0015: * unzip all zip files
0016: * run scripts
0017: * mkdir checkpoint
0018: 
0019: ```scripts
0020: sh run_acm.sh
0021: sh run_dblp.sh
0022: sh run_imdb.sh
0023: sh run_freebase.sh
0024: ```
