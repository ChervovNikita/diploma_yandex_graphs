0001 # Heterogeneous Graph Benchmark
0002 
0003 Revisiting, benchmarking, and refining Heterogeneous Graph Neural Networks.
0004 
0005 **2023.3.2 update**: We make benchmark data including test set pulic. You can download data as follows:
0006 
0007 * Node Classification: https://drive.google.com/drive/folders/10-pf2ADCjq_kpJKFHHLHxr_czNNCJ3aX?usp=sharing
0008 * Link Prediction: https://drive.google.com/drive/folders/1RNOPAQ_jrHlNWMhfNT4wqclnW_O1PqYL?usp=sharing
0009 
0010 Therefore, you can get your metric scores locally. Actually, when you run Simple-HGN in [NC-benchmark](NC/benchmark/methods/baseline/) and [LP-benchmark](LP/benchmark/methods/baseline/), data will be downloaded automatically.
0011 
0012 ***Warning***: As we have opened test data, you should try not to overfit or leak data during training. For example, the order of test data is not random permuted. If you use BatchNorm, you will get a biased norm value.
0013 
0014 ## Roadmap
0015 
