0001: # benchmark
0002: 
0003: benchmark data loader and evaluation scripts
0004: 
0005: ## data
0006: 
0007: Warning: As we have opened test data, you should try not to overfit or leak data during training.
0008: 
0009: ## data format
0010: 
0011: * All ids begin from 0.
0012: * Each node type takes a continuous range of node_id.
0013: * node_id and node_type id are with same order. I.e. nodes with node_type 0 take the first range of node_ids, nodes with node_type 1 take the second range, and so on.
0014: * One-hot node features can be omited.
