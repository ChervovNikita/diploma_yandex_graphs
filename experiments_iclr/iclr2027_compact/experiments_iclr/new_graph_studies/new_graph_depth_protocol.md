# Fixed Cora and Chameleon depth extension

The two graphs and every cell below were fixed before any outcomes from this
extension were viewed. The graph choice was exploratory after the earlier
Roman-Empire, WikiCS, Actor, and ogbn-arxiv results were known. No graph or
cell is to be excluded on the basis of its test outcome.

## Graphs and masks

- `cora`: PyTorch Geometric `Planetoid(name="Cora", split="public")`, the
  published public train/validation/test masks.
- `chameleon`: PyTorch Geometric `WikipediaNetwork(name="chameleon",
  geom_gcn_preprocess=True)`, published Geom-GCN mask 0 of the ten masks.
- PyG downloads only into this study's `data/` directory. All downloaded raw
  files, graph tensors, edge preprocessing, labels, and split indices are
  fingerprinted in each pretraining source manifest. The study uses the
  package's features without extra normalization.
- Directed loader edges are coalesced and symmetrized. No explicit self-loops
  are added. All node features and graph edges are visible during transductive
  full-batch message passing. Only training labels enter the loss. Test labels
  are scored once after validation checkpoint selection.

## Complete 24-cell design

For each graph: SAGE depths 2 and 5, tied GNNM versus fully untied propagation,
and optimization seeds 0, 1, 2. Both arms start from the same member function,
canonical tied weights, and post-construction CPU/CUDA RNG per paired seed.
The architecture uses the original four-member `models.TABMModel` input/output
BatchEnsemble projectors, width 128, residual SAGE blocks, LayerNorm, and
dropout 0.2. Every run trains for exactly 300 epochs with AdamW, learning rate
0.001, zero weight decay, and the mean of four member cross-entropies. Pooled
validation accuracy is checked after every epoch; ties use lower pooled
validation cross-entropy and then the earliest epoch. The selected checkpoint
is restored for test accuracy and cross-entropy.

The 300-epoch recipe and architecture are copied from the earlier external
depth study, not tuned per graph. The unit of replication is an optimization
seed on one fixed graph and published split, not a new dataset sample.

## Audit and interpretation

`verify_new_graph_depth.py` requires all six cells at each graph/depth, checks
the 300 validation rows and selected epoch, source/data fingerprints,
initial member logits and RNG hashes, all checkpoint hashes, and fresh CUDA
replay of validation/test logits and exact member and pooled decisions.
Results from one graph/split with three seeds are descriptive. Late selected
epochs limit conclusions about convergence. The well-known Geom-GCN Chameleon
split is retained as published; conclusions should not be generalized across
its other nine masks without rerunning them.
