# Fourth-graph sharing-position diagnostic: ogbn-arxiv

This exploratory diagnostic was chosen after the Roman-Empire, WikiCS, and Actor sharing-position outcomes were known. `ogbn-arxiv` is a different, official time-split node classification benchmark; no result from this new sharing-position study was viewed before the companion gradient-alignment protocol and predictions were frozen. The dataset choice is not random or preregistered.

## Fixed setup

- Source: official `PygNodePropPredDataset(name="ogbn-arxiv")`, official time train/validation/test indices. The public OGB ZIP is retained in `data/ogb/ogbn_arxiv_official.zip`, and the loader checks every graph, feature, label, and split fingerprint against `ogbn_arxiv_reference_dataset_manifest.json`.
- Graph: `torch_geometric.utils.to_undirected` on the official directed edge index. No explicit self loops are added. The graph and all node features are visible during full-batch message passing, as in transductive OGB node classification. Only official training labels enter the loss. Validation labels select a checkpoint; test labels are scored only after that selected checkpoint is restored.
- Four-member `models.TABMModel` with original BatchEnsemble input/output projections, two residual SAGE blocks, width 128, LayerNorm, dropout 0.2. `private_first` copies only the first residual block for each member; `private_last` copies only the second. Both have identical parameter counts. The `tied` and fully `untied_propagation` reference arms use identical initial tensors and optimization schedules.
- Seeds 0, 1, 2; AdamW, learning rate 0.001, zero weight decay, 300 epochs; each epoch uses mean member cross-entropy. One validation pass follows every epoch. The selected epoch maximizes pooled-logit validation accuracy, breaking ties by lower pooled validation cross-entropy and then earliest epoch. Test accuracy and cross-entropy are reported for that one selected checkpoint.
- The original 300-epoch OGB run configuration and model code are fingerprinted. This is an equal-budget within-study comparison. The exploratory graph choice and three optimization seeds limit inference beyond this graph and split.

## Integrity checks

Each arm starts from a re-created tied-model initialization for its seed. Initial model-state and CPU/CUDA RNG hashes are paired across arms, and initial member logits are checked. Artifacts include all 300 validation rows, selected checkpoint, selected validation/test member logits with official IDs and labels, and SHA-256 hashes. The independent verifier reloads each checkpoint on CUDA, recomputes validation and test logits, checks decisions and metrics, and confirms that `private_first` and `private_last` have the same number of parameters. Incomplete or unverifiable runs are excluded rather than replaced by a favorable score.
