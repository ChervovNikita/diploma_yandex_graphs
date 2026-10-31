# Frozen molecular pilot comparison

Selected VALID results. This validation set also selected checkpoints. TEST remains closed.

Complete paired trials: 3/3. Incomplete trials are not silently omitted from a final claim.

| Method | Paired trials | Mean delta vs base, ROC-AUC points | Mean delta vs I4, ROC-AUC points |
| --- | ---: | ---: | ---: |
| gnnm_boundary_native | 3 | +0.0000 | -7.0696 |
| gnnm_graph_alignment_variance | 3 | +3.3944 | -3.6752 |
| gnnm_private_input_residual | 3 | +3.9805 | -3.0891 |
| gnnm_input_residual_plus_graph_alignment_variance | 3 | +1.5018 | -5.5678 |

All per-trial differences, stopping reasons and source receipts are in COMPARISON.json. Three seeds on one scaffold split do not establish significance or transfer.
