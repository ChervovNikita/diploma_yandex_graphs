# Optional shared-layer gradient interaction diagnostic

Run only after the SAGE control queue and Roman Empire GAT tying pair have completed. This reads the already selected four-member GNNM checkpoints for official masks 0–4: SAGE from `results/projector_controls.csv`, and GAT from `gat_failure_pair_results/projector_controls.csv`. GAT uses five layers on masks 0–1 and four on masks 2–4, as frozen by its pair protocol. The script refuses incomplete rows, missing checkpoints, a missing SAGE split-0 adoption audit, or a GAT pair without its frozen manifest and final marker. It does not modify the queue or training runner.

From the repository root, using the repository environment:

```sh
CUDA_VISIBLE_DEVICES= .venv/bin/python experiments_iclr/gradient_interaction.py --self-test
CUDA_VISIBLE_DEVICES= .venv/bin/python experiments_iclr/gradient_interaction.py
```

Both commands run on CPU. The self-test uses a tiny synthetic graph with both SAGE and GAT, checks that the gradient of mean member loss equals the mean of the four separately calculated shared-layer gradients, and checks the cancellation statistic on known opposite vectors. It reads no study result and writes no artifact. The full run is intentionally separate from the GPU queue.

For each checkpoint, the script reproduces the runner's bidirected Roman Empire graph with self loops and uses only that mask's **training** node indices and labels in multiclass cross-entropy. It never reads validation or test masks, saved predictions, or held-out scores. It restores the complete selected state strictly and calls `eval()` before all four member passes, so dropout is off and every gradient is measured at the same parameter state. The parameter vector contains every trainable tensor under `residual_modules.*`: graph convolution, residual normalization, and feed-forward layers. It excludes boundary projectors and the final output normalization. This matches the shared propagation stack in both backbones.

Let `g_m` be the gradient of member `m`'s full training cross-entropy with respect to that vector. The CSV reports each norm, all six pairwise cosines, and `kappa = ||sum_m g_m|| / sum_m ||g_m||` for each mask and backbone. A value near one indicates aligned member gradients; a value near zero indicates strong cancellation of their sum. The runner scales every member loss by `1/4` before accumulating; applying that common scale to all four vectors leaves kappa and cosine unchanged. The diagnostic calculates each full member gradient separately to make the comparison explicit.

The full run writes `gradient_interaction_results/gradient_interaction.csv` and `gradient_interaction_results/gradient_interaction_provenance.json`. The provenance records source, dataset, selected-row, checkpoint, and training-index/label hashes, plus the result-table and frozen protocol hashes. The ten rows are descriptive measurements on one graph; overlapping official masks are not independent graph replications.

The anonymous draft's cross-entropy identity expresses mean member loss as pooled-logit loss plus a nonnegative reverse-KL term. This gradient diagnostic instead describes how member losses act on the *shared propagation weights* at a fixed checkpoint. It is post hoc: the checkpoint was chosen by validation performance, the measured `eval()` gradients omit training dropout, and a single final-state alignment does not reveal the optimizer's earlier trajectory. GAT depths and this failure-case study were chosen after archived results were inspected. Kappa and cosine cannot establish why SAGE or GAT test scores differ or isolate a causal effect of parameter tying.
