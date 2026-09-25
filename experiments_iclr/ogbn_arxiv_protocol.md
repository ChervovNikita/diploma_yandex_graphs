# ogbn-arxiv external graph pilot

## What this measures

Each model receives the same graph of arXiv papers. One paper has 128 supplied
feature values and one subject label from 40 classes. The model produces 40
class logits per paper. Predictions on the official test papers are compared
only after an epoch has been chosen with the official validation papers.

This is one additional, fixed-configuration benchmark on a graph outside the
five archived heterophilous graph datasets. Its purpose is to check whether
the projector-sharing behavior seen in the archive also appears on this graph.
It is not an independent confirmation across several datasets. Multiple
optimization seeds all use the same one official graph split.

## Dataset and split provenance

- Dataset: `ogbn-arxiv`, from the Open Graph Benchmark (OGB), accessed with
  `ogb.nodeproppred.PygNodePropPredDataset(name="ogbn-arxiv")`.
- Citation: Weihua Hu et al., “Open Graph Benchmark: Datasets for Machine
  Learning on Graphs,” NeurIPS 2020. Dataset page:
  <https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv>.
- Indices: the dataset's unmodified `get_idx_split()` tensors named `train`,
  `valid`, and `test`. The loader checks that they form a disjoint partition.
- All node features and graph edges are available during training, as in the
  OGB transductive task. The loader reads all labels into CPU memory. Only
  training labels create gradients. Validation labels measure checkpoints.
  Test labels stay on the CPU and are used for scoring only after checkpoint
  selection is complete.
- Citation edges are converted to undirected edges with PyG
  `to_undirected`, as in common OGB message-passing baselines. No explicit
  self loops are added. The repository's SAGE layer includes a root transform.
- The run writes tensor SHA256 hashes, graph and split sizes, and OGB/PyG
  versions to `dataset_manifest.json`. Downloads stay in
  `experiments_iclr/data/` inside this repository.

For this local preparation, the OGB package's HTTP downloader stalled before
the response header. The unchanged official archive was fetched over HTTPS
from <https://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip> on the Mac,
then transferred into the repository. Its ZIP integrity check passed, and
its SHA256 is
`49f85c801589ecdcc52cfaca99693aaea7b8af16a9ac3f41dd85a5f3193fe276`.
It contains `arxiv/RELEASE_v1.txt`, the original raw files, and the official
time split. The OGB loader processed those files without changing the split.
The resulting manifest reports 169,343 nodes, 1,166,243 raw citation edges,
and 90,941/29,799/48,603 train/validation/test nodes.

The graph and split are fixed. An optimization seed changes the initial
weights and dropout sequence, not which papers are assigned to train,
validation, or test.

## Predeclared model and training settings

The default pilot uses the repository's residual `SAGE` backbone with two
graph blocks, 128 hidden features, LayerNorm, GELU, and dropout 0.2. Its
input and output dimensions come from OGB. Every variant uses AdamW with
learning rate 0.001, weight decay 0, at most 100 full-batch epochs, and the
same graph and official split. Validation is checked after every epoch.
Training stops after at least 20 epochs when 20 consecutive validation checks
fail to improve the selection score. Seeds 0, 1, and 2 are the intended
minimum complete pilot. These settings are fixed in advance for all variants.
They are a small pilot setting, not a dataset-specific grid search or a claim
of tuned OGB performance. CLI changes must go into a new result directory.

The three variants differ as follows:

| Variant | Learned network | Training loss | Inference logits |
| --- | --- | --- | --- |
| BASE | One ordinary `models.Model` | Cross entropy on training papers | One model |
| ENS | Four independently initialized `models.Model` networks | Mean of the four member cross entropies | Arithmetic mean of four logit matrices |
| GNNM | One `models.TABMModel` with four BatchEnsemble input and output projector members and one shared graph backbone | Mean of the four member cross entropies | Arithmetic mean of four logit matrices |

For ENS and GNNM, one optimizer minimizes the mean member cross entropy.
ENS updates four disjoint parameter sets, while GNNM averages the four
member gradients in its shared parameters. This differs by a constant
loss scale from the earlier ENS runner. All four member forwards
are processed sequentially within an epoch to limit activation memory.

For **each variant and seed**, the selected checkpoint is the epoch with
highest accuracy of the **averaged validation logits**. A tie is broken by
lower pooled validation cross entropy, then by the earlier epoch. All four
ENS members are saved from that same selected epoch. This corrects a
checkpoint-policy mismatch in the archived ENS versus GNNM comparison, where
ENS members were selected separately. It does not make the models share
initialization, training trajectories, or capacity.

The selected checkpoint is restored, its validation score is checked again,
and the official test labels are used for scoring once. Training never uses
test labels or test metrics. The script saves per-epoch training loss and
validation scores, selected member logits and labels for validation and
test, checkpoints, parameter counts, wall times, and peak PyTorch allocated
GPU memory. Times are synchronized around each training and validation step
on CUDA. They include full-batch graph computation and exclude data download.
Wall time also includes checkpoint writing and final evaluation, so it can
vary with file-system load. Peak allocated memory includes the graph already
on the device. It is not total GPU reserved memory.

After a complete run, `ogbn_arxiv_diagnostics.py` reads the already selected
test logits and reports the average accuracy of individual members, accuracy
after logit averaging, and mean pairwise disagreement of member argmax
predictions. It also counts test nodes by how many members classified them
correctly. These calculations do not choose a checkpoint or hyperparameter.

## How to run

From the repository root, after an A100 or another suitable GPU is free:

```bash
mkdir -p experiments_iclr/.tmp experiments_iclr/data
UV_CACHE_DIR="$PWD/.cache/uv" TMPDIR="$PWD/experiments_iclr/.tmp" \
  .tools/uv-x86_64-unknown-linux-gnu/uv pip install \
  --python .venv/bin/python ogb==1.3.6
.venv/bin/python experiments_iclr/ogbn_arxiv_pilot.py \
  --device cuda:0 --seeds 0 1 2 --variants base ens gnnm
.venv/bin/python experiments_iclr/ogbn_arxiv_diagnostics.py
```

The first run downloads OGB files into `experiments_iclr/data/`. A CPU-only
download and provenance check is available with
`--prepare-data-only --device cpu`. The synthetic CPU test needs no OGB
download:

```bash
.venv/bin/python experiments_iclr/test_ogbn_arxiv_pilot.py
```

The output root is `experiments_iclr/ogbn_arxiv_results/`. The script skips
completed seed and variant pairs when resumed. A result is complete only
when its `selected.json` exists. Use a separate output root for a changed
configuration. The script rejects data and output paths outside the repo.
On this machine, the official ZIP has already been placed and processed, so
the pilot runner will load it locally without another download.

## Interpretation limits

The three seed scores describe optimization variation on one official
split. They do not justify a test of general superiority across graph tasks.
The original benchmark used up to five layers and width 512. This smaller
two-layer, width-128 pilot has a different capacity and learning rate, so it
cannot be pooled numerically with the archived results. All variants share
the pilot settings, yet they have different parameter counts and are not
matched by training time. GNNM and ENS process four members, while BASE
processes one. A test score from this pilot should be reported with its
exact configuration and all completed seeds, including unfavorable results
if it is used to support a broad accuracy claim.
