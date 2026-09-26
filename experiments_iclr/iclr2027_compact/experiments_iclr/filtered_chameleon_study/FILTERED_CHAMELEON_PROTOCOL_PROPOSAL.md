# Filtered Chameleon sensitivity: source and fixed-design proposal

Prepared before inspecting any Cora/Chameleon depth or placement outcomes.
This is a dataset and protocol proposal only; it does not assert a result.

## Why this graph

Platonov et al., *A Critical Look at the Evaluation of GNNs under
Heterophily: Are We Really Making Progress?* (ICLR 2023, arXiv:2302.11640),
Section 3.1, found 1,387 duplicate nodes among 2,277 legacy Chameleon nodes
and reported train/test leakage and changed model rankings after filtering.
Their Appendix A says the filtered masks inherit the ten standard splits with
duplicate nodes removed. Testing the fixed GNNM contrast on this official
filtered graph addresses the validity of an observation on the legacy graph.
It does not add an independent application domain or establish a general
sharing rule.

## Exact source and loader

- Official repository: `yandex-research/heterophilous-graphs`, pinned commit
  `a431395582e929d88271309716bea4fe24ce6318`.
- Download URL: `https://raw.githubusercontent.com/yandex-research/heterophilous-graphs/a431395582e929d88271309716bea4fe24ce6318/data/chameleon_filtered.npz`.
- File SHA-256: `bf46f07e1fb5249280447e5fe3100f3e82fc4b93ad1e13ffcfdec924b6ac0bb5`;
  58,924 compressed bytes.
- Use `numpy.load(path, allow_pickle=False)`. Its keys are `node_features`
  `(890, 2325)` float32, `node_labels` `(890,)` int64, `edges` `(8854, 2)`
  int64, and `train_masks`, `val_masks`, `test_masks` each `(10, 890)` bool.
  Choose mask row 0 before training: 409 train, 287 validation, 194 test;
  these are disjoint and cover all 890 nodes.
- This file is not exposed by PyG `HeterophilousGraphDataset` or
  `WikipediaNetwork`. Convert arrays directly to PyTorch tensors. Hash the
  compressed file, source code, feature/label/raw-edge/training-edge tensors,
  and each split index before any optimization. Check finite features,
  contiguous class IDs, mask disjointness, edge endpoints, and all planned
  cells in the freeze.

The official file stores 8,854 directed edge entries, with no self-loops or
reciprocal entries. For comparison with the current GNNM recipe, coalesce and
symmetrize them to 17,708 directed training entries, and add no loops.
Platonov et al. evaluated the directed graph, so our symmetrized scores must
not be compared numerically with their reported model accuracies as if the
graph protocol were identical.

## Proposed complete matrix

Use the same width-128, four-member residual SAGE, AdamW 0.001, zero decay,
dropout 0.2, mean member CE, 300 epochs, pooled-validation checkpoint rule,
and seeds 0, 1, 2 as the current fixed new-graph study. Freeze all source,
data, and protocol hashes before training. Run:

- Depth 2 and 5: tied versus fully untied propagation, all three seeds (12
  cells).
- Depth 2 sharing position: tied, private-first, private-last, and fully
  untied, all three seeds (12 cells, including intentionally repeated depth-2
  endpoint arms if kept as a separate auditable experiment).

Require complete CUDA replay of selected validation/test logits and decisions,
matched initial member functions/RNG, equal partial-arm parameter counts,
and all seed cells before test interpretation. Report each seed, selected
epoch, validation mean, test mean, and paired differences. If using a
validation-selected placement analysis, apply the earlier canonical exact-tie
rule from `POSITION_PRETRAIN_FREEZE.json`; disclose the two-arm training cost.

## Provenance limit and runtime

The Platonov unfiltered NPZ and the current PyG legacy Chameleon have
identical feature, label, and split-0 index hashes. Their symmetrized
**nonloop** edge sets also agree exactly: 62,742 entries. The PyG legacy
graph additionally retains 50 raw self-loops, while the filtered file has
none. Therefore a contrast between their trained results changes duplicate
nodes and those loop entries; it is a sensitivity study, not a strict
one-factor estimate of the duplicate-removal effect. A loop-stripped legacy
bridge would be needed for that causal comparison.

The proposed 24 cells mean 7,200 epochs; filtered Chameleon has 890 nodes
and 17,708 symmetrized entries, versus 2,277 nodes and 62,792 entries in the
legacy run. A planning envelope is roughly 1–3 GPU hours on the same class of
GPU, including selection and replay, but this is an estimate without a timed
filtered run. GPU launch overhead, the 2,325-feature input, and the five-block
backward pass can limit the speedup. Prioritize the complete matrix and audit
over adding more seeds or claims near the submission cutoff.
