# Molecular graph classification

Three fixed recipes cover MolBACE with GCN and GIN, and MolHIV with GIN.
Each recipe has ten paired seed cohorts. Each cohort trains four independent
native models and one four-route GNNM. Native member zero also supplies BASE.
This gives 150 physical training fits and 90 reported cohort-method rows.
The ordinary ensemble averages the logits from the four saved native fits.

## Reference and protocol

The vendored OGB molecular example is pinned at commit
`61e9784ca76edeaa6e259ba0f836099608ff0586` in
`external_sources/ogb_molecular/manifest.json`. Its source is
https://github.com/snap-stanford/ogb/tree/61e9784ca76edeaa6e259ba0f836099608ff0586/examples/graphproppred/mol
The sole source patch makes `gnn.py` import its sibling `.conv` as a package.
The manifest records downloaded and patched SHA-256 hashes and includes the MIT license.
The installed OGB version and other numerical package versions enter every job config.

The native model is the pinned `GNN` with `gnn_type=gcn` or `gin`, no virtual
node, five message-passing layers, width 300, full atom/bond categorical features,
BatchNorm, dropout 0.5, last-layer jumping knowledge, no residual connections,
and graph mean pooling. Adam uses learning rate 0.001 and weight decay zero.
Batch size is 32. The full reference horizon is 100 complete epochs, with constant
learning rate and no scheduler or early stopping. These are the official
`main_pyg.py` defaults for the explicitly chosen nonvirtual GCN/GIN recipes.
The default CLI architecture upstream is GIN with a virtual node. The requested
GCN/GIN recipes explicitly use its supported nonvirtual variants.

Official OGB scaffold indices and the OGB ROC-AUC evaluator are used. Selection
uses the first maximum validation ROC-AUC over the entire declared horizon.
Validation curves and whether the incumbent is in the final quarter are saved.
A 100-epoch horizon is a reference protocol and does not establish convergence.
Test inference occurs on the selected checkpoint at completion. Smoke runs keep
TEST closed and use the full validation split. Scores are fractions in [0, 1].

Each native member selects its own validation checkpoint. Their ensemble averages
those four saved raw logits. GNNM selects a checkpoint using the mean of its four
route logits and trains using the mean of four masked BCE-with-logits losses.
A single optimizer update follows the four sequential route backward passes.
Seeds are `1000 * cohort + member` for native members and `1000 * cohort` for GNNM.
All routes share the native convolution, bond encoder and BatchNorm parameters
and running buffers. Dropout is sampled separately during each route forward.

## Molecular boundary mapping

The native input boundary is OGB `AtomEncoder`, a sum of nine categorical embedding
tables. Concatenating the nine one-hot vectors makes this exactly one bias-free
linear map. The GNNM input boundary reuses `models.BEBlock` on that representation,
copying the native embedding tables into its shared weight matrix. Input R starts
with independent signs, S starts at one and B starts at zero, as in original GNNM.
The official node encoder then consumes the resulting hidden tensor through an
Identity atom encoder. The official message-passing implementation is reused.

The output boundary follows graph mean pooling and replaces the native affine
head with the original `BEBlock`. It retains the native head's weight and copies
its bias into each private B. Output R and S start at one. This bias initialization
preserves the source affine function and is the explicit adaptation of original
GNNM's zero output-B initialization. The categorical one-hot boundary
and graph-level pooling placement adapt the original continuous node-task input
and node output. Identity-factor tests verify native-function equivalence for
both GCN and GIN. Four routes have one shared backbone and four private boundary
factor sets.
## Run and resume

Run from the repository root with the prepared PyTorch, PyG and OGB 1.3.6 environment.
All paths must resolve within the repository. One worker uses the single GPU.

```sh
python -m ablation.p09_molecular.run --root runs/rebuttal/p09 --device cuda:0
python -m ablation.p09_molecular.run --root runs/rebuttal/p09 --status
python -m ablation.p09_molecular.collect --root runs/rebuttal/p09 --out runs/rebuttal/p09_summary
```

The CLI supports dataset/model/cohort/method selection. Full defaults have 150 fits.
The shared `JobStore` and `run_worker` provide claims, heartbeats, atomic state,
retry accounting and completed-job skipping. Config/source/package versions bind
each job, including completed jobs on a later invocation. The data directory
also enters the config. Ordered atom features, edge indices and bond features are
hashed per graph, alongside the official split indices and labels. These hashes
are checked on resume and before completed jobs are skipped. Cohort aggregation
requires matching source, protocol, data and numerical-package bindings. Resumable state contains
the current model, optimizer, explicit null scheduler, selected model, Python/NumPy/
Torch/CUDA RNG, dedicated shuffle RNG, current permutation and next-batch cursor.
A restart repeats at most the unsaved minibatches. Permutations use a dedicated
seeded generator so native member zero and GNNM receive the same graph ordering.
Explicit PyG Batch collation supports exact mid-epoch recovery with zero workers.
Validation is deterministic and can restart after an interruption. The reference
singleton-training-batch skip is retained. Existing checkpoints fall back through
`state.prev.pt` if the latest serialized checkpoint is unreadable.

Bounded numerical qualification uses an independent root and explicit smoke flags.
It changes neither the 100-epoch default nor scientific-run configs. The root role
marker rejects mixing smoke and full runs, and the default scientific collector excludes smoke output.

```sh
python -m ablation.p09_molecular.run --root runs/smoke/p09 --datasets ogbg-molbace --models gin --cohorts 0 --methods ens_m0 gnnm --smoke_updates 2 --device cuda:0
python -B -m unittest ablation.p09_molecular.test_p09
```

The collector emits complete five-fit cohorts, reports missing fits across all 30 planned cohorts,
and supplies paired differences against native member zero. Its summaries contain
sample standard deviations across seed cohorts on the same official scaffold split.

The collector defaults to all 30 planned cohorts. Its `--datasets`, `--models`
and `--cohorts` selectors define the expected groups using the same recipe mapping
as the runner. It returns success when every selected cohort has four native fits
and one GNNM fit. Partial tables retain a nonzero completion status and list the
missing members. Duplicate cohort/member records raise an error.

For engineering qualification, `--include-smoke` explicitly permits smoke fits
and aggregates VALID only. JSON, CSV and summary rows carry an
`engineering_validation` role. The JSON also preserves the input root role marker
and requested selectors. The default scientific output includes VALID and TEST
for completed full-run cohorts.

```sh
python -m ablation.p09_molecular.collect --root runs/smoke/p09 --out runs/smoke/p09_summary --cohorts 0 --include-smoke
```

This engineering command expects all three recipes at cohort zero. A narrower
qualification can select `--datasets ogbg-molbace --models gin --cohorts 0`.
