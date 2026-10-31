# New-study molecular adapter

`gine_adapters.py` implements controls for molecular graph tasks and preserves categorical bond features. It does not reproduce the historical node-task architecture or change submitted results. The coordinator requested width128/three layers/LayerNorm/mean pooling; defaults implement these choices. Each GINE block's internal MLP uses hidden multiplier2. The separate SAGE branch uses multiplier1 per the coordinator's tested source-equivalence specification.

Every graph supplies OGB atom `x: long[N,9]`, `edge_index: long[2,E]`, and bond `edge_attr: long[E,3]`. `features`, when explicit, must be those same categorical atom fields. AtomEncoder sums learned categorical embeddings into width d. The initial projector, dropout and GELU precede three blocks. A block computes

`x <- x + GINEConv(Linear(d,2d), GELU, dropout, Linear(2d,d), dropout)(LayerNorm(x), edges, BondEncoder(bonds))`.

GINE uses a learnable epsilon/root term and PyG's ReLU bond-aware messages. Output LayerNorm acts per node; graph representations are mean pooled; the output projector produces `[graphs,targets]` logits. No graph-batch normalization or virtual node is added. Official inverse edges are preserved. This adapter is a defensible bond-aware control, not a claim of the strongest tuned chemical backbone.

| Arm | Atom/bond/propagation policy | Ensemble computation |
|---|---|---|
| `MolecularSingle` | One AtomEncoder, one input/output linear projector, one GINE stack | One member/path |
| `MolecularProjectorEnsemble` (TIED) | One AtomEncoder; boundary BatchEnsemble W/R/S/B from immutable source snapshot; shared GINE blocks including each block's BondEncoder | Four distinct hidden-state paths; average raw logits |
| `copied_private(TIED)` | Deep-copy current model, then make a separate copy of each chosen GINE block per member; AtomEncoder, boundary W/factors and output LayerNorm remain shared. Each private block includes its copied BondEncoder. Unused shared block parameters are removed. | Four paths; initially identical to TIED, conditional on identical dropout state |
| `MolecularSharedMultihead` | Single atom/bond/GINE trunk, four independently initialized linear output heads | One propagation path; mean-member training loss can train different heads |
| `MolecularIndependentEnsemble` | Four independently initialized single models including separate AtomEncoder/BondEncoders and all projectors | Four independent complete models |

The intervention pair isolates graph-block tying, including bond encoding. It leaves a shared atom lookup in both arms; untying the atom lookup would be a different intervention. Boundary factors operate on embedded atom channels, unlike the original snapshot's factors on raw continuous node features. Report this task adapter difference explicitly.

All classes' `forward(graph, features=None)` and `forward_members(...)` return `[members,graphs,targets]`, matching the coordinator's baseline adapters. `forward_member(graph, features, member)` returns `[graphs,targets]`. `pooled_forward(...)` computes raw-logit mean. Singleton binary outputs retain their final target dimension. The policy-compatible projector ensemble also has `members`, `privatize(indices)` and `probe_blocks()`; the latter returns `{str(layer): tuple(parameters)}` only for fully shared warmup models. Probe groups include LayerNorm, GINE MLP/epsilon and BondEncoder; AtomEncoder and boundary projectors are excluded. Do not privatize after optimizer creation without rebuilding it and logging a chosen warmup-state transfer policy.

The synthetic CPU test checks binary/multitarget shapes, a one-atom graph with no edges, all/selected privatization initial-logit equality, member and pooled interfaces, the sum-of-private-gradients equality for each shared block (including bond encoder), equal atom/boundary gradients, probe coverage, removal of unused shared parameters, finite control gradients, and parameter accounting. It uses double precision with dropout0 for numerical equality. No real dataset or score is accessed.

```sh
PYTHONDONTWRITEBYTECODE=1 python postsubmission_research_20260930/new_tasks/test_gine_adapters.py
```

Local Python lacks torch/PyG/ogb, so only Python syntax was verified here; the coordinator can execute the CPU test in the existing project environment. Publish the resulting JSON and exact module/provider versions before training. Dropout-active paired initial checks additionally require resetting the RNG between arms; the deterministic test establishes equality of the dropout-free function/gradients.
