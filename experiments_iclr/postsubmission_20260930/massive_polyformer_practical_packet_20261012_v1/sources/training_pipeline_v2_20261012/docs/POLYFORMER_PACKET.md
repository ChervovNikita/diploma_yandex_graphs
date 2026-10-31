# PolyFormer mono training packet

This adapter executes the pinned **PolyFormer** implementation from
[air029/PolyFormer](https://github.com/air029/PolyFormer), commit
`d390f39e88d0eaac80318fdc7704bd3bf3cf8b13`. Polynormer is a separate model family.

## Actual source and fidelity

`adapters/polyformer.py` extracts unchanged author definitions from the existing
`coordinate_source_independent_review_v1/strong_backbones_v1/sources/polyformer_code/node_classification/`
archive: `PolyFormerBlock.py`, `mymodels.py` and the mono functions in `utils.py`.
It verifies the existing pinned bytes before execution. This follows the source
loading used by the completed PubMed strong references and private CMCL source.
Author module import seeding, cached pickle loading, dataset acquisition, split
generation and TEST evaluation are omitted. The model definitions and mono
preprocessing functions remain unchanged. This is a source-backed standardized
comparison with prospective roles/budgets, not full published command/split
replication.

The model retains token-specific MLPs, tanh polynomial attention, native degree
decay bias, LayerNorm, residual attention/FFN, token summation, and the native
lin2/ReLU/dropout/lin3 task tail. It does not substitute another transformer or
rename a generic backbone.

## Methods and fitting units

| Method | Physical bodies | Factor rows per affine site | Selected states |
|---|---:|---:|---|
| `polyformer_single` | 1 | none | one own selector |
| `polyformer_factorized_single` | 1 | 1 | one own selector |
| `polyformer_internal_be` | 1 | 4 | one shared-bank selector |
| genuine independent four | 4 separate runs | none or one per body | four separate selectors |

Internal factors use the existing `FactorLinear` source: shared W/bias objects,
`linear(x*r,W)*s + bias`. All native affine sites are equipped, including token
MLPs, Q/K, FFN and the three outer maps. Native slow parameters remain live and
normalizations/bias-scale parameters remain shared. Unit factors preserve the
initial native predictor. Optional first-stem Rademacher initialization is a
separately declared source-supported variant requiring an explicit factor seed.

The full PubMed constructor has 23 affine sites and 2,069,875 ordinary parameters.
Equipped M1 adds 13,943 coordinates (total 2,083,818); equipped M4 adds 55,772
(total 2,125,647). Other declared constructor shapes retain their actual dynamic
site counts rather than being forced into PubMed dimensions.

Each `make_adapter` call owns one body, Adam and validation selector. Independent
four uses four prospective run IDs, construction seeds and separately selected
fits. `assemble_independent` performs only serving and checks disjoint parameter
storage plus matching declared data custody.

## Configuration

Use factory `training_pipeline_v2_20261012/adapters/polyformer.py:make_adapter`
through the study CLI. Paths below are relative to the project root.

```json
{
  "node_inputs": {"graph": "prepared/graph.pt", "train": "prepared/train.pt", "select": "prepared/select.pt"},
  "native": {"K": 2, "hidden": 256, "nlayer": 2, "n_head": 8, "d_ffn": 32, "q": 1.6, "multi": 2.0, "dropout": 0.5, "dprate": 0.8},
  "classes": 3,
  "initialization_seed": 1009,
  "dropout_seeds": [301010, 302019, 303028, 304037],
  "preprocessing": "pinned_mono_base",
  "train_views": 1,
  "fusion": "mean_probabilities",
  "factor_initialization": "unit",
  "learning_rate": 0.005,
  "attn_lr": 0.0005,
  "weight_decay": 0.001,
  "attn_weight_decay": 1e-8
}
```

These numbers describe the archived PubMed shape/optimizer and an explicit seed
example; they are not a universal hyperparameter recommendation. The prospective
study owns all choices. Graph input has exactly `x` and `edge_index`; TRAIN and
SELECT files each have exactly unique ordered `ids` and categorical `y` with
disjoint role authority. No CAL/evaluation targets enter the factory.

`pinned_mono_base` applies the literal archived native normalization to the
supplied graph. `author_undirected_then_mono` additionally applies the author's
undirected conversion before that function. The adapter does not normalize
features or silently change the declared graph policy. It constructs the
target-free `X,A_norm X,...,A_norm^K X` bank using the exact
gcn_norm/SciPy-float32/spmm source path. Actual input-file receipts bind restart
custody; the study must prospectively bind preprocessing and roles.

## Objective, optimizer, RNG and serving

Native own training accumulates separate old-state CE backward calls across
declared views/routes, then makes exactly one Adam update. Ordinary own uses
one view; the strong mean-four-dropout single declares four private factual
streams and four backward calls. Equipped4 own uses one view per factor route.
There is no probability-pool CE in these own procedures.

Adam uses one group per registered parameter: names containing `attnmodule`
receive the declared attention LR/decay; every other parameter receives the
declared ordinary LR/decay. Betas (.9,.999), epsilon1e-8 and ordinary Adam decay
preserve the native optimizer law. AdamW is not substituted.

Each route/view has a caller-declared persistent CPU/CUDA dropout stream.
Construction and stochastic forwards preserve the caller's outer Torch RNG.
Private stream state, factor-member flags and native unregistered attention
bias tensors are captured/restored with the existing classifier state hooks.
Mono tokens remain frozen external inputs bound by actual file/source custody.
The common engine supplies full-budget execution, validation-only selection,
virtual prefixes and exact selected-state restore.

Raw serving logits retain the `[route,node,class]` axes. Dropout-off serving is
`mean_route(softmax(raw_logits))`; a single has one prediction. Independent four
uses the four own-selected predictions with the same probability mean, without
a pooled validation selector or additional optimizer update.

Optional `loss` configuration uses the already qualified scalar-loss callback:
`{"variant":"gncl","coefficients":{"pool_weight":0.5}}`, within-route SupCon,
their combination, or declared paired losses from `node_losses.py`. Paired
variants require `train_views:2`, make two live native forwards per route and
one scalar backward. Their hidden input is the actual live input to `lin3`;
representation auxiliary gradients exclude that head. These transfers are new
standardized ports, not claims of an archived PolyFormer loss recipe. Private
CMCL/OIPG/FoRDE gradient-allocation procedures require their dedicated adapters.

This packet currently supports categorical full-batch node classification and
the mono basis. Author binary-node BCE, Chebyshev basis, sampled training and
full published split/launcher replication require separate explicit adapters.

## Engineering gate

`tests/test_polyformer.py` checks actual author architecture/counts, unit-factor
containment, source normalization, private RNG isolation, live hidden gradients,
native grouped Adam, full-budget execution, unregistered-bias/stream restoration,
paired scalar callbacks and independent probability serving. Tests use small
synthetic tensors and do not constitute benchmark evidence.

Set `GNNM_REFERENCE_PROJECT` to a project root containing the four pinned source
files when running from a standalone QA snapshot. Use the existing approved
Torch/PyG/NumPy/SciPy runtime; missing dependencies/sources cause explicit skips.
No benchmark training or production launch is admitted by creating this packet.
