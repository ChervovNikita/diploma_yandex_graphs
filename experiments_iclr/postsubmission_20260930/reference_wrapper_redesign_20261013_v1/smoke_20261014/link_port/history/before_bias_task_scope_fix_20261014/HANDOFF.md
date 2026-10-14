# Link smoke frontend handoff

## Entry point

Load `callbacks.py` by its file path and call:

```python
session = create_session(recipe, procedure, data_root, device)
```

`recipe.architecture` must be `NCN`, `NCNC`, or `BUDDY`; dataset is
`ogbl-collab`. `native_fixed_config` takes priority over `settings`,
`hyperparameters`, and `config`. Declared settings must match the literal
source-fixed configuration; conflicting or unknown keys fail before Torch
is imported. Seed priority is `_seed`, `seed`, `optimizer_seed`, then zero.

| Campaign procedure | Frontend procedure | Routes |
| --- | --- | --- |
| `native_single` | `native` | 1 |
| `gnnm_boundary_native` | `base` | 4 |
| `boundary_global_half_pool` | `global_half` | 4 |
| `boundary_private_half_pool` | `private_half` | 4 |

The parent's `IndependentBank` constructs four `native_single` sessions.

## Session contract

- `model`, `optimizer`, `optimizers`, and `source_parameter_references` expose
  the actual native modules, optimizer, and retained native Parameters.
- `train_update()` performs one source-sized TRAIN minibatch and one optimizer
  transition. It returns finite-loss diagnostics and declared work counts.
- `validate()` evaluates all 60,084 official VALID positive records and all
  100,000 official VALID negative records. `outputs` is one aligned Tensor,
  positive logits followed by negative logits; it also returns `role="VALID"`,
  `loss`, `diagnostic_loss`, `lossfinite`, and `metrics["hits@50"]`.
- `state()` and `load_state()` preserve native model buffers, optimizer state,
  sampler position/population, boundary factors/route, and four private RNG
  streams. The common controller separately preserves global RNG, module modes,
  and `requires_grad` flags, as its existing checkpoint implementation does.
- `metadata` and `source_metadata` reference the same source/protocol record.
  `allow_missing_grad_names` lists only native inactive NCNC `ptlin` or BUDDY
  `bn_RA` Parameters.

## Required data and staging closure

Bind exact files with `recipe.data_files = {"graph": ..., "train": ...,
"valid": ...}`. Relative paths resolve under `data_root`. No TEST file is
requested or opened.

The parent located these allocation files, relative to the phase directory:

- GRAPH: `buddy_newdata_root_20261012_v1/provider_data/ogbl_collab/processed/geometric_data_processed.pt`
- TRAIN: `training_pipeline_v2_20261012/qa/link_admitted_transfer_v1/cache/ogbl_collab/split/time/train.pt`
- VALID: `training_pipeline_v2_20261012/qa/link_admitted_transfer_v1/cache/ogbl_collab/split/time/valid.pt`

The graph loader accepts a processed `(Data, slices)` tuple or Data dictionary
and requires all 235,868 nodes with 128 feature channels. BUDDY requires TRAIN
`edge`, `year`, and `weight`; its unchanged native `filter_by_year` selects
`year >= 2007` and rebuilds the weighted graph from those TRAIN records with
undirected sum coalescing. Graph fields require the parent's runtime probe;
the located processed graph is absent locally.

Stage all three Python files in this directory, plus the existing sibling
closure `../../native_sources/MANIFEST.json`, the required native snapshots,
and `../../runtime/training_rules.py`. Snapshot bytes are verified against
their frozen source manifest before execution.

Required genuine runtime packages include Torch, PyG, `torch_sparse`,
`torch_scatter`, OGB, NumPy, SciPy, pandas, tqdm, and BUDDY's `datasketch`.
No substitute/stub operators are supplied.

## Native behavior and boundaries

NCN/NCNC preserve native GCN H64, one layer, native TRAIN negative sampling
before the PermIterator permutation, whole-record query masking before
symmetrization, source loss, batch 65,536, and dropped epoch tail. NCNC retains
native recursive completion. The canonical sites are the original
`encoder.convs.0.lin` and the final original affine in `predictor.lin`.

BUDDY preserves H1024, batch 1024, native one-negative-per-positive sampling,
source outer sample permutation followed by shuffled DataLoader ordering,
weighted GRAPH, SIGN-k0 propagation, full-node hashes, and source BCE. Its
original `label_lin_layer`, `lin_feat`, and final `lin` are boundary sites;
original BatchNorm modules remain shared. Query features are hydrated on
demand by the original native hashing function; hydration preserves CPU RNG
and changes cache storage timing only. This staging choice is disclosed in
metadata. No graph, feature, or query subset is substituted.

Hooks preserve native affine classes and Parameters. Input R banks start as
Rademacher signs; final R and all S banks start at one; delta biases start at
zero. Four persistent Python/NumPy/Torch CPU/CUDA source RNG streams are
serialized. Global/private half pooling calls the existing shared training
rules without modifying them.

## Qualification status

Local checks passed for Python compilation, callback import without Torch,
all three campaign config matches, the four campaign procedure aliases, and
rejection of conflicting config values. No local model construction, numerical
forward/backward pass, remote command, fit, or TEST scoring was performed.
Actual smoke, dependency admission, memory, and checkpoint replay qualification
are parent-owned and remain unverified until execution receipts arrive.
