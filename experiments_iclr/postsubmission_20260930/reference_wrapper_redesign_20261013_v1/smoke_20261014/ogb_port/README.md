# Real-data OGB smoke callback

Controller entrypoint: `session.py:create_session(recipe, procedure, data_root, device)`.
The factory constructs the genuine pinned native model. Importing the callback
alone imports stdlib only. This folder contains no remote launcher or full fit.

```python
from pathlib import Path
import importlib.util

path = redesign_root / "smoke_20261014/ogb_port/session.py"
spec = importlib.util.spec_from_file_location("ogb_smoke_callback", path)
callback = importlib.util.module_from_spec(spec)
spec.loader.exec_module(callback)

recipe = {
    "dataset": "ogbg-molhiv",
    "architecture": "gcn-virtual",
    "seed": 0,
    "redesign_root": str(redesign_root),
    "native_fixed_config": {
        "gnn": "gcn-virtual", "drop_ratio": 0.5,
        "num_layer": 5, "emb_dim": 300, "batch_size": 32,
        "feature": "full", "lr": 0.001,
    },
}
session = callback.create_session(recipe, "boundary_private_half_pool", data_root, "cuda:0")
first, second = session.train_update(), session.train_update()
validation = session.validate()                 # entire official VALID, never TEST
checkpoint = session.state()                   # controller separately captures global RNG
```

The controller owns all actual execution, device routing, Python/NumPy/Torch
CPU+CUDA RNG, checkpoint files, engineering replay comparison and stopping after
the authorized 1–2 updates plus the specified third-update replay. No real smoke
pass is claimed by the local syntax checks used to prepare these files.

## Fixed coverage

- Datasets: `ogbg-molhiv`, `ogbg-molbace`.
- Native architectures: `gcn`, `gcn-virtual`, `gin`, `gin-virtual`.
- Source dropout: 0.0 or 0.5; depth5, width300, mean pooling, JK=last,
  residual=False, full categorical features, source batch32.
- Procedures: `native_single`, `gnnm_boundary_native`,
  `boundary_global_half_pool`, `boundary_private_half_pool`,
  `boundary_private_fixed_responsibility`.

These are the 16 OGB source presets and five requested procedures. Ordinary I4
can be checked by the parent controller as four independently constructed
`native_single` sessions; this callback adds no extra I4 architecture.
Molecular SupCon is not an applicable procedure.

## Categorical and native-object preservation

`native_loader.py` verifies the manifest hash/size of the saved author `conv.py`,
`gnn.py`, and `main_pyg.py`. It executes the unchanged definitions from conv/gnn,
temporarily binding the source `from conv import ...` dependency, and never runs
the main program (which would create a TEST loader). Constructor settings are
the exact adopted native defaults. The original source objects and parameters
remain registered inside the model, with no cloning or replacement.

The logical input site is `gnn_node.atom_encoder`, covering **every original
categorical field**. Each native field lookup still receives its original integer
IDs and calls its original Embedding. Its output is multiplied by `R_m,k[ID]`;
the original AtomEncoder still performs the full field sum. One input `S_m` and
one additive bias residual then act on that sum:

`encoded_m = S_in,m * sum_k(Embedding_k(original_ID_k) * R_in,m,k[original_ID_k]) + delta_b_in,m`.

The output site is the original `graph_pred_linear`:

`logits_m = S_out,m * native_graph_pred_linear(prehead_m * R_out,m) + delta_b_out,m`.

All original embedding W, native head W/b, BondEncoders, convolutions,
nonlinearities, pooling and virtual-node modules remain intact. No factor is
placed inside BondEncoders, message layers or the virtual-node path. First
logical-input R tables use one isolated CPU generator in original field order;
other R/S are one and residual biases zero. This sign start differs from the
neutral unit/zero state. There is no fabricated numerical identity check.
Actual installed AtomEncoder/BondEncoder source hashes are reported at runtime.

Native source Parameter references are exposed as
`session.source_parameter_references`; wrapped names use the `native.` prefix.
`allow_missing_grad_names` is empty for these native models. All native objects
survive state restoration; optimizer loading restores state without rebuilding
model parameters.

## Loss, Adam and buffers

Use original BCEWithLogitsLoss with source FP32 casts and the original `y==y`
observed-target mask. Source Adam is constructed over the original parameters
at LR0.001 with its unchanged defaults (including no weight decay). Added factors
are appended to their existing native weight/bias owners' groups. No optimizer
or explicit native regularizer is substituted. Global/private half-pool and
fixed responsibility reuse `runtime/training_rules.py`; native/base use the
ordinary own loss. There is one optimizer transition per update.

BN affine parameters, eps/momentum, running buffers/counters and train/eval calls
remain the original source objects. One native session forwards once per update;
GNNM forwards once per route, sequentially 0–3, through those same shared native
buffers. This honestly advances native BN four times. No private BN copies,
freezing, route batch concatenation or normalization replacement is used.

Fixed responsibility is drawn once over all observed **TRAIN** graph labels and
normalized by TRAIN class, never per smoke minibatch. Native per-observed-target
BCE reduction is retained. Missing targets are never converted into labels.
The complete fixed table belongs to checkpoint state.

## Real provider and full VALID

`data_root` accepts either the native OGB parent `provider_cache` or its
`ogbg_molhiv`/`ogbg_molbace` folder. It requires an existing
`processed/geometric_data_processed.pt`; automatic download is disabled by this
port's preflight. The OGB package/dataset release metadata must also be present
and compatible with the installed native reader. Root's reported stores are:

- MolHIV: `phase/massive_molecular_packet_20261012_v1/provider_cache`.
- MolBACE: `phase/massive_molecular_packet_20261012_v3/provider_cache`.

Only `split/scaffold/train.pt` and `valid.pt` are read (native CSV.gz fallback
for those roles). They must contain complete original integer scaffold indices.
No TEST split file, TEST view, TEST loader, or TEST record indexing is needed.
The source processed OGB container is loaded normally; private labels are selected
by TRAIN indices, validation labels by VALID indices only.

Training uses native RandomSampler without replacement and native PyG
Batch.from_data_list collation, batch32. The sampler's complete permutation,
cursor, epoch and isolated generator are stored, so replay resumes the same
third batch without starting another sampler iterator. This is the declared
paired-smoke RNG adaptation; exact historical advancing-RNG seed equality is
not claimed. Original TRAIN skipped-batch conditions are retained.

`validate()` traverses the **whole official VALID** with source batch32,
shuffle=False and native eval skipped-batch condition. It returns raw-logit
Tensor outputs, observed-target diagnostic loss and official OGB ROC-AUC.
GNNM uses the predetermined mean of member raw logits. M1 source outputs remain
unaltered. VALID prefixes are rejected, and no TEST score is produced.

## Controller state API

- `.model`: native or native-preserving boundary nn.Module.
- `.optimizers`: one source Adam; `.optimizer` aliases it.
- `.train_update()`: finite scalar `loss`, `optimizer_updates=1`, real TRAIN IDs,
  source skipped-batch count and route/work counts.
- `.validate()`: full VALID raw `outputs`, finite diagnostic `loss`, `metrics`,
  VALID IDs and `TEST_accessed=False`.
- `.state()` / `.load_state(...)`: full model/private banks/native BN buffers,
  Adam moments/groups, sampler permutation/cursor/generator, VALID loader generator,
  fixed responsibility table, role IDs, module train/eval flags and update counter.
- `.metadata()`: source/dependency/split pins, fixed settings, factor semantics and
  native optimizer/buffer ownership.
- `.close()`: removes only this adapter's hook handles.

Global Python/NumPy/Torch CPU/CUDA RNG is deliberately controller-owned. The
factory does not reseed it. The controller must restore that global state as
well as `load_state()` before repeating the third update.

Dependencies: the routed native Torch runtime, compatible PyG (including its
native message-passing/scatter dependencies), OGB molecular encoders/dataset
reader/evaluator, and dependencies imported by the unchanged author snapshots.
The controller must stage this entire folder, `native_sources/ogb`,
`native_sources/MANIFEST.json`, and `runtime/training_rules.py` together.
