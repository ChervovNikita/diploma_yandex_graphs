# Public WikiCS context-steering training

Run one fixed method in your own environment with complete caller-supplied
TRAIN/development NPZ files, pinned Polynormer source, an explicit seed/device,
and a fresh output directory. This is a callable public successor to the
reviewed context integration v2. It has no author host/path, GPU UUID, root
approval document or private frozen-target archive requirement. Setup is in
[SETUP.md](SETUP.md); method ancestry and claim limits are in
[ATTRIBUTION.md](ATTRIBUTION.md).

## Run

```sh
python portable_context_steering_public_interface_20261008_v1/train.py \
  --method shared_route --seed 8101 --device cuda:0 \
  --public-interface portable_internal_be_public_interface_20261007_v2 \
  --polynormer external/Polynormer/model.py \
  --train roles/wikics/train.npz --valid roles/wikics/valid.npz \
  --output runs/public_shared_route_8101
```

The public v2 code dependency may live at any caller path. If the two public
folders are siblings, `--public-interface` can be omitted. The dependency’s
published code manifest and every listed file are checked; these are source
identity checks, not author approval. All data/native/output paths belong to
the caller. `--method`, `--seed`, `--device` and `--output` are required. Seeds
are integers in `[0,2**32)`; CPU or any available `cuda:index` is supported by
the public Session. No dependency download, data hydration, next fit, retry,
resume, coefficient search or shortened scientific horizon runs automatically.

## Fixed methods

| Method | Predictor bodies | Context targets | Checkpoint policy |
|---|---:|---|---|
| shared_common | 1 shared, 4 unit-factor routes | Qbar for every route | joint strict-first pooled development |
| shared_route | 1 shared, 4 unit-factor routes | route m gets Qm | joint strict-first pooled development |
| shared_route_permuted | 1 shared, 4 unit-factor routes | fixed class/panel/degree-preserving permuted Qm | joint strict-first pooled development |
| single_native | 1 ordinary model | none | single strict-first development |
| independent4_native | 4 ordinary untied models | none | independently own-selected bank |
| single_common_context | 1 ordinary model | Qbar | single strict-first development |
| independent4_route_context | 4 ordinary untied models | member m gets Qm | independently own-selected bank |

Every method keeps the pinned native Polynormer, two full own-CE stochastic
views, all580 TRAIN labels, original Adam, 1100 epochs (100 local +1000 global),
original fullgraph forward and mean class-probability serving. Shared methods
start with unit factors. Context alignment is symmetric cross-view CE at
weight.05 and temperature.2; residual member repulsion is zero. Shared loss is
mean own CE +.05 mean alignment. Untied4 loss is exactly the sum of four
separable own CE +.05 alignment losses, retaining each model’s unscaled gradient.
Single-common receives its own CE +.05 Qbar alignment. Every live parameter
receives the ordinary scalar objective; no selective gradient permission is added.

Own-selected references restore each member’s own local model/Adam history,
retain live end-local RNG streams, and finish with each own-selected model and
its saved local/global flag. The final mixed-epoch bank is evaluation-only.
Shared/single methods retain the original joint local restore and strict-first
joint selection. Saved joint checkpoints can remain local after1100 training
epochs: restore their `global` flag explicitly when serving. Final own banks
instead carry the per-member `body_global` flags. Those Python flags are absent
from state_dict. No checkpoint reselection or historical-score parity gate is used.

## TRAIN-only target construction

Before constructing a model, the CLI derives four signatures from the caller’s
public graph/features: X, X-PX, PX and P²X. P is directed destination aggregation,
row-normalized after deduplicating ordered edges and including exactly one
self-loop. Predictor edges are unchanged. Each signature selects16 OTHER
same-class TRAIN rows by cosine (or all available if fewer), resolves ties by
fixed TRAIN order, and includes self as a cross-view positive. No development
labels or fitted outcomes enter this function; transductive graph/features
remain the original predictor inputs.

The exact original `torch.linspace(0,579,steps=512,device=...).long()` panel is
checked on CPU and the requested device. Masks restrict to that panel and
normalize afterward. Qbar averages the four normalized route targets. The fixed
permutation991327 preserves class, self, panel membership and every scored
anchor’s restricted positive degree. Both mean target TVs must be at least.05;
a failed differentiation gate stops without changing the seed/recipe.

One immutable base/permuted bundle is prepared for every method. The CLI records
ordered identities, logical array hashes, source hashes, TVs, zero-signature
counts and the archive digest before the first update. For comparisons, predeclare
methods/seeds, keep data/runtime/device fixed, and verify matching logical target
hashes across runs. Newly computed caller targets are separate from the frozen
author study; they do not alter its archive or join its registered roster.

## Outputs and compute

RUN/PROGRESS/VALID_TRACE, selected/local snapshots, own snapshots/bank when
applicable, COMPLETE or FAILURE, TARGET_PREPARATION, TRAIN_TARGETS.npz and
PUBLIC_COST.json are written only in the fresh caller output. The target archive
contains TRAIN labels/IDs and masks; it is a caller artifact, not a packaged
scientific input. Run identities use `public_context_steering__<method>` and
explicitly record that they are outside the registered author family.

The byte-identical v2 replay evaluates2M shadow forwards and2M replay forwards
per update, accumulates2M VJPs, checks exact RNG endpoints and unchanged
parameters, then steps each original Adam once. It reuses the original two
views. Complete call counts are checked and actual attempts/returns retained:

| Predictor | Training member calls | Development member calls | Member VJPs | Adam steps |
|---|---:|---:|---:|---:|
| single | 4,400 | 1,100 | 2,200 | 1,100 |
| shared4 | 17,600 | 4,400 | 8,800 | 1,100 |
| untied4 own bank | 17,600 | 4,404 | 8,800 | 4,400 |

All methods collect1,100 output cotangents. PUBLIC_COST records inclusive wall,
target preparation/update time, actual providers, available CPU time and cumulative
process RSS, CUDA allocated/reserved peaks and storage. Sparse kernels/hardware
and floating-point reverse accumulation can change trajectories; no bitwise
author parity is claimed. The caller manages available memory, timing and
process supervision. Hard termination can prevent a final receipt; preserve
last progress and actual external terminal evidence.

## Scope and status

Only complete WikiCS with the pinned Polynormer is implemented. Arbitrary NPZs
are checked for complete shape/domain/role semantics and their actual hashes;
those checks do not certify official source custody. Use the public converter
and independently obtained official data for an official-data reproduction.
The development population also selects checkpoints, so its scores carry
selection optimism. There is no TEST access or independent-task/split evidence
from this source packet. Different contexts, label-compatible positives and
higher coverage do not guarantee useful complementary predictions or competence.

Preparation performed source/seal/AST inspection and stdlib CLI help only: no
framework import, real data access, model/update, GPU or serving call. The inherited
v2 engine has separate native engineering evidence, but this public construction
and full driver path still require their own runtime qualification. No numerical
performance, external full-training reproduction or novelty claim is made here.
Exact predecessor changes are in `predecessor_diffs/`; SOURCE_ORIGIN records
provenance. Original running/frozen sources are unchanged.
