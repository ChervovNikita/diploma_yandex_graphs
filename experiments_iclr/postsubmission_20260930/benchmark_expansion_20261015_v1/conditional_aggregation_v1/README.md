# Conditional aggregation source v1

Three standalone four-route controls: uniform mean, learned global route
weights, and learned prediction-conditional route weights. All use the same
global J=.5F+.5L, retaining unchanged mean native own risk F and evaluating L
on the actual served raw logits. See [RECIPE.md](RECIPE.md) for the complete
fixed method, numeric conventions, exact native reduction contract, and
interpretation limits. This is mixture/stacking-inspired preparation with no
novelty or predictive benefit claim.

## Files and scope

- `core.py`: Torch-injected cross-task tensor readouts and native scalar risk.
- `molecular_transform.py:transform`: pinned original molecular source bridge.
- `METHODS.json`: method IDs and launch binding requirements.
- `SOURCES.json`, `PROVENANCE.md`: source and prior error-analysis provenance.
- `check_tensor_contract.py`, `CHECKS.json`: bounded local synthetic checks.
- `MANIFEST.json`: frozen package bytes, excluding itself and incidental caches.

The bridge attaches to a fresh `gnnm_boundary_native` OGBSmokeSession only.
It keeps every original Parameter, full AtomEncoder field, boundary factor,
native operator/BN/dropout/graph rule, optimizer group/options, data role and
source reduction. Gate parameters join the existing native head-weight Adam
group. All blocks use one J backward and one original optimizer step; there
is no separate gate optimizer, teacher, backbone copy, VALID fit, capacity
fit or automatic combination with another extension.

Native MolHIV/MolBACE GCN/GIN source shapes are checked by the bridge, but this
source package alone does not qualify those tasks. Node/link native bridges
remain pending. Generic multiclass/binary tensors are not evidence of a
qualified native port. Native task engineering checks requested later must
live in separate qualification files with their own outcomes and costs.

## Runtime v2 method binding

Declare `family=ogb`, `base_procedure=gnnm_boundary_native`, `TEST_access=false`
and the desired exact `procedure` from `METHODS.json`. Set `session_transform`
to this package's `molecular_transform.py`, function `transform`, pinned by its
SHA256. Set `method_settings` exactly to `{manifest_sha256: <actual manifest
SHA256>}` and `recipe._repository_root` explicitly to the actual repository
checkout root. The source session's redesign root must be this package's
sibling phase redesign. Use the original callback/data/controller bindings;
this package defines no launch roster or training horizon.

`validate()` returns actual served outputs/metrics for selection. It also
returns `same_bank_mean_outputs/loss/metrics`, computed from the same four
route forwards, for a diagnostic readout comparison. These diagnostics never
fit a combiner or replace the actual served selection criterion.

Checkpoints have schema `conditional-aggregation-molecular-wrapped-state-v1`,
an exact method/config/seed/source/role binding, and the complete native
source state (including registered gate weights and per-module training
flags). Restoration preserves original Parameter objects. Cross-method
loads and active-route save/restore are rejected. The existing final TEST
backend does not reconstruct this wrapper. Do not strip it to force loading;
selected-state reconstruction remains explicitly unsupported/pending.

## Reproduce local engineering checks

Use the already available local Torch runtime, without installing packages:

```sh
PYTHONPATH=/Users/alex/.cache/codex-research-molecular-tensor-tests/torch251 \
/Users/alex/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
postsubmission_research_20260930/benchmark_expansion_20261015_v1/conditional_aggregation_v1/check_tensor_contract.py
```

The check script imports frozen source interfaces without constructing native
models or loading data. Its molecular harness uses synthetic embeddings and
a linear head under the actual frozen boundary hooks, a recording evaluator
stub, and a no-op Adam.step counter. It verifies gradient/aggregation/role/
checkpoint wiring; it computes no OGB score and performs zero optimizer steps.
Gate finite differences perturb gate parameters only. Native logit gradient
checks hold alpha fixed, respecting the declared stop-gradient features.
