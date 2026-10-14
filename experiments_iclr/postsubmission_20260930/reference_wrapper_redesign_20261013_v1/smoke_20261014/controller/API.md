# Actual-source 2-update smoke API

This controller performs short engineering training on authentic source data. It is not a predictive-quality experiment, a convergence result or GO for the full campaign. It never generates replacement features/edges/labels. Root stages source callbacks and the manifest on the authorized allocation and owns remote execution; no 18.77 operation is part of this package.

## Manifest

A JSON manifest has `cases` and optional `expected_case_count` (167 for the adopted full smoke roster). Each case:

```json
{
  "id": "unique-preset-and-procedure",
  "recipe": {"id": "actual-preset-id", "dataset": "actual-source-dataset"},
  "procedure": "native-or-eligible-wrapper-or-independent4",
  "callback_module": "project/relative/family_callback.py",
  "callback_name": "create_session",
  "callback_sha256": "actual-sha256-after-source-staging",
  "data_root": "project/relative/authentic/source/data",
  "device": "cuda:0",
  "expected_optimizer_owners": 1,
  "rtol": 0.0001,
  "atol": 0.000001,
  "torch_threads": 2
}
```

`recipe` is the actual source-bound parameter dictionary supplied by root, not the abbreviated illustration above. Source slots/caps/data bindings remain the recipe's responsibility. The controller explicitly binds Python/NumPy/Torch CPU/CUDA global RNG immediately before factory construction using `recipe._seed` (fallback `recipe.seed`, then `case.seed`); a missing seed fails instead of silently choosing one. NumPy uses the declared seed modulo2^32. Per-native I4 offsets remain the callback's responsibility and must be recorded. The adopted count is **140 unique single-bank cases** (8 presets×6 procedures +16×5 +3×4), plus **27 genuine independent4 bank cases**, totaling **167**. Source-horizon and longer-track training share this 2-update callback only where their initial operator/optimizer/schedule state is explicitly identical; the full horizon, stopping and later source transitions are not thereby tested.

## Callback

```python
create_session(recipe: dict, procedure: str, data_root: pathlib.Path,
               device: str) -> session
```

Required session interface:

- `model`: actual `torch.nn.Module`; independent4 uses `ModuleList` of four complete native models.
- `optimizers`: list/tuple of real optimizer owners. A single `optimizer` alias is supported; a facade's `.optimizers` is unwrapped when supplied through this alias.
- `train_update() -> dict`: performs actual source TRAIN work and returns finite `loss` (or `objective`/`ce`) plus optional `operation_counts`. A native epoch may contain multiple maps; every actual optimizer `step` is observed and charged. Do not replace the source update with a dummy batch/linear surrogate.
- `validate() -> dict`: returns finite validation `loss` and actual `outputs` tensor/tree from TRAIN/official validation only. No heldout scoring is added. The controller records output metadata/hash and engineering replay checks, not publishable accuracy.
- `state()`: complete plain checkpoint state: model, all optimizers, source scheduler/scaler, training phase/clock, native private RNGs/counters/plain mutable tensors. Global Python/NumPy/Torch CPU/CUDA RNG is independently captured by the controller.
- `load_state(state)`: restores into the **existing** model/optimizer objects. Do not reconstruct/rebind Parameters or optimizer owners during reload. Model modes and `requires_grad` flags are separately restored by the controller.
- `source_parameter_references`: mapping from original source names to actual original Parameter objects captured before wrapper installation.
- `replaced_source_parameter_names`: must be empty for this reference-wrapper campaign. All original native Parameter objects must remain; arbitrary replacement allowances are rejected.
- `allow_missing_grad_names`: preferably a mapping of exact current parameter names to source-specific reasons. Inactive local/global heads and source-delayed branches may be identified from actual source logic. Unexplained missing gradients produce a retained diagnostic failure; all gradient exceptions must be exact names, not a blanket ignore.

Optional:

- `comparison_state()`: extra resume-critical numerical/private state not already in model/optimizer `state_dict`. Exclude wall-time/telemetry fields. The **full** `state()` is still saved/reloaded even when this concise comparison view is used.
- `source_metadata` (or `metadata` fallback): full-graph/batch/role/boundary bindings. A callable metadata method is evaluated before recording its returned receipt dictionary; a method's string representation is not evidence.
- `data_provenance`: actual dataset/input manifest receipts and scope metadata. Without it, the record contains the bound source callback and data_root but flags missing additional tensor-provenance metadata.
- `close()`: source-owned cleanup.

### Independent4

Set `is_independent4=True`, `bank_rule='mean_logits'`, and use four separate optimizer owners. Validation must provide `member_outputs` with leading dimension4 and `outputs` equal to its mean over that axis; alternatively `outputs={'members': ..., 'pooled': ...}`. Link scores follow the same raw-score mean. The controller tests actual aggregation, parameter-storage disjointness and four own clocks/optimizer states; it does not accept a shared model relabeled as I4. Every member performs real source updates, so one I4 callback costs four native bodies.

## What executes

1. Construct the authentic source session; record parameter identities, optimizer owners, construction time/RSS/CUDA allocation.
2. Real TRAIN callbacks1 and2; finite loss/model/optimizer/gradient checks, actual per-owner optimizer calls and state clocks.
3. Validate; save `after_two_updates.pt` containing the complete source state plus all controller RNG and module modes/flags. Record checkpoint size/SHA/time.
4. Run actual third callback uninterrupted and validate; retain compact comparison state/output summaries.
5. Reload the two-update checkpoint and RNG into the same session. Immediately compare the complete restored session state, tensors, scalar clocks, module modes, gradient flags and global RNG with the saved values. The expected session-state copy is isolated before the source loader can mutate its input. Require exact restored values before any resumed training work. Then run the same third callback again and validate.
6. Keep prediction/output and loss **rtol1e-4 / atol1e-6** coordinate criteria unchanged. For post-update floating model/optimizer/private-state tensors use the norm-relative criterion **RMS(delta) <= atol + rtol * RMS(resumed reference)** per tensor; shapes, dtypes, integer tensors and nonnumeric state stay exact. Record both RMS and maximum coordinate differences, policy version and controller-source receipt. Require equal global RNG advancement. This allows ordinary sparse-GPU reduction differences after an exactly restored checkpoint without rewriting an earlier failure or relaxing predictions/loss.

The controller records callback `operation_counts`, falls back to native `work`, or preserves nested I4 `member_updates`/other callback work fields. Missing old source-operation records remain unknown. Compact official VALID scope/count/work receipts are saved beside output tensor receipts; validation metrics are not promoted to scientific accuracy results. The collector separately reports functional update/save/reload completion and strict replay outcomes. A replay failure remains failed even when its four real callbacks completed.

This executes **four actual training callbacks per case** (1,2,3, replayed3), not two uncharged callbacks. Every observed optimizer map—including four independent owners and native multi-map units—is recorded. Validation/posterior draws, serialization and replay costs are included in wall/resource records. Failures preserve logs, partial records and any checkpoint; there are no automatic retries or discarded failure outcomes.

## Entry points

Sequential fresh-process cases avoid global native module/RNG/default-dtype leakage:

```sh
python controller/dispatch_smokes.py --manifest ACTUAL_SMOKE_MANIFEST.json \
  --project-root PROJECT_REPOSITORY --output FRESH_SMOKE_OUTPUT \
  --python PROJECT_REPOSITORY/.venv/bin/python
```

A selected subset uses `--only case-id ...`; each code/source repair needs a fresh evidence output, preserving the previous failure.

After root inspects actual sequential peaks, test two chosen heavy configurations simultaneously on the **same allocation GPU**:

```sh
python controller/pair_memory.py --manifest ACTUAL_SMOKE_MANIFEST.json \
  --project-root PROJECT_REPOSITORY --output FRESH_PAIR_OUTPUT \
  --python PROJECT_REPOSITORY/.venv/bin/python --cases CASE_A CASE_B
```

Both source sessions construct and wait at an evidence-directory barrier, then perform their real smoke work together. Own worker PIDs' GPU memory and device total/free are sampled; per-child CUDA allocated/reserved peaks and process RSS remain in their case records. No other user's process details are stored or changed. Sampling is a physical observation, not a complete mature-training memory upper bound. The pair probe terminates only its own children on failure/timeout.

CLI entry points verify hostname `anogena-2-0` and the sole authorized GPU UUID before reading the source manifest or constructing models. This is a route check, not a filesystem namespace/mount guard. All intentional manifest/data/evidence paths stay inside the repository.

## Verification status

Controller source syntax is checked locally without importing PyTorch or any family module. No synthetic model/data fixture substitutes for the requested real tests. Actual source smoke results are pending root's staging/execution and must be reported per configuration, including failures. Full-campaign convergence, later native phase/scheduler transitions, horizon stopping and dataset-wide scientific performance are not certified by this short test.
