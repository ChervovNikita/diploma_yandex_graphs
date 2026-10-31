# Every-fit native smoke

The user's 16 October instruction requires real engineering iterations for **every one of the 6,460 fits**, fixes for failures, and all queue owners in tmux before full training. This package supplies the per-fit child and exact coverage validator. The separate `campaign_execution_20261016_v1` owner handles tmux, shared GPU custody, current resource admission, failure attempts, and the global training barrier. This source package starts no remote jobs by itself.

## Interfaces

Run metadata preparation once per deployed host, using its qualified environment and exact `BENCHMARK_HOST_CONFIG`, `BENCHMARK_HOST_CONFIG_SHA256`, and physical UUID `CUDA_VISIBLE_DEVICES`:

```text
python -B prepare.py --project-root ROOT --release RELEASE_UNADMITTED.json \
  --release-sha256 SHA --candidates CANDIDATES.json --candidates-sha256 SHA \
  --output FRESH_INVENTORY.json
```

Preparation verifies the source closure once, joins the actual deployed descriptors to immutable canonical tasks and reporting profiles, and writes schema `all-fit-smoke-inventory-v1`. `fits` contains exact `id`, `worker`, `descriptor`, `runtime_entrypoint`, `runtime_job_hash`, canonical hash, science hashes, `preset`, family/recipe/procedure/seed/split/body, and deterministic `smoke_output_relative=sha256(id)`. There are 2,154 allocation fits and 2,153 per server51 GPU. `canonical_fit_ids` lists all 6,460. Rows interleave each worker's cell/method/protocol groups before repeated seeds; every fit remains present. Source/data closures are not repeatedly hashed by smoke children.

```text
python -B run_one.py --project-root ROOT --inventory INVENTORY.json \
  --inventory-sha256 SHA --fit-id EXACT_ID \
  --output ROOT/.../smokes/FIT_SHA/attempts/0001 --updates 2 \
  --wall-seconds 1800
```

The output directory must be fresh; its parent may exist. Every attempt receives durable `RESULT.json`, updated after native callbacks and finalized as `PASS` or `failed`. Any failure returns nonzero. A new attempt uses a new directory and preserves earlier evidence. Each child needs a fresh interpreter. The queue launches it using the descriptor's qualified Python environment, retains its process exit status, and owns resource limits and termination. `--keep-checkpoint` retains a successful reload payload if desired.

## Actual work

The exact descriptor is loaded without editing its policy, data, seed, split, model, callback, optimizer or native minibatch. The matching deployed native runtime validates it and constructs the actual adapter. The existing optimizer observer checks finite gradients. A separate pre-step cap rejects a third transition and accommodates the different existing observer signatures.

`adapter.train_epoch()` executes original native code. The runner wraps the actual `adapter.session.train_update` callback to capture two finite returned losses and stop **before a third native callback**. Full-batch node jobs therefore execute two native epochs; molecular/link jobs execute two original minibatches and no extra epoch to finalize the smoke. Native cleanup runs through the original `finally` blocks. No successful update is fabricated and no representative class replaces a fit ID.

One complete native VALID call checks finite loss, metrics and outputs. The smoke then writes a separate `all-fit-smoke-only-state-v1` payload, reloads its serialized bytes, perturbs one actual model parameter and one initialized optimizer moment, invokes the original `adapter.load_state`, restores RNG, and compares the entire saved adapter state exactly. This adds no optimizer update or epoch. The payload explicitly cannot resume a scientific fit. A successful temporary payload is removed after the reload check; failed attempts retain any payload. Its byte receipt and restoration evidence remain in `RESULT.json`.

`RESULT.json` binds inventory, descriptor, runtime, source manifest, exact fit/worker hashes, two attempted/completed optimizer transitions, two finite TRAIN losses, finite VALID, and checkpoint restoration. Smoke state is never used to initialize full training. No TEST call or scientific quality claim is made.

## Global coverage barrier

Each queue emits schema `allfit-worker-smoke-summary-v1` with `worker`, `status=PASS`, `inventory` receipt, `expected_fit_ids`, `fit_count=N`, `TEST_accessed=false`, and `results=[{id,runtime_job_hash,attempt,result_receipt,result}]`. The earlier agreed `all-fit-worker-smoke-summary-v1` shape with `counts={expected:N,passed:N}` is also accepted. `result` embeds the complete child JSON; `result_receipt` binds its actual serialized bytes. A queue must require child exit zero as well as terminal PASS.

The local/root bundle has exactly three workers:

```json
{"workers":[{"worker":"allocation/gpu0","inventory":{"path":"LOCAL_COPY","bytes":0,"sha256":"SHA"},"summary":{"path":"LOCAL_COPY","bytes":0,"sha256":"SHA"}}]}
```

Add the two server51 entries using the same shape and real receipts. Local copied file paths may differ from embedded remote paths; sizes and hashes must agree.

```text
python -B validate.py --bundle BUNDLE.json --output GLOBAL_SMOKE_PASS.json
```

The validator verifies all three inventory/summary copies, reconstructs each embedded child receipt, rejects stale `running`, failed, missing, duplicate, foreign or incomplete records, and requires exactly 6,460 unique canonical IDs and 12,920 real completed updates. Output schema `all-fit-global-smoke-pass-v1` includes `status`, `fit_count=6460`, `inventory_sha256s`, `worker_summary_sha256s` keyed by worker, exact `coverage`, sorted `canonical_fit_ids`, and the canonical ID-set hash. The owner checks its current inventory, local PASS receipts and summary against this immutable barrier before launching full fits under tmux.

## Local verification limits

`python -B test_source.py` executes only scalar fixtures and exact extracted observer/OGB epoch control flow. It checks all three observer signatures, full-batch/minibatch stopping, original OGB iterator cleanup, hard optimizer limits, failure costs, and exact 6,460-ID coverage refusal cases. No Torch, NumPy, model, dataset payload, GPU, remote operation, Git change or native smoke was executed during source preparation. Actual per-fit PASS is established only by the deployed queue.
