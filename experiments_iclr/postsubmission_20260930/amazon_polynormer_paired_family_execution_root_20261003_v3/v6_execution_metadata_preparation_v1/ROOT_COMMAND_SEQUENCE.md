# Amazon V6 bounded qualification to the original 15 fits

All prepared JSON bodies are disabled. Root reviews this helper and independently publishes each admitted release at its exact `self_path`. No command below was run during preparation. No second registry or source claim is created. Original V5 source, successful qualifier, earlier failures, costs and once-only fit paths remain preserved.

## Exact route and files

Run remote commands only on `anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru:2222`. Root checks the repository and single UUID `GPU-44039938-fd82-41d2-fefd-de71514e2fac` before a launch.

```sh
AMAZON_REPO=/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs
AMAZON_PHASE="$AMAZON_REPO/experiments_iclr/postsubmission_20260930"
AMAZON_SOURCE="$AMAZON_PHASE/amazon_polynormer_paired_family_source_preparation_20261003_v6"
AMAZON_ROOT="$AMAZON_PHASE/amazon_polynormer_paired_family_execution_root_20261003_v3"
AMAZON_HELPER="$AMAZON_ROOT/v6_execution_metadata_preparation_v1/prepare_disabled_metadata.py"
AMAZON_REVIEW=amazon_polynormer_v6_bounded_retention_forecast_independent_source_review_20261003_v1/REVIEW.json
cd "$AMAZON_PHASE"
```

The exact review is SHA `5dcd5bb6ca727d14286b393ab0e1281e9b8990401b88e11f804d81430a850c68`, 8007 bytes. It passes source and forecast scope only. Its `PASS_SOURCE_ONLY` status is represented by the separately sealed compatible source-pass receipt because unchanged `common.gate` requires status `passed`. The compatible receipt is SHA `38141736a944162d333a3cb5c70f9bbee64fbafb94d2e4274db178faa7972946`, 2353 bytes. Set:

```sh
AMAZON_GATE_REVIEW=amazon_polynormer_v6_source_review_runtime_compatibility_receipt_20261003_v1/REVIEW.json
```

The helper checks exact source equality and `passed` before using it in runtime releases. Both receipts stay in custody. This does not admit numerical execution or resources.

## 1. Bind source review and prepare disabled runtime releases

`final_ready_disabled_v1` is already prepared locally with both exact review receipts. Root may inspect and upload that sealed packet directly. To regenerate metadata remotely, use an absent packet name, for example:

```sh
/usr/bin/python3 -I -S -B "$AMAZON_HELPER" --stage initial --packet-name root_inspected_initial_disabled_v1 --review "$AMAZON_REVIEW" --runtime-gate-review "$AMAZON_GATE_REVIEW"
```

Root inspects that packet's manifest/seal and `PLAN.json`. Its consumer rebind changes only source and its disabled authorization flag; all original preprocessing and ordered FIT/control/VAL receipts remain. Root publishes the approved consumer with `execution_authorized=true` at `amazon_polynormer_paired_family_execution_root_20261003_v3/V6_CONSUMER_RELEASE_v1.json`. Set:

```sh
AMAZON_CONSUMER=amazon_polynormer_paired_family_execution_root_20261003_v3/V6_CONSUMER_RELEASE_v1.json
```

Root separately approves the two runtime candidate bodies and publishes them with `execution_authorized=true`, the compatible source review and original exact interpreter descriptor at:

- `v6_releases/runtime_cpu_v1.json` → `v6_runtime_cpu_v1`
- `v6_releases/runtime_gpu_v1.json` → `v6_runtime_cuda0_v1`

Caps are prospective candidates: 900 seconds, 16 GiB RSS, 16 GiB CUDA allocated/reserved. Publishing a consumer/runtime release is a root action; this helper never flips execution flags.

## 2. Execute the two separately admitted runtime captures

Each output must be absent before its one physical attempt. Root launches the unchanged ordinary supervisor using system Python; the worker uses the admitted existing `.venv/bin/python` descriptor. Do not use `-I` for the supervisor, because its guarded local `common` import uses the source directory.

```sh
/usr/bin/python3 -B "$AMAZON_SOURCE/supervise.py" --kind runtime_capture --release "$AMAZON_ROOT/v6_releases/runtime_cpu_v1.json" --output "$AMAZON_ROOT/v6_runtime_cpu_v1"
/usr/bin/python3 -B "$AMAZON_SOURCE/supervise.py" --kind runtime_capture --release "$AMAZON_ROOT/v6_releases/runtime_gpu_v1.json" --output "$AMAZON_ROOT/v6_runtime_cuda0_v1"
```

Stop on any physical failure. Preserve the immutable failure receipt, trace, wall and memory costs; do not restart automatically.

## 3. Bind physical runtime successes and prepare the disabled qualifier

```sh
/usr/bin/python3 -I -S -B "$AMAZON_HELPER" --stage after-runtime --packet-name after_runtime_disabled_v1 --review "$AMAZON_REVIEW" --runtime-gate-review "$AMAZON_GATE_REVIEW" --consumer-release "$AMAZON_CONSUMER"
```

The helper requires exact successful V6 CPU/GPU freezes, zero physical exits, exact admitted release descriptors, full inventory and runtime device identity. It extends the prior-attempt registry with these actual costs and captures. It preserves the existing registry descriptor SHA `6423ae8d8abd605fc754a87c1cd617b50c401dfbbadac09c2642224f34bddea8` (12877 bytes) and the original master source claim.

Root reviews `after_runtime_disabled_v1/disabled_releases/qualification.json`, approves its prospective caps (3600 seconds, 32 GiB RSS, 75 GiB CUDA allocated/reserved) and publishes the admitted body at `v6_releases/qualification_v1.json`. Its consumer/runtime/attempt-registry references must remain the exact inspected descriptors. Then root may run:

```sh
/usr/bin/python3 -B "$AMAZON_SOURCE/supervise.py" --kind qualify --release "$AMAZON_ROOT/v6_releases/qualification_v1.json" --output "$AMAZON_ROOT/v6_qualification_block0_cuda0_v1"
```

This fresh five-form qualification includes native/family local/global full-state replay, selected-local restoration and owned-image retirement with canonical model/Adam/gradients/modes/stage/RNG isolation. It adds no predictive fit or selector. V5's successful qualifier cannot substitute for this V6 gate.

## 4. Bind fresh qualification; obtain actual resource admission

```sh
/usr/bin/python3 -I -S -B "$AMAZON_HELPER" --stage after-qualification --packet-name after_qualification_disabled_v1 --review "$AMAZON_REVIEW" --runtime-gate-review "$AMAZON_GATE_REVIEW" --consumer-release "$AMAZON_CONSUMER"
```

This stage still prepares disabled fit bodies. The source-reviewed forecast helper is pinned from `fit_schedule_resource_candidate_v2_v6_bounded_retention/prepare_v6_metadata_candidate.py`, along with its exact manifest/seal and passed review. Root must replace its explicit unmeasured fsync/retirement allowances using actual same-filesystem V6 evidence and obtain a fresh capacity/quota budget plus allocation lifetime/time budget. The earlier 53.369-hour / 80.054-hour-margin forecast and 32 GiB storage candidate remain planning evidence.

Root writes a separate `amazon_polynormer_resource_admission_v3` JSON bound to exact V6 source, fresh qualification freeze, original registry and measured full 15-fit envelope. It must explicitly authorize all 15 fits, cost local/final replays, and set `all_selected_checkpoint_replays_required=false`. Set `AMAZON_RESOURCE` to its PHASE-relative path. Then bind it:

```sh
/usr/bin/python3 -I -S -B "$AMAZON_HELPER" --stage after-qualification --packet-name admitted_resource_disabled_v1 --review "$AMAZON_REVIEW" --runtime-gate-review "$AMAZON_GATE_REVIEW" --consumer-release "$AMAZON_CONSUMER" --resource-admission "$AMAZON_RESOURCE"
```

No forecast/source/engineering pass authorizes fitting. The helper still leaves every fit and queue body disabled.

## 5. Root publishes the exact original 15 fit releases and closure release

Root approves the 15 bodies from `admitted_resource_disabled_v1/disabled_fit_releases` and publishes them with execution authorization at their original registered `self_path` values under `releases/fits/`. Outputs and claims remain the original `fits/` and `claims/` paths; seeds and recipe remain exact. No fit output or claim is created during release preparation. Publish the separately approved closure at `v6_releases/closure_v1.json` with output `v6_closure_v1`.

After the actual releases exist, bind their hashes into a fresh disabled queue:

```sh
/usr/bin/python3 -I -S -B "$AMAZON_HELPER" --stage after-fit-releases --packet-name queue_bound_disabled_v1 --review "$AMAZON_REVIEW" --runtime-gate-review "$AMAZON_GATE_REVIEW" --consumer-release "$AMAZON_CONSUMER" --resource-admission "$AMAZON_RESOURCE"
```

The helper verifies each published fit's ID, output/self path, source/review, original registry, runtime/consumer, qualification, resource and caps. It binds actual fit/closure descriptors into the queue, which stays disabled. Root reviews and separately publishes the admitted all-fits body at `v6_releases/all_v1.json` with output `v6_full_schedule_v1` before this command:

```sh
/usr/bin/python3 -B "$AMAZON_SOURCE/supervise.py" --kind all --release "$AMAZON_ROOT/v6_releases/all_v1.json" --output "$AMAZON_ROOT/v6_full_schedule_v1"
```

The unchanged runner executes exactly 15 fits sequentially, stops at the first physical failure, and closes only the complete 15-fit/9-family cohort. No automatic retry, replacement cohort, alias fit, TRAIN-control/heldout scoring or TEST launch is part of this sequence. Predictive evaluation requires a later separate complete-cohort root release.
