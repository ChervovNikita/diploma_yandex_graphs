# Artificial CPU metric-cost helper

Source-only implementation of the finalized `METRIC_PROXY_QUALIFICATION_PLAN_v1.json` (SHA-256 `55e0911bea889334336ba194dc97279b6953424b7084de686b0414a266102422`). No calibration or Torch/model/data/label/GPU/remote workload was executed by the author. The draft is unauthorized; this source seal grants no launch or scientific permission.

`metric_cost.py` is a filename entrypoint with stdlib-only bootstrap. It removes script/cwd/PYTHONPATH sources and validates a new request, exact local source manifest, finalized plan and dependencies, fresh root prior accounting, exact interpreter/argv/corrected supervisor, bounded outer START and all linked source hashes before importing Torch. It imports no phase package or scientific entrypoint. It extracts only the unchanged `full_split`, `from_statistics` and `label_free_reductions` function ASTs into a namespace with explicit `math`/`statistics` globals. Whole source-file, exact extracted-source-segment and location-independent AST hashes are recorded in `SOURCE_FUNCTION_BINDINGS_v1.json` and rechecked before compilation.

## Artificial workload

- CPU float32 member logits at Reddit2 `23699 × 41` and products `39323 × 47`, for one and four members; pooled logits are their arithmetic mean. Targets are fresh balanced cyclic artificial classes. Every case uses a fresh CPU generator with seed `20260930`.
- Agreeing pattern: every member argmax equals the artificial cyclic target. Different pattern: member k uses `(target+k+1) mod classes`, giving different argmaxes for four members. The single-member control uses the same offset rule; there is no member pair in that case.
- A CPU validation call verifies exact artificial pooled/member correct counts, pairwise counts and reconstruction by the unchanged `from_statistics`; all returned metric/count values are then discarded. No synthetic loss, accuracy, NLL, disagreement, targets or logits are written.
- Each of eight task/member/pattern cases receives two warmups per kernel and five paired measurements. Measured kernel order alternates. Only source-kernel timings/paired ratios, generation/shape/dtype metadata, validation pass status, pinned runtime, maximum process RSS and provenance are saved.
- Runtime is Torch `2.1.2+cu118`, four CPU threads, CPU float32 inputs, source float64 reductions, no CPU autocast. No GPU tensor, model, checkpoint or provider operation exists. The helper is bounded at 300 whole-process seconds, 2 GiB host RSS and 16 MiB metadata disk. RSS is a lifetime high-water mark, checked after defined work boundaries; it is not an external hard memory limit.

## Request and completed evidence

Root creates a new request inside this directory from `DRAFT_REQUEST_v1.json`, resolves `source_manifest` to the exact finalized local manifest and `root_prior_accounting` to a fresh root receipt, and explicitly sets `root_authorized_artificial_CPU_calibration=true`. Keep all scientific/real-data/heldout/GPU/retained-tensor permission fields false. The helper SHA, plan, CPU policy, exact caps, GPU allocation identity and exclusive output path are frozen in that request. Use `metric_cost.py --request <exact request> --supervisor <exact inner directory>` only under the pinned `bounded_run_v1.py` → corrected `run_authorized_v2.py --require-idle-gpu` → logged filename entrypoint. Torch is imported after admission.

Output schema is `gnnm-shared-message-artificial-CPU-metric-cost-v1`. Its `shape_bounds[task]["1" or "4"]` records worst measured full-kernel time, worst proxy time and worst same-repetition ratio across both patterns. `ARTIFACT_MANIFEST.json` binds exact START/RESULT bytes. Failed partial metadata is preserved and cannot qualify. The child cannot certify its own later TERMINAL: root must audit successful zero-exit bounded TERMINAL, unchanged request, all five inner evidence files, correct GPU UUID, source/runtime/extraction/live/output hashes and absence of FAILURE before calibration enters the ledger. Charge actual complete helper wall/disk into updated prior accounting.

## Excess-CPU accounting

The standalone stdlib `cpu_surcharge(shape_bounds, preflight_rows)` arithmetic expects all ten separately audited preflight report rows, with task/root-arm and their three `three_CPU_label_free_reduction_seconds` values. It is available for exact AST extraction by the root; it imports no source package and does not audit those external preflight chains itself.

For every task/arm, use m=1 for singles and m=4 otherwise:

`3 seeds × 64 calls × max(0, worst_full_split_seconds(task,m) − max(preflight_CPU_proxy_samples(task,arm))) × 1.25`.

The 64 calls per seed/cell are 60 validations plus four selected/final stored/new-forward metric replay calls. Negative excess becomes zero. Add the complete surcharge to fixed `source_protocol_runtime_audits` and name it in the root ledger. Retain the worst paired ratio as diagnostic calibration evidence; the additive time formula controls the surcharge. These artificial patterns and the 1.25 margin are an engineering calibration, not a statistical upper confidence bound or a qualification of scientific quality.

`verify_source.py` only compiles source and exact function definitions, checks the unapproved-request refusal, literal shapes/runtime/repetitions and stdlib surcharge arithmetic. It executes no numerical kernel and imports no scientific package. Preserve this generation after sealing.
