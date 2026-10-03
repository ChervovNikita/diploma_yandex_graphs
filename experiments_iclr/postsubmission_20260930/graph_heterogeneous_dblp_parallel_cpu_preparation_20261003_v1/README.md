# Five-worker DBLP CPU preparation

This separately sealed packet schedules the root-adopted 35-case development study. It does not authorize training. The root must issue a fresh execution release containing the scheduler manifest hash and an explicit positive worker wall budget. The template has `execution_authorized: false` and a zero wall budget, so it cannot launch.

## Unchanged science

Each isolated process executes one paired seed block: 131, 137, 139, 149, 151. All five processes run concurrently; each uses one Torch intra-op/inter-op thread and one OMP/MKL/OpenBLAS/NumExpr thread. Within each process the frozen order is native HGT, global BE, shared relation, CP, unrestricted, untied HGT, wider BE.

The scheduler imports the byte-bound sealed v2 `fit` and `paired_development` functions directly. It uses the original input reader, family builder, implementation and portable OneCycle state code. It does not rewrite optimization, validation scoring, early stopping, checkpoint selection or replay. The existing selected raw FP32 mean-logit predictor is unchanged. Qualification weights are never adopted as study fits. There is no epoch-zero selection, frozen-gate retuning, heldout scoring or temperature fitting here.

Each worker starts fresh models, runs the frozen 300-epoch maximum / patience-30 recipe, and retains the original v2 complete-state checkpoint, logits, selection and trace outputs. Process isolation preserves separate Torch/member RNG state across seeds. Exact runtime: Python 3.11.14, Torch 2.1.2+cu118, CPU only, CUDA hidden. This numerical path uses the native private HGT implementation and Torch scatter operations; it does not use DGL.

## Closure and custody

Every admitted study retains exactly 35 ordered terminal slots, including failed/deferred or missing workers. Controlled SIGINT/SIGTERM and KeyboardInterrupt paths reap owned workers, reconcile available journals, preserve attempted-arm flags and close comparison. Only all 35 selected fits with checkpoint replay and bindings verified, successful worker exits and preserved original sources can produce the original v2 complete development summary. An incomplete cohort is never compared as a successful subset. Raw malformed worker receipts remain evidence; their corresponding slots become failed terminals.

Each selected terminal binds the absolute expected seed/arm path, SHA-256 and byte count of all four files:

- `selected.pt`
- `selected_member_logits.pt`
- `SELECTION.json`
- `TRACE.jsonl`

The controller verifies these descriptors after worker closure, including the selected artifacts of incomplete cohorts, before calling `paired_development`. It checks terminal metadata against the byte-bound original `SELECTION.json`. Sources, inputs, root freeze, release and scheduler payloads are verified before/after execution and immediately before comparison. A root analyzer can verify all 35 descriptors remotely before opening logits. The transport helper returns dispatch metadata only and does not copy tensors or labels locally.

Output is `runs/<released-name>/seed<seed>/<arm>/...`. Worker stdout/stderr, exit/resource records, original arm-start/terminal receipts and worker preservation results are retained. `PARALLEL_STUDY.json` binds every terminal and the all-or-nothing development closure. The canonical v2-compatible `STUDY.json` carries `rows`, the exact original `summary`, `admission`, `original_inputs_verified_unchanged` and `final_labels_closed`. Its summary is unscored/incomplete unless all closure and preservation requirements pass. `STUDY_STARTED.json` prevents automatic replacement/restart of this study or a prior original v2 study.

## Resources

The bound CPU qualification covers all seven full-graph families at the adopted geometry. The sealed qualification launcher applied an 8 GiB `RLIMIT_AS` in the child's `preexec_fn`, before Python/Torch import, together with its 600-second CPU/wall limits. Its retained `REMOTE_CODE.txt` matches that sealed launcher. One disposable TRAIN update and score-free full-target evaluation per family took 62.61 seconds, with process peak RSS 5,520,498,688 bytes (5.1414 GiB). Five times this measured single-process peak projects 27,602,493,440 bytes (25.7068 GiB); this aggregate is a projection, not an observed concurrent run.

Each worker is limited to 8 GiB address space and observed RSS, with its root-released wall budget. The controller requires at least 40 GiB current host/cgroup headroom, five available CPU affinities and a one-minute load no higher than 75% of affinity count. Its source-qualified maximum-cap forecast is 4.6488 hours with five seed workers; the factor-two planning estimate is 9.2976 hours. These timings are forecasts, not guaranteed bounds or GPU measurements. Resource deferment makes no scientific merit decision.

## Preparation checks and launch helper

`scheduler_fixtures.py` uses only stdlib subprocesses and synthetic text artifacts. Its thirteen checks cover five-worker completion/order and exact v2 summary, failed/partial/nonzero workers, malformed/duplicate/order receipts and attempted flags, spawn failure, timeout, four-file tampering/byte counts/path scope, replay/binding guards, actual v2 release/qualification transport admission, controller preservation before/after comparison and canonical study interface, KeyboardInterrupt recovery, actual SIGTERM child reaping, and transport release/inventory checks with SSH mocked. No Torch, real labels, tensors or training main are used.

After sealing, `deploy_cpu_remote.py --stage-only --receipt <fresh-path>` stages every sealed payload plus manifest/seal through the authorized one-GPU SSH account. `--admission <root-release.json> --receipt <fresh-path>` also validates the real sealed admission guard and dispatches a detached CPU controller. It verifies the canonical repository and exact single GPU UUID with a read-only inventory query. Current Git HEAD is recorded, while byte-bound source identities determine admission. Existing packet bytes cannot be overwritten with different content. No environment/package install or GPU computation is performed.

Neither staging nor dispatch has been performed for this preparation. Root review and explicit execution release remain prerequisites for the study launch.
