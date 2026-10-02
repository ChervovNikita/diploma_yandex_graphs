# Exact Squirrel17 native paired qualification preparation

One prepared script, **not executed by its author**: `prototype/qualify_squirrel17.py`. It qualifies the exact frozen Squirrel17 PolyFormer-Mono common warm state, complete 2,223-node/five-class output closure and 512-coordinate stem/head slice. It reuses pinned active Round17 v3 precision source primitives and the sealed paired helper. It adds no training, continuation, coordinator, registry or launch.

**343 stdlib/source/mock checks passed.** They exercised resource success/deferral and fresh output paths using fake source bytes and a mocked GPU probe, checked explicit input categories, and verified all 144 original source/metadata mirrors unchanged before/after. The seven numeric originals were excluded from author-side checks. No dataset/checkpoint/label array, actual GPU probe, remote call or native model was opened/executed by the author.

## Exact inputs and scope

`BOUND_INPUTS.json` contains 151 explicit read-only descriptors. Of these, exactly seven are numeric inputs: full features, canonical topology, the compact TRAIN pack, three unlabeled role-ID arrays, and the frozen warm checkpoint. The only `/labels/` path is `seed17_split0_train.npz`. Validation/final label arrays are neither bound, hashed nor loaded. Recorded label-descriptor digests inside immutable metadata are not recursively opened. The script explicitly refuses any other label-byte path before opening it and never calls the old context/phase/registry orchestration.

Warm checkpoint SHA-256: `e67a0ab44c966f4980c709d0256a10a4a6916daa7dc564a07c98fc1a867a57b2`. The old `execution_root_v2/study_v2` namespace binds the **active v3_precision** source/protocol, not the earlier Round17 v2 protocol. Sealed paired source SHA-256: `e3ff6f134bae639269f284c4d8b0d290a31672787fe662ee829033465ed147d0`.

Original sources, features/topology/TRAIN/role-ID bytes and the checkpoint are hashed before/after future execution. The author has only bound the recorded numeric digests, not rehashed those arrays. At execution, the original v3 source guard verifies its pinned source tree; `source_inputs(validation=False)` loads only allowed graph inputs, role IDs and TRAIN labels. Native preprocessing must equal the exact warm/certificate preprocessing. FP32 native weights/logits/gradients are retained; only per-example CE finite-difference mean/subtraction uses FP64 measurement, with original FP32 records retained.

## Future invocation after root review

The source packet belongs under the canonical repository research root. The process must already expose exactly the frozen UUID `GPU-44039938-fd82-41d2-fefd-de71514e2fac` through `CUDA_VISIBLE_DEVICES` and satisfy the pinned runtime environment, including `CUBLAS_WORKSPACE_CONFIG=:4096:8`. The script reads these settings and does not change them or other processes.

```sh
/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/.venv/bin/python -B /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/graph_paired_native_qualification_preparation_v1/prototype/qualify_squirrel17.py --run-name root01
```

`--preflight-only` emits the read-only fingerprint/resource receipt without runtime/array deserialization or native computation. The fixed-UUID `nvidia-smi` probe checks current total/free memory. Prior exact Squirrel17 scalar qualification costs give a conservative required free memory of **12,561,940,480 bytes** (11.70 GiB), a 60-second planning estimate and a 600-second planning budget. This is not measured paired-native timing/peak memory. Insufficient memory, unavailable/mismatched UUID mapping or planning-budget failure produces `resource_deferred`; an actual OOM also retains explicit resource deferral. No source/resource deferral rejects scientific merit. The budget is a preflight planning screen, not a process timeout implementation.

Each run writes only `runs/<new-name>/` in this new folder and refuses an existing run directory. Outputs are `QUALIFICATION.json`, the pinned ledger's `cost_trace.jsonl` and `interface_trace.jsonl` when native operations occur. Original inputs/checkpoints/cohort outputs/RNG files remain read-only. Restoring RNG and constructing models change only the qualification process's in-memory RNG; no state file is written. The restored optimizer is discarded without an update.

## Native assay sequence

1. Validate preparation/source/state/input fingerprints and resource preflight.
2. Load pinned runtime and TRAIN-only complete-graph inputs; restore the exact checkpoint without updates.
3. Construct fresh K1/K4 warm clones and run pinned native/common/four-route equality; bind the actual full output closure and verify FP32 shape, finite logits, canonical target order and exact TRAIN-row-order hash.
4. Run active `precision_qualification.py` with seed90017, retaining true full-output VJP/JVP and both fixed finite-difference epsilon diagnostics.
5. Normalize the canonical topology; construct seed80017 permutation and verify complete bijection plus exact sparse `Pi S Pi^T` equality, keeping feature/label order fixed.
6. Invoke the sealed four-arm shared-alpha helper on this exact closure. Retain joint failure without replacements.
7. Only on joint acceptance, construct a fresh K4 copy for each arm, verify its common warm outputs, install only that arm's admitted factors, and compare all four actual member outputs with four independent common-closure evaluations at the returned slices. Verify the in-memory native donor and original file fingerprints unchanged.

No validation/test score or label value is emitted. TRAIN CE/derivative/construction diagnostics are retained. Passing establishes source/AD/installation correspondence on this representative exposed warm state; it does not establish later quality, novelty, convergence or independent confirmation. Full training remains dependent on new prospective orchestration.

`COUNTERS_AND_FAILURES.md` explains attempted/scheduled/completed distinctions. The manifest/seal identify this prepared source snapshot, not execution admission.
