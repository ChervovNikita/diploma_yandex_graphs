# Ten actual-input resource cases for root review

This is a disabled engineering proposal. It prepares source bindings and CLIs for root to review and execute. No CUDA execution, actual input decoding, model construction, optimizer update, scientific fit, heldout scoring or GO occurred during preparation. Existing source packets and original GNNM rows are unchanged.

## Finite scope

| Source | Cases | Chosen arm | Exact declared shape |
|---|---:|---|---|
| Common native wrapper v2 | 6 | shared `exchange` | GCN/GAT/SAGE × WikiCS/Photo, H128/L2/M4/rank16 |
| Q21 rotation successor | 4 | `shared4_paired_local` | GAT/SAGE × WikiCS/Photo, H128/L2/M4 |

One exact search row per study: LR .002/dropout .4, seed88101 for common wrapper or77101 for Q21, body0/mask0. These are maximum declared source shapes. All prepared masks have the same TRAIN row count: WikiCS580 and Photo160. No scientific outcomes were used to choose cases. The proposal includes zero already-qualified bootstrap controls, sign-only controls, ordinary/I4 controls or original GNNM cases.

Frozen packets:

- `common_native_wrapper_packet_20261012_v2`: canonical plan `6ca93c2e839ac0354d2594273a6b4ac78727ce0856ce38487bb91f7597003f96`.
- `private_graph_rotation_q21_successor_20261012_v1`: canonical plan `441e59ce9b2f63eafb3b04fb6477c8378ff1d3634cdb928e7d19ec999cc03b5e`.

`PROPOSAL.json` binds every actual frozen study, source run ID/hash, factory entrypoint/file SHA, source-import file, GRAPH/TRAIN receipt and custody metadata. The source-study roster contains 630 original GNNM fits, all retained: 378 common-wrapper and252 prior Q21. None are measured or changed here.

## Reused execution boundary

`qualify_wrapper_rotation.py` derives from the existing reviewed `qualify_exact_sources.py` SHA `8606ab9e6de8561e70ad6b383e46e1ae2cced66853d567688bcb0df091bb0e14`. Only its `node_probe` and `main` top-level function ASTs change. The probe changes work counters to follow the exact source native paths; main additionally binds frozen import receipts and restricts dispatch to the ten source families. `PROBE_REUSE_PROOF.json` records this delta.

The following frozen helpers and behavior are reused:

- h: `private_source_zero_update_helpers_20261012_v2/runtime/qa/practical_zero_update_resources.py`, SHA `effa70e31e64881322ccfb5a6e7e3903ccdb6f887c694a97f6b63d0bd2e267e0`.
- z: `private_source_zero_update_helpers_20261012_v2/runtime/qa/private_source_zero_update_resources.py`, SHA `259ea644b251012da75c7010746e231cc70243cfd462eaef9b04464350adab41`.
- Current-child allocator cap is `min(fresh_free−4GiB,0.4×total)`. No larger-cap retry is proposed.
- Exact concrete/functional optimizer denial, parameter/manual-moment TensorCommitDenial, actual GRAPH/TRAIN RoleIO, fresh manual-zero AdamW moments, and two additional same-shaped workspace tensors per parameter.
- One unused nonTRAIN public node and synthetic zero label satisfy the factory's SELECT interface. Original SELECT/CAL/evaluate bytes are denied, including hash reads. Selectors and evaluators are replaced with denial callbacks and never called.
- Each callback restores the initial complete adapter/private RNG, factor-member and module-mode state. Parameter and manual-moment hashes must stay equal; gradients must be finite.
- Exact full frozen factories/native source callbacks execute. There is no small-model substitution or shortcut of graph convolutions, normalization, FFNs or heads.
- One full-graph, label-free serving pass; actual logits/losses/predictions/scores are not emitted or stored.
- Existing frozen v5 engine captures/clones, atomically writes and verifies/restores a disposable random untrained step0 snapshot. The snapshot contains the full model, manual moments, private/global RNG and modes/selector state. It is removed on success. A four-route CPU bank made of dummy values exercises storage using only metadata CAL/evaluate counts; no real endpoint IDs or targets are opened.

## Work and cost observations

Each case proposes one cold plus three warm callbacks. Every callback has four complete native routes, eight native residual graph-block calls, one backward, zero VJPs and one intercepted source optimizer-step attempt. Actual updates and optimizer kernels stay zero.

Common exchange uses the source's four complete live native prefixes, one node-aligned coupled exchange block and four complete continuing tails. It makes zero calls to the native body's combined `forward` function. Independent hooks on the stem, both residual blocks and head verify the full paths. Paired-local uses four native body forwards, eight corrected dense-map calls and sixteen literal Householder applications. The Householder number is the exact source two-reflection formula per measured corrected-map call.

Totals for40 callbacks:160 complete native routes,320 graph-block calls,24 exchange-block calls,128 local corrected-map calls,256 Householder applications,40 backwards,0 VJPs and40 denied source step attempts. There are ten additional label-free serving passes, ten engine snapshot roundtrips and ten dummy CPU bank roundtrips. Each child has a90-second deadline; the deadline sum900seconds is a bound, not an ETA. No pair/concurrency cases are proposed.

`OBSERVATIONS_SCHEMA.json` specifies the measured callback/serving/constructor/serializer times, CUDA allocated/reserved peaks, cap and external GPU occupancy, host RSS, parameter/gradient/manual-moment/two-P workspace bytes, exact snapshot storage and roundtrip timings, dummy CPU retained/cat-transient geometry, input paths, state hashes, denials and failure custody. These random-state measurements do not measure AdamW kernels, actual learned selected states, quality, CAL/evaluation, concurrent-worker admission or full5000 runtime.

## CLI for root after exact source review

Stage this owned proposal directory using the existing literal guarded allocation SSH transport. Then run metadata staging, which imports no numerical libraries and opens no tensor arrays:

```sh
python native_wrapper_rotation_resource_proposal_20261012_v1/stage_metadata.py --project-root /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930 --source-directory native_wrapper_rotation_resource_proposal_20261012_v1 --output native_wrapper_rotation_resource_stage_20261012_v1 --root-reviewed
```

Omit `--root-reviewed` for a stage with zero approved cases. The reviewed flag is the existing root engineering-review marker; it does not make a scientific GO. The stager preserves a fresh metadata manifest and admission receipt, rechecks all source imports/helpers/custody/study/run identities, and binds the v5 engine/runtime. It does not dispatch workers.

For each exact `case_id` in the proposal, root may separately invoke the bounded helper with a fresh output directory:

```sh
python native_wrapper_rotation_resource_stage_20261012_v1/source/qualify_wrapper_rotation.py --project-root /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930 --admission native_wrapper_rotation_resource_stage_20261012_v1/ENGINEERING_ADMISSION.json --case-id common_native_wrapper-wikics-gcn-exchange --output native_wrapper_rotation_resource_results_20261012_v1/common_native_wrapper-wikics-gcn-exchange
```

The source helper verifies hostname `anogena-2-0` and the sole required UUID before project operations. Root owns engineering child custody and any resource transfer/admission decision. Every failure is retained with no automatic retry. Scientific fits and launch still require the separate comprehensive protocol/resource gates and explicit user GO.

## Local checks

Six static checks passed: ten exact unique source rows; all frozen source-import receipts and ASTs; work-count algebra; ten actual namespaced frozen-v5 factory resolutions; the two-function probe AST delta; and script syntax. No numerical tests were repeated. See `LOCAL_METADATA_PROOF.json` and the sealed `SOURCE_INVENTORY.json`.
