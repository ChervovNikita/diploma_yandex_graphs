# Normalized campaign dispatcher successor

This source extends the frozen `portable_dispatcher_v3` dispatcher to consume
the completed normalization migration. It preserves **6,460 proposed fits**:
2,890 exact composed replacements and 3,570 unchanged fit candidates, plus
270 unchanged native I4 export tasks. It creates a new candidate file,
canonical plan, global ownership file and allocation release. Every task
remains unadmitted. Original paper experiments are not added or rerun;
original GNNM remains a baseline on the new compatible datasets.

## Fixed inputs and integration

`INPUTS.json` binds the actual normalization `generated_v1/PREPARATION.json`
(6,053 bytes; SHA `9d4a9c494b8b6e83d2aeaf4480c30e9628e5ad23c2adb7695bf9bc0c7134b6c7`).
The optional explicit reporting argument supports only the actual rebound
`generated_v1/INDEX.json` (21,008 bytes; SHA
`cd2bc64047e630452eb7b4ce13a0829b205327876c5183d7e28853f6d4445db6`).
Its 27 profiles are checked against all 6,730 candidate/task identities,
including fit descriptors, runtime hashes, entrypoints, output paths, and
ordered native I4 dependencies. Its exact four-key hook is
`{bridge, entrypoint, registry, manifest}`. The reporter manifest is
`e7056159aa1e4f88c0b71595a3822edf687bed9b0e1c472f84af88dee7161420`;
the repaired selected scorer is
`af692038bec60bbb54e9ed45d659289af98a0cb844ab0637c8923ee6ce70b67b`.
The registry stores its scorer pin at the top level, with a bound-source
declaration in `normalization_rebind`; the launcher checks that actual schema.

The composed scientific JOB is used unchanged on the allocation. Its parent
execution descriptor is retained as explicit ancestry, and the same-allocation
binding cannot relocate it to 18.77. Unchanged task cohorts, outputs, workers,
and native dependencies retain the existing ownership namespace. Preparation
never invokes host/task claim operations. Source closure explicitly includes
the genuine producer, helper runtime, normalization source, reporter, profiles,
and repaired scorer. No checkpoint, tensor, dataset, or TEST payload is opened.

`contracts.py` changes only canonical-plan authority, composed job/runtime
binding, and successor source closure. The original eight gate names,
native resource checks, interpreter checks, human GO/readiness checks and
custody requirements remain. The old engine, backend and host ledger are reused.

## Metadata preparation on the allocation

After root review and immutable staging, run the following once with the
reviewed manifest SHA and a fresh direct-child output directory. The entrypoint
performs the literal authorized hostname/GPU guard before project reads.

```sh
python3 -B /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/normalization_portable_dispatch_v1/prepare_projection.py \
  --source-manifest-sha256 REVIEWED_SOURCE_SHA \
  --output /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/normalization_portable_dispatch_v1/generated_v1 \
  --reporting-index /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/benchmark_reporting_normalization_rebind_v1/generated_v1/INDEX.json \
  --reporting-index-sha256 cd2bc64047e630452eb7b4ce13a0829b205327876c5183d7e28853f6d4445db6
```

Omitting both reporting arguments produces metadata with an explicitly pending
reporting hook; the production launcher refuses such a release. Existing output
directories are never overwritten. Partial-generation failures receive
`FAILED.json`. The source package contains no generated release or claim.

## Evidence and execution status

`SOURCE_CHECKS.json` records stdlib-only scalar metadata tests, including a
6,730-record projection and a full 27-profile reporting coverage fixture.
`READ_ONLY_SCHEMA_INSPECTION.json` records the one bounded, guarded allocation
metadata inspection used to check the real index/profile shape. Neither is
native model, convergence, resource-pair, TEST, or scientific accuracy evidence.

All resource/binding admissions and human GO remain absent. The proposed maximum
is two children per physical GPU; no reciprocal pair measurement or admitted
concurrency is claimed. Current metadata assigns every cohort to allocation GPU0;
18.77 remains disabled. Native composed producer/selected-serving applicability,
observer witness, existing claim/checkpoint custody, mature resources and pair
admission, release review, and exact human GO still require their existing
evidence. The preparation receipt retains inherited dependencies without
automatically discharging their gates.
