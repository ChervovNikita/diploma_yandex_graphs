# PubMed/Cheb fit admission candidate — pending CUDA evidence

At **2026-10-15 19:40:46 UTC**, the existing engineering queue remained
`waiting_for_existing_six_fits`: none of its seven children had started.
Owner PID849490/start6089402542 and older GRU peer
PID640428/start6055039165 were live. The guarded read-only observation is
retained in `OBSERVATION.json`. Its queue hash identifies that changing
observation, not a successful terminal qualification.

## Admission status and missing evidence

`FITS_ADMISSION_CANDIDATE.json` is **approved=false** and cannot launch.
It binds the frozen driver manifest
`a6c3bfadc459c2b2afe24e28c80e7365f5f9e0328241687aae66377f89d82d28`
(1,829 bytes) and SPEC
`ad026abb056bfc9e5c7f0c2e65c2e2535a848057d656d52b9fab47ace8a39ff3`
(34,270 bytes). Source and manifests were not modified. Root must add actual
successful terminal CUDA lifecycle receipts for uniform, structured and
permuted, plus production M1, G0, uniform and structured timing receipts.
The three already-passed CPU qualifications in SPEC are reused; no repetition
is requested. Publication, measured memory/time limits, evidence approval
and scientific approval remain absent.

## Controller and comparisons

The study contains 24 cold fits: 12 independent native bodies and three each
of original GNNM/G0, uniform, structured and permuted residuals. M1 reuses
independent body0; I4 averages all four own-selected bodies. Optimizer cohorts
101/202/303, split0 and source split seed1941488137 are fixed. All compared
arms use the same native recipe and matched controller: minimum2000 epochs,
patience100 after the minimum gate, cap10000, whole-VALID accuracy and first
strict maximum. The cap outcome remains unresolved convergence. This is a
generous common training opportunity, not a guarantee of convergence or an
exact replication of the source paper's2000-epoch/patience250 protocol.

The three cohorts are optimizer repetitions on one exposed development split.
VALID counts are [72,1919,1952]; class0 has72 nodes. Retain per-class/macro
harms and every arm, without changing the whole-VALID selector. This design
can test transfer of the fixed residual/structural mechanism and its matched
controls; it cannot establish protected confirmation or broad superiority.

## Cost and resource recommendation

Numeric CUDA costs and peaks are **unknown** until the queued checks finish.
`COST_RESOURCE_RECOMMENDATION.json` leaves every per-kind measurement and
time projection null. The frozen controller entails 48,000 whole TRAIN
updates at2000 epochs for every fit, or240,000 at the10000-epoch cap. Keep
each probe's constructor, TRAIN1, VALID1, TRAIN2 and VALID2 separately.
The conditional per-fit compute scenario is constructor plus E times a
measured TRAIN-plus-VALID pair, with E=2000 and10000. These are expected-cost
scenarios, not guaranteed execution bounds or a steady-state claim.

The real fit runner additionally clones/serializes recovery state every
epoch, saves each new selected checkpoint, and performs final selected VALID
serving. Those costs are excluded from the two-update production probes.
Do not set a watchdog equal to a raw update-only extrapolation. Permuted has
its own lifecycle but no queued production probe; any structured-arm timing
proxy must be labeled as such. The queue runs one new child alongside the
older GRU, so timings do not represent an exclusive GPU.

For later root admission, freshly authenticate the sole GRU peer above;
retain its32GiB GPU/16GiB host growth allowances and the existing one-new-worker
slot. The queue's32GiB GPU/24GiB host reservations are existing engineering
envelopes, not measured scientific fit requirements. Root should choose the
scientific envelope after seeing completed peaks, serialization overhead and
fresh cache-aware headroom. No third GPU model worker is proposed.

## Launch command after root creates an approved actual admission

The command below is a handoff template; it has not been executed. Root must
replace the two admission placeholders with a reviewed repository file and
its SHA, and bind the actual published checkout in that file.

```sh
/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/.venv/bin/python -B -u \
  /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/research/pubmed_cheb_transfer_driver_v1/driver.py \
  --project-root /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs \
  --mode fits --admission ACTUAL_APPROVED_ADMISSION_PATH \
  --admission-sha256 ACTUAL_APPROVED_ADMISSION_SHA
```

Use the existing guarded allocation route and outer resource/deadline owner.
All checks must finish successfully and release the slot before a scientific
fit can be admitted. No scientific launch, GPU work, source edit, Git operation
or 18.77 access was performed for this recommendation.
