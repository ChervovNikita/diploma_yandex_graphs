# PubMed/Cheb measured engineering handoff

All **seven existing CUDA checks passed** by the19:46:05UTC observation:
uniform/structured/permuted lifecycles followed by production
M1/G0/uniform/structured. Every child exited0 without timeout or resource
violation. Their combined child wall was93.324seconds, excluding the owner's
earlier wait. This is engineering cost, not scientific training or accuracy.
Terminal queue:258,888bytes, SHA
`f269d073b59856458b77c8d483a8618f3d8dba9a2bfecf6f3bb1c7d8137df92c`.
Exact individual receipts and full engineering metadata are retained in
`ACTUAL_ENGINEERING.json` and the closed candidate.

## Lifecycle and source

Each residual lifecycle completed two optimizer transitions, three complete
VALID calls, native-equivalent initialization, changed-state rejection and
bit-exact fresh restore. The production probes used exactly one cold
constructor, two whole TRAIN updates and two complete VALID calls apiece;
M1/G0 used two production reverse calls and each residual used four.
Production descriptors join the frozen transfer driver's exact seed101
descriptor hashes. No selector, scientific initialization or TEST evaluation
occurred. Earlier passed CPU authorities are reused without repetition.

The frozen scientific source remains unchanged: driver manifest
`a6c3bfadc459c2b2afe24e28c80e7365f5f9e0328241687aae66377f89d82d28`
(1,829B), SPEC
`ad026abb056bfc9e5c7f0c2e65c2e2535a848057d656d52b9fab47ace8a39ff3`
(34,270B).

## Conditional cost scenarios

The table uses each arm's constructor plus2000or10000 times the observed
minimum/maximum of its two synchronized TRAIN+VALID pairs. Native means one
independent body; the24-fit total contains12 such bodies, three G0 and three
of each residual. These intervals describe two-sample extrapolations, **not
confidence intervals, steady-state measurements, guaranteed bounds or ETA**.

| Kind | TRAIN+VALID pair1 / pair2, seconds | 2000-epoch compute, minutes | 10000-epoch compute, minutes |
| --- | ---: | ---: | ---: |
| Native body |0.453 /0.245|8.2–15.2|40.9–75.6|
| Original GNNM/G0 |1.007 /0.873|29.1–33.6|145.5–168.0|
| Uniform residual |1.527 /1.388|46.3–51.0|231.4–254.5|
| Structured residual |1.398 /1.469|46.7–49.0|233.1–244.9|
| Permuted residual |structured proxy; unmeasured|46.7–49.0|233.1–244.9|

For all24serial fits, the conditional compute totals are10.08–12.16hours
at2000epochs, or50.34–60.73hours at10000. The retained selector cannot stop
earlier than2099epochs: staleness starts at2000and must reach100. That
earliest-stop scenario is10.58–12.76hours before additional runtime costs.
Both production update measurements remain separately recorded.

The real fit runner CPU-clones and serializes recovery state every epoch,
saves new selected checkpoints and performs final selected VALID serving.
The probes exclude those costs. They also ran alongside the allowed older
GRU, rather than on an exclusive GPU. Permuted has its own passed CUDA
lifecycle but no separate production timing; its structured proxy is explicit.

## Resource and deadline proposal for root

Maximum measured CUDA reservation was4.2266GiB across all lifecycles
(production maximum2.9004GiB); maximum host RSS was1.3078GiB. Proposed
scientific envelope: **one new worker,8GiB GPU,8GiB host,8GiB disk and two
CPU threads**, with a **43200-second/12-hour per-child watchdog**. The
watchdog is a conservative policy proposal, about2.8times the slowest
10000-epoch compute projection, not a measured runtime ceiling. Neither
the memory envelope nor deadline is approved by this handoff.

Root must freshly verify the shared slot, cache-aware host/GPU availability,
and older GRU PID640428/start6055039165, retaining its32GiB GPU/16GiB host
growth envelopes. No third GPU model worker is proposed. Admission must wait
for other current slot owners to finish. This report makes no fresh resource
availability claim.

## Scientific adequacy and admission

The fixed three optimizer cohorts101/202/303 compare native M1, genuine I4,
original GNNM/G0 and uniform/structured/permuted residuals. M1 is exact I4
body0; every body retains its own selected checkpoint. Every arm uses the
same recipe and matched minimum2000/cap10000/patience100 controller with
whole-VALID accuracy and first strict maximum. Native operators, Adam groups
and source split remain fixed. A cap hit means unresolved convergence; this
is not an exact source-paper protocol replication or a guarantee of convergence.

This is an exposed development transfer test on one split, with only72class0
VALID nodes. Preserve all arms, per-class and macro harms and three-seed
outcomes. It can test the frozen residual mechanism and its matched controls;
it cannot establish protected confirmation or broad superiority.

`FITS_ADMISSION_CANDIDATE.json` now contains all actual CPU/CUDA lifecycle
and production receipts, finite proposed resources/deadline and existing peer
identity. **approved=false**; publication/evidence/resource approvals remain
false and publication_commit remains null for root review. No source,
manifest, admission or launch on the server was changed.

After root creates and hashes a separate approved admission, the exact command
is:

```sh
/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/.venv/bin/python -B -u \
  /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/research/pubmed_cheb_transfer_driver_v1/driver.py \
  --project-root /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs \
  --mode fits --admission ACTUAL_APPROVED_ADMISSION_PATH \
  --admission-sha256 ACTUAL_APPROVED_ADMISSION_SHA
```

Root owns publication, actual admission, outer resource/deadline supervision
and launch. This agent performed guarded read-only metadata retrieval and
local recommendation work only; no scientific fit, GPU computation, remote
write, Git operation or18.77access occurred.
