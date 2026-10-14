## Real-data smoke checks — 14 October 2026

Completed **167 distinct configuration checks** across the redesigned study:
56 node, 96 molecular and 15 link cases. Each uses real source TRAIN data,
two initial optimizer updates, complete VALID evaluation and checkpoint
save/reload. A third update is run and replayed; all four callbacks and all
four independently owned I4 optimizers are charged. TEST is not scored.

The latest seeded attempts report **94 strict passes and
73 failures**. The separate functional diagnostic confirms
146 cases; incomplete
legacy telemetry remains unknown. Floating replay failures stay failed in
the reports, even when initialization, updates and exact restoration succeed.
These checks establish execution evidence, not prediction quality or convergence.

The source audit corrected link bias scaling and limited NCNC output factors
to the top-level task scorer, retaining its native recursive completion.
Corrected link cases were rerun. All earlier outcomes and repair history remain.

A concurrent Physics Mono/private-ensemble-credit + NCNC/I4 probe passed. Its maximum sampled own GPU usage was 64.7 GiB, with at least 13.6 GiB device memory free. This qualifies the tested allocation pairing, not every combination or the 18.77 runtime.

**Full scientific launch remains disabled.** Complete epoch handling,
source/main selectors, independent I4 stopping/checkpoint ownership, selected
TEST serving and the executable three-GPU launcher still require integration.
The old v29 launcher remains retired; no full-campaign GO has been received.

- [Smoke methodology and limits](experiments_iclr/postsubmission_20260930/reference_wrapper_redesign_20261013_v1/smoke_20261014/README.md)
- [All configuration checks and retained failures](experiments_iclr/postsubmission_20260930/reference_wrapper_redesign_20261013_v1/smoke_20261014/publication/REPORT.md)
- [Remaining full-run implementation](experiments_iclr/postsubmission_20260930/reference_wrapper_redesign_20261013_v1/smoke_20261014/controller/FULL_RUN_GAPS.md)

