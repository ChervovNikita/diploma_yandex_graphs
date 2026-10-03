# V6 bounded-retention disabled resource and queue candidate

Status: prepared metadata only; execution remains disabled.

The unchanged original 15-fit cohort has 40,500 scientific optimizer updates and 64,800 member-trajectory updates. The revised audit scope adds 60 optimizer updates and 96 member-trajectory updates, for totals 40,560 and 64,896. Native-single aliases add no fits. All 2,700 possible strict-improvement snapshot/hash/write events per fit remain charged. Only retained selected-local and final-best states receive next-step replay. Superseded checkpoint bodies are retired with exact descriptor/selection/snapshot-cost and fsynced intent/completion custody; every score and selector decision remains.

## Forecast

- All 15 fits and metadata closure: 53.369 hours; 80.054 hours with 50% margin.
- Retained checkpoint bodies: 4,405,455,504 bytes, at most two per completed fit. One new-image temporary is charged for the sequential queue.
- Incremental free-space estimate: 18,391,721,268 bytes including 25% checkpoint margin, 8 GiB metadata, 1 GiB active/final reserve and 3 GiB fresh qualification reserve. Proposed budget: 32 GiB.
- Candidate per-fit caps: 43,200 seconds; 32 GiB RSS; 75 GiB CUDA allocated/reserved.

The 0.1-second trace-fsync allowance per update and 0.25-second journal/directory-fsync/unlink allowance per retired image are planning assumptions, not measurements. Actual V5 train-plus-VAL, snapshot/write and portable replay intervals are preserved. Safe verify/load and train-plus-VAL intervals are explicit hash/inference charge proxies; no standalone hash/inference measurement is claimed. Fresh V6 retirement timing and resource measurements must replace the assumptions before admission. The scenario counts 1/10/50/100/500/2700 describe possible selection histories and do not predict quality.

## Pending gates

Independent exact V6 source review; exact V6 runtime receipt and consumer release; fresh exact five-form numerical qualification including retirement/full-state/RNG isolation; same-filesystem resource measurement and storage budget/quota; allocation lifetime or user time budget. Historical V5 evidence remains preserved and is not substituted for a current V6 gate.

All 15 fit candidate bodies bind the exact registered paths but are stored only here. Their execution/review flags are false and fresh V6 descriptors are null. The queue stops at the first physical failure, with no retry or replacement cohort. No registered release, output or claim was created. No fits, TRAIN-control scoring, heldout or TEST evaluation, remote request, array/checkpoint/runtime-binary read or numerical import occurred in this preparation.

This engineering packet is not predictive progress or an acceptance argument. The original denied V5 full-retention candidate remains unmodified in its earlier directory.
