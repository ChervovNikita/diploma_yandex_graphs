# Native v2 serial CPU transport supervisor

This separate source-only packet launches and monitors one owned child that executes the unchanged sealed native v2 training main. It adds no scheduler, arm override, dataset, numerical test, scoring or selection function. Root execution release is still required.

The bound native v2 manifest is 5956ed69b1c9646aedda0ce67d950081b23ca6ca77796ac7f98c4b900544af43 and native scientific freeze is 274b293cefa8dc8257ede17a0e5e6b1fb2fa1b53392ce0a34227bf87fc46b87a. Actual all-three full-graph CPU qualification is bound at a2e6e49506005256e02461a2d6d261ec26a2e4b5110b4cf5eec125b8a937cf91 (10,916 bytes), with an immutable copy of its actual transport receipt.

cpu_supervisor.py verifies all original source/resource records, both freezes, the qualification result/transport and exact root release before admitting its child. It invokes the original v2 guard and preserves its complete 15-case frozen protocol. The child uses the exact native train_native.py via runpy; the bootstrap only imports the pinned Torch runtime and sets intra-op/inter-op CPU threads to one. CUDA is hidden and OMP/MKL/OpenBLAS/NumExpr threads are one. All fit, preprocessing, TRAIN, validation, checkpoint selection/replay and native main loops remain in the original file.

The child receives a 16 GiB RLIMIT_AS before Python/Torch import. The supervisor monitors its RSS against 12 GiB and wall time against 18 hours; launch screens current host/cgroup free memory for at least 24 GiB. The earlier one-update resource probe remains a forecast and does not guarantee complete-fit allocator residency.

The detached supervisor installs controlled SIGINT/SIGTERM handlers. Signals are blocked across child creation until its owned PID is registered; cleanup ignores repeated cancellation signals, terminates that child, then kills/reaps it after a five-second grace period if required. It never signals unrelated processes or process groups.

Every child exit, spawn/monitor failure, resource deferment or cancellation is recorded in SUPERVISOR_RECEIPT.json with PID, exit code, stdout/stderr locations, observed resident/virtual peaks and source preservation. If the original native main produces its canonical STUDY.json, the supervisor captures only its path, SHA-256 and byte count. It does not parse its scores, create/modify the study, reconstruct missing fits or compare subsets. Killed children may leave partial original files without a canonical study; these remain evidence for the root. Root validates all fifteen outputs.

deploy_supervisor.py requires the exact execution release, uploads every sealed payload plus manifest/seal, checks the canonical repository and authorized single GPU UUID with a read-only inventory query, validates the real admission guard and detaches cpu_supervisor.py. Existing bytes cannot be overwritten with different content. Current Git HEAD is recorded; byte identities determine admission. No package/environment edit or tensor download is performed.

Use deploy_supervisor.py with --admission pointing to the root execution release and --receipt to a fresh local path. EXECUTION_RELEASE_TEMPLATE.json has execution_authorized false and cannot dispatch. The two scripts were syntax/source inspected locally; no training, dataset-label read, numerical test or remote action was performed during preparation.
