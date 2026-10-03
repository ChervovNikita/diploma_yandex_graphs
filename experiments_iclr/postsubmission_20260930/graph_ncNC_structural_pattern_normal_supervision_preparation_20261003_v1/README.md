# Normal numerical-stage supervision preparation

This packet prepares one ordinary detached numerical child on authorized18.77. No numerical execution is authorized by the packet. V1 source was rejected for a closure receipt variable-scope error and was not uploaded. The repaired V2 source must finish independent review and be sealed before a root release can bind it. The template has empty authorization fields and pending source/support manifest hashes.

The exact repository is `/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git`. Execution uses the existing Python3.12 interpreter and extra sparse/scatter runtime path, with one bound physical A100 UUID in `CUDA_VISIBLE_DEVICES`. The supervisor independently checks the complete expected two-GPU UUID set and repository cwd. It does not create namespaces, install packages, use sudo, mutate the base environment, or interact with existing jobs.

`supervise_numerical.py` starts one `numerical_child.py` with a new process session. It records real `/proc` PID, parent PID, process group, session, start-time ticks, executable, argv, cwd and namespace identities. The child and supervisor must share the ordinary host namespaces. Stdout/stderr are separate external logs. Real wait4 exit status, signal, CPU time and kernel peak RSS are recorded at closure. There is no automatic retry or restart.

Candidate caps are wall1800seconds, child-session host RSS16GiB, Torch peak allocated8GiB and peak reserved8GiB. Wall includes supervisor setup/source reads until child exit. Host RSS is sampled across the dedicated child's session every0.25seconds and the direct child's kernel RSS high-water mark is checked after exit. A transient RSS overshoot can be detected after exit and invalidates supervised success. The supervisor only signals its own new child process group after checking the original PID start-time identity. Existing NCNC/BUDDY jobs are outside that session and are never signaled.

The numerical child executes the sealed driver unchanged through `runpy`, with the same main argv and import path. A separate daemon thread reads existing Torch persistent peak counters only after Torch has initialized CUDA. It does not import Torch, consume RNG, reset counters, wrap the model, or change kernels. The numerical stage does not reset peaks, so the0.25second observations and final observation capture peak allocated/reserved bytes. Cap/monitor violations persist the observation and exit only that numerical child with code88/89. The original driver still owns scientific preflight, exact source/data/runtime checks, model checks and numerical receipts. No official data tensors are loaded by its fabricated numerical stage.

`ROOT_RELEASE_NUMERICAL_TEMPLATE.json` is an example only. Root must bind the independently passed V2 manifest, this support manifest, reviewed authorization reference, exact numerical invocation and candidate caps before execution. It supplies no fit/full-graph authorization. Full-graph resources need a separate measured decision and source-review passage.

## Prospective detached command

`DETACHED_COMMAND.sh` contains the exact command for the external numerical execution root. It is not run by this preparation. All supervision and numerical outputs are fresh siblings outside sealed packets. The caller creates only the fresh external execution root and root release, then starts the detached supervisor through the approved MacLink wrapper. The forwarding allocation is used only for the existing MacLink transport; no computation or filesystem task runs there.

This preparation uses stdlib JSON/hash/AST/compile checks only. It makes no measured cap-feasibility, numerical parity or successful-child claim.
