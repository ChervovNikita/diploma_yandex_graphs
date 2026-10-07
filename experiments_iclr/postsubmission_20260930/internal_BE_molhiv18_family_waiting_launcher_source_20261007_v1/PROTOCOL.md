# One-shot waiting launcher for the frozen MolHIV family

This stdlib source-only packet waits once, for at most 24 hours, and then uses
`os.execve` to replace its waiting process with the exact sealed v3 MolHIV 18-fit
executor. Its template is disabled. Root separately reviews this launcher and
authorizes both the wait and the conditional scientific launch, binding the
launcher manifest and final executor manifest in a separate root approval.

The executor remains unchanged: adopted I, `be_init`, lambda `0.5`, seeds
7101/7203/7307, six fixed conditions, native 100 epochs, original dispatcher and
selection, existing 32,390+10/32,400-second cell caps and derived 583,200-second
family cap. This launcher adds no fit loop, retry, scheduler, TEST reader, outcome
choice, or change to another job. Waiting has no numerical imports or CUDA
context; `CUDA_VISIBLE_DEVICES` must be empty until the single executor handoff.

## Engineering prerequisites

The exact existing O/I/P/G CUDA qualifier owner is PID **521566**, start ticks
**6018676170**. Its `TERMINAL.json` must report complete work, exit 0, actual worker
reap and no cap/monitor failure. `WORK_RECEIPT.json` must bind that terminal,
the sealed qualifier source, existing root approval, original adopted sources and
role hashes, and all four actual CUDA cases. Both saved owner and worker identities
must be terminal. Its admission must be
`whole_Wiki24_closed_all_science_terminal`, with the current whole-Wiki closure
still matching the saved hash. A qualifier failure is retained and prevents any
scientific start; missing terminal/work receipts after its owner ends are failure
evidence, never a restart request.

The exact existing Wiki selected-analysis owner is PID **522237**, start ticks
**6018862135**. Its source-bound owner record must have exactly one COMPLETE or
FAILURE endpoint, and the saved owner must be terminal. Any recorded numerical
child must have its original matching owner identity, integer exit code, an actual
reaped terminal receipt, and no live saved identity. A COMPLETE endpoint must agree
with a successful child terminal. A FAILURE before the source's child-launch
release is retained without inventing a child. Missing child custody after that
launch boundary prevents the scientific start. The launcher does not read the
analysis arrays, error-summary outcomes, or scientific scores.

After those prerequisites, `nvidia-smi` must observe the sole authorized physical
GPU `GPU-44039938-fd82-41d2-fefd-de71514e2fac` with at least **8 GiB actually free**.
The engineering prerequisites and memory observation are repeated immediately
before admission. This is a fresh observation on the normal shared host; it does
not reserve resources or promise future absence of other jobs.

## Records and handoff

The fresh waiting directory records OWNER, WAIT, ADMISSION, a separate exact
EXECUTOR_RELEASE and EXEC_INTENT before `execve`. The generated release truthfully
sets the executor's scientific authority, source review, same-host GPU readiness
and GPU-memory admission declarations from the bound conditional root approval
and actual engineering evidence. Adoption and budget bindings are copied unchanged
from the sealed executor. The intent is not terminal or launch-success evidence;
the executor's actual PARENT_OWNER receipt confirms the handoff. PID/start ticks
persist across a successful `execve`.

Waiting expiry, interruption, missing custody, qualifier failure or an unsuccessful
`execve` produces retained FAILURE evidence. There is no retry or process signal
to another job. All files and fresh outputs remain inside the repository phase.

## Launch after separate root source review

Stage this packet unchanged under the server phase. Save a separate root approval
outside this source packet, bind the launcher manifest, preserve the exact v3
executor binding and closed TEST/retry fields, set the three authority declarations
to true, and specify two fresh phase-relative waiting/family directories whose
parents already exist.

From `/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs`:

```text
CUDA_VISIBLE_DEVICES='' /usr/bin/python3 -I -S -B experiments_iclr/postsubmission_20260930/internal_BE_molhiv18_family_waiting_launcher_source_20261007_v1/wait_and_launch.py --approval <absolute-root-approval.json> --approval-sha256 <exact-root-approval-sha256>
```

The waiting wrapper requires only Linux `/proc`, Python 3.9+, and
`/usr/bin/nvidia-smi`; it accepts only hostname `anogena-2-0` and the authorized
repository working directory. The existing executor/dispatcher retain their
original runtime dependencies. Preparation performs AST/JSON/seal checks and
local CLI help only. No server/GPU operation, numeric payload access, scientific
outcome inspection or launch has occurred for this packet. Runtime remains
unverified.
