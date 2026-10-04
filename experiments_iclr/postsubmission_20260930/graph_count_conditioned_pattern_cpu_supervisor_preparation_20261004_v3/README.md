# Disabled physical supervisor for the fabricated CPU oracle

This new source packet owns one prospective invocation of the sealed
count-conditioned prototype's `qualify_cpu.py`. It addresses F01 in the saved
independent review by putting wall/RSS observation, child ownership and physical
terminal receipts in a separate standard-library process. The candidate and all
earlier artifacts are preserved. No numerical source was imported or executed,
and this packet creates no release or launch.

## Exact future release chain

Root first needs an independent review of this supervisor and a separate exact
candidate source-review disposition. The earlier findings record is preserved;
its `SOURCE_ONLY_FINDINGS_RECORDED` status does not pass the candidate's existing
CPU gate. Any future disposition must account for the actual admitted wrapper
and its remaining limitations. Preparation alone has not closed F01.

The only accepted external supervisor release is
`graph_count_conditioned_pattern_cpu_qualification_execution_root_20261004_v1/ROOT_RELEASE_owned_supervisor.json`.
The existing inner CPU release remains at that directory's
`ROOT_RELEASE_fabricated_cpu.json`. Both release files require separately pinned
hashes, exact source-only PASS reviews, explicit root authorization, frozen
900-second/4-GiB caps, CPU-only stages and no retry/data/science/TEST access.
The supervisor verifies all candidate and external source bindings before and
after the child, including the unchanged review of the original defect.

Root fills the disabled template outside this preparation. It pins the actual
Python executable bytes, observer `/bin/ps` bytes and current platform. The
supervisor and child use the same root-pinned Python path; preserving a virtual
environment's original executable path avoids changing its package context.
The child release also pins its Torch distribution version, which the unchanged
candidate checks itself before importing Torch. The observer never imports
Torch or the candidate modules.

The exact child argv is the pinned Python path, `-I`, `-B`, the candidate's
absolute `qualify_cpu.py` path, `--root-release`, the absolute inner-release
path, `--release-sha256` and its exact SHA256. Its cwd is the candidate directory.
`child_command_sha256` is SHA256 of UTF-8
`json.dumps(argv, separators=(',', ':'), ensure_ascii=False)`. No shell command,
condition selector, numerical worker mode or retry route is exposed. The child
environment hides CUDA devices and disables bytecode writes; these are process
settings, not GPU API calls. `-I` ignores Python environment/path injection.

## Process ownership and physical observation

A persistent exclusive root lock and two fresh run01 paths prevent a second
incurred attempt. Root's execution directory may already contain its releases,
but neither child nor supervisor output may exist. The lock records exact
source/release identities and the owning supervisor PID and is never deleted.

The observer forks exactly one numerical child. That child creates a new
session/group, redirects stdout/stderr and reports its PID/SID/PGID plus command
pin through a bounded pipe before exec. Parent-owned SIGALRM enforces the wall
deadline independently of Torch import/operators. A pinned ps helper with a
0.5-second timeout supplies process birth/RSS/state fields; `getsid` authenticates
session membership. The observer records the identities and sums RSS across
the owned session. Failure to observe ownership or RSS stops qualification.

The direct session leader is not poll/waitpid-reaped while observation or cleanup
can still signal its group. Its unreaped PID keeps the original group identity
from reuse. Only that dedicated group, or the held direct child during setup,
can receive SIGKILL. There are no process-name searches, broad kills or signals
to observer helpers/unrelated sessions. A normal or abruptly exited child is
reaped by exact waitpid after no live owned session member is observed. A final
nonblocking wait can reap an already exited direct child when the observer has
failed, while preserving `REAPED_OWNERSHIP_UNRESOLVED` rather than claiming closure.

Unexpected new child groups or sessions are forbidden and fail as ownership
unresolved. The wrapper does not implement a process-escape sandbox or kill
escaped groups whose identity is not safely held. The sealed candidate directly
creates no subprocesses; root must assess the actual runtime's process behavior
before treating this mechanism as sufficient containment.

RSS is an independent sampled aggregate, with a 0.1-second loop pause and bounded
ps observation work. It can stop an observed overage even when the numerical
process is stalled. It cannot certify an instantaneous RSS ceiling between
samples. Shared pages can be charged more than once. Root must assess that
measurement semantics; this source makes no cgroup/OS RSS-limit claim. The
900-second execution deadline supplies no extra numerical budget. At most
5 seconds after a stop are used to terminate/observe/reap; any breach or unresolved
cleanup excludes PASS. Total observer/bookkeeping time is recorded separately.

## Physical completion and collected oracle results

`SUPERVISOR_TERMINAL.json` records waitpid status/exit code, wall time, observed
aggregate RSS peak, identity ledger, stop/termination actions and observer errors.
`SUPERVISOR_CUSTODY.json` binds that terminal file and preserves actual child and
supervisor output hashes, including partial files. These writes do not depend on
the child writing its own `FINAL_CUSTODY.json`; import failure, SIGKILL, segmentation
fault or another abrupt child exit can still have independent physical evidence.
Ordinary termination signals to the supervisor trigger owned cleanup. Supervisor
SIGKILL, power/storage loss can prevent its final write: durable STARTED/lock show
incurrence, and missing final evidence cannot qualify.

The terminal's `physical_status` is separate from `collected_oracle_result`.
An exit0 and an empty owned session establish physical completion only. A
`COLLECTED_PASS` requires actual `QUALIFICATION.json` and child
`FINAL_CUSTODY.json` links/hashes, the complete declared oracle counts, exact
source/release/runtime identities, fixed caps, claim flags and every child
custody file/hash, with no failure, temporary/unlisted/nested output, source
change or observer stop. A missing child final receipt or partial oracle output
cannot be promoted by this observer. Both supervisor terminal and custody must
have `PASS_FABRICATED_CPU_ONLY` before root can record a fabricated CPU success.
A custody collection failure forces a nonqualifying terminal/custody outcome.

Even an eventual fabricated PASS would establish only those unit law checks.
Actual TRAIN provenance, native float32 single/packing/gradient/state/empty-call
integration, complete full65536-record resource feasibility and a separate
scientific successor release remain unqualified. Classical method ancestry and
post-TEST exploratory status remain unchanged; no ranking, novelty, graph-posterior,
native-fullbatch or fresh consumed-TEST claim follows.

## Minimal v2 successor

The preserved v1 source review found missing final interpreter/observer rehash and a recorded-stop setup race. This separately sealed successor fixes only those observer/lifecycle boundaries. The core, child command, original caps, numerical bodies and physical/collected status distinction remain unchanged. Source review and root admission are pending. No execution occurred.

## Minimal v3 successor

The preserved v2 source review recorded W03: a fork child could run the inherited parent handler before resetting signals, and PID0 could reach a group-wide signal. This successor requires an integer direct-child PID greater than0 before identity lookup or signaling. The inherited handler checks the saved supervisor PID and uses `os._exit(128 + signum)` in the child before any parent stop or cleanup. The parent's positive held-child/group selection and stop behavior are unchanged.

W04 is closed in current source accounting: `SOURCE_BINDING.json` has40 current external pins, and both `STATIC_CHECKS.json` and `INPUT_HASH_RECEIPT.json` record that same rehashed40-row scope. The original v1 receipt checked32 rows; its byte-identical copy in v2 is historical and did not measure the v2 35-row closure. Those predecessor packets and the v2 reviewer are preserved. The core, numerical child bodies, command,900-second/4-GiB caps, PLAN, LIMITS and disabled root template remain unchanged. This source preparation has no execution authority; independent inspection and separate root admission remain pending.
