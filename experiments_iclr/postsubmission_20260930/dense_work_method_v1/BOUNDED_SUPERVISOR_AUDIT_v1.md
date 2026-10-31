# Independent bounded-supervisor audit v1

## Decision and evidence boundary

**No blocking source defect found in the amended bounded supervisor for the declared canonical launch.** It closes the earlier unbounded inner-call gap by placing the unchanged authorization/logging/qualification chain in a separately supervised POSIX process group.

Source: [bounded_run_v1.py](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/protocols/bounded_run_v1.py>)  
SHA256: `b096966c1570da0a998dbb9fb464a9ad71490b74d4ba3406b751390637d19a23`  
Source-bound fixture: [bounded_run_fixture_v2/RESULT.json](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/protocols/bounded_run_fixture_v2/RESULT.json>)  
Fixture SHA256: `77f28106741efba0903225c486642be565130410c5211034cdbd89860610f5d8`

The preserved v1 fixture binds an earlier source hash; it is historical evidence, not the amended source's receipt. The refreshed v2 fixture matches the amended file exactly. I verified source/fixture hashes, parsed and compiled its AST without execution, checked the five reported fixture statuses, and verified unchanged hashes of run_authorized_v2.py, run_logged.py and repo_env.sh against the prior sealed bindings.

I did not execute this wrapper or rerun a process fixture; import Torch/model/scientific libraries; access acquired tensors, labels, checkpoints or predictive results; connect remotely; change source; or spawn agents. The fixture receipt is root-provided local stdlib process evidence, not my independent execution or a real-allocation run. [Audit receipt](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/dense_work_method_v1/BOUNDED_SUPERVISOR_AUDIT_RECEIPT_v1.json>).

## Cap, grace and process-group termination

[execute_group:59](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/protocols/bounded_run_v1.py:59>) launches the command with start_new_session=True and no shell. For this unchanged nested chain, the authorization supervisor, logging supervisor, research child and ordinary descendants inherit the new process group. A wait deadline is derived from cap minus reserved grace and measured launch overhead.

On timeout or interruption it sends SIGTERM to the group, waits no longer than the remaining cap for the immediate child, then sends SIGKILL to the same group even when the immediate supervisor has already exited. This last step handles the concrete case of a TERM-responsive supervisor leaving a TERM-ignoring descendant behind. It then reaps the immediate child.

On normal parent exit, a remaining process group makes the result incomplete and triggers TERM/KILL. A nonzero child exit, timeout or interruption cannot become helper completion. The inherited supervisors propagate research-child failure codes, so the unchanged ordinary inner chain also cannot turn a failed research exit into success.

The v2 fixture reports:

| Fixture | Evidence | Completion |
|---|---|---|
| success | Exit0, no timeout or termination | true |
| failure | Exit7 | false |
| term_timeout | TERM, exit-15, elapsed .7068s for cap1s/grace.3s | false |
| kill_timeout | TERM then KILL, exit-9, elapsed1.0018s | false |
| descendant_timeout | Immediate child exit-15; KILL delivered to surviving group | false |

These fixtures exercise the helper, not the full main admission/envelope path. The tiny measured overrun in the KILL fixture illustrates ordinary scheduling/termination overhead; it does not support a literal absolute deadline guarantee.

## START, TERMINAL and actual launch bindings

[main:105](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/protocols/bounded_run_v1.py:105>) requires finite 0<grace<cap<=28,800, exact repository cwd, the corrected authorized route environment, disabled bytecode writes, the existing phase wrapper, the exact lexical repository venv interpreter, the unchanged run_authorized_v2.py entrypoint and an explicit child argument boundary. The unchanged authorization/logging programs retain their route, actual sole-A100 and cache checks.

Outputs and request/script paths are confined to the exact phase and refuse symlink components. Outer and inner output directories must be new and separate. The research child must use the same repository interpreter. The amended checks require one matching actual child --request, or --binding when no request is supplied, and one --supervisor matching the recorded inner directory. The chosen request flag is recorded.

START uses exclusive creation and records the request SHA, actual complete command/child argv, cwd/phase/route, cap/grace, session mode and wrapper/authorization/logging/research source hashes. The START path is passed directly through GNNM_BOUND_START_JSON. TERMINAL uses exclusive creation, binds START and outer log hashes, records existing inner evidence hashes/bytes, and verifies the request remained unchanged. Missing inner evidence or elapsed time beyond the declared cap makes it incomplete. Main exits zero only for complete results; timeout exits124 and other failure exits1.

The bounded wrapper binds the actual root input bytes and CLI cap; it does not interpret an engineering request's schema, wall cap or prior allocation. The quality_v2 child must verify cap==the consumed request, actual argv/request/supervisor/output matching, source hashes, runtime/tolerance and completed inner/artifact chain. That is the stated separate admission boundary, not an additional process-control defect. Its source audit follows after its seal.

## Timing interpretation

The child-process deadline includes launch overhead and reserved TERM/KILL grace. Outer START creation/hash work reduces the cap supplied to execute_group. After the group ends, inner evidence hashing and outer-log/START hashing contribute to whole_supervised_seconds. The amended main explicitly marks an over-cap value incomplete.

whole_supervised_seconds is sampled while constructing TERMINAL, before the final request rehash field, TERMINAL serialization and final print/exit. Those final report costs still belong to whole allocation accounting through the outer caller's actual elapsed time. This is normal for an immutable report that cannot measure its own completed serialization. The process-group deadline, reported envelope elapsed time and final caller accounting should be described separately.

A hung/long inner training call is now subject to TERM/KILL from an independent parent. Unavoidable OS scheduling, signal delivery and reaping delays do not permit a literal absolute-time guarantee. The source does not need to claim one to establish the intended practical external wall guard.

## Remaining runtime requirement

The code and local helper fixtures support process supervision and failure semantics. They do not establish successful full envelope admission on the authorized machine, CUDA behavior, synthetic memory feasibility, complete training feasibility or a scientific budget. Those remain source-bound root runtime receipts.

