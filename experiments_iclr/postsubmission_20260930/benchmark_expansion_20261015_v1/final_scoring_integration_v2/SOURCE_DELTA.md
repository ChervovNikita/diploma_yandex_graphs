# Minimal source successor delta

The existing portable runner is training/VALID-specific. Direct reuse would
silently misclassify final task dependencies, terminal schemas, export keys
and worker TEST-access accounting. Frozen sources were not edited.

| Source | Reused behavior | Necessary successor |
|---|---|---|
| portablev3 Engine | Actual launch, PID, lease, run/drain loop, two-slot/STOP/failure controls | Literal TEST terminal schemas, TEST exports, retained missing I4 rows, no repeat/resume, truthful TEST/unknown access accounting, post-lease ledger handoff |
| PortableBackend | Actual `admit`, probes, identity, polling, waits, reap/release | Constructor defers ledger establishment; fixed scoring request/capsule argv inside original sole Popen transaction |
| host Ledger | Same HOST.lock, reserve/commit/cancel/reap/STOP/reconciliation mechanics | Establish/load accept explicit final ownership after empty sealed TRAIN handoff, preserving original OWNER and state bytes |
| qualified Python entry | Actual lexical interpreter/prefix/base-prefix/binary/venv validation and host guard | Restrict target to exact final observer entry instead of TRAIN/VALID handler filenames |
| training authority | Full actual host/source/native/environment/readiness/GO checks, once per supervisor | Final-release projection, same-decision host view, derived seal and inference-specific resources |
| scalar observer | Accepted scalar-only copy/receipt helpers; no tensor observation | Observe returned frozen final RESULT scalar metadata without intercepting TEST tensor creation |
| accepted reporter | Exact rows/cohorts/M1-body0/mean identities, unresolved gates, intervals and cost semantics | Separate TEST normalizers plus offline immutable-host report merge |
| frozen scoringv2 | Exact restored factory, private state/readout, selected VALID replay, official TEST serving | No code changes |

The source metadata checks compare unchanged Engine custody methods by AST.
The successor backend still has exactly one Popen call. No independently
starting observer, replacement scheduler, generic checkpoint mover or online
coordinator was introduced. Frozen scorer v2's single-host seal limitation is
handled by authenticated offline host shards producing compatible local seal
views from the complete global terminal partition.

The frozen scorer's one `go.final_scoring.training_release` is compatible with
deterministic per-host execution views of one global user decision. Those
views are used unchanged by that host's training and scoring. No real GO
artifact is manufactured during preparation.

The child does not call full campaign `authority()`. The supervisor validates
all tasks once. Its pinned preparation and own task capsule are committed on
the actual reservation; a metadata sentinel check proves the child capsule
path cannot iterate a campaign inventory and checks only one training task
plus zero/four final dependencies. Shared source/metadata byte hashes and the
frozen scorer's original roster/seal JSON reads remain explicit work.

The integration exposes the existing separate TRAIN and final commands. Offline all-terminal
shard exchange is a real synchronization step. Prepared combined orchestrationv3 sequences those commands. Its actual
engineering qualification remains pending, with no campaign GO claimed.

See SOURCE_REBIND.md for the repaired scorer4 and separately owned inference API pins. The source rebind does not alter native training/selected-state provenance.
