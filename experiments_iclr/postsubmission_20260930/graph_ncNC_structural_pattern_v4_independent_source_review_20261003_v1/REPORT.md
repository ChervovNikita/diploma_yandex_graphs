# Independent V4 source review

## Blockers first

**Material source blockers found in the scoped V3-to-V4 change: none.**

**Execution remains blocked on current evidence:** no V4 numerical or full-graph parity PASS is certified by this review. The candidate still needs its fresh fabricated numerical qualification, then the fixed four-record parity on the complete real graph and the unchanged complete resource qualification. This source verdict is not a launch release, memory-feasibility result, predictive result, or manuscript judgment.

Candidate: `graph_ncNC_structural_pattern_pilot_preparation_20261003_v4`.

- Manifest: `9021a598c7642bf428124de1230a078dfddbf3968095432354ee18a14541104c`.
- Seal: `beffda0bbdbb3d71a232671349bb40b4ff1dec03b40c6ef81706b5ee287a19e3`.

The immutable candidate, predecessor, dependency sources, and physical failure receipts were checked directly. Preparation prose and desired outcomes were not accepted as an oracle. All review checks were stdlib source/hash/AST operations; no project, Torch, model, fixture, or GPU code was imported or executed.

## Findings

### Checkpoint scope and autograd are consistent with the stated change

Only the differentiable captured auxiliary scorer calls use `checkpoint(self.depth_zero, transformed, graph, query, member, use_reentrant=False, preserve_rng_state=True, determinism_check="default")`. Explicit member/input arguments avoid a loop-variable closure. `set_checkpoint_early_stop(False)` is active during forward. The pinned native `depth_zero` body still computes the endpoint product, the second full-node `xlin`, exact neighborhood enumeration, feature sum, and complete decode. No empty-query shortcut or new chunking was introduced; the frozen recipe remains `candidate_split_size=-1`.

The captured score is retained for the auxiliary, while the inherited target forward receives `score.detach()`. The target clamp, private feature weights, endpoint/common decoding, phase order, and raw-logit serving pool are unchanged. Uncaptured target/inference calls return through the original superclass `completion_scores`; auxiliary-disabled calls execute direct `depth_zero` under no-grad. No new parameters, constructor draws, optimizer groups, or trained state keys were introduced.

Official PyTorch v2.7.1 source was independently retrieved at immutable commit `e2d141dbde55c2a4370fac5165b0561b6af4798b`; source SHA `b07acb21f5c727f3eace16249ca126a7c8a0ac8d97c83f2a13fff1b5a59a360f`. Its nonreentrant implementation records the forward autograd graph and supports `autograd.grad`; it captures the early-stop value in the checkpoint frame, so the context need not remain active during backward. Recomputation runs with grad enabled. These mechanics match this use.

### RNG and mutable state have a sound source argument, with explicit runtime checks still required

`depth_zero` and its called native operations read the explicit member and module training flags. They do not read `_pattern_capture` or `_pattern_gradient`; clearing those capture flags in `pattern_forward` before backward therefore does not change the recomputed body. No training-flag change or Adam update occurs between forward and total backward in the actual training/parity trajectories.

The inherited dropout overwrites fresh linear or layer-normalization results. It does not overwrite checkpoint input tensors or parameters. Normalization is private LayerNorm with no running statistics. Graph and query enumeration have no mutable cache, and the inspected graph operations do not modify the graph fields or input queries. The body uses Torch dropout and no Python/NumPy randomness or device moves.

Upstream nonreentrant checkpoint saves Torch CPU state and the initialized CUDA device states discovered from tensor arguments, and recomputes under an RNG fork that restores the caller state afterward. The supplied transformed/query tensors expose the one qualified CUDA device; CUDA is initialized before these calls. This is a source argument for unchanged dropout and backward RNG, not an observed runtime equality. The installed host checkpoint file itself was not read remotely; the existing authority identifies Torch2.7.1, and the new runtime parity probes remain the practical check on its deployed behavior.

`determinism_check="default"` compares saved/recomputed **shape, dtype, and device**, not tensor values. The candidate does not rely on it for value parity: separate explicit forward, full-gradient, state, RNG, and flag comparisons are required.

### The prospective saved-V3 parity is adequate for the scoped engineering change

The reference factory verifies the exact V3 manifest and `pattern_model.py` before loading that saved class. It shares the current qualified prototype and unchanged computational helpers with V4; this is not an independently isolated historical environment. Direct byte checks establish the relevant common computation is unchanged. The changed `pilot_common` section adds receipt admission, not model arithmetic.

For both J and F, reference and candidate trajectories each perform two real Adam updates from the same exact initial model/Adam/RNG/training flags. The saved CPU images compare main/auxiliary/total losses, target logits, raw scorer tensors, native clamped scores, `t`, J/F/count/rho values, query/support/teacher order, all named gradients including `None` status, next model, next Adam, and exact forward/backward/next-step all-RNG and training flags. Main-target score detachment is separately tested. Backward must not advance caller RNG. Reference and candidate GPU graphs are sequential; returned images own CPU copies. The caller RNG is restored after parity, and no probe state enters a fit.

The extra J probes compare the auxiliary-disabled direct path and an empty captured-query forward/backward. These cover arm-independent scorer behavior; omitting an F copy of those same body probes is not a source blocker. Empty queries still execute the native full-node transform.

Limits are explicit: four fixed query records do not prove equality for every query/batch; arithmetic comparisons retain the declared tolerance rather than bitwise CUDA equality. The full real graph is kept in the real parity probe. Actual original-size epoch execution remains a separate mandatory gate, which is the appropriate coverage for shape/resource behavior beyond this bounded comparison.

### Existing resource and scientific work is retained

Independently verified all 78 V4 payloads, 49 V3 payloads, and all payloads in five pinned dependency packets. Exactly three existing Python files changed: `pattern_model.py`, `pattern_qualification.py`, `pilot_common.py`. Fifteen original Python bodies are byte-identical; `PILOT_PLAN.json` is byte-identical. The additions are parity code and a metadata utility.

Parity adds eight audit optimizer updates per qualification stage. It retains six existing serialized-replay updates, and full-graph qualification retains two complete native 17-batch epochs: `8+6+34=48` engineering updates. Both complete VALID positive/shared-negative traversals still compute all five fixed routes and no project metric. Inclusive phase timers cover parity and CPU snapshots/comparisons; the pre-reset CUDA peak is captured before arm resets and combined with later arm peaks. Recomputation and re-enumeration are additional paid work.

The scientific J/F recipe remains seed0, M4/N64, lambda1, original native batch65536, exact original support/teacher semantics, 100 epochs per arm, 1700 updates and 100 complete official VALID selector candidates per arm, plus selected-state replay and fixed-bank diagnostics. There is no support cap, replacement graph, reduced schedule, changed metric, or state donation in this change. Current qualification admission requires V4-bound PASS receipts containing the new V3 parity; the historical V3 numerical result is not accepted as a substitute.

### Both physical cap failures and exact costs are preserved

The 20 original/preserved receipt files were hash/size checked. Both physical terminal records report child exit88, no qualification receipt, and the attempt ledgers show first J batch forward in progress with zero completed batches.

| Attempt | Supervisor-inclusive wall | Peak allocated bytes | Peak reserved bytes | Caps allocated/reserved |
|---|---:|---:|---:|---:|
| 1 | 26.04586512222886 s | 46,954,556,416 | 59,624,128,512 | 42,949,672,960 / 51,539,607,552 |
| 2 | 29.33481625840068 s | 73,043,022,336 | 83,332,431,872 | 75,161,927,680 / 80,530,636,800 |

Paid wall totals 55.38068138062954 s. The second attempt's 70/75GiB ceilings are the candidate's prospective ceiling, subject to separate root admission; V4 does not raise them again. Preservation does not make either attempt a successful epoch or predictive observation. Full-call temporaries and allocator reservations may still exceed those ceilings after checkpointing.

## Verdict and remaining boundary

**Source-only engineering verdict: the scoped change has no identified material blocker and may proceed to separately admitted current numerical parity.** Passing that stage would only permit the separately admitted full-graph stage. Scientific fitting still requires the exact current complete full-graph qualification and fresh physical resource admission. This review establishes neither runtime parity nor memory savings, model quality, novelty, or acceptance.

Exact read scopes, input pins, independent custody checks, and the verdict are preserved in this packet. Candidate/predecessor sources and canonical/index artifacts were not edited.
