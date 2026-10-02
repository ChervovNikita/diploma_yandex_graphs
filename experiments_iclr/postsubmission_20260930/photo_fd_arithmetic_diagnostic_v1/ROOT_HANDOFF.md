# One Photo CE arithmetic diagnostic

This is a new source-only diagnostic packet. The failed Photo17 R17 attempt, original method/source/tolerances and blocked registry are preserved. No diagnostic, scientific import, GPU, SSH, array or checkpoint access was executed by this author. Only stdlib metadata preparation and AST/hash/JSON checks were run. `prepared_v1` contains an explicit false admission, false root decision and false launch request.

## Frozen comparison

Recreate the exact Photo17/cfg0 full-graph TRAIN-only failed cold state with the unchanged native constructor/seed, two dropout-off native Adam population updates (local then global), the unchanged disposable Adam-equivalence copies, native/K1/K4 identity and unchanged `bind_and_qualify`. Capture its expected failed-gate exception and closure; require the complete original failed gate JSON to reproduce exactly. No initializer, warm/fit, useful checkpoint or heldout label operation follows. RNG snapshots are digests only and do not advance or export state. All paired arithmetic must preserve post-gate global RNG.

Use the original three directions (seed90017), two epsilons1e-3/3e-4, original scaling/cotangents, centered-logit FD comparison, CE denominator floor1e-3 and unchanged thresholds2e-4/.05/.05. No mode, direction, epsilon, seed or tolerance is selected adaptively.

| Mode | CE computation on identical FP32 logits | Purpose |
|---|---|---|
| A | Original FP32 reduced `cross_entropy` and FP32 scalar subtraction | Replay the failed arithmetic |
| B | FP32 `cross_entropy(reduction='none')`, then FP64 mean and subtraction | Isolate mean-reduction/subtraction precision |
| C | Cast the same FP32 logits to FP64 before complete CE | Also change softmax/logsumexp arithmetic |

Casting already reduced FP32 scalars is not used. Factors/model/logits stay FP32. Each mode has its own autodifferentiated output cotangent, chained model VJP and loss JVP on the identical FP32 logit tangent. FP32 interface/gradient rounding is explicit. Report the original manual FP32 analytic derivative separately, plus every loss-consistent VJP/JVP, plus/minus scalar, difference and FD error. The paired A/manual-gradient/logit records must exactly reproduce all six original checks again before B/C can be interpreted.

`ARITHMETIC_COMPARISON.json` contains all records and descriptive per-mode checks. A B/C numerical agreement is diagnostic evidence only; it cannot pass the old gate, repair the old registry or admit training. No full-model FP64 branch is included. All old failures stay bound and blocked. Reduction cancellation is supported if B resolves discrepancies while its matched derivative/duality/logit evidence agrees; C alone improving agreement indicates an additional CE softmax/logsumexp arithmetic component. If B/C remain inconsistent, the cause remains unresolved. Root assesses the evidence before any separate precision amendment; this packet does not approve one.

## Root review and one bounded launch

1. Independently review the sealed source. Regenerate complete false external drafts with `prepare_diagnostic.py prepare --output <new external directory>` after sealing; this fills actual packet manifest/seal hashes. The false admission preserves the exact prior failed context and descriptors. The root decision binds that false draft. Do not edit sealed `prepared_v1`.
2. Root copies the external decision to a new signed file, binds the accepted source review descriptor, signer and UTC, and explicitly sets `approved`, `diagnostic_execution_authorized` and `source_review_accepted` true. Keep retry/registry-repair and fit/final labels false. Bind the exact regenerated false `admission_draft` descriptor.
3. Run metadata `prepare_diagnostic.py make-request --decision <signed external decision> --output <new admitted directory>`. It copies the approved false draft and changes only its execution flag, then emits exact `ADMISSION.json` and `ROOT_REQUEST.json`. Source/decision review remains root work.
4. Root explicitly uploads only this sealed source packet, new decision/review and new admission/request files. Existing R17v2/modernv3 and the unchanged Round15 ancestry source must be present; hashes are guarded. Create only the parent supervision directory. Launch the emitted request through unchanged `protocols/launch_modern_root_v1.py`, with phase-relative request path, outer `photo_fd_arithmetic_diagnostic_v1/supervision/diagnostic_v1_outer`, inner `.../diagnostic_v1_inner`, local receipt `.../diagnostic_v1_LAUNCH.json`.

The whole process is bounded at3600s by existing `bounded_run_v1.py`/`run_authorized_v2.py`. Exact repository/venv/cwd/route, active supervisor/whole START, one-GPU UUID and protected source/failure identities are verified before lazy scientific imports. `CUBLAS_WORKSPACE_CONFIG=:4096:8` is set before CUDA initialization; no native defaults or recipe are patched. Source/failure/admission/decision hashes are rechecked after arithmetic. Driver `Ledger` writes operation wall/allocator peaks and interface/cost traces; root/inner/outer receipts include whole-process wall work, guards/imports/metadata/digests and termination. The prior failed Photo qualifier had peak allocated34,285,758,976/reserved36,452,696,064bytes; this is existing evidence, not a promised diagnostic bound.

Diagnostic output is fixed fresh `photo_fd_arithmetic_diagnostic_v1/diagnostic_v1`; root receipts are separate at `root_receipts/diagnostic_v1`. No canonical registry attempt is claimed or retried. Exceptions retain `FAILED_DIAGNOSTIC.json`, partial operation costs and outer process evidence; no automatic retries. Numerical completion is a completed diagnostic, regardless of whether any precision mode agrees. Root receives the result and decides the next scientific action separately.
