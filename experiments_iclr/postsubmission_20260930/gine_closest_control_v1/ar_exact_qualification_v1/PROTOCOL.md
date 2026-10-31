# Prospective exact AR correctness qualification v1

## Status and admission

Authored source only. No source import, syntax compilation, Torch execution, model construction, RNG, dataset generation, server access or diagnostic execution occurred during authoring. Root must independently review, pin and admit execution. The authoring seal certifies bytes and stated scope only. No optimizer, training, original-score recalculation or performance claim is supplied.

The diagnostic uses the existing sealed `new_learning_gap_independent_v1/ar_mixture_source_v1` factories, domain contracts, objective and ledger, plus the separately sealed `shared_transition_heads_source_v1` head-control operations. It does not duplicate the decoder or change either sealed packet. Future admitted research-source imports are limited to those two packets and their pinned mutual dependency paths. The real source `Model` is never imported or constructed. A clearly diagnostic differentiable linear map produces H to test gradient accumulation through a shared encoder-like input; it cannot qualify the real graph encoder.

## Fixed work and numerical rules

CPU only; float64; embedding and recurrent widths both two; Torch intra/inter-op threads one. No random operation or seed is used. Parameters and proxy features are fixed deterministic arithmetic tensors. Matrices/biases/factors/states have distinct injected storage. The factory's registered parameters are all included in gradient comparisons. No width/recipe search or favorable-case replacement is permitted.

Predeclared tolerances:

| Check | Absolute tolerance | Relative tolerance |
|---|---:|---:|
| Complete member/mixture normalization, KL identity, values | 5e-12 | 0 |
| Retained versus forward-only density values | 2e-12 | 0 |
| Every exact/reference gradient coordinate | 2e-10 | 2e-8 |
| Expected explicit density-score cancellation / forbidden gradient paths | 2e-10 | 0 |

All feasible log densities and ordinary probabilities must be finite and positive; infeasible log densities must be negative infinity. No tolerance may be relaxed after inspecting output. Any failure ends the child and is retained; root decides subsequent repair in a new version.

The child has a 110-second internal alarm and the separately authored launcher has a 120-second hard wall limit. Work limits are 16 fixed law/factory cases (14 private-decoder checks and two head-control checks), N<=4, M<=3, at most 16 assignments before feasibility filtering, 900 source objective calls, 512 autograd calls, and 40,000 charged decoder/recurrent steps. These are correctness-work limits, not a performance forecast or relaxed phase cap. The launcher's terminal, stdout and stderr survive child failure or timeout. Torch import cost is included. No peak-memory claim is made.

## Complete tiny laws

All assignments of each complete graph are enumerated. These are deterministic correctness fixtures, not experimental datasets:

1. Four nodes with no edges; order `(2,0,3,1)`; positive unequal fugacities `(0.5,1.25,2,3)`.
2. Complete four-node path; order `(2,0,3,1)`; fugacities `(0.5,2,1.25,3)`.
3. Complete triangle on nodes 0–2 with isolated node 3; order `(2,3,0,1)`; fugacities `(0.4,1.3,2.7,0.8)`.
4. N=0 empty-law boundary, supplied empty order/fugacities.

Each law uses factored M=3, untied M=3 and conventional single M=1 factories. The path additionally uses factored M=1 and untied M=1. Fixed three-member masses `(1,2,4)` give nonuniform pi `(1,2,4)/7`. Single-member masses are `(1,)`. Baseline values `(0.3,-0.4,0.2)` (truncated for M=1) are literal nonzero predeclared constants, independent of all assignments and parameters. They are never estimated from this batch.

## Checks and references

For every case:

- Check parameter names/shapes/counts, CPU/dtype/finiteness, absence of tensor-storage aliases and disjoint private branch slices. The source architecture has `p=d+2`, matrices `W=3hp+3h²+2h`, factored count `W+M(p+15h+4)`, single count `W+7h+2`, untied count `M(W+7h+2)`. At d=h=2: W=40, factored M=3 is 154, single is 56, untied M=3 is 168. The linear H proxy adds eight shared parameters. These are correctness counts, not matched performance controls.
- Enumerate the entire law. Verify source support against an independently written edge-constraint predicate; check positive finite feasible densities, zero infeasible densities, normalization and retained/forward-only equality.
- Normalize the supplied hard-core target exactly. Compute complete finite-sum mixture and mean-member reverse KL, their mutual-information identity and the explicit normalized-density derivative cancellation.
- Evaluate the **actual source surrogate** at each feasible `(member, assignment)` through its iid one-sample API, then weight its gradient integrand by exact `pi_m q_m(y)` with those quadrature weights detached. This is exact finite quadrature of the estimator, not a generated iid batch. Compare against the full finite-sum objective gradients through every decoder parameter and the shared linear proxy parameters. Nonzero independent baselines remain in the source calls. Include M=1 objective/value/gradient coincidence.
- Independently check the actual source estimator on one fixed finite batch with branch counts 1,2,3 (M=3) or count one (M=1). Compare both source objective kinds and iid/stratified designs against literal manual weights `1/B` and `pi_m/count_m`, separately verifying per-branch total weights. This finite batch is deterministic; it checks weighting implementation, not Monte Carlo unbiasedness.
- Confirm rewards, mixture density and target values are detached, generating scores retain gradients, and aggregate non-generating cross entries have zero gradient. For factored/untied private parameter slices, check a branch-only retained density cannot differentiate another branch's slices.
- At three predeclared finite uniform vectors, compare source generation to the sequential decisions implied by enumerated complete member mass. Check original node-ID/order identities, feasibility and positive teacher-forced law. The vectors contain no RNG and are not a frequency experiment. One rejected uniform value 1.0 is checked on a nonempty case.

A successful pointwise generation check does **not** certify continuous ideal uniform draws, finite RNG independence/precision, branch draws from pi, unbiased sampling, saturation behavior across untested parameters, arbitrary future-label conditioning, or scalable resource use. No exact finite-floating-point sampler claim is allowed. Enumeration establishes this diagnostic's complete discrete laws only.

## Bounded shared-transition/head controls

Two additional cases use the separately sealed `shared_transition_heads_source_v1` on the same complete four-node path, order, fugacities, d=h=2, M=3, nonuniform pi and baseline. One uses ordinary private heads and one factored heads. Their actual decoder counts must equal 68 and 72, respectively; these tiny diagnostic widths do not imply an efficiency comparison. All source/dependency hashes are pinned before admission.

The independent reference replays each member separately using the existing source `_transition` and `_head_log_probabilities` operations. It copies no neural equations. It then compares complete densities and every generating-score parameter/proxy gradient with the production all-head shared replay. Exact complete-law gradients are compared with the expected actual source score-surrogate gradients for both objectives. Diagnostic-only wrappers count actual transition/head calls and whether heads execute with gradients enabled; they delegate unchanged source operations. They certify one production recurrent trajectory per supplied configuration, all M heads per step, and only the generating head's score graph. The wrapper changes no source file or learned parameter.

This extension was frozen during authoring before any execution. It adds no frequency experiment, real encoder, fit, task draw or resource probe. The 110/120-second and all existing call/step limits remain fixed. The common proxy H is deliberately tied for this correctness comparison, unlike independently trained scientific arms.

## Literal accounting

The diagnostic records actual API call designs and compares `WorkLedger` counts against independently assembled forward/replay counts. Exact-reference all-member trajectories retain gradients. Forward-only parity trajectories do not. Cross-density matrices replay all members but retain only generating scores. Each objective call performs a full mixture reduction even for the mean-member control. Generation is no-grad and its self score is not reused.

The source counter named `generating_score_backward_trajectories` records **retained density trajectories**, not completed backward invocations. The diagnostic reports these separately from the number of `autograd.grad` calls. Symbolic dense/nonlinearity/mask counters are not complete FLOPs or measured time/memory. Infeasible density returns are uncharged by this source and recorded separately. The proxy performs exactly one shared H forward per case; no encoder or parameter update occurs between sampling/scoring references.

## Proposed launch after root admission

From this folder, root may run its selected installed Python with:

`python3 -B launch_guard.py --admission-id ROOT_REVIEWED_PIN --run-name admitted_run_v1`

The launcher first verifies the authored packet manifest/seal and every pinned source-packet byte, then starts `diagnostic.py` with explicit one-thread environment values and a private new output folder inside this packet. Existing run names are refused. Child result/exception details, timing and limits are recorded. Root must account this invocation against the existing diagnostic phase budget; this proposal grants no admission.

Successful qualification would permit only the tested correctness statements. It does not adopt a representative pilot, prove novelty, establish target-task superiority, qualify the real encoder, justify GPU use or modify the original paper/results.
