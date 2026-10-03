# NCNC joint source-pattern pilot: implementation preparation

This fresh packet prepares the sealed assessment's two-fit seed0 J/F pilot. It is ready for source review and a separate staged execution release. Numerical checks, full-graph resources and predictive effects have not been measured by this preparation. The running frozen predictive family is a separate family.

## V2 narrow repair

Independent V1 review found one blocker: the prior-failure loop could shadow the diagnostics receipt pin in closure. V2 uses a dedicated `diagnostics_pin`. V1 is preserved; the full review is pinned in `V2_REPAIR.json` and copied as `PRESERVED_V1_REVIEW.json`. The original scientific plan and every original Python file except `pattern_close.py` are byte-identical. The command sheet points to V2. A fabricated stdlib-only closure fixture passes with zero, one and two external prior-failure receipts; it imports no model/data/runtime code. V2 still needs independent root review before execution release.

## Fixed experiment

Both arms construct the qualified four-member width64 private-completion model from identical fresh seed0 model, empty Adam, flags and RNG. The existing lexical Rademacher sign law uses dedicated sign seed 1956882699 and does not consume training RNG. Each model has 43,790 total parameters, including unused fixed-pt `ptlin` parameters. Serving remains the equal mean of native raw logits.

The native main loss is unchanged: positive BCE mean over query/member entries plus negative BCE mean over query/member entries. Every branch receives the same native target y. The auxiliary coefficient is 1 for the positive and negative query means. J and F differ only in where the four-component likelihood mixture is reduced. Neither arm pools completion weights during training or ordinary serving.

For a query e, the ordered support is exactly the inherited left residual slots followed by the right residual slots. Its labels are membership in the complete TRAIN observation graph, Z. With four uniform components, t = 2.5(s−6)+log(.1), q = sigmoid(t), and a = Z log q +(1−Z) log(1−q):

- J = −logmeanexp over members of each member's summed slot log likelihood, divided by the query's slot count.
- F = −sum over slots of the logmeanexp over members of each slot log likelihood, divided by the same count.
- An empty support contributes zero and remains in the query average.

The stable log-sigmoid likelihood has no probability epsilon or clipping. The inherited target forward retains the actual native clamp, alpha1.05, private weighted sums and nonlinear decoder. `PatternDecoder` inherits `forward` without overriding it. Only the recursive completion scorer is captured with autograd for the auxiliary loss; the inherited target path receives its detached result. The teacher is never passed to the encoder, candidate enumerator or predictor.

Z=0 means not observed in TRAIN. It is not a verified latent nonlink. J responsibilities weight only the source-pattern auxiliary gradients. This is a representation regularizer; it does not derive conditional-world target predictors, a latent-link posterior, or Bayesian graph marginal prediction. See `ATTRIBUTION_AND_SEMANTICS.md`.

## Stages and commands

`pattern_run.py` accepts `numerical`, `full_graph`, `fit`, `diagnostics`, and `close`. `COMMANDS.sh` contains the exact six prospective invocations on the qualified remote interpreter and normal A100 profile. The command sheet is not a release or a batch launcher. Run each applicable command only after root has supplied its stage-specific release and all dependency receipt hashes.

`ROOT_RELEASE_EXAMPLE.json` has empty authorization fields. Copy it to a fresh external execution root, set its preparation manifest hash, add the reviewed authorization reference and exact invocation, then fill receipt hashes as stages complete. Nothing in the example authorizes execution. Output directories must be fresh, outside the preparation and outside any sealed packet. Keep execution-root directories unsealed until their outputs are complete. The stdlib preflight runs before numerical/source imports; the driver never installs packages or downloads data.

| Stage | Work and result | Required previous receipts |
|---|---|---|
| numerical | Fabricated likelihood/gradient/teacher checks, native forward and main-gradient boundary, actual serialized next-update and serving replay for J/F; `QUALIFICATION.json` | None |
| full_graph | Complete TRAIN teacher, full-node native correspondence and serialized replay probes; fresh separate full17-batch J and F epochs; all official VALID queries with own, all three cyclic crossed and pooled fixed-bank routes; no ranking metric; `QUALIFICATION.json` | Numerical PASS |
| fit J / F | Fresh seed0 fit,100 complete native epochs/1700 steps,100 complete VALID selector candidates, first exact tie, own atomic journals, selected-state serialization and two extra complete served VALID replays; `COMPLETE.json` | Numerical and full_graph PASS |
| diagnostics | Bound selected J/F models; one complete VALID score bank per arm with member competence/error overlap and fixed-bank routes;17 fresh native TRAIN mask events per arm without backward; `COMPLETE.json` and private detail | Both qualifications and both complete fits |
| close | Stdlib custody and all100 epoch streams/RNG/actual supports match; expose `PAIR_RESULTS.json` and write `CLOSURE.json` | Both qualifications, complete fits, complete diagnostics |

The primary pair costs two scientific fits,3400 optimizer steps and200 selector candidates. Fit replay and diagnostics raise complete scientific VALID traversals to206 total. A diagnostic traversal computes five routes from one encoder and one completion scorer bank; it does not create five new selectors. Full-graph qualification adds two complete TRAIN epochs/34 steps and two complete routed VALID traversals, plus six four-record engineering replay updates and small native-correspondence forwards. Numerical qualification adds six fabricated four-record engineering replay updates. Qualification states are removed or discarded and never become scientific donors.

The diagnostics seed is2026100307. Its fixed native negative draw, record masks and support digests must match between J and F. These are new mask events on the same source snapshot, not independent graphs or unseen source-edge truth. Crossed routes give recipient m the donor (m+shift) mod4 weights for shifts1,2,3. Pooled routes average clamped weights while holding each recipient's transformed features/context/decoder fixed, then serve the mean raw logits. They are counterfactual diagnostics, with no new selector or serving rule.

## Resources and receipts

The qualified normal runtime is Python3.12, Torch2.7.1/CUDA12.6, PyG2.4.0, the bound sparse/scatter binaries, numpy1.26.4, pandas2.2.3, ordinary single A100 80GB visibility, float32, no autocast or TF32. `DEPENDENCIES.json` binds exact source/data/runtime authorities. The OGB evaluator is source bound by the existing TRAIN/raw/VALID authority. TRAIN is1,179,052 records and235,868 nodes with128 features. Each epoch uses17 native65,536-record batches and drops the shuffled64,940-record tail. VALID uses all60,084 positives and100,000 shared negatives on the complete TRAIN graph.

Auxiliary scorer activations and backward may cost substantially more than the existing no-grad recursive scorer. A100 fit feasibility, candidate counts, peak memory, time and quality are unknown until qualification. The full-graph stage exercises complete supports and backward for both arms; it cannot reduce graph size, mask size, support, epochs or native negative work as a resource rescue. Failures are retained.

`ATTEMPTS.json` and count-only `STATUS.json` record phase times, completed work and errors. CUDA peaks and process peak host RSS include source/data setup, full-node numerical/replay probes and both arms despite per-arm CUDA peak resets. Terminal JSON-write tails are explicitly unmeasured. Killed attempts retain observed work/time and an unknown remaining cost. Each fit's selected checkpoint, private selection, journal and100 epoch streams/supports are bound in its completion receipt. Root binds those exact receipts before diagnostics or closure. Closure sums the bound stages' attempt ledgers. If a failed stage is retried in a fresh output directory, root lists its exact `FAILED.json` in `prior_failure_receipts`; failures already charged by a completed output's own ledger cannot be counted twice. Unlisted external attempts are outside that cost certificate.

Only fit may use `--resume`, with the same exact released invocation/output and its own journal. It restores model, Adam, Python/NumPy/Torch CPU/selected CUDA RNG and all module flags. No resource or running-family state is accepted as a donor. An incomplete epoch is retried from the last complete journal and the failed/interrupted attempt cost remains in the ledger.

## Source-only verification and limits

`python3 source_check.py` executes stdlib reads, SHA256 verification, JSON parsing, AST parsing and compilation of source text without execution/import. It checks local imported names, inherited target forward, the recorded copied-helper adaptation, and all bound sealed packet payloads. `STATIC_SOURCE_CHECK.json` is source evidence only. The additional `closure_custody_fixture.py` executes only stdlib closure/common metadata code on fabricated receipts. These checks certify no numerical parity, sparse-kernel/autograd behavior, actual selected replay or resource feasibility. Native comparison scope is constructor/RNG, target forward and main gradients; the own J/F Adam serialization replay is separate and does not claim independent native Adam-step parity.

Future numerical comparisons use the qualified tolerance atol=rtol=128×dtype epsilon, with exact RNG, integer state and metadata checks. Stronger mechanistic/quality claims still require a separately competent covariance-aware single and the other controls named in the sealed assessment. This pair is one validation-selected seed0 development pilot. No TEST stage exists.
