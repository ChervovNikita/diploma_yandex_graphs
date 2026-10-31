# Fitted AR screen v2: bootstrap repair qualified for root review

This packet binds root's `protocols/ar_fitted_screen_root_v2/SCIENTIFIC_FREEZE_v2.json`, SHA-256 `210ba66129493a973f8826e207ba0841826b63040bc670701a18004b8f7e6833`. It awaits independent full-source review and separate root execution admission. This separately versioned repair copies only the 12 authored v1 files and preserves the failed v1 reservation. Its stdlib `diagnostics.py` tooling is imported and exercised by `validate_bootstrap.py`; all authored and pinned Python sources are parsed, transformed, stripped and compiled without importing or executing scientific project code. Only artificial fixtures execute to check semantics. No topology, scientific draw, fit or server access occurs during qualification. Earlier packets remain read only. The scientific choices and the `ar_fitted_screen_v1` RNG role string are unchanged.

The original pinned Model, scalar AR generation/scoring/surrogate APIs, complete spatial15 generator, integer/rational oracle and qualified v3 no-gradient batched evaluator remain dependencies. The packet supplies initialization, fixed-factor exclusions, fitted orchestration, exhaustive metrics, classical controls, metering and a launcher. `SOURCE_PINS.json` binds dependency bytes; `AUTHORING_MANIFEST.json` binds authored bytes; `AUTHORING_SEAL.json` binds the manifest.

## Frozen protocol

Use the root freeze unchanged: eight arms in declared order, model seeds17/29/43, training seeds10001–10008, development seeds20001–20008, fugacities4/1,64 AdamW updates/fit, ordered16-cell training cycle repeated four times, eight scalar complete schedules/update, stratified two/member for M4, zero baseline, final update64 checkpoint only. No HPO, clipping, scheduler, early stopping or retry. Complete all24 final checkpoints before development topology draws.

All four private shared-matrix M4 arms receive identical, separately stored iid uncentered N(0,.02²) tables `b_x,b_h,b_o,h0`, with a separate role stream/table. No branch centering is applied. Compatible matrices and encoder state use matched roles. The qualified Torch2.1 initializer advances the explicit stream while restoring caller RNG. Ordinary heads equal initial effective candidate output maps. Features remain graph and homogeneous log fugacity only; the pinned encoder is two-layer GCN, input1, hidden/output16, multiplier2, one head, normalization None, dropout0.

Before topology draws/fits, exactly two required identity implementation gates use first frozen model seed17 and the qualified v3 tiny cases empty0, edgeless3, path3 and triangle+isolate4. They check complete scalar/batched member densities, fixed factors, independent storage and raw decoder counts. Proxy H never becomes fitted input. Unchanged APIs reuse completed v3 qualification.

Evaluate all384 final learned cells on every feasible assignment, with infeasible structural zeros and no renormalization, smoothing or truncation. Perform48 actual-H parity gates: all24 fits on development20001 at both fugacities, all members on the frozen16-slot ordinal spread, atol1e-10/rtol0. Retain checkpoints, complete integer oracles, all per-cell outcomes, full-law digests and explicit pending/incomplete cells. Training never receives oracle probabilities or batched gradients.

Primary metric is KL(q||p) at λ4; TV and per-node signed/mean/max marginal errors are mandatory retained secondary evidence. `summarize.py` implements the frozen all-seven-control KL, seed-sign, λ1 nonregression and TV continuation gate. Report all paired effects and topology means per model seed. Missing cells or invariant failures are terminal no-go. This is development evidence, without original-score, novelty, convergence, confirmation or scalability claims.

Classical controls are a cold complete integer ticket oracle, eight exact draws, an independent8192-draw exact-IID empirical histogram, and eight Gibbs chains with64 burn-in sweeps and1024 retained states/chain. Each sweep makes15 random-site choices with replacement and consumes a coin even when blocked. Both empirical cache laws remain unsmoothed; forward KL is infinity when positive target states are missing. Matched exact-IID histogram metrics separate finite sampling error from chain effects. RNG/setup/cold encoding/warm generation/build/cache/fitting/law timings remain separate.

## Raw parameter inventory

Counts describe executed two-logit heads. Encoder2992 trainable elements are included in the final column. No Bernoulli-equivalent compression is implemented.

| Decoder | Stored | Trainable | Fixed | Encoder+decoder trainable |
|---|---:|---:|---:|---:|
| Factored4×16, mixture/mean | 2712 | 2712 | 0 | 5704 |
| All-factor identity4×16 | 2712 | 2120 | 592 | 5112 |
| Recurrent-factor identity4×16 | 2712 | 2192 | 520 | 5184 |
| Shared transition, ordinary heads4×16 | 1880 | 1880 | 0 | 4872 |
| Single16 | 1778 | 1778 | 0 | 4770 |
| Single22 | 2840 | 2840 | 0 | 5832 |
| Compact untied4×8 | 2792 | 2792 | 0 | 5784 |

Both identities retain trainable private biases/h0. All-factor identity fixes `r_x,s_x,r_h,s_h,r_o,s_o`; recurrent identity fixes the first four and trains output factors. Fixed factors are excluded from fresh AdamW and checked unchanged with no gradients after every update. Runtime records tensor names/shapes, actual counts, storage and optimizer-state elements. All-factor stored encoder+decoder count is5704, despite5112 trainable.

## Diagnostic accounting

The admitted stdlib entrypoint verifies root contracts, authored bytes and dependency pins before model imports. Hash/source/AST/contract setup is phase work. One shared meter and hash-verified in-memory AST loader leave pinned files unchanged.

Only pure if/single-raise guards with no else, exact-name require/_require calls, and assert tests are wrapped. Whole require calls preserve original argument/message order; assert messages remain lazy. Values are evaluated once and returned without added bool coercion. Binding/yield/await hazards and reserved injection-name bindings are rejected; intentional injected-name reads in new orchestration remain allowed. Only calls newly injected by `Transform.wrapped` carry the in-memory `_ar_injected_guard=True` marker. `Strip` validates and removes only marked `invoke(label, zero-argument lambda)` wrappers; original meter `block`, `check_budget` and `invoke` calls are preserved. The marker survives deepcopy and is ignored by AST compilation/digests. Mechanical wrapper stripping must restore the original AST digest; runtime retains hashes/sites. Mathematical and masked-law branches stay unchanged; constructor canonicalization remains algorithm work.

Full blocks cover the two identity gates, parameter storage/inventory, fixed factors, integer/rational equality and final actual-H parity. One depth-aware meter charges outer elapsed spans, without nested double charging. Successful outer whole-require/full-block calls check the60s budget immediately; law chunks/stage boundaries also poll. Test-only if/assert wrappers defer polling/persistence because the original statement may be about to raise. Original exceptions remain primary. Post-call enforcement can overshoot60s; actual measured overrun is retained for root accounting. No exact diagnostic preemption is claimed. Bookkeeping/persistence overhead remains in phase wall time.

Periodic `diagnostics.json` has `final_snapshot:false` and `active_depth`. Only entrypoint finalization after all blocks unwind to depth0 writes `final_snapshot:true`. A hard-kill snapshot omitting an unfinished outer span is a lower bound, not certified final usage. Root must apply its accounting-validity gate and handle unknown usage conservatively.

## Prospective launcher and resources

Root invokes `launch_guard.py --root-freeze PATH --root-freeze-sha256 SHA --root-admission PATH --run-name admitted_run_v1`. No run directory exists in the authored packet. The launcher reserves one fresh `runs/admitted_run_v1`; any previous reservation forbids overwrite/resume/retry. It checks exact SSH destination `anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru` and exactly one GPU0, UUID `GPU-44039938-fd82-41d2-fefd-de71514e2fac`, model NVIDIA A100-SXM4-80GB,81920MiB. No server or GPU query occurred during authoring.

The child is CPU float64, one intra/inter-op thread, empty CUDA visibility and disabled bytecode. Temp/cache roots stay inside the new run; HOME is not repurposed. A new process group receives internal2820s and hard2850s limits, leaving15s for kill/reap inside a2885s launcher envelope. Root reserves upload35+supervision2900+fetch35+orchestration30=3000s.

The launcher polls the child-owned progress journal latest_usage cumulative high-water RSS against4GiB, and aggregate allocated/apparent run artifacts against1GiB, every.25s. RSS observation waits for the next child journal save and is not live RSS or exact preemption; final RUSAGE_CHILDREN provides the terminal cumulative maximum. Unpublished/transient overshoot is possible and violations retain incomplete journals. A separate16GiB RLIMIT_AS permits Torch/shared-library virtual mappings and does not imply16GiB resident memory. Per-file limit is1GiB and core files are disabled. The fitted operation retains its2GiB local+remote reservation and the aggregate200GiB disk cap. Root's immutable `PROSPECTIVE_RESOURCE_ALLOCATION_v1.json` records an additional CPU-only4200s phase/180s diagnostic planning bound and3GiB local+remote reservation; the failed v1 charges remain preserved. Child CPU time is not GPU busy time. Terminal launcher statuses are `child_exited`, `hard_timeout_incomplete` and `launcher_failed_or_interrupted`, with child return code and meter provenance.

Root owns independent full-source review, admission, execution, cumulative resource accounting and scientific interpretation. Resource or scientific completion is not certified by source preparation.

## Bootstrap qualification

Use Python >=3.10 (the existing reserved-binding checker uses `ast.MatchAs`). Run `python3 -B validate_bootstrap.py --result qualification/validation_result.json` locally from this packet; this retained qualification uses the bundled Python3.12.14 runtime. The stdlib-only suite verifies copied-source and dependency pins, AST parse/Transform/deepcopy/Strip digest restoration, and compilation of every authored `.py` file plus every pinned `.py` dependency. It excludes Markdown/JSON from Python compilation, parses authored JSON, and retains an inventory with source and AST hashes. It covers unmarked original meter calls, nested injected wrappers, return/exception/evaluation order on artificial fixtures, malformed marked wrappers and original reserved-name binding rejection. Bytecode writes are disabled. This qualification never calls the scientific loader, executes transformed project code, imports Torch/model modules, draws data or fits.

`qualification/validation_stdout.txt`, `validation_stderr.txt` and `validation_result.json` retain test output and measured test/process wall time for root diagnostic accounting. `REPAIR_DIFF.patch` and `REPAIR_CHANGE_RECORD.json` give the copied-file changes and exact line spans. The source manifest includes the 12 copied authored files and the new validation script; the seal also binds retained qualification evidence. The v2 scientific freeze is immutable after sealing. Root still owns review, admission, coordination and accounting; passing AST/compile qualification does not certify the scientific experiment.
