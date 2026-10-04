# Isolated accuracy-first graph-view source preparation

**Source implementation only; no scientific freeze, execution admission or production release.** Fixed graph views are a known diversification baseline/probe. The hypothesis concerns whether sharing weights improves useful native-graph ensemble predictions relative to otherwise matched untied trajectories. Parameter saving is secondary. This package establishes no accuracy, specialization or novelty result.

V2 is a minimal device-handling successor. CUDA detection uses `torch.device(device).type`; implicit CUDA requests resolve to an explicit current index before allocation/seed/state handling. RNG images, construction and driver bindings record selected-device provenance, and state schema v2 rejects mismatched device provenance. The four-bank protocol, 10% views, objectives and immutable neural operations are unchanged. V2 numerical verification remains pending; root's reported five passing v1 CPU tests are historical evidence for sealed v1 only. The missing reference plan is separate in `REFERENCE_IMPLEMENTATION_PLAN.md` and adds no bank or executable control.

## Implemented

- `views.py`: all official TRAIN labels, no extra FIT/control reservation; equal/different TRAIN-TRAIN unordered categories; floor 10% deletion per category; bidirected deletion; every native self-loop retained; surviving edge order preserved. Random nulls match semantic deletion counts in native endpoint-degree-bin strata. The null pool ignores classes and may overlap the semantic deletion set; overlap is reported. Undersupply selects available pairs, records deficits and rejects scientific freezing, without borrowing/retry. Empty semantic deletion masks also reject freezing.
- `bank.py`: one tied complete M4 family or four independent copies of exactly the initialized factorized trajectories. Each untied copy owns its interior, boundary W and corresponding one-row R/S/B. Copies consume no constructor/reset RNG. Both wrappers call the byte-bound immutable v6 member forward. Model/Adam ownership, native primitives, stage flags, modes, gradients and Python/NumPy/Torch CPU/selected CUDA streams are stored/restored.
- `schedule.py`: one bank clock and strict earliest-tie selector; two probes per view at every update; fixed member roles or complementary hash-indexed assignments balanced per member separately over 200 local and 2,500 global updates.
- `driver.py`: four native passes followed by four probe passes; each CE/4 backpropagates immediately, then one Adam step. This preserves pre-step parameters and avoids retaining eight full graph tapes. Dropout stays enabled at native locations/rates. All conditions start from the same seeded constructor/device-reset/factor-wrap stream and use the same pass order/opportunities, with no per-pass reseed. Numerical equivalence and any shape-induced stream differences still require verification.
- Native-graph inference uses the fixed mean of four class probability vectors. One pooled VALIDATION correct-count selector spans both stages. At actual update 200, selected-local model+Adam rewind while live RNG and actual schedule cursor remain; every core then enters global mode. Final restoration can select a local checkpoint. TRAIN reports are fitted-function diagnostics; selected VALIDATION accuracy/NLL/macro-F1 and member accuracy/NLL are exploratory development evidence.

Global mode still executes all local GAT layers. Masks act on their edge support in both stages; they do not directly mask global attention. Local weights remain trainable in global mode. No view-local representation is cached. Exact operation, initialization, stage-transition and block bindings are in `SOURCE_BINDINGS.json`; immutable v6 is neither copied nor edited. Verified source bytes are compiled without creating bytecode caches inside v6.

Source/context descriptors use paths relative to this folder's parent phase directory. Runtime resolution uses `ROOT.parent`, and records the actual resolved neural paths in provenance. The same bytes can be verified on the authorized server without embedded Mac path dependencies.

## Default and evidence status

`python3 -B driver.py` prints source-only metadata and imports no numerical packages. Numerical entry is a deliberate external call to `load_runtime(execute=True, manifest_sha256=..., protocol_sha256=...)`, after root admission. Training additionally requires its explicit `execute=True` and an external frozen protocol binding. There are no host restrictions, custom filesystem sandboxes, registrations or automatic launch/qualification stages.

`COVERAGE_STATUS.json` leaves **real fixed-mask/null-stratum coverage pending**. Root's separate TRAIN-only eligible-edge census is referenced there; no masks were created by that census, and its counts do not prove probe strength. `SYNTHETIC_COVERAGE.json` is a 30-node implementation fixture and cannot satisfy actual mask coverage. A future admitted caller must prepare the graph with native preprocessing, save actual TRAIN coverage first, and bind its exact descriptor in an external freeze. Scientific entry checks that saved coverage equals the live roles/masks, is eligible, and binds the declared split/seeds, raw feature bytes, validation targets, all four conditions and this source/protocol. No actual freeze is supplied here.

The stdlib suite checks label-domain/mask exclusion, bidirected identity/loops/order, coverage custody, deterministic null undersupply, both-stage exposure balance, one strict selector, clock semantics and default import exclusion. `STDLIB_TEST_RESULTS.json` records the run. These are implementation checks, not predictive experiments.

`test_numerical.py` remains **unrun**. After explicit root admission in the pinned CPU runtime, it tests:

1. Exact tied/untied initial functions and factor-row correspondence, separate storage, both stages, dropout-off and paired training streams.
2. CE/4 accumulation versus the summed-loss objective, gradients and one-step parameters in both stages/banks, using declared FP32 comparison tolerances; RNG equality and absence of mutable forward-buffer changes. Tolerance checks do not certify bitwise arithmetic equivalence.
3. Full model/Adam/RNG/flag/mode/gradient restoration and exact next-update replay in both stages/banks; primitive mismatch rejection.
4. Selected-local model/Adam transition with live RNG and actual update cursor preserved, plus valid local final restoration.
5. Correct probability pooling rather than logit averaging.

The numerical fixtures use the exact operations with smaller artificial feature/hidden/layer dimensions for CPU implementation verification. They do not qualify the full Amazon shape, resource cost or predictive recipe.

## Fatal confounds and remaining work

- Immutable v6 uses a smaller FIT budget and independently selected conventional members. Its current fits cannot be decisive matched controls for this all-TRAIN pooled-bank comparison.
- This patch implements **four bank arms only**. A competent same-budget modern single and conventionally initialized independent reference, paired outcome synthesis, worthwhile-gain decision rule and separate confirmation/TEST-history protocol remain unimplemented. Four-bank results alone cannot establish the complete quality claim in the decision memo.
- Acquisition and verification of the existing dataset/compact-label projection provenance remain the admitted caller's responsibility. This package has no data acquisition or TEST-label reader. No real labels, graph tensors or checkpoints were inspected during preparation.
- Numerical tests, full-shape runtime behavior, memory/time measurements and resource admission remain pending. Every update charges eight forwards and eight backwards plus four native selection forwards. Driver timing includes its preparation/updates/restoration/metric/checkpoint outputs; external coverage preparation and complete host process costs require separate measurement. No memory-cap increase is authorized.
- No scientific mid-run restart orchestration is included. The full selected image supports function/state restoration; selection events are retained in the trace, while only selected-local/final images are saved. There is no promise to reopen every transient strict-improvement image.
- Degree strata match coarse endpoint bins and counts, not exact per-node degree losses or identical perturbation semantics. Coverage reveals availability, not whether either probe is useful. VALIDATION is reused for checkpoint selection and development assessment; it supplies no independent heldout or unseen TEST claim.

All work stays in this new folder. Existing v6 and current studies remain unchanged.
