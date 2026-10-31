# Independent analysis companion v2 source review

The statistical formulas, real registry mapping and closure-before-report ordering are correct. **One provenance correction is required before the companion can claim independently verified outer supervision costs.** Its `closed_cohort` guard does not bind each accepted outer terminal to the corresponding attempt. This review establishes a source defect, not an error in any actual experiment cost or outcome.

The inspected analysis source SHA256 is `8e8816cc01e3ef595fb54785d81c2a72f473f9da2e8cd9ab4ae609138eeac770`; the actual registered cohort SHA256 is `715c361c82b543d4c436a565ccaa86470a9204433998d410954f2f268300307f`. Exact inspected hashes, the source snapshot, synthetic check script and results are preserved here. The F1 source review remains separately sealed.

Only local source, source-preparation JSON and the prospective registry JSON were read. No real final report, completion/phase/whole terminal, phase freeze, label, logit, checkpoint or scientific outcome was opened. No SSH, remote action, scientific import or scientific execution occurred. All executed checks use stdlib and synthetic differences/in-memory metadata. The reviewed source packet was not edited.

## Required correction: outer cost identity

`analyze_complete_report.py:116–123` verifies the descriptor hash, successful boolean fields and equality between completion-receipt seconds and the referenced terminal's seconds. It does not require the terminal's canonical per-key outer path, uniqueness across the 72 attempts, the terminal's `START_sha256`, or a START/root-request binding to the key's launcher request. Hash equality proves the identity of the referenced file; it does not prove that this file supervised the current attempt.

The independent check reproduces the problem using an entirely in-memory synthetic 72-phase cohort with the real registry's context/key structure. All canonical phase terminal/claim/output identities and all phase/arm counts are retained. Every completion references **one** synthetic successful outer terminal containing seven seconds. `closed_cohort` accepts all 72 rows and the companion's cohort sum becomes **504 seconds**, while only one distinct outer terminal was supplied. No physical fixture or real experiment receipt was read or written. `SOURCE_CHECKS.json` records this counterexample, and `review_source_checks.py` reproduces it.

The coordinator source writes each outer descriptor from the specific request's `outer_supervisor_directory` (`graph_init_precision_continuation_v3_cap_binding/finite_coordinator.py:225–235`). Request preparation and the continuation entry bind that directory to `supervision/<key>_outer`, the request to `requests/<key>/REQUEST.json`, and the inner directory to `supervision/<key>_inner` (`build_continuation.py:105–127`; `continuation_entry.py:64–82`). The outer supervisor writes both the request descriptor into START and `START_sha256` into TERMINAL (`protocols/bounded_run_v1.py:139–177`). Those bindings are available for independent verification.

A specific correction is to:

1. Require every `whole_terminal` to resolve to `run/supervision/<key>_outer/TERMINAL.json`, and require 72 distinct outer paths.
2. Validate the outer terminal schema and bind its `START_sha256` to the corresponding canonical `START.json`.
3. Retain the full launch-claim document, hash-check its request descriptor, and require the canonical `run/requests/<key>/REQUEST.json` path. Check its attempt/output against the registered attempt and the launch claim.
4. Require START's `root_request` path/hash, `outer_output`, inner supervisor path and cap to match that exact request/key; require the observed seconds to lie within the verified cap rather than trusting only a boolean flag.
5. Save these request/START/terminal identities in the cost bindings. Add a synthetic rejection check for a shared or swapped outer terminal.

This preserves the scientific recipe, statistics and outcomes. The correction belongs in a new source revision; the v2 review and its inspected snapshot should remain unchanged.

## Statistics

The source computes the sample standard deviation across the three context differences and uses `t_(.975,df=2) × sd / sqrt(3)` for the mean interval (`analyze_complete_report.py:61–73`). The closed-form quantile is correct: the df=2 CDF is `1/2 + t/(2 sqrt(t²+2))`; setting it to .975 gives `sqrt(2 × .95² / (1−.95²)) = 4.302652729749464`. Independent CDF bisection reproduces the interval. The preparation failure preserved in `FAILED_PREPARATION_01.json` concerns an over-tight comparison to a rounded numeric reference, not an estimator defect.

The sign reference excludes exact zeros and calculates the two-sided exact fair-sign tail, capped at one. All 27 three-entry vectors over `{-1,0,1}` agree with independent enumeration. Three same-sign nonzero differences produce .25; fewer nonzero pairs have still larger minimum p values. The eight-contrast Holm block correctly sorts, applies decreasing multipliers and takes a cumulative maximum (`analyze_complete_report.py:207–212`). Because every possible p value in this cohort is at least .25, **all eight Holm-adjusted p values necessarily equal one**, for any outcomes. The rejection flags can never establish significance at .05 in this cohort.

Holm does not require independent tests across the eight overlapping controls; marginal sign-test validity still needs independent symmetric signs within each contrast. The registered seeds also couple seed and split index, and the graph nodes are shared. The plan correctly limits interpretation to conditional, exploratory setting-specific context summaries, requires approximate normality/independence for the t interval, forbids a new-graph confidence claim, and does not turn the two settings into six independent graphs. Accuracy is secondary and descriptive. Degenerate intervals from three identical observed differences are an arithmetic result, not evidence of zero population uncertainty.

## Actual registry and report guard

The actual six contexts are Squirrel/polyformer_mono and Photo/polynormer_r, each at seeds 17/29/43 paired with split indices 0/1/2 and config zero. For every actual attempt, independently recomputed context hashes and native keys `hash({context_sha256, phase, arm})` match the registry. Each context has one qualify, one warm, five initialize and five fit entries. Thus the companion's context hash mapping and its expected 6/6/30/30 phase counts are valid. The source does not mistakenly hash a reduced graph/seed record or infer keys from filenames.

`main` invokes `closed_cohort` before `validated_report` (`analyze_complete_report.py:196–197`). Closure checks the exact coordinator path, absence of FAILED, the successful closure schema/count, the pinned registry, all 72 receipt keys and their canonical phase identities before report access. A synthetic pending-closure check confirms that the report reader is never called. This is a source/synthetic verification; genuine cohort closure was not inspected.

Report validation requires the registry's canonical report/comparison paths, separate admission metadata, copied source-selection agreement, exactly 30 setting/seed/arm rows and 24 contrasts, finite/domain-valid metrics and recomputed graph-minus-control differences. The native reporter source emits those schemas and fields (`prototype/graph_init_driver.py:913–1029, 1068–1173`). Selected tensor integrity remains with that original admitted reporter; the companion must not open its label/array/checkpoint descriptors. No final report was available to this review or executed through the companion.

## Cost interpretation and remaining qualification

Given correctly bound distinct supervision terminals, summing one duration per registered attempt and charging qualification/warm once per setting is correct. Per-arm own costs include its initialize and fit phases. Each standalone account adds that setting's shared qualification/warm costs; those overlapping accounts cannot be summed as actual cohort expenditure. The source and plan state this explicitly (`analyze_complete_report.py:213–228`). The narrow counter excludes acquisition, environment/deployment work, earlier failed studies, coordinator transport/admission overhead outside the outer processes, and final scoring. These excluded costs still require disclosure when discussing full project expenditure. Parameter storage does not establish runtime, memory or latency advantage.

After the outer identity correction and independent source recheck, runtime qualification still requires genuine 72-attempt closure, complete local mirrors of the required metadata, the existing separate final-report admission, and successful execution on the canonical report. No source preparation or synthetic test substitutes for those gates. This review introduces no significance, superiority, originality or acceptance claim.
