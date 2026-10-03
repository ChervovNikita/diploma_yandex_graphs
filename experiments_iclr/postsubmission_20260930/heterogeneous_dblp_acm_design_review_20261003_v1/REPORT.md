# Prospective DBLP → ACM scientific design review

2026-10-03. **Recommendation only; no study adoption or execution clearance.**

## Decision

The seven HGT configurations form a useful prospective comparison of configured predictors. Freeze all five seeds and all seven arms before reading any new score. Treat DBLP as a continuation screen, then require complete, independently frozen ACM and heldout confirmation plus competent native challengers before a broader predictive-superiority claim. A strict negative mean alone is insufficient: a numerical near-tie must remain a near-tie.

The current contrast cannot identify learned member × relation interaction as the cause of improvement. CP starts from a different function than global BE and the shared-relation control. The closest-prior report also rules out a defensible new-operator claim. The defensible hypothesis concerns utility of a known conditional diagonal adapter with CP sharing, composed with HGT and ordinary BatchEnsemble.

## What is qualified by this review

The training-preparation v1 manifest and all 11 payload hashes match; its 15 nonnumeric provenance bindings match. The closest-prior manifest and all 60 payload hashes match. These checks establish packet integrity only. The reviewed v1 preparation and template remain unadopted and retain placeholders; no source execution qualification is granted by this review. During completion the root reported that its five separate development splits were successfully frozen and the repaired v2 CPU replay passed. Those later source/admission receipts were not redundantly checked here. The root should bind the exact repaired v2 source, replay and frozen-split release before training admission; the earlier v1 scheduler serialization issue is not a scientific objection to that qualified continuation.

Only saved source and schema metadata were read. No archive, original label payload, fitted checkpoint, new study outcome, remote operation, or native/Torch execution was performed. Prior conclusions were reused from the sealed prior-art report; its primary papers were not newly method-read.

## Data and paired design

The actual-schema receipt, parsed through its two nested stdout JSON wrappers, binds archive SHA `0d3ea4a74399f9cd3e83af206e8e0b67e1844fe2c8463b424189884dd58ad7c8` (2,567,741 bytes): 26,128 nodes; type counts 4,057/14,328/7,723/20; six raw directed relations; 239,566 unit-weight records; no duplicates or self edges. Public metadata doubles edge counts and misstates target features. Use the payload schema without synthetic duplication, reverse relations, or self edges.

Native feature-type2 uses 334 target attributes and sparse identities of widths 14,328/7,723/20 elsewhere, giving input-width sum 22,405. The source development pool contains 1,217 targets. Each fixed seed 131/137/139/149/151 shuffles the same sorted pool using NumPy RandomState; 243 become validation and 974 TRAIN. All arms must use the same byte-bound split per seed. Source checks TRAIN class coverage over classes 0–3. These are five paired split/seed blocks on one transductive graph: split, core/factor initialization, and dropout variability are coupled. They do not separate these sources of variation or represent five independent datasets.

Freeze the exact arm list: `native_HGT`, `global_BE`, `shared_relation`, `CP`, `unrestricted`, `untied_HGT`, `wider_BE`. This means 35 HGT terminals. Although the v1 guard accepts a five-arm minimum, the proposed study requires all seven; do not shrink the denominator after a failure. Retain resource deferrals and failed fits and withhold comparison eligibility until all 35 complete under the source and selection gates.

## What the controls identify

| Contrast | Useful interpretation | Remaining limitation |
| --- | --- | --- |
| CP vs global BE | Incremental utility of the configured relation-conditioned predictor | Different initial functions; interaction, perturbation, optimizer and decay effects are combined. |
| CP vs shared relation | Utility beyond a nearly equal-capacity common relation residual | Shared rho₀=.01 matches CP RMS c, not its initial served function; this is not a causal isolation of learned private c. |
| CP vs unrestricted table | Utility of a restriction relative to more flexible same-site conditioning | Initial functions match exactly in FP32, but coordinates, optimization and weight decay differ. A shared gain supports relation conditioning, not a uniquely beneficial CP restriction. |
| CP vs wider BE 72 | Whether generic extra HGT capacity is a competitive explanation | Width 72 is a coarse overcapacity control, not a closely matched budget. |
| CP vs untied HGT | Utility relative to four complete independent parameter sets | One joint mean-member objective and common mean-logit selection; not four independently selected native jobs. |
| CP vs native HGT | Utility relative to the paired original backbone | Four private trajectories versus one; neither parameter sharing nor rank proves predictive or computational superiority. |

Source-reported trainable parameters are: native 1,654,240; BE 1,655,776; shared 1,655,989; CP 1,655,998; unrestricted 1,659,616; wider 1,892,968; untied 6,616,960. CP adds 222 parameters to BE and 9 to shared. CP saves 3,618 versus unrestricted—about 0.22% of total parameters. These small differences cannot establish practically meaningful whole-model efficiency. Wider adds 236,970 versus CP. Four full private hidden-state/attention/dropout trajectories remain.

Before training, save epoch 0 served validation metrics and prediction fingerprints for the existing arms as an initialization diagnostic, while preserving epoch 0's ineligibility under the native recipe. Initial-to-selected changes remain descriptive; subtraction does not remove initialization confounding. Add no new arm, aggregation rule, or tuning search. If a causal learned-interaction claim is necessary, it needs a separately prospectively designed study; narrow the present claim instead.

## Recommended prospective DBLP continuation gate

Freeze this gate, its signed contrasts, and its thresholds before any new outcome. Let `d_s = NLL_CP,s − NLL_control,s` at the selected raw mean-logit predictor. For **each** primary control, global BE and shared relation, require:

1. All35 frozen HGT terminals are valid, with exact split/source bindings, complete-state replay, native post-update/latest-tie selection, and complete paid-cost/failure receipts.
2. Mean paired NLL difference is at most **−0.005 nats**.
3. CP has a strictly lower NLL in **at least 4 of 5** paired blocks.
4. Mean paired macro-F1 difference is **≥0** (strict mean nondecline), over the fixed four-class schema.

Retain the existing illustrative paired t95 interval as a descriptive stability diagnostic; its assumptions are not qualified here and it is not a significance/power guarantee. An upper endpoint <0 can be frozen now as an additional conservative filter if the root elects it, but it is not required by this recommended minimum continuation gate. Never promote the interval to proof of methodological success.

This is a policy margin, not a measured minimal meaningful effect or a power-derived value. A 0.005-nat gain corresponds to about 0.50% higher geometric-mean assigned probability of the true class; it intentionally excludes tiny numerical differences. If a domain-specific margin is preferred, replace it prospectively once, before any scores, and record the rationale. Do not reduce the margin after outcomes.

A failure, deferral, incomplete denominator, or near-tie means **DBLP continuation gate not passed**. Preserve all results and do not claim methodological success or retune the gate. A pass authorizes the already planned complete ACM/heldout confirmation only. Report CP-minus-control vectors for every control, not just the two gated controls, plus mean, SD/SE, individual signs, illustrative intervals and leave-one-block-out mean sensitivity. These remain conditional development descriptions after validation selection.

Freeze ACM schema/split/recipe/arms, external-native protocols, calibration and final-evaluation rules before DBLP outcome inspection where feasible. Require the same two primary practical NLL margins and macro-F1 guard separately on DBLP and ACM final evaluation, with all five paired seeds retained. Do not pool a gain on one graph against a loss on the other. Development selection and graph continuation cannot count as independent heldout confirmation. This review does not itself release final labels.

## Competent native challengers and calibration

**GAT, Simple-HGN and SeHGNN** are required native challengers; Simple-HGN and SeHGNN are the essential heterogeneous competence checks. Run their qualified native preprocessing/recipes on the complete graph and identical frozen target splits and available labels, five seeds each. Disclose native feature and selection differences; never substitute published scores. Any supervised propagation or label embeddings must obey TRAIN-only labels during development scoring. This adds 15 DBLP configurations, for 50 including the HGT family; incomplete native challengers prevent a broad benchmark-superiority claim. A CP win within HGT is only an HGT-family result if stronger natives remain competitive.

Keep checkpoint selection and the DBLP continuation gate on the currently served **raw mean logits**. The driver already records raw NLL, micro-F1 (accuracy here), and fixed-class macro-F1. Mean logits and mean probabilities are different predictors; keep the prescribed aggregation throughout and make no aggregation choice from outcomes.

Freeze one scalar-temperature fit per selected served predictor, fit only using the source validation labels after selection, and apply it to independent heldout evaluation. Freeze the objective, solver/bounds/fallback, and shared protocol across arms. The same validation observations used for selection and temperature fitting cannot supply an unbiased calibrated development score. Preserve raw and calibrated NLL and multiclass Brier score, plus micro/macro-F1; report ECE only as a fixed-bin diagnostic. With 243 validation examples, ECE is too fragile to be the principal calibration claim. For a final probabilistic-utility claim, require the same 0.005-nat mean gain after temperature calibration against both primary controls on each graph. Raw success with a calibrated near-tie does not satisfy that claim. Brier and fixed-bin reliability diagnostics remain fully reported auxiliary checks, without post-hoc metric substitution.

## Cost-reporting concern discovered in source

`families.build` places every requested model on the GPU before the loop over arms. `fit` then resets/reports device-wide peak allocated/reserved memory while untrained remaining arms stay resident. Peaks therefore include other arms and depend on arm order; reserved caching can also carry over. Keep these as process receipts, but do not label them isolated per-arm memory or infer arm-memory efficiency from their differences. For a memory-efficiency claim, qualify isolated arm measurement with a common graph/features baseline and prospectively defined timing scope. Current wall time covers fitting/selection/checkpoint work but excludes family construction; disclose that scope. Include all attempted/failing work and private trajectory count in any cost claim.

## Claim boundary

A complete positive final study could support a narrowly stated, graph-conditioned predictive benefit of the tested configuration and a descriptive parameter/cost comparison. It would not establish a new modulation operator, semantic understanding of relations, general superiority across graphs, a uniquely causal role for learned CP interaction, or reduced computation merely from matrix sharing. If unrestricted or wider controls match the gains, report that explanation directly. No outcomes are available in this review. **Concrete verdict: GO for the complete seven-arm HGT development comparison under the root-qualified repaired v2 release and the prospective gate above; NO-GO for methodological or broad superiority claims until native-challenger and independent ACM/heldout closure.**
