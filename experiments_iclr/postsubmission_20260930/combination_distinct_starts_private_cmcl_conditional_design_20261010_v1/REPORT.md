# One conditional combination: distinct starts × private CMCL

10 October 2026. **Do not run now.** This is the sole prospective combination
retained here, with no scientific activation, executable patch or launch. It
uses one shared backbone. No partial M1, CMCL18 or current WikiCS quality was
opened or used. Original scores and failed gates remain unchanged.

## What the completed experiments actually support

| Completed evidence | Consequence for a combination |
| --- | --- |
| Newly closed M1-six: native 89.937426%, M1-native 90.816844%, M1-four-dropout 90.783020%, shared M4 90.901404% | Factorized singles largely reproduce the old shared-versus-native gain. Require a good factorized single before attributing usefulness to four routes. |
| Shared−M1-native +.253678/+.202943/−.202943pp, mean+.084559pp; shared−M1-four-dropout mean+.118383pp and NLL+.005442 worse | There is no stable all-seed accuracy gain over M1, nor a joint accuracy/NLL advantage over the four-loss M1. Another ensemble-specific branch has lower priority. |
| PubMed complete12: shared mean-member accuracy exceeds own-selected I4 by .97666pp, worst-member by 1.13310pp; shared pooled gain over I4 is .68493pp | The shared recipe already has competent members. Preserve this before creating alternatives. |
| Shared PubMed common-wrong counts 343/356/363; pooling loses an available correct member on only 4/8/2 nodes | The main target is missing correct alternatives, not recovery of many existing alternatives by a new serving rule. |
| Wiki15 alignment-only repairs 148/113/99 and harms 167/147/85; residual-only repairs 95/112/145 and harms 99/131/125 | Negative aggregate ingredients can still make many real repairs. These counts do not identify their repair-set intersections or stable distinct roles. |
| The actual Wiki15 alignment + residual factorial repairs 392 and harms 403 seed-node instances; mean interaction +.19593pp accompanies final delta −.06952pp | Interaction alone cannot justify a package. This completed combination must not be repeated with another coefficient. |
| Wiki24 first-adapter sign starts lose .8532/.8343/.0758pp to unit starts; adding its contrastive package still trails unit + package by 1.0997/1.0049/.6447pp | Breaking starting symmetry can weaken learning. Sign starts are an established, risky ingredient; broader coverage is not sufficient. |
| Wiki24 signs versus units: oracle coverage +2.667pp, mean-member competence −1.199pp, pool harms12→183.7 per seed; init+contrast versus init acquires315 correct alternatives on2503 baseline common-rival seed-node instances, serves128 and loses187 | This separates acquiring alternatives from keeping members competent and converting alternatives into served repairs. The315/128/187 counts are for contrast added to sign starts, not A/B repair intersections or unique cross-seed IDs. |
| Complete relation18 and QK36 fail their original joint quality comparisons; direct12 head refits add harms exceeding repairs in every seed | Gradient placement, geometric differences or readout freedom do not constitute an empirical competence remedy. |

The 15 selected WikiCS geometry endpoints refine the interpretation: alignment
and the combined objective improve class cosine gaps in all three seeds, yet
their serving remains unsupported. The residual-only endpoint is more aligned
than plain in every seed. SupCon/6203 has substantially more classifier-visible
separation but unchanged correct coverage, 23 additional pooling harms and
1.70170 worse NLL. An explanation based only on hidden null directions is
insufficient; useful predictions and harms must be measured directly.

The completed evidence identifies a sensible division of work—maintain ordinary
member learning and acquire correct alternatives—but **does not establish that
the two ingredients below repair distinct PubMed failures**. No cross-task repair
counts or independent backbones are combined to claim otherwise.

The M1 result is a competing optimization/reparameterization explanation, not
a causal identification: M1 adds13,943 factor coordinates, not the55,772 of M4,
and uses its own selected endpoint. Mixed per-class scalar differences do not
identify complementary repaired nodes. The complete report was opened only after
root confirmed all six successful owner closures; currentCMCL quality remains closed.

## Sole candidate and closest prior

Representative task: full PubMed node classification, the already encountered
19717-node graph, 500 features, all three classes, split seed190111, TRAIN11829
and VALID3942; TEST closed. Four private routes share the same native monoK2
PolyFormer, all23 FactorLinear sites and all original slow matrices. Total
coordinates remain 2,125,647: 2,069,875 shared/native plus55,772 private factors.

**A: the existing first-adapter Rademacher start.** After the native constructor
and factor wrapping, initialize only `model.lin1.r` (4×500) to independent ±1
draws using the existing `initialize_first_factor` helper and isolated CPU seed
`optimizer_seed + 710000001`. Keep every s and every later r at one. Use the
same signs in A and A+B. This changes initial route functions without adding
parameters, examples, a teacher, masks or backbones. It does not preserve native
initial predictions and is not a new graph-specific initializer.

**B: exact competence-anchored private CMCL credit.** Keep the currently frozen
M4/K3/beta.75/lambda1 rule: stable, stopped top3 ownership by CE−.75KL(U||P);
owner CE and nonowner .75KL; member-summed C, mean-own E. Shared data gradients
receive E, private factors E+C, one native Adam commit after all member
contributions. All members retain all TRAIN own supervision. The own-loss term
does not guarantee preserved competence; that remains a measured constraint.

A supplies the alternative-acquisition role suggested by the old sign-start
coverage gain. B's nonowner confidence penalty may help those new correct
alternatives survive averaging, while its own-floor/owner CE must preserve
competence and repair missing alternatives. This is a plausible division of
roles, not observed complementarity on PubMed. A also provides different initial
trajectories for B's current owner ranking. At copied deterministic starts, its stopped ownership
can instead favor member indices; dropout already supplies asymmetry. A can
damage competence, and B can merely flatten confidence or reinforce an early
bad assignment. These are falsifiable interaction hypotheses, not findings.

Closest ancestry is BatchEnsemble/TabM first-adapter initialization plus
[Confident Multiple Choice Learning](https://proceedings.mlr.press/v70/lee17b/lee17b.pdf),
with the already prepared selective shared/private gradient rule. The earlier
Wiki24 starts × contrastive factorial is the closest empirical collision and
failed. CMCL's confidence penalty/assignment and block-selective backward are
established; this note claims no new objective, function class or novelty.

## Exactly one full-task factorial

| Arm | Start | Data gradients |
| --- | --- | --- |
| 0 | Unit | Shared E; private E |
| A | First-input signs | Shared E; private E |
| B | Unit | Shared E; private E+C |
| A+B | Same first-input signs as A | Shared E; private E+C |

All four arms use seeds9101/9203/9307, unchanged native full-graph preprocessing,
dropout/member RNG, Adam groups, max2000/patience250, earliest strict maximum
complete pooled VALID accuracy and mean-probability serving. Use the current
CMCL source's CPU log-mean-softmax selection arithmetic for **every** arm.
It differs from the earlier complete12 direct FP32 probability selector; retain
that historical report as disclosed context, without altering its endpoints.
The newly completed M1-native and M1-four-dropout endpoints are mandatory external
factorized-single references. Preserve their own-selector/source identities and
costs, and keep comparisons exploratory; do not blend their predictions into A+B.

This is twelve complete arm-seed records. The six unit 0/B records may be
immutable anchors only after the entire current18 closes successfully and exact
source, task, runtime, RNG, selector, selected outputs and owner custody match.
Then A/A+B require six new full fits. A successor changes only declared start
construction/metadata, never a stored anchor or old source. If parity is not
established, the same factorial requires twelve fresh fits; do not silently mix
recipes. No further strength, site, seed or initializer grid is retained.

The factorial tests package utility and interaction. It does not isolate
state-dependent assignment from stronger private CE/KL under changed starts;
the current18 uniform/constant-credit controls apply to unit starts. A gain
cannot alone attribute a new selective-credit mechanism or support a paper claim.

## Fixed prospective decision

Record every seed and failure before interpreting quality. In percentage points,
require A+B minus0 positive in all three seeds and mean at least .2pp; require
A+B minusA and A+B minusB nonnegative in each seed and mean at least .1pp each.
Its mean pooled NLL must not exceed any of0/A/B. Mean-member accuracy may lose
at most .1pp and worst-member at most .2pp versus0 in each seed. Mean per-class
accuracy loss versus0 may not exceed .5pp, with no seed-class loss above1pp.
Macro F1, every class NLL and every member risk remain visible. These are proposed
prospective utility criteria, not modifications to any existing frozen screen.
Practical advancement also requires A+B accuracy positive versus each M1 reference
in all three seeds, mean at least .2pp, and mean NLL no worse than either. A
four-arm interaction alone cannot establish an ensemble-specific explanation;
source/selector matching and unused confirmation remain required for such a claim.

On the same selected VALID node IDs, save the16 joint pool-correctness patterns
for0/A/B/A+B, overall and by class. Define R_A/R_B as the respective repairs of
0's mistakes. Report R_A-only, R_B-only and shared repairs, all new harms,
which component repairs A+B retains, and A+B repairs absent from both components.
Also report exact node repair persistence across seeds on the common baseline
error cohort. Both exclusive component repair sets must contain at least4 nodes
per seed retained by A+B. An ingredient can have a negative total delta; these
criteria do not require A or B to be a standalone winner.

Measure the same flows for correct-member coverage and the fixed baseline false
rival; rival clearance is not top1 correctness. Separate member gains from pool
lift, lost alternatives and all-wrong rescues. Report the paired interaction
`accuracy(A+B)−accuracy(A)−accuracy(B)+accuracy(0)` and the analogous NLL contrast,
but it cannot override worse final quality. The chosen development endpoints
and three optimizer seeds remain exploratory, not unused confirmation or
independent graph-population replications.

## Minimal implementation and cost

No new model or owner is needed. A reviewed successor of the existing CMCL
package would add a bound `start_mode` to `native.build`, Session identity,
snapshot and fresh reconstruction. Call the existing first-factor helper on
the explicit PolyFormer stem after wrapping and before original Adam creation;
replace the all-unit guard with an exact stem-only sign/all-other-unit guard.
Reuse current own_floor/private_cmcl steps, same-state score/replay and fit
routines. The current R/S ownership and shared native parameter identities stay
exact. Construction and sign draws must be charged. Two changed-start full-TRAIN
update/fresh-restore qualifications resolve the concrete source delta before fits;
no tiny-population or reduced-task surrogate is useful here.

| Additional scientific work | Six new A/A+B fits if anchors qualify | Twelve fresh fits otherwise |
| --- | ---: | ---: |
| Maximum bank/Adam updates |12,000|24,000|
| Gradient TRAIN forwards |48,000|96,000|
| Stopped-assignment TRAIN forwards |24,000|48,000|
| Parameter VJPs |72,000|144,000|
| Regular full-graph serving forwards |48,000|96,000|
| Fresh selected reconstruction forwards |24|48|
| Full body constructors including reconstruction |12|24|
| Preprocessing banks |6|12|

The existing per-fit finite bounds are active9000s + cleanup10s, GPU32GiB,
RSS16GiB, output512MiB, log8MiB, no retry. Six new fits have a54,060s (15.02h)
sum of active/cleanup caps; two600s+10s qualifiers add1220s. These are ceilings,
not an ETA or isolated speed estimate. There is no qualified wall estimate for
this changed-start combination. Charge both qualifications, all fits/failures,
checkpoint/HISTORY rewrites, saved outputs, reconstruction, reader, transport
and owner cleanup; retained existing costs are not free or double-added. Serving
requires four full native forwards, with no certified latency ratio to historical
shared anchors.

## When not to run

M1-six is now closed; current CMCL/initializer quality remains unavailable.
Interpret those complete frozen comparisons before another run. M1 largely
reproduces the shared gain, so prioritize the simpler factorized learner's
references and unused confirmation unless exact useful-alternative evidence
separately warrants M4. A supported existing rule needing confirmation takes
priority over this conditional combination.

Do not infer a combination from softened confidence, hidden separation, positive
interaction, broader oracle counts or many repairs offset by uncontrolled harms.
Do not reopen the failed masked-context continuation, graph-filtered initializer,
direct head refit or original contrastive factorial. Do not blend independent
predictors into this candidate. Crucially, the currently read evidence lacks
matched PubMed A/B repair intersections: it supplies a conditional sketch, not
an admission case. Without newly completed evidence supporting distinct stable
repairs and the proposed competence protection, leave this sole design inactive.

## Evidence used

Only completed compact reports, selected ledger dispositions and source text were
read. Principal files: `historical_pubmed_strong_reference_followup_20261010_v1/REPORT.md`;
`Wiki15_functional_diversity_analysis_20261008_v1/REPORT.md` and its
`PAIRED_FLOW_DIAGNOSTICS.csv`; `Wiki24_analysis_after_closure_execution_root_20261007_v1/readout/REPORT.md`;
`Wiki15_selected_member_geometry_descriptive_execution_root_20261010_v2/INTERPRETATION.md`;
`wikics24_scientific_interpretation_independent_20261007_v1/REPORT.md`;
`relation18_completed_scientific_decision_root_20261010_v1/INTERPRETATION.md`;
`Wiki24_direct12_closed_result_independent_assessment_20261008_v1/CONCLUSIONS.json`;
the complete QK36 disposition in `research_ledger.json`; and the frozen private
CMCL protocol/native/session source plus existing `factors.py`. No source
numerical import, model, raw prediction/label array, checkpoint, scientific remote
operation or score recalculation was performed for this report.

The newly closed M1 source is
`pubmed_factor1_complete6_scalar_comparison_root_20261010_v1/fetched/pubmed_factor1_complete6_scalar_comparison_execution_20261010_v1/COMPARISON.json`,
SHA256 `ab4887437af18f2e11cb9b4d6bcbaace4b41636b440c6873fc9ccbbc559b692d`.
Its complete-six/reference12 flags were verified; it contains stored scalar
readouts, no new model calls or array reads.
