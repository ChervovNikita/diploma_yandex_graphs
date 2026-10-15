# A concrete prospective primary question

**Proposal for the user's final design review; not adopted or launched.**
The full method-by-preset matrix remains visible. This option asks whether one
frozen shared-ensemble recipe improves average quality on this chosen benchmark.
It does not make a claim of superiority on every cell or on unseen graph tasks.
It changes the scientific estimand prospectively, rather than changing a test to
make the old 226-cell family significant.

## Selection, then confirmation

1. Use TRAIN and selected VALID outputs to develop and screen candidate recipes.
   Preserve every failure and every candidate's VALID result. Freeze one complete
   recipe, including its task adaptations, coefficients, aggregation, source
   presets and applicability, before confirmation training and TEST readout.
   Use the same recipe across its declared compatible backbones. Do not choose
   a different winning recipe for each preset. A recipe that only applies to
   architectures with BatchNorm must use a prospectively restricted estimand;
   it cannot silently replace the all-backbone candidate.
2. Freeze new independent paired confirmation seeds, complete fit counts and
   source-preserving selection rules. Original paper scores and separate source
   reproduction fits are historical checks, not paired confirmation samples.
   TEST cannot set a coefficient, candidate, seed count or inclusion rule.
3. Evaluate the selected shared candidate, base GNNM, native M1 and genuine I4
   at their independently frozen checkpoints. Report all cells and per-trial
   differences. The single is exact native body0; I4 pools its four own selected
   native fits. The candidate never receives the independent ensemble's weights.

## Two primary contrasts

For each confirmation trial, compute candidate-minus-I4 and candidate-minus-M1
on the native primary metric, with all metrics on their native [0,1] scale.
Average first over the prespecified source presets **within each dataset**, then
average those dataset differences with equal dataset weight. Thus sixteen OGB
presets do not give molecules sixteen times the weight of one link dataset.
The resulting quantity is an explicitly chosen mean percentage-point change
on this finite benchmark. Accuracy, ROC-AUC and Hits@50 have different meanings;
their weighted average is a declared summary, not a common real-world utility.
Publish the seven component differences alongside it, including all losses.

Apply the already proposed exact two-sided sign-flip mean test to each of these
two trial-level contrast vectors. Set alpha=.05 and use Holm across these **two**
primary questions. Give marginal paired confidence intervals and all individual
differences. Candidate-minus-base GNNM and per-cell/model/dataset breakdowns
remain descriptive secondary results unless a separate valid prospective error
allocation is frozen. Do not call their unadjusted intervals simultaneous.

At ten effective nonzero trials the sign-flip floor is .001953125, below the
first Holm threshold .025 for two questions. This removes the old design's
mathematical impossibility; it says nothing about adequate power or likely
success. Trial count still needs an effect target and an honest operating-
characteristic assessment before adoption. A null result is not equivalence.

## Assumptions and limits

- One primary observation is a whole paired confirmation trial, not an edge,
  node, molecule, route, epoch, preset, or pooled seed-row count. Repeated
  presets on the same graph must not become independent observations.
- The sign-flip test assumes independent, sign-symmetric trial differences
  under its null. Pairing, fixed common data and shared seed labels alone do
  not prove that assumption. The full source split/seed roster must be reviewed
  before choosing this analysis, especially where public splits differ by trial.
- Inference is conditional on the chosen datasets, source settings and TEST
  populations. More training seeds do not create new datasets or populations.
- Record prior TEST exposure in earlier research. New frozen seeds and a closed
  current TEST gate do not erase that history or make the whole programme a
  first-use holdout study.
- Missing/failed fits have no favorable substitute and no automatic retry.
  Report coverage and costs; an incomplete bank prevents an unqualified complete-
  benchmark primary conclusion. A recovery or design amendment must be recorded
  independently of the unseen TEST result.
- No method is selected by the size of a TEST effect or its p-value. The complete
  expanded fit roster, screening/confirmation separation and source readiness
  still require the user's review before the large launch.
