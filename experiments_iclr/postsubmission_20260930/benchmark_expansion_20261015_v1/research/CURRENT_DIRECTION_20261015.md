# Research direction and next decisions

We want several trainable prediction routes through one shared native GNN.
The routes should learn useful alternatives, while adding few parameters.
The target is better predictions than both the native single model and four
independently trained copies. Portability across backbones is a design goal;
it remains something to test. A result on one backbone would establish a
limited useful setting, not universal improvement.

## What the completed experiments have taught us

On the official MolBACE scaffold split, ordinary GNNM is substantially weaker
than the independent ensemble. Giving each route its own batch-normalization
running statistics recovers much of the loss. Those statistics have known
precedents and are an implementation control, not our novelty claim.

Replacing the multiplicative BatchEnsemble factors with small additive input
corrections further improves the mean ranking score. This version retains one
native GIN and adds 1.0183% learned parameters. Its mean validation AUC is
74.518%, compared with 74.847% for the independent ensemble. It wins one of
three paired seeds, has much more seed variation, and has worse probability
loss in all three. It has not met the research goal.

The routes themselves now rank reasonably well, but averaging adds almost
nothing. About 95.54% of the pool's ranking reversals are wrong in every
route. More hidden-state disagreement would not, by itself, repair these
errors. This motivates supervised specialization that preserves individual
quality, followed by direct measurement of what averaging repairs and harms.

These findings use selected validation results from a small, repeatedly
examined split. All unsuccessful combinations, paired outcomes and costs
remain in the [complete result](../pilot_20261015_residual_only_private_bn_v1/complete_results_v1/INTERPRETATION.md).
TEST remains closed.

A complete saved-prediction diagnostic now identifies a class-specific
probability failure. Compared with I4, residual-only increases BCE on positive
molecules and reduces it on negatives in every seed. At the fixed diagnostic
0.5 threshold, false negatives rise from 67/46/60 to 105/95/78 out of 130
positives. The five largest loss increases explain only 12.3/17.7/38.2% of net
excess BCE; the problem is not confined to a few outliers. These facts support
ranking recovery, not general recovery of prediction quality. TRAIN and VALID
positive prevalence differ (39.67% versus 86.09%), but this does not establish
pure label shift or explain the paired method difference. No correction was
fitted. See the [complete diagnostic interpretation](../analysis/completed_probability_failure_v1/INTERPRETATION.md).

## The current sequence

| Study | Question and comparison | Status |
| --- | --- | --- |
| MolBACE structural responsibilities | Does making small private corrections specialize by TRAIN graph size improve on uniform responsibilities and a shuffled-assignment control? Native shared parameters still receive the ordinary loss. | Six full fits running over three seeds; existing controls reused. Interpret only after all six finish. |
| PubMed / native PolyFormer-Cheb | Does the residual and responsibility design work on a different backbone and node-classification task without batch normalization? Compare native M1/I4, original GNNM, uniform, structural and shuffled variants. | Source and three CPU lifecycle checks complete. Seven GPU engineering checks wait for the molecular study. Budgeting the proposed 24 fits requires measured production costs. |
| MolBACE pairwise supervision | Can routes learn different useful ranking comparisons while keeping the native classification loss? Compare common, structural and shuffled pair weights; include independently trained native ensembles with the same common ranking loss. | All four CPU lifecycle checks passed. The complete 21-fit comparison is frozen and allocation preflight passed; the CUDA cost probe is prepared. No scientific fit has launched. |
| Broad reference-wrapper campaign | Which methods transfer across the fixed native paper settings? Preserve source reproductions, ordinary independent ensembles and original GNNM throughout. | 6,460 proposed fits; serving coverage and measured placement/cost remain incomplete. Requires user review and GO. |

The pairwise idea uses the model's existing logits. No teacher or independently
trained donor is acquired, and inference remains the same four-route mean.
The structural pair roles use only TRAIN molecular inputs. The common-loss
independent ensemble is essential: improving the training loss alone must not
be mistaken for an advantage of the shared wrapper. Prior work already
establishes pairwise AUC learning and shared/private specialist rankers; the
[literature note](pairwise_competence_prior_followup_v1/NOTE.md) records the
remaining empirical question and unresolved prior-art leads.

Our default next priority is the existing PubMed comparison, after complete
MolBACE outcomes and measured GPU costs permit it. This tests another native
backbone and prediction task without batch normalization. It is more informative
for transfer than another round on the same 151 molecular validation records.
The pairwise study remains a coherent secondary hypothesis: it directly trains
ranking decisions and gives the independent ensemble the same auxiliary loss.
PubMed cannot validate that ranking-specific mechanism. The
[priority assessment](next_study_priority_20261015_v1/NOTE.md) records the
alternatives, fixed comparisons and conditions that would reject each claim.
Neither setting is untouched confirmation. No frozen protocol is changed.

We will admit the next scientific study using completed comparisons and
measured cost. We will not tune a running study after individual seed results.
Improvement must appear in the actual served task metric, with member quality,
probability loss, convergence, parameter count and costs reported alongside it.
Any promising recipe needs a separately frozen transfer or confirmation test
before a paper-level superiority claim. Manuscript review follows substantial
supported improvement; setup checks do not justify another acceptance review.

Only the dedicated one-GPU allocation is in use. Original paper scores are
unchanged. There is no full-campaign launch authorization and no current
authorization to use 18.77.

## 18:50 UTC follow-up: fixed pooling does not repair residual probability loss

The completed all 33-bank comparison applies both fixed rules to every bank.
Residual BCE changes 1.952035→1.950576; native I4 changes 0.890168→.820155.
Residual AUC slips 74.518→74.493%, while I4 changes 74.847→74.896%.
Its positive-class excess therefore grows under the matched probability readout.
No fitted calibration or new selected serving rule is introduced. The
[complete interpretation](residual_positive_bce_pooling_v1/complete_results_v1/INTERPRETATION.md)
records every seed and limits. This strengthens the priority of testing useful
member changes; it does not reject every learned readout or training objective.
The [bounded common-error audit](shared_error_credit_audit_v1/NOTE.md) adds no
new training branch because its concrete realizations are already accounted for.

## Prepared alternative: learned private weight noise

A bounded review retains a six-fit learned-versus-fixed private-noise contrast.
Both preserve a native zero-residual start, use the same Gaussian mean anchor,
and serve deterministic route means. The learned-scale contrast tests adaptive
perturbation under its declared KL objective. It is not a posterior-correctness
or novel-Bayesian-principle claim. Exact paid point/native references must match.
See [proposal and limits](stochastic_private_adapter_audit_v1/NOTE.md). Source
preparation is accepted; no stochastic scientific fit is launched. PubMed24
remains the next scientific priority.

## Release30: actual preparation and private-noise qualification

The normalization migration, reporting registry and dispatch plan were generated
on the allocation with zero model work. The current roster remains 6,460 fits
and 270 I4 export tasks; 2,890 fit descriptors use composed normalization.
All 27 reporting profiles cross-check against the complete dispatch roster.
Actual generation took 34.73, 49.02 and 55.11 seconds respectively. The initial
migration attempt failed before creating outputs because two decision files
were absent; the dependency repair and failure remain recorded.

The fixed- and learned-private-noise implementations both passed two genuine
CPU updates, fresh VALID restore and next-noise replay. This is engineering
evidence only. New stochastic training remains unlaunched; the prepared
PubMed transfer comparison remains the next scientific priority.

No massive-run admission follows from metadata success. Composed BUDDY, NCNC
selected serving, runtime reporting witnesses, custody review and measured
resource placement still require completion. All 18.77 execution is disabled.
