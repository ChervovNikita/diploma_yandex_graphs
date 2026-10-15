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

## The current sequence

| Study | Question and comparison | Status |
| --- | --- | --- |
| MolBACE structural responsibilities | Does making small private corrections specialize by TRAIN graph size improve on uniform responsibilities and a shuffled-assignment control? Native shared parameters still receive the ordinary loss. | Six full fits running over three seeds; existing controls reused. Interpret only after all six finish. |
| PubMed / native PolyFormer-Cheb | Does the residual and responsibility design work on a different backbone and node-classification task without batch normalization? Compare native M1/I4, original GNNM, uniform, structural and shuffled variants. | Source and three CPU lifecycle checks complete. Seven GPU engineering checks wait for the molecular study. Budgeting the proposed 24 fits requires measured production costs. |
| MolBACE pairwise supervision | Can routes learn different useful ranking comparisons while keeping the native classification loss? Compare common, structural and shuffled pair weights; include independently trained native ensembles with the same common ranking loss. | All four CPU lifecycle checks passed. TRAIN-only support is adequate under the native sampler. Proposed 21 fits are not launched. |
| Broad reference-wrapper campaign | Which methods transfer across the fixed native paper settings? Preserve source reproductions, ordinary independent ensembles and original GNNM throughout. | 6,460 proposed fits; serving coverage and measured placement/cost remain incomplete. Requires user review and GO. |

The pairwise idea uses the model's existing logits. No teacher or independently
trained donor is acquired, and inference remains the same four-route mean.
The structural pair roles use only TRAIN molecular inputs. The common-loss
independent ensemble is essential: improving the training loss alone must not
be mistaken for an advantage of the shared wrapper. Prior work already
establishes pairwise AUC learning and shared/private specialist rankers; the
[literature note](pairwise_competence_prior_followup_v1/NOTE.md) records the
remaining empirical question and unresolved prior-art leads.

We will select the next scientific study using completed comparisons and
measured cost. We will not tune a running study after individual seed results.
Improvement must appear in the actual served task metric, with member quality,
probability loss, convergence, parameter count and costs reported alongside it.
Any promising recipe needs a separately frozen transfer or confirmation test
before a paper-level superiority claim. Manuscript review follows substantial
supported improvement; setup checks do not justify another acceptance review.

Only the dedicated one-GPU allocation is in use. Original paper scores are
unchanged. There is no full-campaign launch authorization and no current
authorization to use 18.77.
