# Bounded molecular pilot proposal

15 October 2026. Root owns allocation execution. This is a concrete proposal for
the optional objective hook, not a launcher or authorization for a giant matrix.
The separate 270 native paper-reproduction fits remain retained and unchanged.

## Fixed question and arms

Test whether class-balanced graph alignment with a within-class variance hinge
improves served quality beyond both ordinary shared supervision and alignment
alone. The intended candidate is fixed in advance, not whichever auxiliary wins.

| Arm | Hook mode | Gradient rule | Scientific role |
| --- | --- | --- | --- |
| B0 | `none` | F for all native/private parameters | Same boundary GNNM, no auxiliary |
| B1 | `alignment` | F + .05 A through the live prehead | Does alignment alone explain any gain? |
| C | `alignment_variance` | F + .05(A+V) through the same prehead | Does conditional variation add useful quality? |

Use existing four shared native routes, original first-input Rademacher start,
native GCN/GIN settings (5 layers, width300, mean pooling, full categorical atom
and bond inputs, batch32), dropout .5, and original Adam LR .001/defaults/no WD.
No projection head, altered graph, rebalancing sampler, extra forward or output
combiner is fitted. All arms use the same original TRAIN/whole VALID roles and
native BCE. Private-only credit, queues, virtual-node variants and parameter
grids are outside this first pilot.

## Finite run set and convergence

1. **MolBACE / native GIN, no virtual node:** B0, B1 and C on paired seeds 0 and 1:
   six fits. Whole-epoch cap300, minimum100, patience50 whole-VALID checks beginning
   after epoch100; reset on strict ROC-AUC improvement. Native source has no warmup.
2. **MolHIV / native GCN, no virtual node:** B0 and the already fixed C, seed0:
   two additional fits, for the rare-positive coverage question. Cap300/min100/
   patience50 unchanged. Activation is subject to root's measured time/memory
   admission, not selection of a favorable MolBACE outcome or seed.

Maximum eight new fits; no automatic retry, third seed, new backbone, coefficient
search or phase after this set. Root should quote the one-epoch cost before
dispatch and enforce a four-hour wall cap per fit. A resource stop before
convergence, missing member result, failed replay or cap without an adequate
plateau is retained as incomplete/inconclusive. A short fit is not relabelled
as an epoch-complete source replication. The first stage alone is an executable
representative panel if the MolHIV cost is not admitted.

Save the first maximum whole-VALID source OGB ROC-AUC checkpoint among completed
epochs1..cap; no checkpoint0. Minimum100 gates stopping, not checkpoint eligibility.
Save curves and selected/completed epochs. Cold-start paired arms from matching
initial native weights/sampler streams; save factors, BN, optimizer, sampler,
global RNG, epoch phase and hook config for resume. No TEST split or scoring is
needed. The nonvirtual choices avoid treating unresolved native virtual replay
as a passed engineering gate.

## Readout and advancement rule

At every epoch record TRAIN loss, unscaled A/V, auxiliary dose and time, both-class
batch frequency, observed-class counts, positive coverage, same-graph-only
positive rate, variance support by class and actual optimizer/route counts.
Log the fraction of updates on which each term is active; an inactive auxiliary
is a coverage result, not evidence against a correctly exercised method.

At each selected whole-VALID state retain aligned member and pooled raw logits,
source evaluator ROC-AUC, BCE, mean/worst member strength, and threshold error
coverage/unique correct members/common errors/pool repairs and harms. Retain
full-VALID class-conditional feature variances and source role IDs. Charge total
wall/GPU time, peaks and replay work. Do not use node-objective metrics or latent
distance as substitutes for served prediction quality.

The strict exploratory screen for C is positive pooled AUC change versus **both**
B0 and B1 in each of the two paired MolBACE seeds, no pooled BCE increase versus
B0, and no loss in mean or worst member AUC versus B0. If this fails, retain the
negative or mixed outcome; do not promote based on variance, loss or a favorable
seed. Report the MolHIV pair separately, including positive-class coverage. Two
development seeds do not establish significance or generalization, and whole
VALID is already used for epoch selection. Any promising result needs frozen
confirmation and capable objective-equipped M1 / genuine independently selected
I4 controls before a sharing or superiority claim; those controls are not
silently charged as part of these eight exploratory fits.

