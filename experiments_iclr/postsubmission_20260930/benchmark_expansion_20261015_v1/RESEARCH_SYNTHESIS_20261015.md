# Current scientific direction

The target remains better prediction quality from several learned routes with
one shared native backbone. The benchmark keeps that backbone's operators,
features, graph, optimizer settings and task evaluator. It measures new wrapper
mechanisms against native single models, independent ensembles and base GNNM.
Separate source-reference reproductions remain in the design.

## Mechanisms being made testable

**Preserve the native initial function, then learn different corrections.**
The residual-only four-route candidate initializes every route at exactly the
native evaluation function. Each route has a different low-rank down projection
and a zero up projection. Initially, task gradients can update the up matrices
through their different projected features, while down-matrix gradients are zero
at an exactly zero up matrix. This gives a concrete source of different learning
directions without initially changing the native predictor. Both matrices then
learn. It does not guarantee that the routes stay different or improve quality.
Real MolBACE and CiteSeer construction/update/checkpoint tests passed; no trained
quality comparison for this new residual-only candidate is available yet.

**Give private routes different, declared training responsibilities.** The new
structural assignment proposal emphasizes different degree/graph-size ranges
within each TRAIN class, retains positive supervision everywhere and keeps
class/member total mass fixed. An exact within-class permutation control removes
the association with structure while keeping those masses and weight values.
The shared body receives ordinary own risk; private coordinates receive the
declared weighted risk. This is a graph/context specialization hypothesis, not
a claim that shared trajectories or original risk remain unchanged. Its bridge
and semantic checks are under development. Source batches/reductions stay native.

**Learn aggregation only with the right training control.** The prepared
mean/global/conditional readouts all use the same half-own, half-served-pool
training objective. The conditional model learns from detached confidence and
disagreement descriptors of existing predictions; its logits remain live in the
served sum. It starts at the exact mean, adds20 parameters and retains a positive
weight floor. Actual molecular update/VALID/checkpoint checks passed. Native
node/link ports and scientific efficacy remain separate questions. These are
known mixture/stacking ingredients; no novelty is claimed for weighted averaging.

**Separate normalization failure from lack of useful alternatives.** Route-
private BN state changes evaluation state ownership without adding learned
parameters. Three frozen candidate fits reuse the complete parent controls and
will run after the current24-fit queue, with the same native settings/selection.
This tests a specific possible cause of weak routes. It does not assume the
cause has been established or pretend that private histories can be recovered
from one old shared EMA.

## How these mechanisms relate

Different initialization supplies different private gradient features;
responsibility supplies different task emphasis; aggregation tests whether the
resulting alternatives can be used from available information. They address
different potential failure modes. Combining them is a later explicit factorial
question, with ingredient controls, rather than an automatic claim that several
small effects add. Normalization is a separate state mechanism. The current
frozen contrastive/input-adapter2×2 remains intact and is not retrospectively
replaced by this synthesis.

The first-seed diagnostic already shows that disagreement can accompany poor
ranking and class-biased errors. It supports measuring competence, calibrated
task loss where meaningful, repairs/harms and actual pooled ranking. It does not
support increasing latent distance indiscriminately. A variance penalty has no
general collapse-avoidance theorem, and an almost-complete member-correct union
does not establish a learnable near-perfect predictor.

## Evidence needed before a paper claim

Complete paired runs and retain negative outcomes. First establish a benefit
over base GNNM, then evaluate the same candidate against capable native singles
and independently selected ensembles. Check a second task/backbone setting;
avoid another broad hyperparameter grid. Attribute adapter/low-rank, contrastive,
normalization and mixture ideas to their actual predecessors. Fresh primary
readings and the historical catalogue constrain what can be called new.

The complete proposed benchmark still needs concrete task ports, immutable job
bindings, full-epoch/runtime/resource qualification and a prospective confirmation
policy. The old2,480-fit matrix is preserved as a predecessor, not relabeled as
the broadened final count. No full campaign starts before the user's review/GO.
The current selected-VALID work is exploration, not independent TEST confirmation,
novelty proof or a reviewer acceptance verdict. Original paper scores stay fixed.
