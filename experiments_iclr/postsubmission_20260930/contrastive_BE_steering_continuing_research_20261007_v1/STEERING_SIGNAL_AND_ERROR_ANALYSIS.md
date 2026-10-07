# Steering means useful changes in predictions

Status: source reasoning and a prospective analysis specification. No new fit,
checkpoint, measured margin response or superiority claim is present.

## Why the common-target comparison isolates a concrete change

For route m, let ell_m be its symmetric cross-view contrastive cross-entropy
with fixed target Q_m. The common-target arm replaces Q_m by Qbar, the average
of the four row-normalized targets after the identical TRAIN-panel restriction.
For a fixed score matrix, cross-entropy is linear in its target. Therefore, when
all four routes have exactly equal scores in both views, their average contrast
loss equals the common-target loss. The average score cotangent is also equal.

When route factors are unit, member stochastic views are identical, and the
shared map is the same, the shared-parameter Jacobians also agree. Under these
conditions the shared gradient is equal between the two arms. Their private
factor gradients may differ because each private route receives its own Q_m.
This gives a precise way to break symmetry without changing average target
mass. It does not show that the resulting differences will be useful.

The actual native training uses independent dropout streams. Equal factors do
not imply equal outputs or Jacobians under those streams. The equal-gradient
statement is a local deterministic identity, not an identity between whole
training trajectories. Adam can also map equal-average raw gradients to
unequal-average updates. No shared-gradient cancellation theorem is claimed
for the native stochastic Adam trajectory.

## Why hidden separation can be irrelevant or harmful

For a TRAIN example, a contrast gradient can change a hidden direction that the
classifier does not use. It can also emphasize a nuisance direction or alter a
correct margin in the wrong direction. For an ordinary SGD step, the local
response of a truth-versus-competitor logit margin f to an auxiliary factor
update is proportional to - grad_factor(f) dot grad_factor(ell_m). That overlap
can be zero, positive or negative. This observation is not a guarantee for
Adam, finite updates, selected-development objects or heldout accuracy.

Scalar rescaling can additionally be removed by representation normalization.
Nonuniform factor changes inside nonlinear graph propagation may alter
rankings, but the private Jacobian can still erase most of the target signal.
The prepared CPU helper only verifies that one nonlinear fixture has different
private gradients and matched shared gradients. It is not a native-model
measurement and must not be cited as empirical model evidence.

Unselected same-class objects are denominator distractors in the candidate.
This deliberate within-class discrimination supplies more structure than the
class label, while creating an explicit risk to member competence. TRAIN labels
make positives compatible with the task; they do not prove that graph-signature
neighbours are meaningful subclasses or that contrast preserves accuracy.

## A serving limitation identifies the required prediction change

If some wrong class has higher probability than the true class in EVERY member
for an object, any convex probability pool using those same member predictions
also puts that wrong class above truth. A learned nonnegative router cannot
repair that object by selecting among unchanged predictions. It must acquire a
better member ranking, use a different transformed representation, or change
the predictors. This is why the present candidate changes training rather than
adding a complex router. It does not claim routers are generally useless.

## Prospective complete-family analysis

Open outcomes only after the entire nine-fit Stage1 family closes under the
root-owned protocol. Retain selected checkpoints with native global/local flags,
not an arbitrary final epoch. Preserve each member probability tensor and labels
server-side. Use the complete5274 selected-development population.

Primary contrasts remain route minus common and route minus permuted. For each
seed, report pooled accuracy/NLL/Brier, mean and worst member accuracy/NLL,
selected epoch, training/inference costs, coverage and the pool-minus-mean-member
benefit. Descriptive paired seed differences are optimizer variability on one
selected graph. They are not independent-graph evidence or confirmation.

For error diagnosis, freeze all reference cohorts from the COMMON arm before
opening corresponding candidate tensors. Count pooled wrong-to-right repairs
and right-to-wrong introduced errors, across the full population and each
reference cohort. Report the truth-margin changes and compare improved mean
member margins with changed pool-minus-member benefit. Do not equate member
coverage with served accuracy or report a favorable subgroup as the endpoint.

On objects where common members share a wrong competitor above truth, record
whether at least one candidate member reverses that competitor and whether the
served candidate repairs the object. Also count newly acquired common wrong
competitors on reference-correct objects. These paired counts can distinguish
useful rank acquisition from merely larger hidden distance. They cannot alone
identify a causal factor-gradient mechanism.

Stage2, if admitted by the whole-family gate, compares capable ordinary and
objective-matched single/untied controls. If a single Qbar regularizer explains
the gains or untied Q_m members are better, retain that interpretation. Do not
call the effect a sharing-specific benefit. A positive Stage1 attribution does
not bypass Stage2 or unused-task/split confirmation.

## Boundaries on methodological claims

Graph-filter signatures, supervised contrast, positive mining, member diversity,
and diagonal BatchEnsemble factors are all established ingredients. The current
candidate asks whether fixed different label-compatible graph contexts are useful
through restricted private factors in a shared predictor, relative to exact
aggregate-target and randomized-target controls. That is a controlled empirical
question, not a proven new primitive or an accuracy guarantee.

Learnable diversity weights can collapse to zero under ordinary joint loss
minimization unless an explicit outer criterion or constraint is supplied.
Reinforcement learning adds no demonstrated advantage to this differentiable
training objective. Keep both as unadmitted alternatives until a concrete
learning problem and nonredundant control justify their cost.
