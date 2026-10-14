# Molecular graph alignment with conditional variance

15 October 2026. Implemented objective extension for full-feature MolHIV/MolBACE
native GCN/GIN, including their virtual-node variants when that native runtime
is qualified. Shared native modules, categorical fields, mean graph pooling,
head, batch32, optimizer groups, Adam settings and source BCE remain intact.
Frozen `reference_wrapper_redesign_20261013_v1/smoke_20261014` files are untouched.

## Question and intended prediction effect

Can label-aware alignment of the **native pooled graph vector**, while penalizing
loss of its within-class variation, improve pooled VALID ROC-AUC and BCE without
weakening members? Binary labels identify compatible task classes, not chemical
equivalence: an active and another active molecule may have different relevant
substructures. The variance term tests whether retaining some of that variation
helps the task. It is not a claim that members specialize, use different graph
evidence, or gain useful diversity merely because their vectors spread.

Retained notes already contain ordinary class alignment, class-centered member
repulsion and private gradient routing. They also exhibit a classifier-null
counterexample and the exact binary degeneration of wrong-class-output contrast.
Those recipes are not renamed as new methods here. The present composition
adapts SupCon positive averaging, BCL class-averaged denominators, and a VICReg-
style variance hinge to existing graph-level member views. It is not a complete
reproduction of any of those papers. [Reading scopes](reading/READING.md) separate
abstracts, method sections, retained project notes and unexamined material.

## Exact objective

For an ordinary source TRAIN minibatch, collect each existing route's live
`graph_pred_linear` prehook capture, before its output R: `H[m,g,d]`, with
`M=4`, at most 32 graphs and native width300. This is after unchanged native mean
pooling. Each graph has one loss row regardless of its atom count. Normalize
only the auxiliary copy: `Z = H / max(||H||_2, 1e-12)`.

Only observed native binary labels participate. NaN labels are omitted from
pairing and variance; they are never a third class or pseudo-label. For anchor
`a=(m,g)`, define `s_ab = dot(Z_a,Z_b)/0.2`, exclude the exact same anchor, and
let `B_c(a)` contain remaining observed entries of class `c`. With
`P(a)=B_y(a)`, use

```
D(a) = sum over present classes c of mean_{b in B_c(a)} exp(s_ab)
A(a) = log D(a) - mean_{p in P(a)} s_ap
A    = mean over eligible anchor classes of mean anchors in that class A(a)
```

This averages the positive **log probabilities**, not the positive probabilities
inside the log. The denominator averages exponentials within each class before
summing classes. The anchor reduction is class-balanced too. This changes the
auxiliary weighting only; native BCE is not class-reweighted or resampled.

Other members on the **same graph** are positives. Thus even a singleton active
graph has three positives when M=4. Distinct same-label graphs are also positives;
coverage reports distinguish those cases. Other-class entries are negatives.
If only one observed class occurs, `A=0`: no instance-negative or synthetic
prototype substitutes for an absent class. There is no cross-batch memory bank.
An optional M1 control skips anchors lacking a nonself positive; its coverage
therefore differs explicitly from M4.

For each member and observed class having at least two **distinct graphs**, take
the feature standard deviation across those graphs, with population variance:

```
gamma = 0.5 / sqrt(D)
V = mean over supported classes, members and dimensions of
      max(0, gamma - sqrt(Var_graph(Z | class, member) + 1e-8)) / gamma
```

Unsupported singleton classes contribute no V; they retain native BCE and any
same-graph cross-member alignment. V may remain active on a one-class batch.
Member offsets do not satisfy this across-graph variance target. L2 normalization
prevents satisfying it merely by increasing the raw feature scale. The mean and
variance remain live, with no detached center or added projector.

The candidate is `F + .05 (A + V)`, where F is unchanged mean-member native BCE
with native observed-label mask and FP32 casts. Fixed ablations are F and
`F + .05 A`; see [CANDIDATES.json](CANDIDATES.json). There is no strength grid.

## Gradients and limitations

Default `gradient_scope='all'` uses one scalar backward and one unchanged native
Adam transition. A and V reach native encoder weights and input-private factors
through H. The native head and output-private factors have no direct auxiliary
path because capture precedes them; they still receive F gradients. Four existing
route forwards, their native dropout and sequential shared-BN updates are retained.
No additional stochastic forward or graph alteration is requested.

`gradient_scope='private'` is a separately labelled future credit ablation. Its
result cannot be added as an ordinary scalar: `scalar_total()` rejects that use.
A block-aware controller would request F derivatives for shared parameters and
F+auxiliary derivatives for inventoried private parameters before one Adam step.
It is outside the initial pilot and does not create additional combinations.

V penalizes constant within-class vectors, but does not impose full-rank
covariance; batch32 cannot support a rank300 covariance target. Exact constant
vectors can still be stationary for the smooth variance expression. Class-null
or otherwise prediction-irrelevant variation can satisfy this penalty. Neither
the penalty nor private credit guarantees no harm after shared-parameter updates.
Missing-label H rows have zero direct auxiliary gradient; native BN can still
couple parameter gradients across graphs. No claim of causal graph specialization
or benefit follows from a lower auxiliary loss.

## Optional full-epoch objective hook

The full-epoch adapter can import `MolecularObjective` and `ObjectiveConfig`
from `molecular.objectives` in this expansion package. For an existing four-route
TRAIN forward, after forming native F:

```python
hook = MolecularObjective(ObjectiveConfig(mode="alignment_variance"))
result = hook(prehead=torch.stack([r[2] for r in routes]),
              labels=batch.y, graph_ids=batch.smoke_graph_id.reshape(-1),
              role="TRAIN")
loss = result.scalar_total(F)
optimizer.zero_grad()
loss.backward()
optimizer.step()                   # the existing source Adam; exactly once
```

Graph IDs must be the actual aligned native TRAIN IDs, int64 and unique within
the batch. H must stay attached. `result.loss` is already weighted; `result.terms`
are unscaled live tensors and `result.diagnostics` are detached coverage/variance
records. Do not apply the coefficient twice. The hook owns no parameters, RNG,
queue or optimizer. Save its config/schema via `state_dict()` alongside the full
native/controller checkpoint and reject a changed config on resume.

Whole-VALID prediction analysis uses `prediction_diagnostics(..., evaluator=
session.evaluator)` on aligned actual raw member logits. It calls the supplied
source OGB evaluator for each member and the predetermined raw-logit mean; it
reports BCE, mean/worst member AUC, pooling lift, common mistakes, unique correct
members, and pool repairs/harms versus member0. Its threshold diagnostics use
the native zero-logit binary decision and supplement ROC-AUC. Full-VALID
`embedding_diagnostics()` computes conditional variance without a quadratic
pair matrix. Both analysis functions reject TEST. The training hook rejects VALID.

## Verification and next executable work

`test_objectives.py` passes 12 local CPU tensor checks, including an independent
NumPy formula and finite-difference gradient verification. [SEMANTIC_CHECKS.json](SEMANTIC_CHECKS.json)
records exact source/test hashes and runtime. No native molecular model, data,
fit, server operation or accuracy claim occurred. Test dependencies live in a
local runtime cache, outside this deliverable. Root must connect the accepted
hook protocol to the new full-epoch adapter and run its ordinary native/provider,
optimizer, replay and resource gates before the [bounded pilot](PILOT_PROPOSAL.md).

