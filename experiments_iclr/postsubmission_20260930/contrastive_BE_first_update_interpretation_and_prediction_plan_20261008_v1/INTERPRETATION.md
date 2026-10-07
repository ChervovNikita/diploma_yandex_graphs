# The first native updates differ; prediction quality remains unknown

Evidence is the completed CPU diagnostic of two already-created, discarded LOCAL
qualifier snapshots. It performed no fit, model forward or GPU operation. Seed,
model/optimizer recipe, source identities and member RNG endpoints matched.
All153 parameter objects were accounted for; no parameter/storage aliases were
found, and39 inactive LOCAL parameter objects remained at their reconstructed
initial values.

| Block | Entries differing exactly | Difference L2 | Common-update L2 | Difference/common-update L2 |
|---|---:|---:|---:|---:|
| Private r/s |3799 /122112 |0.016996 |0.302139 |5.63% |
| Shared native parameters |1193211 /7537172 |0.125163 |2.384859 |5.25% |

All four members' private updates differed. Their difference L2 values were
0.00867,0.01011,0.00734 and0.00758. The maximum private difference was0.001994,
close to twice the nominal0.001 Adam step scale. Common and route private update
norms were nearly equal,0.302139 and0.302146. Thus the result concerns changed
update directions/coordinates, rather than a substantial overall norm increase.
These ratios describe parameter updates; they are not percentages of explained
gradients, improved predictions or repaired errors.

The strongest supported statement is narrow: the two target rules produce
different first native parameter updates at this initialization, including
private factors. This removes the concern that distinct target arrays were
necessarily inert here. It does not establish that every initial state has
nonzero contrastive gradients. Parallel normalized same-class embeddings remain
a possible stationary configuration despite nontrivial target variation.

Shared weights also differ. The deterministic helper's equal-shared-gradient
identity requires equal member outputs/Jacobians. Native members use independent
dropout, and Adam further transforms the mixed own/auxiliary gradients. That
identity must not be asserted for this actual trajectory. The diagnostic does
not decompose auxiliary gradients from ordinary supervision or certify native
floating execution bitwise.

Exact nonzero counts are not meaningful-effect counts. For example,247 local
classifier-weight entries differ, but their maximum difference is only about
8.4e-9. The private classifier input factors differ in two entries by at most
6.0e-8, while its output factors are identical. Magnitudes and locations therefore
matter; counting every different bit as specialization would overstate the
evidence. No extra precision audit or threshold-based hypothesis rescue is
recommended.

Qm versus Qbar also changes per-route target concentration/entropy. This update
comparison cannot attribute the effect solely to persistent assignment or graph
semantics. The permutation preserves positive counts/class/self/panel, not public
graph degree. X is one context, so a route-permuted advantage would not by itself
establish topology-specific causality.

No logits, rankings or competence were evaluated in this diagnostic. Useful
prediction differences must be established at whole9 closure, followed by the
unchanged12 competent references if the frozen gate passes. Neither this result
nor the target-eligibility check establishes model quality, novelty or acceptance.
