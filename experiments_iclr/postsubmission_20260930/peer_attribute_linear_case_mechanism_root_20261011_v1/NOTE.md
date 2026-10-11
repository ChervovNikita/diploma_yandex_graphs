# A restricted explanation of peer attribute feedback

This is algebra for a simplified operation, saved before any new pilot outcome. It is not a theorem about the actual nonlinear learner, an accuracy guarantee or a new sampling/graph primitive. The frozen thirty-bank proposal and core remain unchanged. No numerical experiment or new condition was run.

## What the simple case predicts

Treat each attribute as a hub joining the nodes that contain it. A node sends its current representation to those hubs and receives their average representation back. Let B be the binary node–attribute incidence matrix, r its row sums and c its column sums. On positive-degree rows and nonempty columns, the complete two-step feature operator is

G = diag(r)^(-1) B diag(c)^(-1) B^T.

It is a row-stochastic two-step walk. The symmetric matrix similar to G is

diag(r)^(-1/2) B diag(c)^(-1) B^T diag(r)^(-1/2).

This matrix is Q Q^T for Q=diag(r)^(-1/2)Bdiag(c)^(-1/2), so its eigenvalues are nonnegative. Row-stochasticity bounds their maximum by one. Empty rows/columns are zero under the core's declared convention; the argument restricts to the active support. This is a standard normalized incidence/hypergraph construction, not a new operator.

Now consider four member states H_m, replacing the learned nonlinear map by the identity and assuming the outer clipping is inactive. The recipient excludes itself:

H'_m = H_m + alpha G (sum_{n != m} H_n)/3.

Write the common state Hbar=(sum_m H_m)/4 and each member deviation d_m=H_m−Hbar. Direct summation gives

Hbar' = (I + alpha G) Hbar,
d'_m = (I − alpha G/3) d_m.

On a feature-kernel eigenmode with eigenvalue lambda, the common component is multiplied by 1+alpha*lambda and the deviation by 1−alpha*lambda/3. At the fixed alpha=.25, those factors are [1,1.25] and [11/12,1]. In the node-degree-weighted norm, the deviation cannot grow in this simplified case.

Thus peer exchange need not push members apart. It can strengthen their common feature-associated signal and damp their disagreement. That may help when disagreement is nuisance noise; it may harm when the common direction is a shared wrong class or when the differing route contains the only correct alternative.

## Limits and scientific use

The actual core uses a learned rank16 ReLU map, row clipping, private factors, source dropout and joint training. The map can change signs/directions and its local derivatives can differ across attributes and routes. Projection is often active because source hidden states already reach the unit ball. The linear decomposition above therefore does not prove contraction, competence or calibration for SX4. It does not identify the cause of the earlier Physics errors.

The existing frozen measurements are sufficient to distinguish the practical outcomes: mean/worst member quality, common wrong rivals, available correct alternatives, pooling losses, gross repairs/harms and final raw/proper quality. If the candidate strengthens members but removes useful alternatives, the final served result and same-information single/I4 controls still determine utility. No new metric, threshold or cohort is selected from this note.

The nonlinear-order control SM4 is useful because applying a nonlinear map to each donor before averaging can preserve information that vanishes when donor states are averaged first. Ordinary nonlinear noncommutation is known. A positive empirical interaction would support this particular operation, not a general ensemble expressivity advantage; the entire bank still has a structured wider-single representation.

Do not insert this algebra into the main manuscript merely to make it look theoretical. If complete experiments support the mechanism, a short plain-language explanation and an optional restricted derivation can clarify the actual result. Otherwise retain it as a prospective explanation and limitation.
