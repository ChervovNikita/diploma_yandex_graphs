# Local derivative qualification for finite private learning through piecewise activations

5 October 2026. **Retain the native finite private-SGD response and its graph/class-competitor targets.** The latest immutable engineering receipt gives two-sided agreement at the original state and direction: central derivative discrepancies are1.0036441e-10 at1e-6 and4.7633011e-9 at1e-7. This supports a local derivative interpretation at that tested state. It supplies no reason to change the backbone, the response objective or the state to obtain a pass.

Finite SGD through piecewise activations can nevertheless produce a discontinuous learning response at a switch. That mathematical possibility limits a global smoothness claim; it has not been established as the cause of this fixture's coarse mismatch. The original qualifier remains **FAIL_ENGINEERING**, and the newer diagnosis remains **COMPLETE_BRANCH_DIAGNOSIS_NOT_QUALIFICATION**. No training or original qualification pass is granted here.

## Actionable continuation criterion

Continue only as a separately recorded engineering continuation with the current sources, synthetic state, direction, graph, roles, serving and G0 fixed. Preserve the original failure and bind all three engineering receipts. A reviewed continuation specification should state that its derivative evidence is local and retrospectively informed by the same-state diagnosis; it must not overwrite the original receipt or supply it to an accessor expecting a synthetic PASS.

1. **Bind the local derivative evidence.** The ordinary-autograd and torch.func objective values agree, their complete shared-coordinate gradients differ by at most1.1102230e-16, and the later central differences agree at both1e-6 and1e-7. The base value and direction reproduce the earlier receipt exactly. Retain the fixed coarse failures, including1e-5 discrepancy0.0899636569. Apply a stated continuation criterion to the complete function; a one-sided match or activation-mask observation alone is insufficient.
2. **Complete the originally unreached checks.** Source lines270–296 show that the original derivative assertion fired before the exact unused-local-head finite tangent, equal forward private response for live/stop-Q, and stop-Q versus independently fixed-Q derivative checks. Source lines437–439 show that the complete public episode, post-core recomputation from the original private state, and native state/input/RNG restoration were also unreached. Their requirements remain intact. The separate diagnosis's restoration check does not qualify the complete public episode.
3. **Keep graph-scale qualification separate.** After the synthetic continuation is reviewed and complete, full-graph resource/higher-order qualification still requires the reviewed enabled safe accessor with exact native preprocessing. It must record the actual complete episode and recomputation costs and whether full-graph directional FD was performed. The synthetic local receipt does not certify full-graph derivatives, floating feasibility or resources.
4. **Preserve the scientific boundary.** Engineering continuation fits nothing and grants no predictive value or novelty. Prediction-error analysis continues to motivate class-competitor correction learning, all-member competence and complete probability-pool evaluation. It does not justify replacing the correction target with activation stability or an easier numerical objective.

This criterion requests completion of the existing qualification obligations. It adds no exploratory scan, easier state, altered tolerance, new backbone, scientific arm or fit. Root owns any continuation source and its independent review.

## What the same-state evidence establishes

The first qualifier failed at central displacement1e-3: secant0.0285137476 versus analytic direction0.00387111866. Its earlier checks established member callback reconstruction and dense/sparse value/gradient parity. Those facts do not replace the failed complete response check.

The independent ordinary-autograd diagnosis reproduced the objective and all shared gradient coordinates. It used an equal-valued independent private proxy for the main private partial while retaining Q's connection to the original private state and shared core. Thus it tested the intended derivative ownership rather than silently adding a private Q-chain term.

The later receipt reports:

| Displacement | Central derivative | Absolute difference from analytic direction | Observed ordinary-callback transitions from base |
|---|---:|---:|---|
| 1e-5 | 0.0938347756 | 0.0899636569 | 2 coordinates on the positive side;7 on the negative side |
| 1e-6 | 0.00387111876 | 1.0036441e-10 | 1 coordinate on the positive side;0 on the negative side |
| 1e-7 | 0.00387112342 | 4.7633011e-9 | 0 on both sides |

There are372 captured ordinary activation sites and248 explicitly skipped transformed-tensor calls per point. Zero observed ordinary transitions at1e-7 therefore does not establish stability of every private-gradient activation. A transition at1e-6 coexists with excellent complete-objective agreement; a changed mask alone neither proves a jump nor identifies a harmful derivative path. Raw mask files were not opened in this assessment.

These are two-sided numerical checks of one direction at one synthetic state. They support the local derivative without proving every-direction differentiability, global continuity, a causal explanation of the coarse secants or predictive usefulness. The receipt explicitly leaves the cause unresolved. Smaller displacements were applied to the original point; the state was not substituted.

## Why a continuous network can have a discontinuous SGD response

Let native logits be f(theta,phi), with shared theta and private phi. On an open region where every relevant activation decision is fixed and the other operations have nonsingular smooth denominators, the support gradient and the finite private update

    U(theta,phi) = phi - eta * partial_phi L_support(theta,phi)

are ordinary smooth compositions. Autograd computes that branch's chain rule. At a switching surface, continuous logits and continuous CE do not imply a continuous private gradient. A finite update can amplify a gradient jump into a parameter jump.

A scalar analytical counterexample uses two distinct support/query items with the same scalar feature, private start phi=0, binary correct-class logit ReLU(theta+phi), competitor logit0 and correct label1. The support CE and competitor softplus margin are both

    L(theta,phi) = log(1 + exp(-ReLU(theta+phi))).

For theta<0, the private gradient is0 and the SGD response is U=0. For theta>0, the private gradient is −1/(1+exp(theta)), giving

    U(theta,0) = eta / (1+exp(theta)).

As theta approaches0 from the right, U approaches eta/2; from the left it approaches0. Consequently post-adaptation query CE approaches log(1+exp(−eta/2)) from the right and log2 from the left. The finite competitor response cost also jumps from0 to a strictly negative value. Selecting ReLU's derivative0 at equality defines a deterministic update value but cannot make these limits equal. This is a symbolic construction, not a numerical run or a claim about the native fixture.

The native source contains ReLU in local and global processing, and the engineering metadata observes GAT LeakyReLU. LeakyReLU also has a slope switch, so replacing ReLU with it would not establish global response smoothness. No activation replacement is proposed.

In the current operator, a probe-gradient jump can enter finite margin costs and hence Q; the main private gradient can also jump directly. Member centering, a positive RMS stabilizer, softplus, entropy and the finite smooth assignment solver do not generically restore continuity of a discontinuous input map. Balanced constraints or downstream cancellation can hide a particular jump, so the complete objective must be distinguished from any intermediate mask or gradient. Uniform assignments and stopped-Q derivatives retain the same native private-learning switches.

If the complete objective is continuous and locally Lipschitz at a genuine kink, generalized nonsmooth calculus may be applicable under its conditions. If it jumps, a Clarke subgradient of that complete objective is not justified because local Lipschitz continuity fails. Averaging the two branch derivatives does not repair a jump. None of these alternatives is needed to explain the currently observed fine-scale local agreement.

## What the close meta-learning scopes actually assume

| Saved or newly extended primary scope | Relevant statement | Limit for this native response |
|---|---|---|
| MAML v1, saved§2.1–2.2 | Explicitly assumes the loss is “smooth enough in theta” for gradient-based learning, then differentiates a post-SGD objective. | This supplies the local adaptation/hypergradient ancestry. The inspected scope contains no boundary contribution or theorem for a discontinuous finite-SGD response. |
| Ren learning to reweight, saved§§3.1–3.3 | Computes a validation derivative with respect to infinitesimal example weights, evaluates it at weight0, and implements backward-on-backward AD. | At fixed classifier state, its virtual step is linear in those weights. The candidate additionally changes the shared state inside private support gradients; Ren's weighting derivative is not a continuity certificate for that dependence. |
| Meta-Weight-Net, saved§§2.1–2.3 and new bounded§2.4 assumption scope | Uses a ReLU hidden weighting network. Theorems1–2 assume a Lipschitz-smooth loss and a differentiable, twice differentiable weight map with bounded gradient/Hessian. | These are conditional smoothness assumptions requiring justification for actual switching surfaces. The displayed theorem statements do not establish arbitrary ReLU response continuity; proofs were not audited. |
| sMCL v1, saved method scope | The oracle min-loss objective gives winner-only SGD, with arbitrary tie breaking. | The minimum of finitely many continuous losses is continuous but can be nonsmooth; its selected gradient/update can switch. Learning through an update is a stronger regularity question than continuity of the scalar training loss. |

The inspected close priors already establish adaptation, weighting, specialist assignments and differentiation through learning. They do not clear a new nonsmooth learning principle or furnish a global theorem for the candidate. This is a scope-bounded conclusion; unread papers are not absence evidence.

## Response formulation and disposition

Keep the actual finite native-SGD response, finite graph-balanced responsibility map, independent-Q private partial, retained shared credit, original-state recomputation and fixed probability serving. State derivative evidence as **local to the qualified native finite-learning map**, rather than as a global smoothness theorem for piecewise networks.

A genuine unresolved jump at a required operating state would require a separate method decision. A deliberately smoothed neighborhood objective would change the mathematical target and require its own information, estimator, serving and paid-cost specification. The new fine-scale receipt provides no justification for adopting such a change here. Gaussian smoothing is not proposed as a way to pass the preserved coarse gate.

Only one new substantive primary scope was needed: Meta-Weight-Net§2.4 theorem assumptions from its saved immutable HTML. An exact-v1 Evolution Strategies HTML request returned a PDF placeholder; its method remained unread and was abandoned when the new local engineering result removed the reason to pursue an alternative formulation. Its fetched metadata/abstract supplies no method claim. Saved MAML/Ren/MW/MCL scopes were consulted first. Source hashes, exact read ranges, authorized engineering bindings and read limits are in READ_SCOPES.json, SOURCE_BINDINGS.json and ENGINEERING_SUMMARY.json.

No native or qualifier code was changed, no branch tracing was rerun, and no scientific dataset, checkpoint, output, raw mask, SSH, fit or additional agent was accessed. The counterexample limits global claims; the engineering receipts support completing the remaining native qualification work.
