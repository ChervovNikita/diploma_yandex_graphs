# Shared graph ensemble correction under fixed context challenges

This is one concrete, unadopted candidate: retain ordinary own-label learning of a live shared backbone, and intermittently train private graph-message corrections on a fixed graph-context challenge. Assign corrective credit to different paths only when their old-state derivatives worsen a common wrong ranking in every member. Check each proposed correction against clean TRAIN competence before accepting it.

The precise operational difference is **context-conditioned common-competitor path credit with finite competence checks**. Gates, graph views, paired ranking losses, gradient allocation and constrained optimization are established ingredients. I cannot presently defend this complete rule as novel: GEENI's primary method remains unresolved, and saved graph-diversity and functional-gradient priors are close. This is a falsifiable utility hypothesis, not a new framework or novelty verdict.

## One architecture and sharing rule

Use the existing four-member, unit-initialized shared WikiCS backbone and complete native schedule. Share its dense weights, norm coefficients and other common parameters `theta`. Retain the existing member factors `phi_m`. Add a private scalar log-bias `b_m,e` on each original nonself edge-message position of **the first local attention layer**, before its incoming aggregation. Keep the original support and row normalization:

```text
a'_m,vu = a_m,vu exp(b_m,vu) / sum_w a_m,vw exp(b_m,vw)
b_m,e initially 0; |b_m,e| <= log(2); self-loop biases fixed at 0.
```

Here `a` denotes attention probabilities before native attention dropout; ordinary training retains that dropout after the biased softmax. This is an attention-bias parameterization with direct ancestry, adding at most `4E` parameters. It is a transductive edge table for the representative graph; no inductive or cross-graph gate-generalization claim follows. Source qualification must establish the exact attention hook, edge ordering, normalization and native stage/snapshot behavior before any implementation or execution. This proposal does not edit the existing source.

Every ordinary update trains `theta`, all `phi_m`, and all `b_m` using the mean of the four members' CE over the same two stochastic TRAIN views. All members see all TRAIN labels. Use one native Adam transition and the original schedule, then project biases to their declared bounds. There is no teacher, learned prediction router, embedding repulsion, mask-overlap penalty or new GNCL mixture. Shared parameters stay live throughout.

## A fixed graph context challenge

Late clean TRAIN classifications can already be perfect. A common-error cohort drawn only from that fitted graph can therefore be empty or irrelevant to development errors. Do not import development labels, development error cohorts, or alter the cohort after results.

The frozen alternative hypothesis is **robustness to a bounded change in relative neighbor contribution**. At a correction opportunity, make one deterministic clean forward in serving mode. Temporarily introduce a common first-layer edge log-bias `delta_e=0` shared across members, and collect the gradient of full-TRAIN mean own CE with respect to it. For each receiving row with at least two nonself edges and a nonconstant gradient, give its maximum-gradient edge `+log(2)` and minimum-gradient edge `-log(2)`. Leave other entries and self loops at zero. Re-normalize each member's attention row as above. Stop the derivative through this one-step challenge construction.

The rule uses TRAIN supervision and the current predictor, with no label information at serving. It preserves graph support and row mass, but it can remove effective access to necessary evidence. Keeping labels fixed under this change is a **training robustness hypothesis**, not a claim that the edited context is semantically equivalent or causal. The view is not a certified worst-case adversary. Its construction, bound and schedule are frozen before any new outcomes.

## Corrective objective and path credit

On that challenged view `V`, compute member probabilities and their arithmetic mean `pbar`. For each TRAIN node `i`, define:

```text
K_i = {k != y_i : p_m(k | V,i) > p_m(y_i | V,i) for every m}
U = {i : K_i is nonempty}
k_i = highest-pbar class in K_i
C_V = mean_i_in_U softplus(log pbar(k_i | V,i) - log pbar(y_i | V,i)).
```

Freeze `U`, competitors and the view during this correction. This is an established paired logistic ranking risk applied to an explicitly defined common-error context. If `U` is empty, skip the correction and record the null opportunity.

Collect `s_m,e = derivative(C_V)/derivative(b_m,e)` at the same old state. An edge is eligible only if `s_m,e>0` for every member: increasing its attention bias locally worsens this cohort risk in all members. This is a local derivative criterion, not evidence that the edge is intrinsically wrong. Partition eligible edges into disjoint member recipient sets using a predeclared balanced seeded assignment. Each member receives corrective gradients only on its assigned edges. The ordinary own gradients remain unrestricted.

For the selected coordinates use the negative corrective gradient, normalized to maximum absolute coordinate one. Its first-order dot product with the cohort gradient is nonpositive. Propose a gate-only step at the inherited native learning rate, clipped to the declared bias bounds. Other parameters remain at the already completed ordinary update. Try at most four scales: `1`, `1/2`, `1/4`, `1/8`. Use deterministic clean and challenged forwards, with the same frozen view and cohort, for each trial.

Accept only if `C_V` strictly decreases, every member's full clean-TRAIN CE and the clean pool's CE do not increase, and no previously correct clean-TRAIN member or pool decision becomes wrong. Otherwise restore the biases and take no correction. There are no hidden retries or relaxed fallback criteria. The gate-only correction is stateless projected gradient descent; it does not advance Adam moments. Corrective gate steps are additional gradient steps; they are not another native Adam transition or a scalar loss jointly minimized by all blocks.

## What the competence checks do and do not establish

The checks protect measured clean TRAIN behavior **only for the incremental correction**, relative to the state after the normal update. They do not protect development competence, make the whole trajectory monotone, or prevent memorization. Ordinary updates can still change decisions. Finite checks are generic constrained optimization, not the novelty claim.

They may reject every nonzero step. Clean CE near a strict local minimum can make asymmetric corrections infeasible; the fixed challenge can also yield no common errors or no consensual adverse paths. Any of those outcomes makes this candidate inapplicable on the representative graph. Record eligibility, accepted steps, rejected steps, zero gradients and clipped directions. Do not increase the view bound, weaken the guard, move to a development-defined cohort or add another mechanism to manufacture activity.

The intended graph mechanism is learning different corrections to over-relied-on context paths while retaining clean competence, with useful effects on the unchanged serving graph. This transfer is unproven. Ordinary shared attention can already learn neighbor weighting, and nonlinear/private-attention members already have different effective propagation. The candidate does not assume a universal channel-factor capacity failure.

## Closest priors and exact boundaries

| Prior and saved scope | Direct overlap | Proposed operational difference and remaining limit |
| --- | --- | --- |
| [GNN-FiLM](https://arxiv.org/abs/1906.12192v5) §2.1; ordinary GAT | Data-dependent incoming-message modulation and attention before aggregation. | The operation is prior. The candidate specifies when a paired ranking signal is routed to graph paths and checked, rather than claiming a new gate. |
| [DIVE](https://arxiv.org/abs/2408.04400v1) §§2–3 | Continuing learned edge masks, own task losses and overlap regularization. | This rule uses no overlap objective; recipient partition concerns corrective learning, not disjoint served masks. DIVE explicitly uses separate encoders and selected-single inference. Shared fixed-pool utility remains a separate question. |
| [GEENI](https://doi.org/10.1145/3489517.3530416) indexed abstract | Suppression of outgoing likely-error-node messages and diverse ensemble creation. | Error-conditioned message correction is particularly close. Exact path credit, constraints, conditioning and sharing are unresolved because primary retrieval returned403; no complete difference is certified. |
| [SuGAr](https://arxiv.org/abs/2410.22228v2) §4 | Supervised predicted edge-weight diversity and label-aware learning of graph evidence. | No cross-member edge-weight dot-product penalty or selected-subgraph ENS stage is added. Both pursue supervised graph evidence; existing operational ancestry remains. |
| [GNN-Ensemble](https://arxiv.org/abs/2303.11376v1) selected §§III–IV; GRACE/GRAND view ancestry | Graph/feature context variation, supervised or consistency learning, and ensemble robustness. | The challenge is one gradient-derived normalized context, and correction uses a specific ranking cohort and recipients. Generic graph robustness is a necessary alternative explanation. |
| [FoRDE](https://arxiv.org/abs/2306.02775v3) §3.2; prediction-tangent and Local Ensembles scopes | Task-dependent functional gradients and direction selection. | The derivative coordinates are attention-edge biases and the action is continuing correction, not input-gradient repulsion or one-time covariance initialization. Prediction geometry and constrained updates are already established. |
| [GNCL](https://arxiv.org/abs/2011.02952v2) and existing block supervision | Own/pool objectives and different gradient recipients. | The target here is a frozen common-competitor context risk with path-specific selection and a finite guard. A broad claim still requires GNCL controls; a new scalar loss is not claimed. |

No new primary retrieval or whole-paper/code credit is taken. The saved scopes do not resolve all recent counterfactual, adversarial-view or error-specialist methods. The candidate's exact operational rule is explicit; originality remains unestablished.

## One bounded paired mechanism screen

Propose one future paired seed block using the [full public WikiCS source recipe](/Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/portable_internal_be_public_interface_20261007_v2/recipes/wikics.json): `be_unit`, four members, hidden512, seven local/two global layers, dropout.5, lr.001, 100 local plus1000 global epochs, two ordinary views, mean-probability serving and original strict-first complete-VALID joint selection. Use seed6101 solely as the inherited first pilot seed. The existing portable source is not a qualified author-runtime reproduction. This is a proposed screen, not admission or a population claim; the development population also selects checkpoints and TEST remains closed.

Corrections occur only every tenth epoch in the final half of the1100-epoch horizon:55 possible opportunities. No warm acquisition or continuation is added outside that horizon. Compare exactly two conditions:

1. **Data-dependent path credit:** the rule above.
2. **Matched random path credit:** identical architecture, challenge, cohort risk, guard, proposal norm and opportunities. Match each member's recipient count within each receiving row, but choose its disjoint recipients randomly from that row's original nonself edges. Do not select recipients from development outcomes. This tests the claimed credit-selection advantage against equally sparse corrective learning.

Both conditions compute the quantities needed for those matched counts; this is an attribution control, not padding for a speed claim. Finite trial counts and admissions can differ. If the data-dependent rule has no eligible coordinates, record the null rather than silently substituting another mask.

Each cell has8800 ordinary training member/view forwards and1100 native Adam transitions. An opportunity requires at most four clean member forwards, four challenged member forwards and four trials of clean-plus-challenged four-member forwards:40 total. Thus the bounded additional training work is at most2200 member forwards,110 reverse collections for challenge/corrective gradients, and55 accepted gate-only steps. Validation work remains complete and separately charged. Actual work, rejected trials and memory for `4E` biases must be reported; counts do not imply equal reverse cost or a measured speed advantage.

This pair can reject the path-credit mechanism; it cannot establish the whole recipe's practical superiority. Before any method claim, obligations include the same gated bank with ordinary own learning, unrestricted corrective credit with the same challenge/guard, a same-architecture GNCL reference, the competent unit-plus-contrast/single references, and a competent independently acquired four-model bank with disclosed checkpoint-selection differences. DIVE/SuGAr or another qualified learned-evidence comparator is material for a broad graph-diversity claim. These are claim obligations, not an admitted grid. Reuse existing results only when source, recipe, selection and readout really match.

## Existing evidence and decisive falsifiers

The [existing Wiki24 interpretation](/Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/wikics24_scientific_interpretation_independent_20261007_v1/REPORT.md) already rejects the rationale that more coverage or a larger pool-minus-member gain is sufficient: Rademacher improves coverage while weakening members and the served bank; unit-plus-contrast improves mostly competence. The candidate must outperform that simpler competence explanation. The [saved DIVE assessment](/Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/graph_subgraph_disagreement_dive_primary_scope_20261005_v1/REPORT.md) shows that disjoint copies of the same spurious motif can have zero mask overlap and identical errors. Different corrective paths likewise cannot guarantee complementarity. Citeseer initialization21's failed fixed-RMS recipes, supplied by root, rule out repeating that initialization family under a new description; this candidate changes continuing message learning and does not reuse it as a success claim.

Those results constrain the rationale. They are not an empirical test of this unexecuted rule. The paired screen falsifies its specific mechanism if matched random credit equals or exceeds its useful net repairs, if guards stall, or if the fixed challenge/cohort is inactive. A method claim fails if apparent cohort repairs are canceled by new full-population errors, mean or worst-member competence declines, the clean served bank fails to improve beyond competent references, or gains come only from graph-view regularization/calibration without changed relative neighborhood-margin responses. Report full-population repairs and harms, common-competitor support, member mean/worst scores, pool accuracy/NLL and a frozen label-blind neighborhood-response panel; neither favorable cohorts nor oracle coverage replace the serving result.

No code, dataset, active Mol18 source, root status, ledger or remote-server state is changed. The candidate, pilot, constants and any later qualification remain subject to root's separate scientific admission.
