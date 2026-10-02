# Graph-conditioned correction of shared updates

3 October 2026. Source/literature theory packet only. Retained v21 and the sealed SAM/MAML conclusions were consulted first. No source driver, data/model inspection, launch, protocol/index change or measured gain is produced.

## Decision

Retain **one conditional composition**: minimally correct a proposed shared-parameter displacement so that it preserves current, graph-weighted correct-class margin advantages of useful members, while preserving the proposed pooled-loss directional change. Private updates and the native moment transition stay explicit and fixed. This is response-preserving projected optimization with a graph-conditioned choice of protected responses. OGD supplies the closest consulted projection operator; PCGrad supplies the closest consulted pairwise loss-gradient comparator. Known ingredients do not reject this composition, and no new projection principle or exact-equivalent-method absence certificate is claimed.

The scientific opening is that **aligned member loss gradients can still erase a correct member's logit-margin advantage** when member shared Jacobians differ. Pairwise PCGrad need not act in that case. Conversely, identical member shared Jacobians cannot erase a centered logit contrast to first order. A generic claim that sharing always destroys complementarity is false. No source/state evidence establishing harmful erasure in the current recipe was accessed.

The graph prior is conditional: useful TRAIN specialist advantages might cluster along graph neighborhoods. Protecting those particular response functionals could be better than protecting topology-free or permuted counterparts. Preservation itself is not predictive evidence, and current TRAIN winners can be overfitted. Current GPUs do not decide the hypothesis's scientific merit.

## One exact operation

Use the same prospective native four-member warm state, initializer, modern backbone, continuation and mean-raw-logit pool for all arms. No new initializer or inference router is added. Correct only the **first 32 native continuation optimizer steps**, with this window frozen before outcomes; a source with fewer remaining steps is ineligible.

At a step, obtain the ordinary noisy own-CE optimizer proposal, including the private proposal phi_plus, shared displacement d0, and native moments/step transition. Build a deterministic dropout-off reference B=(theta_old,phi_plus). Thus the diagnostic isolates the shared proposal after the same private proposal. Source qualification must bind buffers, dropout-off map, native RNG progression and exact copied state. Native training gradients remain stochastic; the response constraints and finite checks refer to this declared served-prediction surrogate.

At TRAIN node v define the class-centered correct margin vector t_y=e_y−mean_(k!=y)e_k and the member contrast margin

    a_m(v) = t_y^T (z_m(v) − mean_j z_j(v)).

Let I_m(v) mark TRAIN nodes where member m predicts the correct class, has larger target probability than the pool, and a_m(v)>0, under fixed ties. Set u_m=I_m*a_m. For the released graph use one fixed self-looped row-normalized nonnegative adjacency P, and

    w_m = I_m * (P^2 u_m) / sum_TRAIN [I_m * (P^2 u_m)].

Zero denominator means an absent specialist constraint; no substitute or redraw. All labels enter only on TRAIN. Unlabeled nodes may carry the two-hop transport but are not labeled roots. Stop gradients through the support and weights at this step. P is used only for this training correction; the native forward graph and inference pool do not change.

Protect S_m=sum_v w_m(v)*a_m(v), with shared gradient h_m=gradient_theta S_m at B. This is a weighted **relative logit response**, rather than an embedding norm or member CE task loss. Using log-probability differences themselves as protected responses would confuse ordinary common-confidence saturation with contrast erasure; they select competence support here but are not the differentiated protected quantity.

Let a=gradient_theta L_pool(B), and H contain the at most four eligible h_m. Require a^T d0<0 under the declared nonzero/rank tolerance. Find the minimum-Euclidean-norm correction

    minimize_c 0.5 ||c||^2
    subject to a^T c=0, H^T(d0+c)>=0.

This tiny constraint QP uses parameter VJPs and an at-most-4x4 Gram; it needs no full Jacobian or Hessian. Fixed active-set order/tolerances are required. If infeasible, numerically unresolved, or ||c||>||d0||, use the native proposal and retain the reason. There is no cap enlargement, alternate block or multiplier search.

Evaluate the full proposed corrected state (theta_old+d0+c,phi_plus), native joint proposal, and B under the same dropout-off map. Accept only if every eligible frozen S_m is at least its value at B, corrected pooled TRAIN CE is no worse than **both** native joint proposal and B, and corrected mean-own TRAIN CE is no worse than B, within prospectively bound finite tolerances. Otherwise commit the native proposal. A stricter requirement that own CE beat the native proposal can defeat the intended protection: at equal pooled logits, preserving a contrast can retain a larger Jensen penalty. Report that cost rather than hide it.

Commit phi_plus and the original native moment/step transition, with only the shared parameter displacement corrected. This is an explicitly projected optimizer map; it is not raw-gradient PCGrad followed by AdamW. Moments track the original own gradient, and this convention can itself be unfavorable. All trial state is discarded and native RNG advances only once. After step 32 resume the unchanged optimizer map. Source implementation remains unqualified and unauthored.

## Mathematical distinction and closest priors

[DERIVATION.md](DERIVATION.md) gives the shared-Jacobian null, aligned-gradient witness, minimum correction and feasibility obstruction. The leading contrast change is h_m^T d0. Pairwise products g_i^T g_j do not determine it. The correction preserves a^T d0 exactly only in the local diagnostic; finite pooled checks do not imply later quality.

| Consulted prior | Complete operation and difference |
|---|---|
| **PCGrad v1**, arXiv:2001.06782v1 | Computes task loss gradients, sequentially removes negative pairwise components in random task order, sums them, and passes that gradient to an optimizer. Applying it to ensemble members is a known control. Our constraints use graph-weighted member-minus-pool **logit response gradients**, operate on one already-proposed shared displacement, and preserve its pooled directional effect. This is not the same criterion or optimizer convention. The exact v1 listing was read; latest v4/code was not qualified. |
| **OGD v1**, arXiv:1910.07104v1 | Projects updates away from stored previous-task logit-gradient spans, retaining outputs locally while learning a new task. It is the closest consulted response-preserving operator. Here responses are current graph-weighted relative margins, constraints are one-sided, refreshed at each step, and a pooled-progress equality is added. There is no past-task replay or full-output invariance claim. |
| Retained GNCL, TreeNets and SEA conclusions | Attribute own/pool and negative-correlation objectives, shared branches and error-directed complementarity. The CE identity own=pool+Jensen penalty remains. This operation changes the update map; it does not claim a new ensemble objective or a faithful shared-CE SEA port. |
| Retained graph-error/C&S/BernNet and shared/private graph expert conclusions | Attribute graph-conditioned residual/response weights, structural specialization and shared/private expert machinery. They do not establish equivalence to this particular constrained shared-displacement composition. No task head or inference gate is introduced. |
| Saved SAM/MAML notes | Perturbation-steered gradients and post-update initialization criteria are prior. Those methods were **not reread or counted as new** here. This operation corrects ongoing shared updates rather than selecting initialization by lookahead. |

The OGD scope also mentions A-GEM's loss-gradient inequality constraints. Its primary method was not read here; no exact A-GEM comparison or global nearest-prior closure is claimed. Broader projected/multiobjective optimization is attributed. A small graph-conditioned application can still be useful if the exact graph construction beats the same topology-free correction and standard controls.

## Falsifier and representative paired test

Complete **Squirrel and Photo**, one root-bound modern native configuration and initializer, fresh paired seeds **101/103/107** subject to root use-history checks. Use six arms:

1. Native own-CE continuation.
2. Shared pooled-CE gradient for the same 32-step window; private updates use the unchanged own-CE rule.
3. Standard member-task PCGrad on shared gradients for that window; private updates use the unchanged own-CE rule.
4. The specified graph-conditioned shared-displacement correction.
5. The same correction using one fixed node-permuted P only in its weight construction.
6. The same correction with P=I, retaining competence support, cap, QP, finite guards and window.

Arms 2/3 are ordinary block-objective and member-task PCGrad comparators with explicitly different moment conventions. The shared-pooled arm is a matched block control, not a faithful reproduction of full pooled training from TreeNets. Arms 4/5/6 isolate the conditional graph construction under the **same** projected-displacement convention. Candidate ranks, protected supports and realized correction norms may differ; report them. A common cap does not isolate orientation alone. Native inference, training topology, parameterization and served cost remain identical.

At every one of the 32 declared steps, retain shared versus private score changes, h_m^T d0, pooled a^T d0/a^T c, eligible support coverage, norm/rank/feasibility, all finite values and fallback reasons. Retain the whole paired cohort, including native fallbacks. Initial preservation and TRAIN guard success are construction evidence.

Primary utility is later pooled VALIDATION NLL under the unchanged native selector, fixed final endpoint secondary. A proposed gate for root to freeze is mean gain at least **0.01 nats over each of the five controls**, at most **0.5 percentage point** mean accuracy loss, and the same gain sign in all three seeds. These are practical screening constants, not a power or significance claim. Retain member competence, Brier, pool/own/Jensen curves and paid cost. Competent single/native BE/packed-independent references remain necessary for any broader GNNM utility claim.

The shared-erasure explanation fails if the native shared contribution does not erode the eligible response scores, if private changes dominate the disappearance, or if correction mainly preserves overfitted TRAIN advantages without later benefit. Topology-specific promotion stops if P=I/permuted correction matches the graph arm. Improvement only over PCGrad establishes neither a new projection principle nor graph value. Infeasibility, negligible use or finite rejection is retained without changing window, cap, protected functional, graph kernel, source block, tasks or seeds after outcomes.

If the development screen passes, freeze all definitions/state/tolerances and confirm on one previously unused complete compatible graph, e.g. Computers only if root use-history/source qualification permits, with seeds **109/113/127** and one heldout opening. No papers100M or scale claim is required for this narrow mechanism. No existing test score or original manuscript claim changes.

## Resource judgment

The permitted outcome-free receipt gives a **27.02 device-hour baseline continuation forecast** for six arms x two graphs x three seeds, before acquisition, source qualification or correction overhead. This is not measured runtime for the changed algorithm.

Across the three projected arms there are at most **576 correction attempts**. Each conservatively adds three four-member surrogate forwards and up to five aggregate VJPs: **12 member-forward and 20 member-backward route equivalents**, before state copies, graph transport and rejected QPs. PCGrad and the shared-pooled control add further backwards. Packing can change execution cost; route equivalents are not wall-time measurements.

At most seven shared parameter vectors (four h, pooled gradient, native displacement, correction) add **28*P_shared bytes in float32**, plus snapshots/optimizer transition, graph weights, activations and buffers. Profile actual source memory and all paid maps before a resource request. Scientific merit remains conditional independently of present device availability.

## Accounting and status

Two primary **method scopes** were read: **one retained abstract-only PCGrad citation upgraded to a scoped method**, and **one first scoped OGD identity**. Both are absent from v21 paper records; absence there does not make PCGrad a previously unknown paper. Zero full papers, author implementations or model/data artifacts were read/executed. Truncated overbroad locator/extractor previews are disclosed in READ_SCOPES.json. No numerical paper benefit is transferred.

One conditional operation saved, zero methods/drivers/launches adopted. Root owns source qualification, the concrete protocol, nearest-prior followup, resource request and any promotion. Integrity sealing certifies bytes and declared accounting only.
