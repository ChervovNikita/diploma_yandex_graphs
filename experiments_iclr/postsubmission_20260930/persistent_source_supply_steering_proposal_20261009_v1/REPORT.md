# Persistent source-supply steering

9 October2026. **One inactive proposal for root review.** No new numerical/model/data/checkpoint work, research-host action, new primary method reading, frozen-source edit or running-recipe change occurred. This is an attributed learning-policy extension, not a requested paper verdict.

## The obstacle and the proposed intervention

A shared-weight committee can have competent paths that make the same wrong decision. A convex probability pool cannot repair a query when every member assigns a strict wrong rival more probability than the correct class. Larger factor distances, hidden angles or normalized transport differences need not create a classifier-visible correct alternative. The existing initializer selects directions once; the known centered condition changes the factor prior. Neither supplies continuing credit for which member brings useful raw graph evidence to a prediction.

**Proposal:** after an ordinary own-loss update, give different persistent routes private credit for supplying different fixed semantic source families that are absent from the rest of a counterfactual committee. Compare two genuine probability mixtures: all members lack the **same family assigned to the current recipient**; then only that recipient receives it. Differentiate both that member's factual and source-ablated predictions. Restrict the correction to private message/transport factors and accept it only with local competence and full-pool safeguards. The ordinary predictor always retains an explicit all-full-input view.

The graph-specific obstacle is attribution through a live nonlinear message system: a source affects messages, normalization and states at several layers. A basis probe or map norm can miss the supplied classifier evidence. The concrete setting to assess is typed IMDB neighborhoods: actor/director/keyword families **only where the dataset/backbone agent verifies those families exist**. Semantic relation removal gives a meaningful access intervention even when source-node features are zero. It does not identify a causal real-world source, irreducible geometry or holonomy. Any useful effect may still be explained by semantic attention, node adapters or a capable joint model. No duplicate dataset/backbone survey was performed.

## Precise source and predictor definition

For the representative setting, use M4 complete paths of a separately qualified contemporary typed backbone. Let θ denote shared native learned weights and η_m member-private multiplicative factors at the qualified relation-message/transport affine sites. Stems and heads remain shared; the correction cannot add new private heads or semantic gates. Exact eligible sites, native objective/metric, native self paths, source schema, baseline competence and resource eligibility belong to the separate dataset/backbone assessment and must be frozen before source implementation or a run. The objective/update below is precise; this is **not yet an executable IMDB recipe**. In a compatible factorized bank, keep the known centered own objective:

`F = mean_m NLL_m(X,TRAIN) + 0.0005/(2M) * sum_m ||eta_m−1||²`,

with native slow Adam/decay and zero fast optimizer decay, qualified at the actual benchmark. The coefficient0.0005 is a proposed fixed carryover, not an attributed IMDB optimum; root must freeze that base against a competent ordinary reference. This proposal adds no teacher, router or initializer. The stated categorical-CE pilot requires a verified single-label probability task; a multilabel IMDB variant needs a separately specified proper-score amendment rather than a silent port.

Use G=2 or3 verified semantic source families, ordered actor/director/keyword where available. Do not fabricate keyword nodes from a feature vocabulary, or assume that masking actor/director features changes a model. Assign family a to member a for a=0,…,G−1, fixed throughout training. Remaining members receive F only and remain in the four-path pool. No TRAIN/VALID labels choose these assignments.

Let `V^−a` be the complete native graph view with **all edges of source family a and their required reverse relations removed for every member**. Retain the node set, all node features, movie/query own inputs and native mandatory self/projection paths. Rebuild the native graph cache and its normalization/attention over that view; do not zero already-normalized messages without renormalization. If a backbone takes precomputed meta-path views, the equivalent removed-family construction and leakage through derived paths must be source-qualified. It must not leave a path containing family a in the probe accidentally. Other source families remain available. The factual view V retains every family. The loss population T is all TRAIN movie queries, the same for every a.

At one post-own-update reference state `(θ+,η+)`, obtain native class probabilities:

`p_m = p_(θ+,eta_m)(V)`, `p_m^−a = p_(θ+,eta_m)(V^−a)`.

Use a fixed native TRAIN-mode RNG token per member/view, recorded and replayed exactly, retaining native stochastic operations. The entire steering block restores the ordinary training RNG afterward. Native support changes may change random draw counts, so common randomness is not assumed between factual and relation-removed views. The same tokens/views are used by every control, and pre/post guard comparisons use the same realization. Replay/state equivalence must be qualified. Eval-mode source responses are separately measured after restoration.

## One source-supply objective

For each active member m assigned family a=m, freeze the other members' ablated predictions from the reference state:

`C_m = sum_(k != m) stopgrad(p_k^−a)`.

With θ frozen during the correction, define

`Q_m^supply(eta_m) = [p_m(V;eta_m) + C_m]/M`,

`Q_m^absent(eta_m) = [p_m(V^−a;eta_m) + C_m]/M`,

`J_m = mean_(i in T) [−log Q_m^supply(i,y_i) + log Q_m^absent(i,y_i)]`,

`J = mean_(m=0,…,G−1) J_m`.

These are normalized probability pools, not an unconstrained synthetic logit correction. `J_m` is the log-risk change when member m alone supplies that source to a committee otherwise missing **that same source**, not all sources and not each peer's own different source. Minimize J. All factual and ablated dependence of m's native propagation is differentiated; shared θ and the other members' reference predictions are frozen. Recompute detached references at every steering update. This is a state-dependent surrogate/Jacobi rule, not gradient descent on one time-invariant joint objective.

Compute mixture log likelihoods by log-sum-exp of native log probabilities. This avoids subtracting near-zero probabilities. Let `rho_m^supply = p_m(y)/(M Q_m^supply(y))` and `rho_m^absent = p_m^−a(y)/(M Q_m^absent(y))`. The exact private derivative is

`grad_eta_m J_m = mean_T [rho_m^supply * grad CE(p_m,y) − rho_m^absent * grad CE(p_m^−a,y)]`.

Both responsibilities lie in [0,1]. This equality holds for categorical multiclass CE as well as binary CE. The rule is a **difference of adaptively weighted factual/probe CE gradients**, not a new contrast primitive. An implementation with those exact detached weights, scalar J values, RNG, state and guards is an identity and needs no duplicate fitted arm.

The negative probe coefficient is a real gaming risk: unconstrained descent can increase absent-view NLL instead of improving supplied-view NLL. The counterfactual-pool anchor below is therefore indispensable. The second differentiated prediction distinguishes this rule from the saved frozen-teacher/no-message offset, whose binary private derivative reduces to one positively weighted context CE. It also distinguishes J from unsigned/angular function-space repulsion: only correct-class pool likelihood supplies the target, and opposite responses are not rewarded simply because they differ.

Different fixed recipient families do not mathematically force complementary full-input predictors. All members could still learn one common correction or ignore the supplied family. Only the assigned member receives that family's source gradient; peers serve as detached reference predictions. The COMMON assignment control and restored all-full-input quality/source assays below test whether that allocation makes a useful difference.

## Exact persistent update and competence safeguards

Use a new common100-epoch warmup under centered F with **all factors trainable**, then branch from the exact epoch100 model/Adam/RNG state. This warmup is distinct from the sealed initializer's frozen-factor warmup. Train each branch through epoch500. Apply the steering block after the ordinary F update every fifth continuation epoch: `100+5*(u+1)`, u=0,…,U−1, with `U=G*floor(80/G)`. This gives80 opportunities for G2 or78 for G3 and complete COMMON source cycles. Remaining epochs use F only, with no adaptive horizon extension. The cadence is a proposed fixed resource/optimization choice, not an adopted winning schedule.

At such an epoch:

1. Retain the private displacement `Delta_eta_F` of that ordinary Adam step. Keep all Adam moments as produced by F.
2. Form the old-state surrogate J and private gradient g over active recipients. Let R_j be each competence risk below (full-input own risks, full pool risk, assigned-probe own risks and assigned absent-pool risks). Define the closed cone `K={v: dot(grad R_j,v)<=0 for every j}` and the unique Euclidean projection `v*=argmin_(v in K) ||v+g||²/2`. If v* or the active own-displacement norm is zero, record a zero correction. Otherwise set `d = 0.1 * ||Delta_eta_F,active|| * v*/||v*||`. This fixes the initial correction dose relative to the actual own step. There is no learned weight or strength search. Cone projection/Armijo are known constrained optimization machinery; solver and derivative qualification remain necessary. Projection can legitimately yield zero if source improvement conflicts with competence.
3. Try `eta'=eta+ + t*d` for t=1,1/2,1/4,1/8, in that order. Every attempt starts from η+, with θ+, Adam state and RNG tokens unchanged. Accept the first finite attempt satisfying all conditions below; otherwise retain η+ and log a rejected correction. There is no retry, reseeding, extra candidate or horizon extension.
4. Keep shared parameters and all Adam moments unchanged by this external private step. Ordinary F training resumes. The stale own-loss moments after a private correction are part of this split learning policy and can affect later learning; no optimizer-invariance claim follows.

For the exact cone projection, `dot(g,v*) = −||v*||²`; nonzero v* is a first-order J descent direction while every declared competence-risk derivative is nonpositive. This does not certify a finite nonlinear step, which is why the guards remain. The cone has `M+1+2G` scalar risk constraints (at most11 for M4/G3), not one constraint per TRAIN query. Its gradient storage and repeated VJPs are charged.

Acceptance conditions, evaluated at the same pre/post reference realizations:

- `J(eta') <= J(eta+) + 1e−4 * t * dot(g,d)` (Armijo decrease of the declared surrogate).
- Every member's **full-TRAIN evaluation-mode own NLL** is no larger than at η+.
- The full-TRAIN evaluation-mode **actual mean-probability pool NLL** is no larger than at η+.
- Each active member's assigned source-ablated native TRAIN-mode own NLL on T, using its fixed token, is no larger than at η+.
- For each active m, `A_m=NLL(Q_m^absent,T)` is no larger than at η+, with the same detached peer references. This directly anchors the denominator the objective would otherwise sabotage.
- Each recipient's J_m is no larger than at η+; the aggregate also satisfies the Armijo decrease.

Mathematical inequalities are exact. Root must prospectively fix any implementation tolerances and abort/finite policy during qualification, before scientific outcomes. A nonfinite reference or gradient makes the recipe unqualified; it must not be hidden as a favorable skipped seed. Finite guard rejection is a recorded algorithm event.

Write `S_m=NLL(Q_m^supply,T)`, so J_m=S_m−A_m. `Delta J_m<=0` and `Delta A_m<=0` imply `Delta S_m=Delta J_m+Delta A_m<=0`; strict J decrease gives strict supplied-pool improvement. Thus the correction cannot obtain its mean contrast gain solely by worsening its declared reference absent-pool NLL. An own-probe CE anchor alone would be insufficient because pool weights differ. Each source contrast freezes peers at the pre-correction reference; simultaneous corrections may change peers on other removed-family views. The anchor does not certify the all-updated source-absent committee, which requires fresh measurement. Guards protect full-input risk **relative to that post-own state on TRAIN**. They do not guarantee per-query probe retention, heldout competence, primary-metric improvement, stability of later Adam steps, or protection from redistribution/overfitting. If the permitted factors cannot improve source supply feasibly, zero projection or frequent rejection is a meaningful falsifier. A negative source term without these anchors is not the proposed method.

## Memory and work: retain one native tape

The demonstrated four-tape memory cost is relevant. Do not materialize a joint differentiable four-member/source graph.

- Run the reference bank sequentially under no-grad: M factual and M*G family-removed TRAIN-mode forwards, then M factual evaluation-mode forwards. Retain log probabilities/RNG tokens and detached reference responses, not their tapes: `M*(G+2)` reference paths. At G3/M4 this is20, not four probes with peers losing their different assigned families.
- Treat required output log probabilities as small leaves to obtain J and R_j cotangents. Replay factual/assigned-probe TRAIN-mode paths plus factual evaluation-mode paths sequentially. Keep the fixed audit schedule of `2M+G` paths (11 at G3/M4), including zero-recipient factual work, for matched controls. Multiple required VJPs may retain a single path's tape briefly; release it before the next path. **Only one native autograd tape is live**, with separate gradient vectors for the small set of constraints. Parameters stay at the same old state throughout. The declared gradients are exact if replay matches; projection/storage/VJP costs are additional.
- A trial needs M factual TRAIN-mode, G assigned family-removed TRAIN-mode and M factual evaluation-mode paths: `2M+G`. Peer responses stay frozen at the reference state for J/A; they are not reevaluated or silently changed. At most four trials are allowed. Charge all rejected paths.

The upper bound is `M*(G+2) + (2M+G) + 4*(2M+G)` additional native paths per opportunity, plus repeated VJPs, cone projection, gradient/output/RNG storage, native view caches, guard arithmetic, snapshots and serialization. This is a work count, not a measured runtime/memory result. Four persistent model states and normalization workspaces still consume memory; a sheaf instantiation also retains native detached `.L` diagnostics, which are not operator measurements. Replay reduces retained activation tapes; it does not make graph work free or certify a feasible peak. Any cache release/checkpointing requires a separately qualified source amendment. Resource limits warrant scheduling/measurement, not dismissal of the scientific question.

## Strongest published collisions and exact boundary

All conclusions below are reused from bound saved method scopes; zero new primary scopes were necessary.

| Prior | Collision and boundary |
| --- | --- |
| GNCL, arXiv:2011.02952v2, saved Eq5/pinned fit path; existing internal-own/pool credit policy | Actual probability-pool risk and own/pool objective mixtures are prior. J adds fixed source removal and a **negative source-absent derivative** with restricted private recipients and finite guards. This is not a new pooling or responsibility principle. |
| AdaGCN1908.05081v3; BGNN2101.08543v1; B3F-GNN; C&S2010.13993v2 | Graph-conditioned residual/error specialization and correction are established. Here the residual is a contemporaneous source-absent committee, not a fixed local teacher, propagated label residual, sequential classifier or fitted tree. Simultaneous sharing and this intervention are an operation boundary, not novelty clearance. |
| sMCL1606.07839v1; saved graph expert/context routing and meta-weighting | Specialist task allocation, current-loss credit and balanced responsibilities are prior. The assignment here is fixed and label-independent; no oracle/winner router is served. Correct source credit need not yield a useful uniform pool. |
| HGEN2509.09843v1/IJCAI2025 scoped method; LHGEL2510.03432v1; CHoE2605.15888v1; HetSheaf2409.08036v3 | Semantic/meta-path experts, residual/semantic attention, typed relation transforms and heterogeneous ensembling are direct collisions for IMDB. CHoE's inspected experts are frozen/pretrained and routed by edge similarity; HGEN has independent path learners and learned fusion/Gram penalty. The proposed current-label source-supply credit, shared parameter placement and fixed probability pool differ operationally. These differences do not clear novelty or outperform semantic attention. |
| FoRDE2306.02775v3; DICE2101.05544v1; conditional graph-response kernel prior; SuGAr/HGEN | Predictive/conditional diversity and graph-view specialization are prior. J uses supervised source-supply log-risk rather than response orthogonality, MI or hidden/map distance. A classifier-facing objective alone is not a novel diversity principle. |
| CF-GNNExplainer2102.03322v4; AD-GCL2106.05819v4; GRAND/GraphMix | Finite graph removal, native degree reconstruction and supervised/contrastive views are prior. Here semantic relation families are fixed without labels, query own features remain, and predictor parameters—not a mask generator—are steered. No causal explanation or invariance theorem is inherited. |
| BSNN2410.09590v1; BatchEnsemble/Rank-1 BNN; BuNN/HetSheaf/LMGC | Shared weights/private geometry, correlated global factors and multiple transports are established. Source-supply credit does not establish a new sheaf principle or a uniquely useful persistent representation. |

The potentially identifiable mechanism is **useful supply of an actual semantic neighbor family to a committee otherwise missing that family**, under competent all-full-input paths. A pointwise model with preserved query features has factual/probe functions and private derivatives identical on T, hence J and its gradient are identically zero. Mere output equality at one parameter state is insufficient for that derivative conclusion. This null identity supplies a graph-dependence check; it does not distinguish message transport from all other graph-conditioned capacities.

## One representative pilot and indispensable controls

Propose **one typed IMDB representative split only if the separate dataset/backbone assessment establishes eligibility and competent native/independent references**. M4, G2/3 verified semantic families, three prospectively frozen optimizer seeds, the fixed centered base F, and the new common warm100 followed by400 branch epochs are the proposed schedule. Do not claim IMDB unused until its local history is checked; proposed seeds may remain7409/8501/9607, which are not unused confirmation. The horizon/Adam must be qualified for this backbone rather than treated as a published IMDB optimum. No original18/centered3 artifact is opened, modified or silently reused. There are six shared-bank branches and one independent-bank counterpart per seed; independent and shared warm states are separately fitted. No fitting is admitted here.

| Branch | Private correction objective; question isolated |
| --- | --- |
| Centered own only | d=0. Is any persistent correction useful? |
| Native pool credit | NLL of the factual probability pool; θ still receives only F. Does ordinary predictive ensemble credit explain the effect? |
| Source-view supervision | Mean assigned source-ablated own CE on T. Does masked-view supervision explain the effect? |
| Uncoupled source contrast | Mean `[CE(p_m(V),y)−CE(p_m(V^−a),y)]` on T. Is the effect generic supervised source contrast rather than committee-conditioned supply? |
| COMMON source-supply assignment | All G active recipients receive the **same** family a(u)=u mod G at steering opportunity u, cycling through identical families; other members still receive F only. Is fixed member-specific assignment necessary? |
| Proposed source-supply J | The precise surrogate and recipients above. |
| Genuine untied independent4 plus J | Four complete factorized member models, every learned parameter object/storage disjoint. Same F, views, fixed assignments, source-supply objective, private recipients, projection/guards, horizon, selection, probability pool and charged native-path/VJP opportunity. Does sharing provide any advantage? |

Use the same semantic views/full T, reference calls, replay method, correction normalization/cadence, risk-cone constraints, four-trial cap and validation/checkpoint/serving contract. All methods retain full-input own supervision. Require each nonzero objective's own Armijo decrease plus identical factual, assigned-probe and absent-pool competence anchors. Source-supply branches also require their J_m guard. COMMON gives each active recipient all families over complete G-cycles with the same number of source-gradient opportunities; the U formula fixes equal aggregate family exposure without outcome-guided extension. Own-only makes no private proposal; shadow measurements remain disclosed. Rejected calls, VJP work, unequal accepted dose and all caches are charged, not assumed equal in wall time. Equal computation opportunity cannot stand in for matched actual resource reporting. There is no label-shuffled arm or raw map penalty.

The independent4 counterpart is genuinely untied in its complete learned bodies, with the same incidence/message factors per body so its steering scope is comparable. It is not claimed identical to the ordinary native independent4, which remains a competence/reference requirement. Loss coupling through detached committee references does not create parameter sharing. Charge four bodies/moments, every source/replay/guard path and selection/restoration at the same resource accounting as the shared candidate. No cheaper weak independent reference licenses superiority.

Full-VALID selection uses the verified benchmark's native primary metric, then NLL/earlier epoch, and fresh restored **all-full-input mean-probability** serving. The common warm-best remains eligible; selection at epoch≤100 is no selected steering effect. Do not use source-view validation to choose families or assignments. Root must freeze practical own-member and pooled-risk success tolerances, uncertainty/reporting and all cost contracts before release. The one-source-supplied surrogate is not the deployment predictor, so a positive J or source-supply virtual-pool score cannot count as serving improvement.

Require better actual **all-full-input** served pooling, preserved own-member competence, correct alternatives with sufficient probability to repair the served pool, and favorable repair/harm balance over the complete population. Require a gain over both COMMON assignment and matched source-view supervision before a route-specific source-credit claim. Measure all-VALID fresh family ablation and within-family typed-neighborhood reassignment (native pairing/normalization/degree conditions qualified), with fixed movie/query own inputs. Report correct-class source-supply log-risk, full-input class margins, common wrong rivals, error coverage, accepted/rejected corrections and costs. Every member/seed and all VALID rows remain in the report. A gain only in source-supplied surrogate views fails the intended claim.

Promising results still require competent native single/independent4, global-latent BSNN/Rank-1 and matched node/frame/joint multi-transport capacity controls before a sharing/geometry advantage. Independent or joint-capacity alternatives should receive the same source information/steering opportunity when claiming that the parameter-sharing placement matters. An unused graph/split and frozen confirmation protocol are later requirements for general claims. The containing untied class has no automatic accuracy disadvantage.

## Candid assessment and root recommendation

This is scientifically distinct from the sealed initializer and factor prior because it introduces persistent, signed, classifier-visible credit through two complete source-dependent predictors and a counterfactual committee. Its ingredients and exact weighted-CE algebra have strong prior ancestry. No complete published equivalence was proved from the saved scopes, and no novelty gap was cleared. The defensible contribution, if demonstrated, would be a bounded source-credit/sharing result with measured competence and served utility.

Keep it inactive. Root may review the objective, necessary controls, replay qualifications and cost before deciding whether a new source amendment is worthwhile after the existing panels close. Avoid adding an unqualified penalty/grid to running recipes. A failed pilot cannot be rescued by larger J separation, map spread or source-response magnitude alone.

Historical exposure is recorded precisely in `READ_SCOPES.json`: the completed, already-opened Wiki24 report was displayed while retrieving saved competence/prior conclusions. Root subsequently confirmed that such historical negative lessons may inform this hypothesis. No current original18/centered3 or other incomplete-panel outcomes, raw predictions, datasets, labels or checkpoints were accessed. The historical lesson used is the need to acquire correct evidence with protected members, not a new numerical result or unused confirmation claim.
