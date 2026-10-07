# P note: the staged correction gradient and useful complementarity

7 October 2026. **Theory and source review only.** This note analyzes the actual private graph correction loss, connects it to the saved common-error diagnosis, and retains one known objective as a possible future diagnostic. It also gives limits relevant to the separately owned trainable internal BatchEnsemble/contrastive suite. It changes no study, implementation or execution plan.

The correction can learn useful common-error repair when TRAIN labels and reachable private functions support it. Its pooled Brier term supplies label-residual pressure, not an assignment of complementary tasks. Own CE already gives a strong gradient on a labelled confident mistake. Feature distance, a learned scalar or aggregation cannot by themselves establish useful error complementarity.

## 1. The actual source contract

The bound source is [method.py](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/wikics_staged_private_graph_residual_method_preparation_20261007_v4/method.py:191>), SHA-256 `d5874791fa6b6a323f1a672553ce22297df2fb45d083afc50dc7b16947bf8a1c`. `correction_loss` averages over TRAIN examples, sums Brier over classes, and uses own CE plus twice pooled Brier. `bank_epoch` obtains a deterministic evaluation-mode peer bank, detaches it, then computes each requested route's gradient with that route's training dropout. Every requested gradient is computed before any requested optimizer step.

For one labelled example, write the active logits as z, its probabilities as p=softmax(z), its true-class one-hot vector as e_y, and the frozen peer sum as s. With M members,

\[
q=(p+s)/M,\qquad L_m=-\log p_y+2\|q-e_y\|_2^2.
\]

The source has M=4. The formulas below are per example; the implemented gradient is their TRAIN mean. The training pool uses an active dropout prediction and deterministic peers. It has the probability-mean form of the serving operator, but is not the all-evaluation-mode serving prediction on that step.

E_stage permanently freezes earlier fitted paths. E_joint interleaves the same per-route correction against a common pre-update peer bank. Its donor is frozen and its peers detached. E_joint is therefore an interleaved coordinate comparison, not fully corrective GrowNet or end-to-end learning of a live common trunk.

## 2. Gradient, saturation and the pointwise optimum

Let J(p)=diag(p)−ppᵀ be the softmax Jacobian. Direct differentiation gives

\[
\boxed{\nabla_z L_m=(p-e_y)+\frac4M J(p)(q-e_y).}
\]

For the actual four-member source this is

\[
\nabla_z L_m=(p-e_y)+J(p)(q-e_y).
\]

The pooled term sends the same labelled pool residual through the active member's softmax Jacobian. The private parameter gradient additionally passes through that route's logit Jacobian. Two routes can consequently have different parameter gradients even when their logit cotangents match.

For M=4, set r=q−e_y and a=pᵀr. Each wrong-class logit has gradient

\[
g_k=p_k[1+q_k-a]\ge0,\qquad k\ne y,
\]

because a=pᵀq−p_y≤1. The gradients sum to zero, so g_y≤0. Thus an unconstrained independent-logit descent step increases the true logit and decreases wrong logits. The pooled term can change the relative pressure on wrong classes; it does not reverse the combined source gradient into a distinct wrong-class target. Shared neural parameters, cross-example coupling and Adam do not imply monotone improvement of every example from these logit signs.

At a confident wrong vector p→e_c with c≠y, J(p)→0. The Brier contribution saturates while the own-CE contribution tends to e_c−e_y. **A labelled confident error is already strongly corrected by own CE.** Low TRAIN CE supplies no label or gradient on an unseen held-out error, and cannot establish that unseen common errors are correctable.

The pointwise target is also explicit. Holding peers fixed,

\[
\|q-e_y\|^2=M^{-2}\|p-(M e_y-s)\|^2.
\]

The residual target t=M e_y−s has t_y≥1, t_k≤0 for k≠y, and components summing to one. Its Euclidean projection onto the probability simplex is e_y: moving probability from any wrong component to the true component cannot increase the squared distance. Own CE has the same simplex optimum. Hence the combined pointwise optimum is p=e_y, regardless of which member is active. With finite softmax logits the boundary optimum is approached as an infimum. It is not a member-specific role or a complementary label assignment.

This does not forbid complementarity under limited capacity and different private Jacobians. Different routes may generalize differently or repair different examples. It means useful complementarity is a possible outcome of learning, not a conclusion supplied by this pointwise loss.

### Initial representation differences are not initial prediction differences

The actual residual route is z_m=z₀+U_m R_m(x,H₀,A)+c_m, with U_m=c_m=0 initially. All members therefore initially predict the donor probabilities in real arithmetic, despite independently drawn graph representations R_m. At that instant,

\[
\partial L/\partial U_m=g_m R_m^\top,
\qquad \partial L/\partial c_m=g_m,
\qquad \partial L/\partial R_m=U_m^\top g_m=0.
\]

Different representations can yield different first output-weight updates; the private graph itself has zero first-step gradient through this zero output map. Zero gradients are compatible with connected source autograd. Representation distance alone cannot certify functional or error diversity.

## 3. Brier ambiguity is an accounting identity

For evaluation-mode member probabilities p_m and their mean q,

\[
\|q-e_y\|^2
=\frac1M\sum_m\|p_m-e_y\|^2
-\frac1M\sum_m\|p_m-q\|^2.
\]

This identity gives a useful interpretation of pool Brier. A spread term appears only together with changing member errors. Increasing spread while degrading the members can leave the pool worse. It is not a guarantee that repelling features, maximizing disagreement or decreasing error overlap improves classification. The source's detached-coordinate training and own-CE anchor must still be interpreted according to their actual updates.

The [saved scientific synthesis](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/shared_backbone_common_error_scientific_synthesis_20261007_v1/NOTE.md>) already records completed cases in which more pooling benefit or lower common-error overlap accompanied weaker members and worse served quality. That qualitative conclusion motivates complete served accuracy, retained member competence and net error repair. It does not predict a WikiCS result or prove that sharing caused the errors. No raw or ongoing outcome is used here.

## 4. The probability-mean dilution bound survives a loss change

Suppose the unchanged donor favors one common competitor c over truth y by d=p₀,c−p₀,y>0. If t of M copies change, each changed probability vector contributes at most +1 to the y-minus-c margin. Therefore

\[
q_y-q_c\le\frac{t-(M-t)d}{M}.
\]

A necessary condition for strict repair of that ordering is t>(M−t)d. For M=4 and t=1, d>1/3 makes repair impossible for the first route alone, whatever its capacity or objective; at d=1/3 the upper bound is a tie. Changing all four routes can remove this particular unchanged-peer obstruction. Changing only the loss cannot remove the bound under the same mean-probability serving rule.

More generally, if **every** member has p_m,c>p_m,y for the same competitor c, any nonnegative convex mixture obeys

\[
\sum_m w_m p_{m,c}>\sum_m w_m p_{m,y},
\qquad w_m\ge0,\quad\sum_mw_m=1.
\]

This includes node-dependent learned weights and one-member selection. It differs from the case where all members are individually wrong with different competitors, which a pool can sometimes repair. A nonlinear or class-specific prediction map may correct the unanimous case, but then it learns an additional classifier rather than merely selecting existing member evidence.

## 5. Closest saved priors, before proposing an objective

The [closest-prior P note](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/staged_private_graph_residual_closest_prior_P_note_20261007_v1/P.md>) and the diversity/learning notes were consulted before retaining the control below. These are reused scoped conclusions; no primary paper or author implementation was newly read.

| Saved prior | Relevant established operation | Limit for this question |
| --- | --- | --- |
| Deep Sub-Ensembles, 1910.08168v2, §2 | Train a competent full learner, freeze its learned trunk, fit private task networks on the same labelled data, average probabilities. | Direct architectural ancestry. Raw-x/H new graph paths and z₀ anchoring do not create a new frozen-trunk ensemble principle. |
| TreeNets, 1511.06314v1, §§5/6.1; GNCL, 2011.02952v2 | Shared initial/private later computation; individual versus pooled supervised objectives, including average-score/probability CE and scalar tradeoffs. | A pool-CE control is a known objective family. Different private Jacobians still matter; identical averaging-layer cotangents do not imply identical parameter gradients. |
| Hydra, 2001.04694v2 | Shared-body/private-head growth, each head trained against a separate teacher-member target. | The teacher targets and their acquisition differ. The control below acquires no separate teacher bank; its frozen donor acquisition still costs work. |
| GrowNet, 2002.07971v2; BoostResNet, 1706.04964v2; saved AdaGCN/BGNN/B3F-GNN | Raw/prior hidden features, sequential residual/error learning, additive or fully corrective operations. | Strong residual/graph-boosting ancestry. Raw mean-logit prediction and fully corrective earlier learners are materially different from permanently frozen probability-mean coordinates. |
| DICE; FoRDE | Label-conditional representation redundancy with same-class negatives/noise/discriminator; task-score input-sensitivity kernel repulsion. | Class-compatible or functional diversity has direct prior and extra sampling, auxiliary or mixed-derivative costs. Arbitrary feature cosine is not an equivalent estimator and neither method supplies an accuracy guarantee. |
| BatchEnsemble, 2002.06715v2; saved TabM/graph-view assessments | Learned input/output feature scaling around shared matrices; established graph-view/channel/contrastive diversification. | Supports a useful trainable internal adaptation, with complete member trajectories still charged. It does not prove common-error repair or create novelty by combining these ingredients. |

Saved adaptive GNCL, DOI 10.12792/iciae2026.015, explicitly uses previous-step loss ratios, moving averages and confidence variants. A learned or adaptive diversity coefficient is therefore not an unoccupied principle. The unread AdaGNN/GENNN/FAGEL method gaps remain unresolved; this note makes no exhaustive absence or exact-duplicate claim.

The [separate internal-factor follow-up](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/trainable_internal_graph_batchensemble_contrastive_mechanism_followup_20261007_v1/P.md>) already compares retained DICE, CDLG, SuGAr, HGEN and FoRDE operations. Its graph-evidence and class-compatible pairing requirements remain applicable. This note adds source-gradient limits rather than another proposed suite.

## 6. One retained teacher-free objective: probability-pool CE

The single retained loss substitution is

\[
\boxed{L_m^{\mathrm{poolCE}}=-\log p_{m,y}-2\log q_y,\qquad q=\frac1M\sum_jp_j,}
\]

using the same frozen-peer, active-dropout and probability-mean conventions. It changes the objective while preserving serving. It is an attributed known-method diagnostic, not a novel complementarity mechanism.

Differentiating with peers fixed gives

\[
\nabla_{z_m}L_m^{\mathrm{poolCE}}
=\left(1+2\rho_m\right)(p_m-e_y),
\qquad
\rho_m=\frac{p_{m,y}}{\sum_jp_{j,y}}.
\]

The extra term is exactly a positive, peer-dependent reweighting of ordinary own CE on each example. It provides no new logit direction or private target. At identical predictions, ρ_m=1/M, so for four routes it scales that example's CE gradient by 1.5. Adam history, epsilon and changing per-example scales prevent treating all finite training as a simple learning-rate identity.

There is a useful saturation distinction. If the routes remain identical while approaching a confident wrong prediction, their responsibility remains 1/M and the added pool-CE cotangent remains a CE multiple; the Brier cotangent tends to zero. If only the active member has almost no true-class mass while peers have appreciable mass, its pool-CE responsibility tends to zero. Its own CE remains active. Thus the weighting rewards existing true-class contribution, rather than identifying which private route has the greatest ability to repair the error.

This loss can support common improvement using permitted labels and learnable private functions. It supplies no supervision for unseen common errors, removes no staged dilution bound, and guarantees no error complementarity. If the question requires a substantively new complementary learning target, this tempting replacement does not satisfy it. Its defensible purpose is to test whether the current Brier curvature/saturation is a useful part of the known correction design.

### Why raw mean-logit CE is not this substitution

Replacing the pool term by CE of raw mean logits while still serving mean probabilities would optimize a different combiner. One active member can send its true-versus-c margin to infinity, making mean-logit CE tend to zero, while its probability contribution stays bounded by one. Three sufficiently confident unchanged wrong peers can keep the four-probability pool wrong. Serving an additive/logit predictor instead would change the predictor and bring the comparison into known boosting/residual-network ancestry. It is not an objective-only fix to the current probability pool.

## 7. Limits for the user's internal BE, contrastive, learned-strength and aggregation ideas

### Internal trainable modulation is a real learning change

W_m=diag(r_m)W diag(s_m) implements known BatchEnsemble scaling. Placing live factors before nonlinear attention and message passing can change member neighborhood evidence, while W receives gradients from all trajectories. This differs from adding private paths after a frozen H₀. Shared matrices do not imply one common encoder forward: M member states and message trajectories still consume work, whether batched or looped.

Across-member feature separation is not output separation. A private output nullspace can hide it; an invertible feature change can be canceled by its classifier; a member-identity code can be ignored by prediction. The zero-output initialization above is an actual source example. Useful diversity must earn complete served accuracy and net common-error repair with competence retained. Within-member contrastive invariance and across-member redundancy reduction should have declared, class-compatible pairing and evidence semantics. They are different objectives, not interchangeable labels for cosine distance.

### A freely learned positive coefficient has a degenerate incentive

For ordinary joint minimization

\[
L(\theta,a)=L_{\rm task}(\theta)+\lambda(a)L_{\rm div}(\theta),
\quad L_{\rm div}\ge0,
\]

with λ=softplus(a) or an increasing sigmoid,

\[
\partial L/\partial a=\lambda'(a)L_{\rm div}\ge0.
\]

Descent drives a downward and λ toward zero unless the diversity penalty is already zero or another declared term supplies opposing credit. Conversely, an unbounded nonnegative λ multiplying a subtracted diversity reward can run to infinity when that reward is positive; a bounded coefficient is pushed toward its upper boundary. The sign and offset of a signed regularizer matter, so this nonnegative-penalty proof must not be silently transferred to every named diversity loss.

A meaningful controller needs a specified constraint, regularized normalization, primal-dual interpretation or outer prediction objective. These are established methods with data, optimization and cost choices. A free scalar does not automatically learn the accuracy/diversity tradeoff. This is an implementation constraint for the separately owned suite, not an instruction to add another search.

### Learned aggregation and RL require prediction credit

Convex weighting or selection obeys the unanimous-competitor bound in §4. A richer supervised aggregator can learn new decisions, but should be compared with ordinary differentiable stacking/gating and an equally capable single predictor. It adds trained capacity and information use. Modulation, soft contrastive objectives and soft weighting are already differentiable. RL becomes a separate concrete proposal when an action is genuinely discrete, such as an edge subset under a fixed message budget, with a predeclared predictive reward and charged sampling work. RL supplies neither missing labels nor a guarantee that disagreement is useful.

## 8. One falsifiable prospective comparison, without adopting it

If the parent later adopts this known-objective diagnostic, use **two objectives × three paired seeds (17, 29, 43)**, not an additional diversity or aggregation grid:

1. Current own CE + 2 pool Brier.
2. Own CE + 2 probability-pool CE.

Use the existing E_joint-style interleaved schedule: 100 rounds, all four route gradients computed against the common pre-update deterministic detached peer bank, then all four private optimizer steps. Match the official TRAIN/serving populations, frozen donor/cache, corresponding initial private tensors, route RNG states, active dropout, fresh Adam, own coefficient 1, pool coefficient 2, update counts and evaluation-mode mean-probability endpoint. Fix numerical pool-CE evaluation and stopping/endpoint conventions before a fit. All planned fits must reach their fixed endpoint before comparative outcome inspection.

Reuse existing E_joint fits only if their full contracts match; otherwise this is six separately adopted future fits. There is no new teacher acquisition. Charge donor acquisition, deterministic bank forwards, active route forwards/backwards, optimizer work, wall time and peak memory. Equal update counts are not a claim of equal measured cost. The dedicated internal BE/contrastive allocation remains a separate question; this note adds no cells to it.

At the complete fixed endpoint, report served accuracy and NLL, each member's accuracy/NLL/Brier, pool gain over mean member accuracy, and donor-error flows: wrong→correct, correct→wrong and wrong→different-wrong. Report net common-competitor repairs on the same predeclared donor cohort, alongside the full population so cohort improvements cannot conceal introduced errors. Initially copied donor predictions define one donor-error cohort, not four independent observations.

The common-accuracy hypothesis is supported only by a predeclared useful paired accuracy gain with positive net common-error repair and retained competence. Lower Brier/NLL alone rejects an accuracy explanation. More disagreement or lower overlap alone rejects a useful-complementarity explanation. If pool CE wins, attribute the result to the known peer-dependent CE weighting under this architecture/schedule; it does not demonstrate a new complementary target. If it fails, close this control rather than choose another coefficient, objective, checkpoint or combiner after seeing outcomes. A favorable development result still needs separately frozen confirmation for generalization.

## Read accounting and disposition

SOURCE_BINDINGS.json binds the implementation and reused notes/index. READ_SCOPES.json records the bounded source and summary scopes. This packet claims zero new primary paper reads, zero new primary method scopes, zero full-paper certifications, zero author-code reproductions and zero numerical model executions. Previously read source summaries were reused; old completed numerical summaries encountered in those notes supply only their already saved qualitative interpretation here. No dataset, label payload, checkpoint, prediction/logit payload, partial ongoing outcome, scoring operation, GPU/SSH job or active source edit was accessed or performed.

**Disposition:** retain the exact gradient and dilution limits; treat probability-pool CE as one known diagnostic, not a new complementarity solution; pass the class-compatibility, coefficient and aggregation constraints to the separately owned internal BE/contrastive work. The original studies remain governed by their existing plans.
