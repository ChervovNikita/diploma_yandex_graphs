# NCNC structural ambiguity and completion-pattern supervision

3 October 2026. Static scientific assessment of the root-bound, running NCNC family. No project outcome, dataset, checkpoint, prediction, SSH session or scientific execution was opened. Only this new packet is written.

## Decision

**Retain one unadopted incremental hypothesis and one representative paired pilot proposal:** train the four existing shared-factor NCNC completion scorers with a likelihood of an entire masked **TRAIN observation-incidence pattern**, instead of a product of its member-mean marginal likelihoods. The proposed difference is where the member mixture occurs relative to the product over residual-neighbor slots. It can allocate a branch to a coherent completion pattern; a marginal objective cannot identify dependence from marginal reconstruction supervision alone.

This is a transplant of an established mixture-of-Bernoulli construction, with **GRAN explicitly attributed**. The possible contribution is its use as structural supervision for the tied NCNC scorer/decoder bank under restricted sharing, with detached target-completion routing and unchanged mean-logit serving. The inspected sources do not establish an exact duplicate of this complete composition. They also do not certify that it is unpublished or predicts better.

The hypothesis concerns a **denoising proxy for observed topology**, not calibrated uncertainty about true missing links or a graph posterior. Source nonedges are unknown latent links. Their known absence of observation in TRAIN must be named separately from a verified nonlink. A positive pilot result would first support this particular regularizer against an equally supervised marginal control. Advantage over a competent covariance-aware single, independent NCNC, or stronger link predictors would remain unestablished.

Earlier rejection of scalar uncertainty-weighted repulsion remains correct; broad graph-uncertainty ancestry alone is not a reason to reject the smaller, specified operation here. There is no execution release, source adaptation, grid, canonical-memory adoption or alteration of the frozen running family.

## 1. What the running ensemble actually learns

The local root releases bind driver manifest `a59c669356e1a1437ce90107a7c76eb8f0b5579dc48d67b4378a7e237e956b7e` and qualified prototype v2. The local driver-manifest hash agrees with both releases. This is a preserved-source/root-binding assessment, not live verification of remote bytes or progress. Exact paths, hashes and line scopes are in `SOURCE_SCOPES.json`.

The current family has N64, independent I4, F4-private, its F4-pooled twin, and capacity-reference N70: five base seeds, 35 unique 100-epoch fits and 25 served arm/seed cells. The current primary scientific contrast is preserving each member's completion/feature/decoder association versus giving each receiving branch the mean of four already-clamped completion weights. The F4 twins have the same initialization, candidate work, stochastic schedule and loss. They are separately selected by complete official VALID Hits@50.

F4 has one learned encoder, shared dense decoder weights/biases, private input/output factors `r,s`, private LayerNorm affine parameters and private beta. A factor map is `diag(s_m) W diag(r_m)` with shared bias. Each branch transforms the node features, recursively scores missing counterpart links at depth zero, applies the native clamp, sums observed common-neighbor features plus the two weighted unilateral-neighbor feature sums, and nonlinearly decodes the result. Recursive scores are under `no_grad`; the downstream feature/decoder path remains differentiable. Training is the native positive and negative log-sigmoid loss averaged across all query/member entries. There is no ambiguity estimator, posterior law, structural reconstruction objective or diversity regularizer.

For a target pair `e=(i,j)`, candidates are neighbors observed on exactly one side in the record-masked graph. Nodes with both endpoint links unobserved are outside this support. Removing selected positive **records** before coalescing can leave an edge present through duplicate records. Encoder edge dropout is additional; completion uses the original masked graph. Native source pin is `GraphPKU/NeuralCommonNeighbor@11d597013750da17ce7468e344bec756a7af39a4`.

Serving is equal mean **raw logits**, with strict OGB shared-negative-pool Hits@50. It is not an arithmetic probability mixture. Independent NCNC already preserves each model's completion/decoder association. The running family tests preservation under sharing, not a new completion principle or universally larger ensemble function class.

## 2. What structural ambiguity would have to mean

Let `C_e` be a vector of latent counterpart-link incidences over the residual support. Meaningful structural ambiguity concerns a conditional law of the **whole vector**: which neighbors appear together, which alternatives compete, and how those configurations affect a nonlinear decoder. Marginal uncertainty, number of residual neighbors, and output disagreement do not determine that law.

The native weight is

    w = alpha * sigmoid(scale*(s-offset) + log(pt)).

Here alpha=1.05, scale=2.5, offset=6 and pt=.1. Thus native weights can exceed one. Applying Bernoulli entropy directly to `w` is invalid; normalizing `q=w/alpha` supplies a bounded logistic parameter, but supplies no calibration or Bayesian interpretation. Any new Bernoulli likelihood must explicitly use `q`, not the native weight as a probability.

### A competent alternative's precise information limitation

For two residual bits, exclusive alternatives `{10,01}` and co-occurring alternatives `{00,11}`, each equally likely, have the same marginals `(1/2,1/2)`. A marginal entropy gate and a factorized completion law cannot tell them apart. A nonlinear response sensitive to `C1*C2` has expected raw response zero versus one half; the independent law and response to the mean bits both give one quarter. These are analytical witnesses, not measured NCNC errors or a ranking advantage.

The stronger covariance-aware control deserves a stronger witness. Uniform **even parity** on `{000,011,101,110}` and uniform **odd parity** on `{001,010,100,111}` have identical marginals, identical pairwise joint moments, zero off-diagonal covariance, and identical entropy. Nevertheless, `E[C1*C2*C3]` is zero versus one quarter. Four near-deterministic mixture components can represent either law. A predictor restricted to the same context plus first and second completion moments cannot distinguish them.

That is not a failure theorem for a capable single receiving richer graph context or a full mode bank. Such a single can encode higher-order dependence or implement the complete grouped bank exactly. The witnesses explain a possible benefit of retaining learned modes instead of only moments. They do not prove that these ambiguous configurations occur usefully on collab, that the current decoder expresses the witness exactly, or that a covariance-aware learned single will lose.

For comparison, set binary target residuals `r_m(e)=sigmoid(z_m(e))-y_e` and a nonnegative, detached structural-ambiguity weight A_e. The pair-overlap term satisfies exactly

    O_A = 2/[M(M-1)] * sum_(m<k,e) A_e*r_m(e)*r_k(e)
        = [M*||mean_m r_m||_A^2 - mean_m ||r_m||_A^2]/(M-1).

This is weighted quadratic NCL, with graph transforms giving the previously derived PSD kernel form. Native BCE plus this auxiliary is not exactly a BCE interpolation, and its probability-mean identity is not the current raw-logit serving rule. Learning A supplies extra gating gradients but does not identify the missing-neighbor joint law. This candidate is rejected as a distinct operation; it is not the proposed J/F pair.

## 3. Concrete proposed operation

Use the same residual supports, branch features, recursive score calls, clamp and private completion routing as F4-private. Add one reconstruction auxiliary. The two future arms differ only in its joint-versus-factorial formula. No gate, new neural map, topology sampler, entropy reward or repulsion is introduced.

### Labels and their scope

Define **Z**, not C: `Z_er=1` iff the counterpart edge for residual slot r was observed in the complete supplied TRAIN snapshot before the native minibatch record mask. The auxiliary estimates the law of these observation-incidence bits under that known masking process.

| Entry | Known fact and proposed treatment |
| --- | --- |
| Observed counterpart link made absent by the minibatch record mask | Known synthetic removal; Z=1 in the auxiliary. The edge must actually disappear after all surviving duplicate records are coalesced. |
| Retained observed common neighbor | Both endpoint links remain observed. It stays in the native common-neighbor sum with native weight one; it is outside the residual reconstruction vector. |
| Counterpart nonedge of complete TRAIN | Z=0 means **not observed in this snapshot**. Whether it is a true missing link is unknown. The proposal explicitly reconstructs the observation bit zero; it never asserts latent C=0. This adds a dense source-absence proxy beyond native sampled negative supervision. Both arms receive exactly this proxy. |
| Native sampled negative target pair | Retains its conventional native target y=0 in the main link loss. This is a sampled training-negative convention, not a verified nonlink. It does not license calling all latent missing links negative. |
| Candidate outside the residual support | Not reconstructed; no new candidate inflation. Both endpoint-unobserved neighbors remain absent from the method. |

The complete TRAIN graph is a **label-only teacher** for Z. It must not enter encoder inputs, candidate enumeration or target completion features. VALID/TEST edges never supply auxiliary labels. The masking law is the existing native record mask, not an added mask distribution or a silently repaired unique-edge mask.

This distinction is substantive. If all unknown source-zero bits were excluded and only synthetic positives were reconstructed, `q_mr=1` for every supervised slot would minimize both objectives. Positive-only reconstruction would not justify learned mode diversity. Conversely, treating source zeros as a calibrated latent absence would be scientifically unsupported. The proposed operation uses an explicit observed-snapshot denoising proxy and must be evaluated as such.

### Joint and equally supervised factorial objectives

Let R_e be all residual slots, including both sides, and let

    t_mer = 2.5*(s_mer-6) + log(.1),
    q_mer = sigmoid(t_mer),
    a_mer = Z_er*log(q_mer) + (1-Z_er)*log(1-q_mer),
    S_me = sum_r a_mer.

The joint auxiliary is

    L_J(e) = -log[(1/4) sum_m exp(S_me)] / |R_e|.

The strong marginal/factorial control is

    L_F(e) = -sum_r log[(1/4) sum_m exp(a_mer)] / |R_e|.

Equivalently F is the likelihood of independent Z bits with member-mean probabilities. It is **not** four individually supervised BCE copies, which would force every branch to fit every bit separately and be a weaker diversity comparator. Both use the same per-slot labels, candidate count, q bank and source masking. Empty supports contribute zero and remain in the all-query average. Use stable log-sigmoid/logsumexp forms, without a probability epsilon or a hard winner.

For each arm, add the positive-query mean plus negative-query mean reconstruction loss to the unchanged native main loss with the single fixed coefficient **lambda=1**. No coefficient search is proposed. This is an unmeasured representative setting; a null/adverse result closes that setting. Normalizing by support size prevents high degree alone from scaling the query loss, while the product inside J still defines a coherent pattern. Four components and equal prior weights are fixed. Unequal conditional mode frequencies may require duplicate components; no learned mixture gate is included.

### How responsibilities change learning

For J, the training-pattern responsibility is `rho_me=softmax_m(S_me)`. Its score-logit derivative is

    d L_J(e) / d t_mer = rho_me*(q_mer-Z_er)/|R_e|.

One responsibility applies to all slots of the same query, so a branch must explain a coherent joint pattern. F instead assigns a component separately for each bit; it can fit correct marginals while leaving their dependence unidentified. There is no rewarded minimum separation: if the conditional source pattern is effectively factorial or the bank collapses, J can reduce exactly to F. At identical branch q values, the losses and score gradients coincide.

Auxiliary gradients must pass through the complete recursive depth-zero scorer, its second node-feature transform, the branch factors/norm/beta, shared dense maps and shared encoder. The shared parameters receive the sum of responsibility-weighted branch gradients. Native main target gradients still average all member losses. The main completion route must use **detached** q and `w=1.05*q`, preserving the native downstream feature gradient while preventing an unannounced target-loss path through the completion scorer. Enable scorer autograd for the auxiliary only; do not compute a detached auxiliary that has no effect. The forward/dropout schedule can remain identical across J/F; enabling autograd increases saved activations and backward work even if it adds no score calls.

At inference, retain each branch's own soft native weights, branch features and nonlinear decoder, then serve mean raw logits. No responsibilities computed from Z are available or used at inference. No sampled hard graph or probability-mixture posterior is served. Each component is decoded at its conditional mean; within-component nonlinear uncertainty remains collapsed. The potential benefit is only preserving learned between-component alternatives in this finite bank.

## 4. Why it can help, and what existing methods already supply

If residual observation patterns have nonlinear predictive consequences and higher-order dependence learnable from the masked source context, J has a way to assign branch completion patterns coherently. F's bitwise likelihood does not identify that coherence. The tied decoder gives those patterns a direct route to the final score. This is a reason for a **prospective** improvement, not a guarantee: auxiliary negative density, mask-to-future shift, branch capacity, shared-encoder information loss, component collapse and the individual main BCE agreement pressure can all remove or reverse the benefit.

In particular, native target BCE still supervises **every** branch on the same target label; rho weights only reconstruction gradients. The bank is not trained as correctly labeled predictors conditional on different latent graph worlds. The proposed mechanism is structural representation regularization, not a derivation of ideal mode-conditional prediction. The analytical expectation witnesses establish an information possibility, not that this tied objective learns the witness response. Responsibility-weighting the target loss would introduce a separate MoE/multiple-choice intervention and is outside this one pair.

| Prior / alternative | Exact overlap and bounded distinction |
| --- | --- |
| NCNC and independent NCNC | Native completion, scorer/decoder tying and own pairing are prior. The proposed delta is the joint source-pattern auxiliary and its responsibility gradients, which are absent from the inspected native/running losses. |
| **GRAN, arXiv:1910.00760v1** | Direct prior for a parallel mixture of factorial Bernoullis capturing within-block edge dependence; the author loss sums edge BCE within component and then uses component logsumexp. That likelihood and its mode responsibilities are attributed. GRAN generates adjacency blocks with an attention GNN and learned component weights; the read source does not specify a tied shared-factor NCNC residual-completion bank, native detached completion, or mean-logit LP training/serving. This is broad and exact ingredient overlap, not proof of an exact full-operation duplicate. |
| BGCN, LDS and node copying | Joint/conditional graph-model uncertainty, nonlinear prediction averaging and correlated neighborhood alternatives are direct prior. They prohibit broad novelty claims, but do not by themselves duplicate the specified scorer-supervision/parameter-sharing operation. |
| CORE | Target-link-specific stochastic reduction, missing-edge inflation, shared encoder and nonlinear link prediction are prior. Its scoped native inference uses expected edge probabilities; it is not this four-component reconstruction likelihood. |
| IECNC | Learned probability/entropy transforms of completion features are prior. Its printed `-p log p` is not Bernoulli entropy; the method does not identify a joint latent incidence law in the retained scope. |
| Link-MoE | Pair-specific learned mixing of complete expert scores is prior. J's responsibility is supervised by a whole structural observation pattern during training and does not gate the final target prediction. A generic latent MoE can represent the mixture law; the proposed distinction is its local tied implementation and supervision, not a new mixture principle. |
| Weighted graph NCL / generic uncertainty gating | A detached ambiguity weight multiplying residual overlap is exactly diagonal-weighted NCL; adding graph transforms remains kernel NCL. A learned gate multiplying final member scores is ordinary conditional expert mixing. Neither recovers the joint pattern from marginal uncertainty. J/F instead compare two normalized structural likelihoods. |
| Function-space repulsion / FoRDE | Repulsion at completion vectors is function-space repulsion in another metric; input-gradient diversity has direct prior. J does not repel predictions or gradients. It fits source observation patterns, and may legitimately prefer no diversity when dependence is absent. |
| Competent covariance-aware single | Must receive the same usable graph context plus completion means and full second moments, with enough nonlinear capacity and paid work. It can succeed on actual data despite the moment witness. It remains required before claiming higher-order bank utility over strong singles. A full-mode grouped single can reproduce the bank exactly. |

One new scoped primary method was read: GRAN §§2.1–2.3 and the pinned author likelihood routine. The printed Eq. (7) omits explicit bit/complement powers; the source loss resolves the component likelihood construction. Pinned code allows weighted BCE and averages gate logits where the paper prints a sum. The proposed unweighted, uniform-component loss is explicitly a local adaptation, not a native GRAN reproduction. All other papers are retained scoped conclusions from index_v36 and their bound source packets. No new broad search or inaccessible route was repeated.

## 5. One representative paired pilot, if separately prepared

**Primary pair:** fresh J versus F, shared-factor NCNC width64, M=4, base seed **0**, 100 complete native epochs, lambda=1, all residual candidates and the same fixed current source recipe. Start from identical fresh model/empty Adam/RNG states and dedicated sign initialization; use matching native negative/permutation/dropout schedules. These are two new prospective fits, not running-family replacements, warm starts or outcome-selected donors. Freeze the observed-Z label contract and all formulas before any predictive outcome. No grid, seed expansion or automatic continuation is proposed.

Select each arm on its served complete official VALID Hits@50 over epochs 1..100, first ties, TRAIN-only graph and equal mean raw logits. Report both values, selected epochs and the single paired difference. One pair is a mechanism screen, not optimizer-population inference. Existing TEST remains unreleased; this packet proposes no TEST access and no changes to the current family. A later confirmation needs its own frozen design.

**Mechanism diagnostics at selected states:** record marginal and joint source-pattern log scores under prebound fresh TRAIN mask events, q marginal competence, between-component pattern spread, responsibility concentration, member competence and served ranking-error overlap. A fresh mask tests transfer across mask events, not unseen graph truths or independent edge labels; do not call it posterior validation. Charge every diagnostic forward. No metric other than served Hits@50 selects the checkpoint. Diagnostic strata such as no observed common neighbors or higher source-pattern ambiguity are descriptive and cannot replace the complete endpoint.

Apply the previously defined fixed-bank own/crossed/pooled score diagnostics when useful, keeping the recipient's own transformed features and exact clamp. They separate branch association from nonlinear averaging; they are not new learning arms. Final score averaging and the diagnostic probability or logit operation must be named explicitly.

The first pair isolates joint-versus-marginal supervision with matched observation targets and scorer gradients. Before attributing a positive result to useful higher-order completion over strong singles, prepare a **separately trained competent covariance-aware single** given the same graph context and full completion second moments, plus the grouped/full-mode equivalent and independent NCNC capability references. Those controls are not qualified here and are not silently replaced by a weak terminal affine head or a second-order Taylor approximation. This pilot alone cannot establish that advantage.

### Falsifiers and interpretation

1. **Null/adverse served Hits@50 difference:** no quality support for this representative J-versus-F configuration, even if J fits the joint source pattern better.
2. **Only improved marginal reconstruction:** evidence is compatible with stronger ordinary auxiliary link supervision/calibration, not specifically learned dependence. A dependence interpretation requires improved joint fit beyond matched marginal competence and useful target quality.
3. **Collapsed components / no coherent pattern benefit:** if branch patterns collapse, J=F in the equality limit; lowered loss or gradient magnitude alone does not support structural diversity.
4. **Proxy failure:** poorer native target/member competence, bias toward source-sparse completions or weak mask-to-future transfer defeats the proposed mechanism. Known source absence never becomes proof of latent absence.
5. **Covariance-aware single matches/exceeds the useful effect:** no evidence that retaining higher-order completion modes is needed at the tested capacity/work. A grouped single's exact equivalence already rules out a universal ensemble function-class claim.
6. **Own/crossed/pooled diagnostics erase the supposed association:** credit generic nonlinear representation averaging or auxiliary supervision as appropriate; do not rename them a new uncertainty mechanism.

### Work and remaining preparation

Every arm pays full native graph masks, shared encoder, four full-node transformations, all left/right recursive completion scores, all sparse sums, native target backward/Adam, and complete selection evaluations. Additionally pay full-TRAIN teacher graph/index construction, every residual-label membership lookup, all bit log-likelihoods and query/member reductions, saved recursive scorer activations and auxiliary backward through both transforms/encoder. J adds pattern component reduction; F performs component reduction per bit. Both pay the same four scorer bank, not a pooled-scorer shortcut. Static label lookup can be cached by missing-link identity, but changing masked graph features and learned scorers cannot be cached across updates.

No timing, peak memory, candidate-volume or effective synthetic-removal frequency has been measured. Scorer-gradient admission, finite/logsumexp behavior, duplicate semantics, label-only teacher isolation, completion detachment and selected-state replay need a new, bounded source preparation and engineering qualification. Cost is a reporting/implementation requirement, not a scientific reason to reject the hypothesis. A full covariance control may be costly and must have its actual construction, capacity and complete work disclosed before use.

## 6. Custody and claim boundary

The packet binds actual local source scopes and reused conclusions, records one new primary identity and bounded author-source read, and separates first reading from extraction. GRAN HTML was mechanically extracted in full; only the declared method blocks plus incidental locators were read. Historical reports contain already-permitted qualitative/outcome summaries; no original historical or running outcome files were reopened or used to choose the hypothesis.

The outcome is **one conditional, unadopted incremental proposal**. There is no measured gain, exact Bayesian inference claim, new general mixture/diversity principle, global novelty certificate, strong-single advantage or manuscript conclusion. The current running private-versus-pooled family remains unchanged. Index_v36 and canonical ledgers are not edited.
