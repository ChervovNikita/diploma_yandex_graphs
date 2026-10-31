# Private structural corrections after shared graph propagation

2 October 2026. Literature and mathematical assessment only.

## Decision

**No new learner or pilot is promoted.** A shared graph operation followed by cheap private corrections is feasible in a restricted linear setting. It can either mix an existing common evidence bank or acquire additional sparse structural measurements. The first is already a multiscale/shared-backbone modeling operation; the second still incurs private propagation and lacks a supported new rule for learning useful, complementary errors.

The saved assessments already cover the substantive variants: private polynomial filters and learned edge views; structural expert gates; task-informed graph sampling/repair; graph-filtered supervised residual initialization; and exact destination-row member-contrast correction. Combining those operations with low-cost private factors supplies no new learner distinction. This is a no-go for the proposed mechanism family on the evidence available, not a claim that every possible structural correction is impossible or previously published.

The existing **node-selective exact contrast correction** retains its narrow allocation principle: a discarded-message-energy bound chooses destination rows under actual sparse cost. It approximates an already trained exact ensemble and does not learn new structural routes. Its prior proposal and later GEENI resolution are preserved; it is not relabeled as a fresh learner here.

## Evidence consulted first

`literature_memory/index_v15` was the latest available index. Saved conclusions were consulted before considering acquisition. The following are the relied-upon dispositions; exact paths, hashes and report passages are saved in this packet.

| Saved assessment | Consequence for this question |
|---|---|
| Adaptive sharing novelty | Shared trunks, private low-rank capacity and gradient-driven unsharing are established. Mean member loss and deployed pooled loss can favor opposite sharing decisions. |
| Recurrent shared-message closest prior | The common neighbor channel preserves the member mean and deletes transmitted member contrasts. Private roots and nonlinear states survive. Fixed stream mixing and shared propagation are established primitives. |
| Complementary graph/filter evidence gap | FSGNN/SIGN, ACM, HLCL, ASPECT, HiLoMix and H2SGNN cover shared banks, private/learned filters, nodewise channel allocation and complementary edge views. Structural diversity does not establish useful error complementarity. |
| Shared propagation/receptive fields | Member-specific polynomial filters over a shared bank are a falsifiable operating-point hypothesis, with methodological novelty unestablished. This question is not new. |
| Propagation cost and coordinated sampling | Mean-plus-residual transport and coordinated masks have local linear identities. Nonlinear prediction/gradient guarantees and automatic cost savings do not follow. |
| Graph-band route initialization; round21 curvature | Supervised graph-residual VJPs and private-factor expansion already have a precise saved hypothesis. The curvature extension was not promoted. These are not new structural edge learners. |
| GraphMoRE; Chimaera | Topology/geometry-specialized experts and heterogeneous frozen experts with adaptive fusion are direct precedents. Their topology-specific information and total expert costs remain. |
| Link-MoE | Pair-specific structural/feature gates, independently trained experts and second-stage score fusion are direct prior. Its collab gate uses additional validation supervision. |
| GEENI; node-selective correction | Error/consensus-guided outgoing-message suppression and confidence-based edge pruning are established. Exact fixed-operator destination contrast correction with its energy/cost certificate is a narrower saved difference, not a structural-route learner. |
| Conditional graph repair audit | Valid dependent edit laws, corruption reconstruction and task-informed graph sampling have primary precedents. Shared solves are a computational enabler rather than an established new learning principle. |
| Efficient ensemble readiness | A common feature cache is available to untied predictors as well. Strong packed untied members and the valid cached-row MIMO construction are controls, not benefits exclusive to factorized members. |

No primary method was newly retrieved or reread. Increments: **0 full papers, 0 new scoped primary methods, 0 retained-primary revisits**. Reading saved conclusions and provenance is recorded separately. Remaining MORGAN/FAGEL full-paper access limits are preserved and cannot support a positive novelty inference.

## 1. Separate evidence reuse from evidence acquisition

### Private channel maps and common filter banks

For fixed linear node operator `P`, common state `H` and a global channel map `B_m`,

`P(H B_m) = (P H) B_m`.

One graph application can therefore feed private channel maps. Those maps transform the common measurements; they do not create a new node operator. Channels may already carry different frequencies, so this does not deny useful dataset-specific responses.

For a fixed bank `S_k = T_k(P)H`, a private response `R_m = sum_k S_k B_mk` is exact mixing of shared measurements. Constructing a degree-K polynomial bank still requires K graph applications, plus storage and backward work when H evolves. “Computed once for all members” is different from “one graph application.” These are the saved SIGN/FSGNN/GPR/BernNet-style operations, with stronger learned-channel and edge-view precedents in the filter audit.

If each final logit is linear in the bank, averaging raw logits collapses to the mean coefficient map. Private nonlinear branches can enlarge the function class beyond one linear head; an ordinary multistream predictor with the same forward capacity and pooled loss remains a necessary control. Nonlinearity does not by itself supply a new learning rule.

### Private sparse operator corrections

For **common input H**, let `P_m = P_0 + S_m`. Then

`P_m H = P_0 H + S_m H`.

This is exact and permits one common full propagation plus private sparse updates. It can add information absent from the receiver's common aggregate. For example, a receiver with two equally weighted neighbor values `(1,-1)` and one with `(0,0)` both receive mean zero. A correction that reads the first neighbor separates them. The example concerns a readout limited to the receiver's common aggregate; it is not an information impossibility for an unrestricted model allowed to reread every neighbor feature.

The correction is a new measurement and has a price: at width D, at least its actual nonzero-channel work, endpoint access, aggregation and reverse pass must be counted. Selecting/scoring edges also costs work. Disjoint corrections may increase the union of touched edges and feature reads. A low-rank representation `S_m = U_m V_m^T` gives `U_m(V_m^T H)` but constrains the operator family and charges construction/storage of its factors; low rank supplies no structural-learning novelty.

`S_m` must be the difference of the **actual normalized operators**. Changing degrees in a normalized mean/GCN view can change retained-edge weights beyond the edited edges. If degrees and original denominators are held fixed, that is an explicitly different weighted-message rule. HiLoMix already learns feature-based complementary edge weights and distinct normalized views; graph-learning/repair priors already learn input/task-conditioned edge alternatives.

### Evolving private states require the missing terms

With `H_m = Hbar + Delta_m`, the exact private message is

`(P_0 + S_m) H_m = P_0 Hbar + P_0 Delta_m + S_m Hbar + S_m Delta_m`.

Computing only `P_0 Hbar + S_m Hbar` omits original-topology transport of member contrast as well as private-operator transport of that contrast. A correction supported only on edited destinations cannot recover `P_0 Delta_m` on unedited destinations in general. Adding private root or nonlinear heads does not reconstruct this term by an operator identity.

Thus sparse edge corrections can be exact after a common state, or approximate after private recurrent states. They do not make arbitrary recurrent private graph trajectories exact at the cost of one full-width common propagation. The existing sampled-contrast and destination-row correction packets already address that approximation explicitly, with limited certificates and full work accounting.

## 2. Structural separation does not certify different useful errors

Even if `sum_m S_m = 0`, only the mean **linear message on common H** is preserved. For member-dependent nonlinear updates `f_m`, in general

`mean_m f_m(P_0 H + S_m H) != mean_m f_m(P_0 H)`.

Orthogonal filters, different edge supports, zero-sum messages and negative sampling correlation do not imply negative prediction-error covariance. A predictor may ignore the correction, or a common wrong logit can dominate all route increments. For binary logits `z_m = b + epsilon_m`, all members predict the same wrong class whenever `b` has the wrong sign and `abs(b) > max_m abs(epsilon_m)`, even with distinct zero-sum increments. This is a counterexample to a structural guarantee, not a prediction of a trained model's behavior.

A proper pooled predictive loss is therefore the decision target. Disagreement, filter distance and contrast energy are diagnostics. Mean member loss, individual strength, simultaneous errors and pooled gain must be distinguished. The saved adaptive-sharing counterexample already shows why an own-member local improvement cannot replace the deployed pooled-loss gate.

Task-residual targeting does not supply the missing distinction by itself. Graph-residual correction, graph-dependent gradient ensembles, supervised factor seeding and functional cotangent pullback already have saved precedents. A routine covariance/repulsion term or private nonlinear head is not promoted as a new learner.

## 3. Closest method boundaries

| Proposed change | Established overlap | What remains unsupported |
|---|---|---|
| Route-private coefficients over a common bank | SIGN/FSGNN/GPR/BernNet; learned allocation in ACM/ASPECT/H2SGNN | A distinct learner beyond shared multiscale features plus multiple predictors. |
| Learned complementary sparse edge views | HiLoMix; GEENI suppression/pruning; conditional graph learning/repair methods | A new estimator/objective or a demonstrated useful-error advantage of the proposed constrained family. |
| Topology-dependent expert assignment/fusion | GraphMoRE, Chimaera, Link-MoE; saved spectral-MoE source evidence | A routing principle distinguished from structural gates and ordinary mixture learning. Homogeneous shared parameters alone do not provide it. |
| Graph-residual specialization of private factors | Saved graph-band initializer; BGNN/C&S/LoRA-GA/VJP ancestry; rejected curvature extension | A new edge/filter learner and held-out utility beyond generic or topology-null directions. |
| Exact full contrast on selected destinations | Saved node-selective proposal and GEENI comparison | This is a bounded transport approximation/allocation hypothesis, not newly acquired route-specific structure or a new supervised learner. |
| Shared cache/solve and private predictors | BUDDY cache boundary, SIGN and saved resolvent assessments; packed untied predictors | A statistical method difference attributable to shared execution. Equal deterministic reuse belongs in every valid control. |

This table does not assert an exact complete-assembly predecessor. The no-go follows because the specified substantive operations are established and no justified learner delta is supplied. A missing exact assembly is insufficient positive evidence.

## 4. One precise falsifiable question survives as a comparison

**With a fixed common backbone and a disclosed extra structural-measurement budget, do separately supervised route corrections improve deployed pooled prediction beyond the same evidence and forward capacity trained as a single pooled predictor?**

This is an established-method comparison question, not an admitted pilot or new method. A future reopening would need an explicit edge/filter selection law, parameterization, objective, gradient target, inference reducer and budget before fitting. Its essential controls would include:

1. The identical backbone with tied corrections, separating additional evidence/capacity from route separation.
2. The same private forward tensors and correction union trained by pooled loss instead of mean member loss, separating supervision from structural measurements.
3. A competent single multiscale/edge-aware predictor with the same permitted information and complete forward capacity, plus ordinary independent predictors using identical deterministic cache reuse.
4. A disclosed cost-matched generic/random allocation and topology-null allocation when a graph-specific learned allocation is claimed. Match actual executed sparse work, not only row or sparse-call counts.
5. The identical learned function executed by shared updates versus separately computed operators. This isolates execution; any predictive difference is an equivalence error.

The route explanation fails if its pooled benefit vanishes against tied or pooled-loss controls, individual quality is merely traded for disagreement without a useful pooled gain, graph alignment has no advantage over matched null allocation, or private work removes the practical benefit. A learner distinction would require more than passing such controls: it must specify the substantive rule absent from the cited priors. No such rule is established here.

No thresholds, tasks, optimizer fits or new search grid are added. Current studies remain the root's responsibility; their outcomes were not inspected or used.

## 5. Current backbone limits and disposition

The retained PolyFormer path consumes fixed polynomial-token rows. Private mixing over those rows is the existing-bank case. An arbitrary private edge operator changes the preprocessing evidence; it is not a free post-cache coefficient change, and coherent private token construction must be charged.

The retained Polynormer local path uses GAT attention and the global path has node-axis numerator/denominator reductions. Member-specific states or edits can change attention scores, softmax normalization and later global summaries. Fixed-linear `P_0 + S_m` algebra does not establish native attention parity or a cheap update. Sharing an unchanged attention tensor would be a different architecture unless exact equality is qualified.

**Stop this research branch at the source-level no-go.** Preserve sparse-correction feasibility and the exact common/private-state expansion as useful boundaries. Keep the existing contrast-allocation proposal separate and retain strong efficient ensemble comparisons. Reopen only for a concrete estimator, target, update or identification result that differs from the saved multiscale, MoE, graph-learning and residual-specialization operations.

## Scope and provenance

All new artifacts are in this packet. Work comprised saved report/index reading, mathematical reasoning, administrative JSON generation and hashing. There was no implementation, numerical scientific execution, model/data/label/checkpoint access, SSH, GPU activity, new primary acquisition, manuscript change, study amendment or subagent. One saved assessment includes prior engineering scalars; those were incidentally visible and were not used for this decision or a resource estimate.

`REUSED_CONCLUSIONS.json`, `READ_SCOPES.json`, `INSPECTED_REPORT_PASSAGES.json` and `REUSE_BINDINGS.json` preserve exact reused evidence. `RETRIEVAL_DISPOSITION.json` records why no acquisition was needed. `CONCLUSIONS.json` records the bounded no-go and algebraic conditions. `ARTIFACT_MANIFEST.json` and `SEAL.json` bind the completed packet. No predictive or measured runtime result is supplied.
