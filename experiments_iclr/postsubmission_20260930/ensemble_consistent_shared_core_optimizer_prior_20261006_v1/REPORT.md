# Shared-core memberwise Adam: prior and mathematical assessment

Date: 6 October 2026. Scope: static assessment of the proposed shared-coordinate optimizer, preserving the ordinary joint supervised loss, four predictions, backbone and private factors. No model implementation, fits, data or tensor reads, held-out evaluation, server contacts, or manuscript changes were performed. The scalar check below evaluates elementary formulas, without a model or dataset.

## Decision

**Close the generic optimizer novelty route.** Adam with AdaTask already maintains separate first and second moments for each task on shared coordinates, bias-corrects each pair, and sums the individually preconditioned directions. The proposed mean is the same shared-block update on the same input vectors after setting AdaTask's learning rate to one fourth of the proposed rate. This is an exact optimizer-kernel identity under the assumptions below, rather than a vague similarity.

The coupled pooled-loss route VJPs are a legitimate way of constructing those vectors. The inspected AdaTask algorithm does not explicitly prescribe this pooled ensemble loss. Consequently this report does **not** claim that AdaTask literally published the full current graph recipe. However, ordinary chain-rule attribution through each output route, a tied shared parameter, and a different task-vector source do not establish a distinct adaptive optimizer mechanism. Applying this known rule to shared graph members would be an attributed empirical adaptation, with no predictive benefit established here.

The candidate is generally different from Adam on the actual joint gradient. It can produce a shared-block ascent direction even on its first step with zero moments. It therefore has no unconditional descent or member-competence guarantee for the unchanged joint loss. It also requires substantially more optimizer state.

## 1. Saved memory and bounded primary scope

Saved index_v72 was checked before selecting primary methods. Exact searches for MTAdam, AdaTask, local Adam, FedAdam and task-aware adaptation found no matching paper record. Index record 247 and its saved `persistent_optimizer_history_shared_graph_literature_20261005_v1/PAPER_CONCLUSIONS.json` were read: that prior is **Can We Learn Communication-Efficient Optimizers?**, arXiv:2312.02204v1. Its retained scope uses SGD workers, a learned global optimizer and optimizer accumulators; it is not a direct instance of the present per-member Adam rule. No saved primary passages were reopened. The saved recent shared/meta-ensemble synthesis was also checked for optimizer-history conclusions, and BatchEnsemble record 8 was reused for shared-weight/private-factor ancestry and complete-member inference costs.

Three new primary algorithm scopes were assessed, with zero full-paper reads and zero author-code reads:

| Source | Exact assessed scope | Relation |
|---|---|---|
| Yang et al., **AdaTask: A Task-aware Adaptive Learning Rate Approach to Multi-task Learning**, arXiv:2211.15055v2; metadata associates AAAI 2023, DOI 10.1609/aaai.v37i9.26275 | PDF pp. 4–5: proposed-method text, Eqs. 4–9, Algorithms 1–2, separate first moments explanation, efficient-state and implementation paragraphs. Algorithm 2's Adam branch, lines 21–29, is decisive. | Exact shared optimizer-kernel match after mean/sum rate normalization and matching conventions. |
| Malkiel and Wolf, **MTAdam: Automatic Balancing of Multiple Training Loss Terms**, EMNLP 2021, DOI 10.18653/v1/2021.emnlp-main.837, ACL publisher PDF | PDF p. 3 §4 method; p. 4 Algorithm 1, accompanying explanation and visible memory/run-time paragraph. | Separate moments are prior. Its layer norm balancing and shared maximum second-moment denominator differ from this proposal. |
| Reddi et al., **Adaptive Federated Optimization**, arXiv:2003.00295v5, primary identifies ICLR 2021 | PDF p. 3 FedAvg/model-delta algebra, Algorithm 1 FedOpt and explanation; p. 4 continuation and opening of §3 identifying server-adaptive/client-SGD specialization. | One local Adam step then ordinary model averaging has the same algebraic kernel under an explicitly matched client-state convention. FedAdam is different. |

Algorithm pages were rendered and visually inspected. AdaTask's Algorithm 2 denominator was additionally rendered at high resolution: epsilon is outside the square root, as in the candidate's formula. Its preliminary printed aggregate-gradient formulas contain apparent notation inconsistencies; the exact identity rests on Algorithm 2's coherent Adam recurrence and final sum, not on the aggregate expression in Eq. 6. Some empirical figures/text were incidentally visible on rendered or extracted pages; no reported numerical benefit was audited or adopted. Mechanical extraction of every PDF page for locating methods is not semantic full-paper reading. Source identities, hashes, read limits and retrieval failures are recorded in `PRIMARY_SCOPES.json` and `RETRIEVAL_LEDGER.json`.

The handoff's provisional identification of MTAdam as ICML was incorrect. The PMLR guessed page returned 404, its volume index contained no MTAdam title, and an additional unverified CVF guess returned 404. Exact-title Crossref metadata located the EMNLP DOI and the ACL primary PDF. These failures are preserved; they are neither method evidence nor negative novelty evidence.

## 2. Exact route decomposition of the unchanged objective

Let shared parameters be \(W\), private parameters of route \(m\) be \(\phi_m\), and the four output tensors be

\[
z_m=z_m(W,\phi_m),\qquad M=4.
\]

The notation accommodates logits or probabilities without changing which current pooling rule is used. Write the actual ordinary data objective as

\[
L(W,\phi)=\frac{a}{M}\sum_{m=1}^M\ell_m(z_m)
             +b\,\ell_{\rm pool}(C(z_1,\ldots,z_M)).
\]

Here \(a,b\), losses and pooling map \(C\) stay at their current values; this is notation, not a new loss or a claim about the native source's precise coefficients. A pure member objective or pure pooled objective is included by zeroing one coefficient. Any additional direct shared regularizer must be accounted for explicitly once rather than silently duplicated.

On the **same forward realization**, define output cotangents

\[
c_m=\frac{a}{M}\nabla_{z_m}\ell_m
 +b\,(D_m C)^\top\nabla_C\ell_{\rm pool},
\qquad
q_m=J_{W}z_m^\top c_m.
\]

Then

\[
\sum_m q_m=\nabla_W L.
\]

For a mean-output pool \(C=\bar z\), the pooled contribution is \(b\,J_Wz_m^\top\nabla_{\bar z}\ell_{\rm pool}/M\). This contribution must be kept. Computing just each member's individual loss gradient drops pooled credit; computing the full pooled shared gradient separately for every member duplicates it.

To compare a **mean** of route vectors with the current gradient without a factor-of-four ambiguity, set

\[
g_m=Mq_m,\qquad G=\frac1M\sum_mg_m=\nabla_W L.
\]

Private parameters retain their actual ordinary objective gradient
\(\nabla_{\phi_m}L=J_{\phi_m}z_m^\top c_m\), and their current optimizer/learning-rate convention. They must not receive the artificial factor \(M\) used to express the shared update as a mean.

This decomposition assumes that direct dependence on shared coordinates is represented in the routed output graph or separately declared regularizers. Shared upstream features can be traversed by each route VJP. If another term depends directly on \(W\), it needs an explicit decomposition. Re-forwarding different dropout masks or updating batch-normalization buffers between VJPs invalidates equality to the original realized joint gradient.

### Coupled pooling and task-objective limits

The vectors \(q_m\) are partial derivatives of a replicated-variable objective

\[
F(W_1,\ldots,W_M,\phi)
=\frac{a}{M}\sum_m\ell_m(z_m(W_m,\phi_m))
+b\ell_{\rm pool}(C(z_1,\ldots,z_M))
\]

evaluated on the diagonal \(W_1=\cdots=W_M=W\). Since \(L=F(W,\ldots,W,\phi)\), differentiation through tying gives their sum. In general \(F\) is coupled across copies. Thus these route VJPs need not be gradients of fixed, independent scalar tasks on a single \(W\). At each iteration they can be produced by varying one replica while freezing peers, but those surrogate objectives change with the peers.

This qualification limits claims of objective-level equivalence to independent-task MTL or federated client objectives. It does **not** alter an optimizer-kernel identity on a supplied vector sequence, and it provides no automatic unbiased-task, cross-fitting, federated-convergence or graph-generalization theorem.

## 3. Adam after averaging versus averaging Adam directions

Suppress coordinate indices. Ordinary Adam on the joint gradient has

\[
m_t^{J}=\beta_1m_{t-1}^{J}+(1-\beta_1)G_t,
\qquad
v_t^{J}=\beta_2v_{t-1}^{J}+(1-\beta_2)G_t^2,
\]

\[
d_t^{J}=\frac{\widehat m_t^{J}}{\sqrt{\widehat v_t^{J}}+\epsilon},
\qquad W_{t+1}=W_t-\eta_t d_t^{J}.
\]

The proposal maintains

\[
m_{m,t}=\beta_1m_{m,t-1}+(1-\beta_1)g_{m,t},
\quad
v_{m,t}=\beta_2v_{m,t-1}+(1-\beta_2)g_{m,t}^2,
\]

\[
d_t^{R}=\frac1M\sum_m
\frac{\widehat m_{m,t}}{\sqrt{\widehat v_{m,t}}+\epsilon},
\qquad W_{t+1}=W_t-\eta_t d_t^{R}.
\]

With identical beta values, clocks and compatible initial states, the linear first-moment recurrence gives

\[
m_t^J=\frac1M\sum_m m_{m,t}.
\]

The second moments do not commute. If \(\bar v_t=M^{-1}\sum_m v_{m,t}\) and initialization matches, then

\[
\bar v_t-v_t^J
=\beta_2(\bar v_{t-1}-v_{t-1}^J)
 +(1-\beta_2)\frac1M\sum_m(g_{m,t}-G_t)^2\ge0.
\]

The candidate therefore keeps route-gradient dispersion information absent from the joint second moment. But its update uses **each** denominator, not simply \(\bar v_t\), and division is nonlinear. This establishes an actual numerical difference from the current optimizer; it does not establish novelty, descent or predictive improvement.

If all route gradients and moment states are identical, the two updates coincide. If all route streams are fixed positive multiples of one common stream, states are initialized with matching linear/quadratic scaling and epsilon is zero, Adam's scale cancellation also gives equality. More general equality is contingent or accidental. A common frozen preconditioner lets averaging commute with the first moments, but independently accumulated second moments usually do not remain that common preconditioner.

### Exact AdaTask correspondence

AdaTask Algorithm 2 lines 21–29 takes \(g^k\), separately updates \(m^k,G^k\), bias-corrects them and applies

\[
W_{t+1}=W_t-\eta_{A,t}\sum_{k=1}^M
\frac{\widehat m^k_t}{\sqrt{\widehat G^k_t}+\epsilon_A}.
\]

Set \(k=m\), supply the **same** \(g_m=Mq_m\), choose its first/second decay factors to match the candidate (the paper uses \(\gamma_2\) for first moments and \(\gamma_1\) for second moments), use the same epsilon and clock, and take \(\eta_{A,t}=\eta_t/M\). The shared update is exactly \(d_t^R\). Keeping ordinary Adam in the private parameter group is an ordinary optimizer-group composition; the claim here is exact equivalence of the modified shared block, not identity to every private parameter update in an unmodified AdaTask implementation.

If instead AdaTask is supplied \(q_m\) while the candidate uses \(Mq_m\), the directions are equal at epsilon zero, with correspondingly scaled moments. At positive epsilon, exact matching requires scaling epsilon as well: the direction from \(Mq_m\) with \(\epsilon\) equals that from \(q_m\) with \(\epsilon/M\). Uniform gradient rescaling must therefore be stated rather than hidden behind Adam's approximate scale invariance.

### MTAdam correspondence and difference

MTAdam Algorithm 1 normalizes each loss gradient in layer \(\ell\) by \(n_{\ell,t}^1/n_{\ell,t}^i\), where the \(n\)'s are exponential averages of layer gradient norms anchored to the first loss. It then updates separate first/second moments of these **normalized** gradients. Its final denominator for every loss is the same coordinatewise
\(\sqrt{\max_i\widehat v^i_t}+\epsilon\).

The proposal has neither the anchored layer balancing nor that common maximum denominator. Therefore it is not MTAdam as published, although separate loss moments are explicit older ancestry. Removing both features would move it toward the AdaTask kernel; those removals are not a new mechanism.

### Conditional one-step local-Adam consensus identity

Start each replica/client at \(W_t\), retain its own matched moment history, and take **one** local step on the supplied vector \(g_{m,t}\):

\[
W^{\rm loc}_{m,t+1}=W_t-\eta_t
\frac{\widehat m_{m,t}}{\sqrt{\widehat v_{m,t}}+\epsilon}.
\]

Then ordinary model averaging yields

\[
\frac1M\sum_m W^{\rm loc}_{m,t+1}=W_t-\eta_t d_t^R.
\]

This is a specialization of FedOpt Algorithm 1 with all four clients, \(K=1\), ClientOpt=Adam, and ServerOpt=SGD at rate one on the negative averaged local delta. The inspected template does not fully specify persistent local Adam state across rounds; retaining rather than resetting it is an explicit assumption of this identity. More local steps, partial participation, client-state reset, unequal weights or an adaptive server generally change the rule. In particular, the paper's FedAdam specialization uses client SGD and **server** Adam after averaging deltas; it is not memberwise Adam followed by an ordinary mean. No independent-client objective or FedOpt convergence theorem is transferred to the coupled pooled ensemble loss.

## 4. First-step ascent counterexample under the same type of joint supervision

Take one shared scalar \(W\), fixed member slopes \((-2,-2,-2,8)\), logits \(z_m=a_mW\), target \(y=1\), and zero moments. Use a joint loss with half mean member BCE and half pooled BCE. The pool may be either mean logits passed to logistic BCE, or mean probabilities passed to probability BCE. At \(W=0\), both versions have

\[
g=(1,1,1,-4),\qquad G=-\tfrac14.
\]

The pooled route cotangents are included in these vectors. Bias correction makes the first-step member directions \(g_m/(|g_m|+\epsilon)\). At epsilon zero,

\[
d^R=\tfrac14(1+1+1-1)=\tfrac12,
\quad \Delta W=-\eta/2,
\quad G\Delta W=\eta/8>0.
\]

For positive epsilon,
\(d^R=\tfrac14[3/(1+\epsilon)-4/(4+\epsilon)]>0\) whenever \(0\le\epsilon<8\). Thus this is not an artifact of choosing epsilon zero. Ordinary first-step Adam on \(G\) moves in the opposite direction, toward descent.

The scalar check `evidence/ASCENT_COUNTEREXAMPLE.json` uses epsilon \(10^{-8}\), learning rate \(10^{-4}\). Both losses start at 0.69314718056. After the candidate step the mean-logit version is 0.69315968357 and the mean-probability version is 0.69315968357; ordinary first-step Adam reduces each to approximately 0.69312219259. These are elementary formula checks, not experimental outcomes.

This refutes a universal **shared-block** descent or joint-competence guarantee. It is not a claim that the full simultaneous private/shared update always ascends, that this example occurs in the current trained graph, or that ordinary Adam with nonzero history always descends. Private variables can be held fixed in this construction; their favorable simultaneous movement cannot restore a universal guarantee for the proposed shared rule alone.

## 5. Matching assumptions that matter in practice

- **Common state and timing:** route updates are computed from the same old shared weights and realized forward graph; moments and bias-correction steps advance once per route per ordinary update. Missing-gradient versus zero-gradient semantics must match. Applying an optimizer sequentially to already-mutated shared weights is a different algorithm.
- **Objective coefficients:** individual and pooled cotangents retain the existing loss coefficients. Normalizing away each vector's magnitude changes the direction even though the scalar loss definition is unchanged. Learned loss weights or other shared variables require their own ordinary gradients unless explicitly included in the proposal.
- **Initialization and carry:** equivalence is on matched moment histories. Copying a preexisting joint moment into all routes, resetting at different times, permuting route histories or using different beta values changes the trajectory. None is prescribed here.
- **Epsilon, dtype and clipping:** epsilon placement, scaling, precision, loss scaling, clipping before versus after route attribution, and clipping before versus after averaging must match. Separately clipping routes is not identical to clipping the joint gradient.
- **Weight decay:** decoupled AdamW decay should be applied once to the shared weights. With one local AdamW step from the same old \(W\), averaging produces that same single decay. A literal sequence of four shared AdamW optimizer steps generally repeats decay and is not the proposed mean. For coupled L2 regularization, a declared equal decomposition can add \(\lambda W\) to each scaled \(g_m\), giving the correct averaged raw regularizer gradient; nonlinear route preconditioning still changes its effect.
- **Private updates:** preserve the actual original gradient and update. The factor four in shared \(g_m\) notation does not authorize scaling private gradients or averaging private updates across unrelated parameter sets.

## 6. Costs and what is actually preserved

For \(p_W\) shared parameters and \(p_\phi\) total private parameters, the original moment bank has \(2p_W+2p_\phi\) scalars. The candidate has \(2Mp_W+2p_\phi\). Its additional moment storage is

\[
2(M-1)p_W\quad\text{scalars}=24p_W\quad\text{bytes for }M=4\text{ and FP32 moments}.
\]

That is six additional FP32 tensors on the shared block. It also increases uncompressed optimizer checkpoint storage by that amount. For one million shared parameters it is 24 MB, approximately 22.9 MiB, of added moment state alone.

Separate route-gradient buffers, retained forward graphs, batched VJP workspace, snapshots and optimizer overhead can add more. Sequentially processing VJPs can reduce simultaneous gradient buffers but requires a qualified execution strategy. No four-times-backward or equal-runtime claim is made: shared upstream graph reuse, batching and runtime details determine actual costs. Shared moment updates and their reduction require \(O(Mp_W)\) elementwise optimizer work instead of \(O(p_W)\).

The weight count, four forward predictions, architecture and serving computation can remain the same. Moment state is training state and need not be loaded for serving. This does not create an inference efficiency gain, conditional execution, or a cheaper optimization procedure. BatchEnsemble's saved vectorization ancestry does not establish equal training costs for an added per-route moment bank.

## 7. Consensus interpretation and follow-on disposition

The mean is the Euclidean consensus projection of independently preconditioned replica steps: it minimizes \(\sum_m\|\delta+\eta D_m\widehat m_m\|^2\), where \(D_m=\operatorname{diag}[(\sqrt{\widehat v_m}+\epsilon)^{-1}]\). This is an algebraic interpretation of the same known rule.

It differs from projecting in each route's adaptive metric. Minimizing

\[
\sum_m\tfrac12(\delta+\eta D_m\widehat m_m)^\top
D_m^{-1}(\delta+\eta D_m\widehat m_m)
\]

gives

\[
\delta=-\eta\left(\sum_mD_m^{-1}\right)^{-1}
\sum_m\widehat m_m.
\]

At a fresh-gradient step this is a positive-metric update of the summed raw gradient, whereas the proposed Euclidean average can oppose it. With momentum history it does not acquire a general descent guarantee. This elementary constrained quadratic calculation is recorded only to explain the geometry; it is not promoted as a new optimizer, implementation proposal or novelty claim. Adaptive metric methods and generic gradient-conflict or descent safeguards would need their own prior assessment, and adding one by name does not create a distinct graph mechanism.

**No genuinely distinct new follow-on ingredient is established by the assessed sources.** The recommendation is to retain this construction only as an explicitly attributed AdaTask-style shared-block training variant if there is a separate, prospective empirical reason to evaluate it. The current task and literature result do not justify implementation or a fit, and no new scientific cell, threshold or acceptance gate is proposed. Graph application, preserved four-output serving, persistent route moments, and pooled route cotangents alone should not be advertised as a new optimization principle.

## Evidence and limits

The strongest conclusion is exact shared-block kernel ancestry plus a concrete failure of universal descent. There is no exhaustive literature absence claim, publication-version equivalence certificate for the arXiv papers, native runtime qualification, graph-specific benefit, predictive superiority, convergence theorem, manuscript adoption or execution authorization. Canonical literature indices and sealed scientific packets were left unchanged.

The packet's `FORMULAS_AND_ASSUMPTIONS.json`, `CONCLUSIONS.json`, `PRIMARY_SCOPES.json`, retrieval ledger, counterexample and manifest make the assessment reviewable without changing the current study.
