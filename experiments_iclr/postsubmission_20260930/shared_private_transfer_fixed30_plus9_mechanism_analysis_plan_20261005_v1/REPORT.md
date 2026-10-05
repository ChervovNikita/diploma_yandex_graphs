# Mechanism and analysis plan for the fixed 30+9 cohort

5 October 2026. Prospective source-only plan. No fit outcome, checkpoint or prediction payload was accessed, no server was contacted, and no numerical/model program was run. The original thirty cells, nine fixed-row0 companions, selector and promotion policy remain as frozen. The current withdrawal of GPU77 access leaves its six companion fits unreleased and unobserved; saved provider manifests are provenance, not permission or terminal evidence. This plan authorizes no computation or launch.

## 1. What the intervention actually changes

Let `theta` denote the shared encoder and dense head bases/biases (1,409,026 coordinates), and `phi_m` one private factor, normalization and beta row (6,402 coordinates). F4 has four private rows and separate Adam histories; F1 has one. F1 constructs the original four-row bank with the same seed/factor seed and retains fixed row0. Its initial function matches F4 route0, not the pooled F4 function. Both use four explicit inner query/dropout streams per inner pass, the same support construction, and nine functional forwards per episode.

In the bound `transfer_step.py`, with each stream's separately normalized loss `L_I,j`,

\[
g_m=\nabla_{\phi_m}L_{I,m}(\theta,\phi_m),\qquad
\phi'_m=\phi_m+A_{s_m}(g_m)\quad\text{(F4)},
\]

\[
\bar g=\tfrac14\sum_{j=1}^4\nabla_\phi L_{I,j}(\theta,\phi),\qquad
\phi'=\phi+A_s(\bar g)\quad\text{(F1)}.
\]

The private responses are virtual. Native shared Adam then commits one shared update; private gradients are recomputed at the updated shared parameters, starting from the unchanged old private parameters and moments, and one private update commits. The virtual moments are discarded. Dropout masks replay within virtual/recomputed work; each stream advances once on recomputation. Outer labels commit no private gradient. Serving requires no adaptation. The ordinary controls instead commit three native joint Adam updates (inner, outer, repeated inner), so their comparison concerns the declared complete learning rules.

The actual F4–F1 intervention changes private row count, initial private functions, separate histories, pooled serving and the outer objective. It is a strong role/exposure match, not a pure isolated parameter-count intervention. The richer capable single remains a required quality baseline.

## 2. Why separate responses need not collapse to a mean response

At frozen previous state `s=(m0,v0,t)`, the bound functional Adam uses

\[
A_s(g)=-\eta\frac{\beta_1m_0+(1-\beta_1)g}
 {(1-\beta_1^{t+1})\left[\sqrt{(\beta_2v_0+(1-\beta_2)g^2)/(1-\beta_2^{t+1})}+\epsilon\right]},
\]

coordinatewise, with `eta=.001`, `beta1=.9`, `beta2=.999`, `epsilon=1e-8`, no decay, AMSGrad or clipping. At the first zero-history step this reduces to `-eta*g/(abs(g)+epsilon)`. Thus, generally,

\[
\tfrac14\sum_j A_s(g_j)\ne A_s(\tfrac14\sum_j g_j).
\]

Different real F4 histories add another distinction. Prediction nonlinearity also gives, generally, `mean_j f(theta,phi+A_s(g_j)) != f(theta,phi+A_s(mean_j g_j))`. These synchronized-state comparisons explain an available mechanism; averaging the actual trained F4 rows is not an exact F1 counterfactual.

Four private states can retain opposing corrections which a mean gradient suppresses, and can retain different preconditioning and momentum. Those corrections may help other queries, remain useful only for their own stream, or amplify noise and harmful transfer. Noncommutativity supplies no favorable sign. This is a general optimizer/ensemble property, not a graph theorem.

For the virtual outer objective `F_O`, the current live derivative decomposes exactly as

\[
g_{\rm live}=d+c,\quad d=\partial_\theta F_O(\theta,\Phi'),\quad
c=\sum_m(H_{\phi_m,\theta}L_{I,m})^T
       (D_gA_{s_m})^T\nabla_{\phi'_m}F_O.
\]

The detached path has derivative `d` **at the same virtual adapted state**. F1 has the analogous term with the Adam Jacobian at `bar g` and the mean mixed derivative. Averaging response Jacobians/mixed derivatives generally does not commute with evaluating them after averaging gradients. Previous optimizer states are constants in this differentiation: it is not differentiation through the whole training history. The exact stable zero-history root has a finite first derivative; no everywhere second-differentiability claim is needed.

At the first zero-history step, the coordinatewise sensitivity is exactly

\[
\frac{d A_0(g)}{dg}=-\eta\frac{\epsilon}{(|g|+\epsilon)^2},
\]

including derivative `-eta/epsilon` at zero. For `abs(g)` much larger than epsilon, sign-like normalization can make sensitivity small; near zero the conclusion differs. With positive history, write `r=sqrt(beta2*v0+(1-beta2)*g^2)`, `a=sqrt(1-beta2^(t+1))`, `D=r/a+epsilon`, and `m1=beta1*m0+(1-beta1)*g`. For `r>0`, the fixed-history derivative is

`-eta/(1-beta1^(t+1))*[(1-beta1)/D - m1*(1-beta2)*g/(a*r*D^2)]`.

The source's stable zero-history case applies where this expression would divide by zero. Jacobian sensitivity alone does not determine the mixed term: mixed derivatives and outer private gradients also matter. A technically correct live derivative may nevertheless be practically negligible. No observed magnitude is inferred from this derivation or the qualification receipts.

For a small SGD inner step, the familiar first-order outer-loss change is `-alpha * outer_private_gradient dot inner_private_gradient`; differentiating that alignment yields the familiar mixed-derivative meta-learning term. Native Adam requires its actual response/Jacobian, and the shared update requires native shared Adam. A negative raw gradient dot alone therefore proves neither harmful Adam transfer nor a need for multiple members.

## 3. General identities versus graph-conditioned utility

The source outer loss is `0.5*BCE(mean raw logits)+0.5*mean member BCE`. By BCE convexity it equals aggregate BCE plus half the nonnegative Jensen gap. This pressures member competence; it does not reward dispersion. Classical squared-error pooling also depends on error cross-moments, `R_pool=M^-2*sum_mk E[e_m e_k]`, rather than dispersion alone. That identity does not become an Adam, GNN or MRR guarantee.

The graph-specific hypothesis is narrower: with this masked support and endpoint-conditioned episode sampler, different private corrections may cover different label-relevant response directions and improve the pooled predictor. The encoder, common-neighbor/pair head, query allocation and support jointly determine the mixed derivatives. Merely placing a standard meta-learning derivative on a graph does not establish the hypothesis. Endpoint-versus-matched-random contrasts test geometry dependence within this graph; one graph cannot establish graph-general specialization or separate every sampling/exposure explanation.

The saved prior reviews already locate shared/private adaptation in ANIL/BMAML, training-only transfer in MLDG/MetaReg and episodic domain generalization, and virtual→meta→recomputed persistent graph updates in SELAR. H-GRAM and Meta-iKG supply graph meta-learning precedents; BatchEnsemble/TabM supply private factor allocation precedents. The exact inspected recipe was not found in the saved scoped search, but that is no global novelty clearance. This plan derives standard identities and proposes measurements; it claims no new theorem or result.

Output dispersion may be confidently wrong, and embedding dispersion is sensitive to basis, scale and irrelevant directions. Negative gradient dots can reflect stochastic masks, loss normalization or optimizer geometry. Useful specialization needs member competence, complementary errors or transferable corrections, and a pooled quality gain over the fixed strong controls. No member, seed or graph subgroup is selected because it looks favorable.

## 4. Five fixed diagnostics after complete family release

All analysis waits for root release of the complete fixed 39-cell family. Preserve unavailable cells as unobserved and report incompleteness; do not shrink to completed/favorable cells. Retain the existing first maximum complete VALID MRR rounded to four decimals, every-five-cycle cadence, horizon and miss limit. Use every block and every fixed VALID query with its fixed 500 negatives. Report block values and their arithmetic mean; no new threshold, checkpoint choice, CI/p-value acceptance gate, subgroup rescue or tuning is introduced. These are exploratory, selection-conditioned development diagnostics.

### D1. Fixed quality and attribution contrasts

Apply every required quality and attribution contrast in the bound original `ROOT_RELEASE.json` unchanged. Add descriptive role-matched contrasts for each rule, `R_F4,end,r - R_F1,end,r`, and the member-count credit interaction

`(R_F4,end,live - R_F4,end,detached) - (R_F1,end,live - R_F1,end,detached)`.

Use the already fixed F4 geometry interaction

`(R_F4,end,live - R_F4,end,detached) - (R_F4,random,live - R_F4,random,detached)`.

Here `R` is the frozen selected complete VALID MRR; also report frozen Hits10 and selected cycles/horizon separately. F1 has no random-geometry cells, so no F1 geometry interaction or count×geometry×credit interaction is identifiable. Live/detached trained checkpoints have diverged histories; their cohort contrast is not the same-state derivative identity above. Host and seed remain confounded across only three blocks.

### D2. Competence, complementarity and pooling from saved VALID logits

Use only each arm's retained `selected_VALID_logits.pt`, verifying its checkpoint/input bindings. Reproduce the source tie-aware rank `1+0.5*(count(negative>=positive)+count(negative>positive))`. Report all member MRR/Hits10, pooled MRR/Hits10, pooled-minus-mean-member reciprocal-rank gain, and fractions of queries where pooled reciprocal rank is higher/equal/lower than the member mean. Report pairwise correlations of member reciprocal-rank errors `1-RR` (undefined correlations remain undefined), and all six pairwise Hit10 error-overlap rates for F4. Report the aggregate BCE, mean member BCE and Jensen gap from all saved positive/negative logits with the source's separate positive/negative normalization. These identify error complementarity and pooling behavior at selected states, not which historical private update caused them; correlations/Jensen gaps alone cannot establish specialization.

### D3. Geometry, exposure and paid training history

From all completed `CYCLE_DRAWS`, `EPISODE_HISTORY` and `CYCLE_HISTORY` records, report retained route exposure/repetition, outer coverage, masks, forward/update accounting, cycle counts and the virtual/recomputed loss summaries. Compare paired endpoint/random metadata across all available cycles; unequal horizons remain visible. These records can check whether the declared mechanism had the intended opportunity and exposure. Their scalar inner/outer losses do not supply a before/after outer-loss comparison or a historical mixed-credit trace.

### D4. Synchronized Adam/response comparison — later TRAIN probe only

If separately authorized after release, use **all six F1 live/detached selected checkpoints**, and the first paired episode of the first saved cycle (source cycle index0), with its saved support/query identities and the checkpoint's four frozen stream states. At each checkpoint compute its four same-state gradients, `mean_j A_s(g_j)` and `A_s(mean_j g_j)`. Report their norm difference and the difference between the mean raw logits of four separately adapted copies and the one mean-gradient-adapted copy on that episode's full TRAIN outer query set. Freeze the starting state, moments, masks and queries; commit nothing. This is a selected-state synchronized response diagnostic, not a reconstructed training episode or proof of the trained F4–F1 quality mechanism.

### D5. Transferable/harmful corrections and mixed credit — later TRAIN probe only

If separately authorized, use **all eighteen F4 endpoint/random and F1 endpoint live/detached cells**. Fix TRAIN episode indices `1`, `ceil(N/2)` and `N` from each cell's first saved complete cycle, where `N` is its episode count; use each arm's recorded geometry/support/query identities. At each episode evaluate two independently frozen states: reconstruction of the declared initialization with zero moments and initial stream states, and its selected checkpoint with trained moments/stream states. Initialization reconstruction is part of the later authorized probe. Never carry a probe update to the next episode. These span positions within one fixed episode cycle and two states; they do not reconstruct intermediate training states or claim a representative sample of the full trajectory.

Report the exact TRAIN outer BCE change caused by the virtual private response before any shared step. For F4 also report the full 4×4 matrix of each row's normalized stream loss change when that row receives its own private response, evaluating old/new row on every stream with identical masks; negative off-diagonal means local cross-stream transfer and positive means local harm. F1 reports all four stream changes after its mean-gradient response. Neither describes held-out generalization.

**Concrete learning-credit diagnostic:** at each fixed episode/state, compute `d` and the source mixed-derivative credit `c` above, retaining the fixed-history Adam Jacobian. Report `norm(d)`, `norm(c)`, `norm(c)/norm(d)` and `cos(d,c)`. For F4 also report the four `norm(c_m)` to expose cancellation in their sum. Clone the frozen shared Adam state to obtain `U(d+c)` and `U(d)`; report `norm(U(d+c)-U(d))/norm(U(d))`, `d dot [U(d+c)-U(d)]`, and exact TRAIN outer loss changes after both full source update rules, including recomputed private commitment in discarded copies. Zero-denominator ratios/cosines remain undefined, with zero norms reported; do not introduce a tuned stabilizer. Report every fixed episode/state/block value and their summaries without choosing a favorable probe.

A negative first-order displacement projection suggests locally helpful shared movement; the exact loss comparison can disagree because of curvature/recomputation. A consistently negligible mixed credit or optimizer displacement at these probes would constrain the proposed practically influential learned-transfer interpretation even if source qualification passes. A large term alone would not establish useful credit. No sign/norm threshold selects a model or rescues a failed quality gate, and these probes cannot establish what occurred along the missing historical trajectory.

## 5. Evidence limits and paper use

The source retains selected checkpoints/logits and scalar histories, not per-episode states, embeddings, gradients, mixed derivatives or pre/post-update logits. D1–D3 can use released saved artifacts; D4–D5 require later authorization for separate discarded TRAIN computation (including initial-state reconstruction) and sufficient bound checkpoint/source/data custody. If custody or required state is unavailable, mark the diagnostic unavailable. Do not add training logging, rerun fits, tune probes, or treat this plan as execution approval.

Only if the **existing prospective quality gate passes** may this become a supporting mechanism analysis for an honest exploratory paper. Even then, separate Adam responses are an established optimizer/meta-learning explanation; graph-conditioned usefulness must be supported by the fixed contrasts and measured corrections. If a required attribution condition fails, retain the root policy's narrower claim. If the quality gate fails or the family is incomplete, do not promote dispersion, conflict, a favorable probe or a single block into evidence for ensemble necessity or a methodological contribution.

`INPUT_BINDINGS.json` binds the exact sources, deployed manifests, plans, original promotion release and saved closest-prior reports. `MANIFEST.json`/`SEAL.json` seal this plan only. Original study and execution artifacts were not modified.
