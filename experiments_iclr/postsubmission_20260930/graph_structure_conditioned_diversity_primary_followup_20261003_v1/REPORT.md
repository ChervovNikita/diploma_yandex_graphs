# Graph structure-conditioned diversity: bounded primary follow-up

**No supported distinct gap was found, and zero pilots are promoted.** Fixed graph-frequency or neighbor-error weighting can define a useful empirical metric, but the central residual-correlation operation is exactly graph-kernel negative-correlation learning (NCL). The retained literature already supplies topology-aware routing, learned spectral-filter diversity, graph-error propagation and function-space repulsion. One new primary paper further establishes adaptive accuracy/diversity loss balancing. No evidence here justifies a new science queue or manuscript claim.

This is a bounded literature and symbolic assessment, not an exhaustive novelty exclusion or a result about target-graph quality. The sealed V9 audit packet and its review remain untouched. No experiment, model/data load, GPU or remote scientific execution, SSH, author contact, code execution from papers, index adoption or canonical ledger edit was performed. Public literature HTTP requests, PDF/text inspection and this packet's preparation were performed.

## Memory first and access limits

The root-adopted `literature_memory/index_v36/LITERATURE_INDEX.json`, retrieval ledger and saved scout were consulted before retrieval. REUSED_CONCLUSIONS.json retains exact indexed conclusion records/scopes for BernNet, C&S, FoRDE, graph-expert studies, GraphMoRE/MoE-NP, the unified diversity theory, TFE-GNN, BankGCN/Specformer, HGEN and recent structural-response priors. Saved GNCL, graph-diffused-error and adjacent-label-pair assessments were also consulted. These are **reused saved conclusions**, not new primary rereads. Their identifiers, source document hashes and limits remain explicit.

The ledger says GENN, *Graph ensemble neural network*, DOI [10.1016/j.inffus.2024.102461](https://doi.org/10.1016/j.inffus.2024.102461), remains primary-unresolved with repeat requests disabled. Its six-route failures and earlier blocked ScienceDirect landing were not retried. Its exact sharing, neighborhood/filter maps, losses, gradient paths and deployment pool remain unresolved; that cannot be treated as novelty evidence.

The requested Tsukiashi–Ono paper, *Robustifying Graph Laplacian Regularization Against Edge Weight Uncertainties: An Infimal Convolution Approach*, [ICASSP2026 DOI 10.1109/icassp55912.2026.11460998](https://doi.org/10.1109/icassp55912.2026.11460998), was checked as a possible robust-metric lead. Crossref verifies the title, two authors and registered publication date 2026-05-03. The IEEE landing gave HTTP202 with an empty body; the registered staging PDF URL gave HTTP405. Exact-title and author/Laplacian arXiv queries returned zero. A public search response was an unusable JavaScript page. Semantic Scholar and OpenAIRE mark the record closed and provide no manuscript locator. **No primary abstract, method or paper was obtained or read.** We cannot infer its infimal-convolution formula, uncertainty set, statistical guarantee, ensemble relevance or overlap from its title. Stop these unchanged routes; reopen for a supplied accessible manuscript or new specific locator. All 12 HTTP routes for this follow-up, including three successful DNCL retrieval steps and the initial broad Crossref discovery, are preserved individually.

## One new primary: dynamic NCL

Hiromu Takama and Tatsuhito Hasegawa, *Dynamic Negative Correlation Learning in Deep Ensemble Learning*, Proceedings of ICIAE2026, [DOI 10.12792/iciae2026.015](https://doi.org/10.12792/iciae2026.015). The exact publisher PDF was obtained from its observed viewFile link, SHA256 `879da965df987c60f3679dc75419d5e36e315469e030d6889fb7c2d728170a0f`.

Exact scope: all extracted text on PDF pages 1–8 was exposed and read, including §§2.1–3.3.2, method Eqs1–11, §§4.1–4.2.4, conclusion and references. Pixels on pages 2–4 were inspected for Figure1 and Eqs1–11. Other experimental figure pixels, author code, gradient/detachment implementation and numeric reproduction were not inspected. This is one first scoped primary-method read relative to the consulted index, **zero full-paper certifications**, and zero retained primary-method rereads. Downloading or extracting a page is not itself counted as reading it.

The method uses GNCL's loss interpolation

\[
L_t=\lambda_t L_{\mathrm{cat},t}+(1-\lambda_t)L_{\mathrm{dis},t},\qquad
L_{\mathrm{dis}}=\frac1M\sum_m L_m,
\]

with the basic coefficient set from losses observed one training step earlier,

\[
\lambda_t=\frac{L_{\mathrm{cat},t-1}}{L_{\mathrm{cat},t-1}+L_{\mathrm{dis},t-1}}.
\]

Moving-average and confidence variants are also described. Four models of the same architecture are evaluated on image classification; same architecture does not establish shared parameters. Nothing in the inspected method conditions the loss on graph frequencies, neighborhoods or member error fields. The useful baseline implication is narrow: changing a scalar accuracy/diversity weight over training is already published and cannot supply the proposed gap by itself.

The results discussion reports DNCL's lowest KL-diversity in its ResNet18/CIFAR10 comparison while reporting higher accuracy there. It also acknowledges lower accuracy than some baselines for EfficientNet-B0/VGGNet16. We adopt the caution that greater measured spread need not mean better predictions; no image scores, optimality or graph utility transfer is adopted.

The source has reproducibility ambiguities. Printed NCL Eq2 uses `k != j` and `h_k(x_k)`; Eq3 divides an M-member sum by the sample count N. These are visible in page2, not extraction repairs. Page4 states a lagged rule, whereas Eq8's same-symbol substitution gives `(L_cat^2+L_dis^2)/(L_cat+L_dis)` only for an instantaneous coefficient; it is not the exact current objective/gradient under the lagged rule. Step/epoch language is mixed. Eq11 prints a sigmoid of x_j while the prose calls it prediction confidence. The ratio expression is homogeneous of degree one, so its resemblance to L2 does not establish a regularization theorem. Native classification aggregation and source update order remain unverified. Attribute the adaptive-loss family; do not silently repair these equations into a qualified native implementation.

## Why frequency or neighbor-error conditioning alone is insufficient

The retained `graph_diffused_error_diversity_prior_v1/REPORT.md` already gives the relevant exact identity. Its extension to several fixed graph-band maps is immediate. Let P_m be member probability matrices, Y known training one-hot targets, T the training mask, and

\[
R_m=\operatorname{concat}_b\{W_b^{1/2}D_bT(P_m-Y)\},\qquad W_b\succeq0.
\]

The fixed diffusion or spectral maps D_b induce

\[
K=T\Big(\sum_bD_b^\top W_bD_b\Big)T\succeq0,
\qquad
\langle R_m,R_l\rangle_F
=\operatorname{tr}[(P_m-Y)^\top K(P_l-Y)].
\]

Thus penalizing cross-member overlap is NCL in a fixed transformed-output metric. Training-class masks give a label-conditioned kernel; they do not break the identity. Define B as mean squared member R-norm, R-bar as mean R, and O as average pair overlap. Then

\[
O=\frac{M\|\bar R\|_F^2-B}{M-1}.
\]

This is an exact symbolic equality, requiring no numerical/model run. With squared-error supervision it yields the familiar individual/pool interpolation in the same metric. **CE plus this quadratic auxiliary is not identical to GNCL's CE interpolation.** Also, the quadratic identity uses an arithmetic probability pool; the current mean-raw-logit predictor is the normalized geometric probability pool. These pooling and loss distinctions are material and cannot be hidden by the word diversity.

A filter bank whose operators sum to identity need not satisfy `sum_b D_b^T D_b = I`; spectral-band energies do not automatically form an orthogonal quality decomposition. Signed/high-pass filters also lose the same-class sign safeguard available to entrywise-nonnegative diffusion. Live cosine normalization can reward worsening a member's probabilities, as the retained report's explicit two-node counterexample shows. Parameter spread, filter-coefficient cosine or band occupancy therefore cannot stand in for deployed pooled NLL/accuracy and individual competence.

An adaptive graph-conditioned metric could be a different implementation, and nonlinear/task-conditioned safeguards could change the operation. This follow-up found no named regime, demonstrated capable-baseline failure, or verified mechanism establishing a useful increment from those broad possibilities. FoRDE already conditions its function-space similarity on training samples and true-label input gradients; repulsive ensembles already pull back finite function-space interactions. A new metric requires a concrete utility case, not a renamed repulsion term.

## Closest quality controls and their limits

| Baseline or prior | What it controls | What this follow-up does not infer |
|---|---|---|
| Ordinary/fixed GNCL; new DNCL | Individual competence versus pooled supervised objective, including dynamic scalar balance | Exact native DNCL aggregation/code qualification or shared-graph benefit |
| Exact graph-kernel NCL spelling | Algebraically identical fixed band/neighbor-error overlap | A different implementation is a new learning principle |
| CE plus member graph-residual norm; C&S | Ordinary smoothing/correction versus cross-member effect | Train-error diffusion is itself new or heldout quality is known |
| FoRDE and repulsive function-space ensembles | Task-dependent explanatory/function diversity and paid extra differentiation | Bayesian/convergence inheritance for an arbitrary graph penalty |
| BernNet, TFE-GNN, BankGCN, Specformer, MORGAN | Spectral/filter capacity, pre-nonlinearity graph evidence and spectral gating | Filter streams are independently supervised predictive members or useful error diversity |
| GraphMoRE, MoE-NP, Chimaera; HGEN | Topology-aware experts/routing, graph views and learned representation diversity | Their complete recipes match a tied-factor method or representation spread proves competence |
| Unchanged competent shared-factor ensemble and full independent members | Existing model family and the quality reference | A storage saving or restricted capacity change supplies a quality contribution |

An ordinary NCL control, topology/degree-matched shuffled metric, self-error smoothing control and the **exact same kernel-NCL objective** would be essential if the particular metric were later tested strictly for utility. Other controls must respect their already recorded qualification/access limits; an inaccessible or unqualified method cannot be represented by a weak stand-in. All member forwards, filters, residual propagation, differentiation, tuning and selection costs remain charged. These are comparison obligations, not a launched grid.

## Pilot disposition

**Zero promoted pilots; no representative training pilot is warranted on this evidence.** The operation most directly suggested by graph-frequency/neighbor-error conditioning already has an exact NCL reduction, and the new paper supplies an adaptive NCL baseline rather than a missing graph principle. The inaccessible ICASSP and GENN sources do not reverse that conclusion. A particular kernel's empirical utility is possible, but no supported increment was established that merits opening another science lane.

Reopening would require a precise graph/error regime, a named competent alternative that fails for a mechanistically explained reason, and a proposed operation beyond fixed transformed-output correlation. A separately authorized representative falsifier would then freeze the deployment pool, matched starting states, complete native schedule, per-member competence, pooled NLL/accuracy benefit, topology-shuffle/self-error controls and paid physical-cost gates before outcomes. A lower diversity proxy, only one favorable graph, or a gain caused by capacity/pooling/label correction would not support the intended claim. No threshold, seed, pilot or training admission is selected here.

PAPER_CONCLUSIONS.json, READ_SCOPES.json, CITATIONS.json and PILOT_DISPOSITION.json preserve the operation-level claims and limits. No global novelty absence, manuscript acceptance verdict, current-audit result or original-score revision is claimed.
