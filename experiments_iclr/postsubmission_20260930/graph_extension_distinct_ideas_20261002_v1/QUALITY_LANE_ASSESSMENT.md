# Quality lane: graph regimes and correlated ensemble errors

2 October 2026. Unadopted research only. The adopted v19 index and verified but unadopted v20 draft were consulted before this assessment. Saved conclusions were reused; two previously unindexed primary methods received new scoped reads. No active model, dataset, checkpoint, logits, training source or current experiment output was inspected. No experiment is authorized or recommended as a new GNNM method by this packet.

## Decision and scientific potential

**No distinct predictive-quality mechanism survived the generic regime/conflict review below.** The precise candidate below is ordinary conditional stacking, and its flexible version is graph-conditioned mixture-of-experts routing. A second candidate, relocating shared-gradient harm into private factors, has an elementary reduction to increasing each member's ordinary private-gradient step. These are scientific attribution/equivalence decisions. They were not rejected because available GPUs are small, because execution would be costly, or because a positive result is impossible.

Root subsequently requested a specific support amendment to the existing graph-error initializer. `FULL_NODE_COTANGENT_LIFT.md` retains that separate conditional question: omit the final TRAIN remask and use the full-node output Jacobian. It is an extension of the current initializer, with a precise extra J_U^T q_U direction and unestablished quality. It is not a second independently discovered GNNM method. This generic review does not reject that amendment.

Conditional pooling could still improve a competent ensemble if member error covariance varies predictably with graph regimes. That empirical question is unresolved. It is a useful ordinary baseline, not a demonstrated benefit or a new graph learning principle. Establishing a new contribution would require a distinct estimator/operation or a graph-specific statistical result that does more than recast supervised stacking. Larger datasets and more GPUs alone do not change the operation's equivalence.

The independent general maintenance lead and shared-W arithmetic hypothesis remain parked in their own notes. Their reasons are also separate: maintenance is not factor-specific; compression needs an unestablished trained rank/work property and is ordinary low-rank multiplication. The symbolic rank counterexample does not prove that trained compression can never help.

## 1. A precise conditional error-covariance algorithm

Use M=4 competent frozen predictors with class probabilities p_m(v). Partition nodes into K graph regimes using a fixed label-free map g(v), based, for example, on degree and feature-neighborhood disagreement. Establish a TRAIN-label fit/calibration separation before expert training. Experts may use the complete transductive graph/covariates allowed by the native task, but receive only fit-role labels. Obtain their predictions on the calibration-role TRAIN nodes; official VALIDATION and TEST labels never enter the error estimates.

For calibration node v and class c, define e_m(v,c)=p_m(v,c)-1[y_v=c]. For regime k form the **uncentered** residual Gram

    S_k[m,n] = mean_(v:g(v)=k) sum_c e_m(v,c)e_n(v,c).

The mean-error contribution matters: a centered covariance alone omits squared bias. Shrink a poorly supported S_k toward the global Gram with one predeclared estimator rule. Solve the small convex program

    w_k = argmin_(w>=0, sum w=1) w^T S_k w,
    p_pool(v) = sum_m w_g(v),m p_m(v).

Define empty/small-regime fallback before outcomes, such as the global simplex solution. No per-test label, homophily measured from heldout labels, outcome-selected partition or graph-error pulse is required. This changes the deployed pool to an arithmetic probability mixture; it must not be compared as though it preserved a mean-raw-logit model. Both pooling conventions need explicit reference arms.

### Exact equivalence

For fixed regime k and fixed expert outputs,

    mean_v sum_c (sum_m w_m p_m(v,c)-1[y_v=c])^2
      = w^T S_k w.

Thus the covariance program is exactly simplex-constrained least-squares stacking with piecewise constant graph-feature weights. Calling S_k an ensemble error-correlation object does not produce another learning objective. Shrinkage adds an established statistical regularizer. With a learned w(v)=softmax(h(graph_features(v))), directly minimizing Brier or CE gives an ordinary supervised mixture gate. The same head can consume independent or shared-factor experts. BatchEnsemble changes predictor storage/parameterization, not this reduction.

Graph dependence across **different nodes** is a separate object from covariance across members at the **same node**. Replacing the node average with a nonidentity positive graph kernel K yields a quadratic error-field objective of the form sum_mn w_m w_n tr(E_m^T K E_n). It can emphasize graph-coherent errors, but it is no longer the native pointwise Brier risk unless K is the corresponding diagonal weighting. No improvement of CE, accuracy or nodewise Brier follows just because K is positive. A node-resampling confidence interval cannot silently treat connected nodes as independent replicates.

### Closest priors

| Prior | Consequence |
|---|---|
| Saved graph link stacking and Link-MoE conclusions | Supervised graph-predictor combinations and structural/input-conditioned expert weights are established. Pair versus node domain is a task distinction, not enough methodological novelty. |
| New scoped **Mixture of Experts for Node Classification**, arXiv:2412.00418v1 | Direct node-domain prior: random-walk local context, feature-based edge discriminator, degree information, averaged global context, and nodewise softmax expert weights. The main objective fits weighted predictions; the appendix explicitly discusses pretraining/fine-tuning and expert training on fewer labels before gate training. |
| MCL/TreeNets saved closure | If the idea changes from pooling to lowest-loss member assignment or marginal oracle credit, specialization and sharing are already established. |
| AdaGCN and B3F-GNN saved conclusions | If regimes are error-reweighted sequential experts or hop classifiers, error-guided graph specialization is prior. |
| New scoped **GraphMix**, arXiv:1909.11715v1; saved GRAND conclusion | Graph-context versus feature-only shared-weight training, graph views, pseudotargets and consistency/mixup are established. GraphMix shares an FCN and GNN, alternates manifold-mixup and native GNN losses, and serves the GNN alone; GRAND uses stochastic propagated-feature views and consistency, rather than this served mixture. |
| Saved DICE/FoRDE/DivDis/D-BAT/GNCL/repulsion conclusions | Replacing residual Gram fitting with a spread, response-diversity or task-plus-repulsion penalty invokes established diversity methods; larger spread does not certify complementary correct predictions. |

MoE-NP's scoped stylized CSBM theorems concern particular linear low-pass models and shifts. They do not prove that one sufficiently expressive modern backbone cannot represent heterogeneous node regimes. Its author-reported tables were incidentally displayed in the selected block ranges; no score was adopted or used to choose a threshold. The final published/latest versions and author implementations were not qualified here. The already-existing earlier version is sufficient to establish the routing ingredient; no global absence-of-prior claim is made.

## 2. Shared-step harm compensated by private gradients

Consider member loss L_m(W,phi_m), where W is shared and phi_m denotes private factors. For a planned shared update d_W, ask for the smallest Euclidean private update that makes each member's **first-order** loss change nonpositive:

    minimize ||d_phi,m||^2
    subject to g_W,m^T d_W + g_phi,m^T d_phi,m <= 0.

If g_phi,m is nonzero, the solution is

    d_phi,m = -max(0, g_W,m^T d_W)/||g_phi,m||^2 * g_phi,m.

It is an ordinary member-private loss-gradient update with a member-dependent magnitude. If a nominal private update already exists, apply the same formula to its remaining directional deficit. If g_phi,m=0 and the shared directional term is positive, the constraint is infeasible without changing the shared step or admitting other private capacity. A radius cap can likewise make compensation infeasible.

This does not create graph-specific credit, useful member disagreement, or a finite-step guarantee under AdamW, curvature and coupled shared-message paths. Generalized metrics or joint constraints produce standard constrained/block or multitask optimization subproblems, with PCGrad/MGDA-style methods as close controls. Calling members tasks or adding graph strata does not establish a new principle. The saved selected Products state had strongly aligned shared own-CE gradients; that is a bounded adverse diagnostic for a blanket conflict story, not proof about every state or graph. No gradient observation was independently reacquired in this packet.

**Disposition:** no new private-compensation method or pilot. A future conflict explanation first needs qualified actual optimizer directional effects and the simpler own-versus-pooled objective control. Generic conflict surgery is not the missing quality hypothesis by default.

## 3. Optional representative ordinary-baseline test

This section is an archived conditional design for root review, **not a proposed new-method launch**. It records what would be needed to test the empirical utility question without disguising limited compute as a scientific rejection. Existing initializer, pulse and BUDDY studies are not altered or duplicated.

### Complete task and controls

Use complete **ogbn-products**, native official splits and the same transductive graph/features for every arm. Qualify runtime dataset version and graph treatment before fitting. Declare seeds 17/29/43 prospectively; these are optimizer seeds on one fixed graph, not independent graph-population replicates. Use a competent modern three-layer residual GraphSAGE with h=256, normalization and a declared fixed neighbor sampler (for example fanouts 15/10/5), with a competent recipe established without selecting from this study's outcomes. A differently chosen modern backbone is permitted only before the protocol freezes; no rescue switch after results.

Within official TRAIN, make one reproducible 80/20 fit/calibration label separation, stratified on TRAIN classes only. Expert supervision uses the 80% fit labels. The 20% calibration labels fit every learned pool; they are not official VALIDATION. Every reference receives this same supervision contract. Singles may also receive a separately labeled full-TRAIN reference, but it cannot replace the matched 80% reference. No current fitted bank is inherited.

Train four predictor families in every seed: competent single, parameter-matched single, M=4 all-layer BatchEnsemble with **member-resolved graph messages**, and M=4 independent ensemble of the same backbone. The common-message approximation is not assumed. Offer independent members competent packing/batching and the same neighbor-sampling policy. Pair prescribed initial/RNG components where the architectures permit and disclose those that cannot be paired.

For each ensemble bank, reuse fixed predictions for: mean probabilities, mean raw logits, global simplex Brier stacking, graph-regime simplex Brier stacking, and a supervised softmax gate on the exact same declared graph/context features. Include a regime-identity-permuted control with the same support sizes and estimator capacity. A covariance gain over uniform pooling alone is weak; the global and graph-feature gate controls are decisive. The gate's labels, selection and preprocessing costs must match the covariance pool's. Do not optimize the expert bank jointly for one favored pool.

One precise regime choice is four degree quantile bins crossed with two feature-neighbor-disagreement bins, yielding K=8. Fit bin cutpoints only from allowed TRAIN covariates, without labels; fix normalization, isolated-node fallback, small-bin shrinkage and the permutation seed before outcomes. Do not scan alternative K, signals or partitions. This is illustrative and must be frozen by root if this optional study is elected.

### Three-seed falsifier and stop

Primary endpoint: pooled VALIDATION NLL under one unchanged native selector. Accuracy, Brier, mean/member competence, cost and the secondary fixed final endpoint are retained. A possible practical screen, frozen before outcomes, is at least 0.01 nats mean NLL improvement over **both** uniform and global stacking, no more than 0.2 percentage point mean accuracy loss, the same improvement sign in all three seeds, and improvement over the matched graph-feature gate. These are planning constants, not power calculations or a significance guarantee.

Stop graph-regime specificity if permutation/global stacking matches it, and stop a proposed covariance-estimator contribution if the matched expressive gate matches it. If shared-factor and independent banks benefit similarly, attribute the result to pooling rather than factor sharing. Stop task-utility promotion if competent single controls remain better at the paid training/serving budget. Mixed seed signs are inconclusive; poor member competence is retained as an adverse result. Increased disagreement, a favorable fitted S_k, or a lower calibration Brier cannot substitute for later utility. Do not alter bins, loss, seeds, backbone, test role or thresholds after outcomes.

### Realistic planning budget

The four families across three seeds require 6 single-model fits plus 6 M=4 ensemble-bank fits: conservatively **30 single-member fit equivalents**, before gates, inference caches, qualifying the recipes or failures. Common W does not eliminate M private message/state passes. No timing was measured and no precise completion time is promised.

For a qualified neighbor-sampled h=256 recipe, reserve **40–160 A100-80GB GPU-hours** for this Products matrix, including roughly 25% for qualification, prediction materialization/gates, diagnostics and failed fits. This allowance assumes approximately 1–4 GPU-hours per single-member fit equivalent; it is a planning assumption requiring a full-data throughput/memory preflight, not a performance observation. A preflight outside that range changes the resource reservation; it is not a scientific failure and must not cause graph/label subsets to masquerade as the complete task. An H100 may change wall time, not the scientific comparisons.

Provision at least one 80GB accelerator, 256GB host RAM and sufficient local SSD for released graph/features, samplers and all checkpoints/predictions; qualify actual peaks before launch. Additional accelerators can run banks/seeds concurrently. Three-seed confirmation on complete **ogbn-papers100M** would conservatively reserve **300–1,000 A100-80GB GPU-hours**, approximately 30 member-fit equivalents at 8–24 GPU-hours each plus overhead, and 512GB–1TB host RAM with 4x80GB GPUs available. Those deliberately broad reservations are not source timings or a guarantee of feasibility; actual complete-data preflight settles representation, memory and throughput. Lack of present resources would make this ordinary utility test resource-deferred, not scientifically rejected.

### Heldout confirmation

If the exploratory graph-regime estimator beats all qualified controls, freeze the expert recipe, role split, bins, shrinkage, gate capacity, selector, costs and thresholds before opening Products TEST once. That test is a within-task heldout role, not a new population. For cross-task confirmation use complete papers100M, fresh seeds 47/59/71 and native splits, with predeclared fit/calibration role rules and no outcome-based changes. Qualify its feature/split/version contract separately. Charge fresh full expert acquisition, preprocessing, all gate fits and prediction/serving passes. No result, superiority, novelty or reviewer verdict follows from this prospective design.

## Source/read accounting

- New quality scoped methods: **2**, GraphMix arXiv:1909.11715v1 (2019-09-25) and MoE-NP arXiv:2412.00418v1 (2024-11-30).
- Earlier new maintenance scoped methods in this same packet: **2**, CoRe-GNN and dynamic nonlinear propagation.
- Total new scoped paper methods in this packet: **4**; new full-paper reads **0**; retained-primary method rereads **0**; author-source reads/executions **0**.
- GraphMix paragraph blocks 15–32 and display-math blocks 0–2; MoE-NP paragraph blocks 15–18, 23–50, 89–96 and display-math blocks 0–12. Isolated appendix proof formulas in the latter display are disclosed; complete proofs are not audited. Complete listings/figure pixels/results/native source are not claimed.
- Latest metadata versions were v3 for both quality papers; only exact v1 sources were inspected. Existing earlier-version mechanisms establish ancestry; final-version implementation equivalence remains unqualified.
- Exact versioned bytes, hashes, paragraph paths and selected passages are in the JSON accounting. Six bounded quality discovery queries are metadata only; unsuccessful/fuzzy results establish no absence.

Parent review owns adoption, any optional baseline qualification and all later execution. This packet promotes zero GNNM methods and zero pilots.
