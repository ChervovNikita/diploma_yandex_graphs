# Full-node support for the graph-error private-factor lift

Root-requested addition, 2 October 2026. Prospective support amendment only. No active initializer, pulse, source, protocol, status, index or experiment was changed. This is an extension of the existing graph-error initializer, not a distinct second ensemble method. Saved Correct & Smooth, BernNet, GNTK and NTK-ensemble conclusions were consulted; no retained primary was reopened and no further new primary was acquired.

## Decision

**Retain one conditional quality hypothesis for root review:** the current TRAIN remask may remove useful graph-propagated cotangents on unlabeled **output roots**. Pulling these cotangents back through the full-node Jacobian can expose additional private-factor directions. The exact support change is defined below and has a simple algebraic witness. Utility, finite constructibility and novelty remain unestablished. This merits a separately frozen support comparison if root elects it; no execution is admitted here.

An important premise is narrower than “TRAIN masking discards unlabeled graph learning.” A TRAIN-logit Jacobian already differentiates through unlabeled neighbors, their features and intermediate states wherever they affect the labeled predictions. The proposal adds derivatives of **unlabeled-node output logits**. It need not recover information missing from TRAIN-root backpropagation, and sparse labels alone do not prove an advantage.

The closest retained sources establish the ingredients, and the pullback has the exact one-step soft-target distillation representation below. Correct & Smooth diffuses supervised errors into output correction; this method can distill related complementary corrections into a finite private tangent. GNTK establishes graph-dependent tangent kernels; it does not make this finite CE intervention Bayesian, useful or new. No exact equivalence to the complete support-amended, projected/capped private-slice initializer was verified in the retained scopes. The appropriate contribution scope would be a useful answer about the cotangent-support boundary, rather than “graph error diffusion,” “Jacobian-based ensembling” or a new pullback primitive.

## Exact operation at the common warm closure

Let C denote TRAIN nodes, U the other released nodes, E the N by |C| injection, P_C=EE^T, and z_all(phi) the N by K class-logit output under the existing bound private slice. E^T z_all gives TRAIN logits. All four routes start at the identical warm closure phi_0 with the same J=∂vec(z_all)/∂phi. Class indices are carried unchanged by the graph operators. Define r=∇_(z_C) CE(z_C(phi_0),y_C), with the existing loss normalization, and g=J^T E r. No y_U exists in this definition.

Use the same normalized symmetric topology and degree-3 Bernstein bank H_m as the existing initializer, with sum_m H_m=I and M=4. Let B_m=H_m-I/M. The two alternatives are

    q_remask,m = P_C B_m E r,
    h_remask,m = J^T q_remask,m,

    q_all,m = B_m E r,
    h_all,m = J^T q_all,m.

This remasked expression equals the existing TRAIN-logit lift J_C^T(E^T H_m E-I/M)r. The exact difference is

    h_all,m-h_remask,m = J^T(I-P_C)B_m E r = J_U^T q_U,m.

The unlabelled-node q_U is a signed training-error cotangent transported through graph filters. It is not an unlabeled ground-truth vector or an asserted correct pseudolabel. Ordinary backpropagation through the true full-node closure is required. A custom backward that suppresses cross-node paths changes the operation.

**Parent-reported static source boundary, not independently re-read here:** the current closure already returns full [N,K] logits through `select_target_logits`; the current h helper explicitly zeros cotangents outside `train_rows`. Thus the anticipated source change is small. Before a prospective run, qualify that `target_nodes` covers exactly the declared released output universe, that graph-node and output-row orders are identical or related by one explicit permutation, and that E/`train_rows`, graph filters and q all use that same ordering. A closure returning only some target nodes does not establish full-graph coverage. Save forward equality and true VJP/JVP checks under the full target; do not silently broaden a typed target universe or change graph normalization.

Detach r and H at phi_0. For g nonzero, project with P_g=I-gg^T/||g||², set t_m=P_g h_m, and apply the unchanged shared centering, cap and direction sign convention. Since sum_m B_m=0, sum_m h_m=0 and sum_m t_m=0 at this common closure, before a common scalar cap. The directions d_m=-g-lambda t_m satisfy

    g^T d_m=-||g||²,
    mean_m J d_m=-Jg.

The first identity is TRAIN-CE first-order descent for each common route. The second shows that the mean first-order full-node logit movement is common descent, so the extra contrasts do not directly improve the initial mean-logit predictor at first order. They can change individual trajectories and later pooled decisions; that is the empirical hypothesis. Any useful initial pooled effect can involve finite curvature and is not supplied by PSD alone.

The identities require the common warm route Jacobian, one linear common projection and one common scalar cap. They do not extend unchanged to already divergent routes or to independent per-member rescaling. A divergent pulse needs the separately saved joint constraints; this packet adds no pulse.

### Frozen linear squared-loss limit

In a frozen common linearized predictor with squared TRAIN loss, ordinary full-batch gradient descent uses the same affine update map for every route. Relative to the common baseline, a route perturbation obeys delta_m^(t+1)=(I-eta_t J_C^T J_C)delta_m^t. If the initial mean perturbation is zero, it remains zero at every update; the mean full-node prediction trajectory is exactly the common baseline. The same conclusion holds with one common fixed linear preconditioner. It does not generally hold for route-specific sampling, nonlinear optimizer transformations/moments, evolving nonlinear body Jacobians or classification CE.

Thus a larger projected full-support tangent is not itself an ensemble-quality improvement or a convergence benefit. Possible useful mechanisms include classification curvature, nonlinear body learning and finite-time basin/trajectory selection, but they require measurement in the declared continuation. Local alignment, an unlabeled-root response or tangent rank alone cannot establish them. The later pooled-quality comparison is the actual utility falsifier.

### A local objective, with clear limits

For either support choice, h_m is the gradient at phi_0 of the detached linear objective

    F_m(phi)=<q_m,z_all(phi)-z_all(phi_0)>.

Equivalently, define an auxiliary TRAIN logit map a_m(phi)=z_C(phi_0)+E^T B_m[z_all(phi)-z_all(phi_0)]. At phi_0, the gradient of CE(a_m(phi),y_C) is h_all,m. This is a local pullback interpretation, not a recommendation to fit that auxiliary predictor as the deployed model. Recomputing r or phi_0 on the fly changes the local objective. It does not generally give a fixed global task objective for persistent updates.

In the tangent regime, the full-output contrast response is J P_g J^T B_m E r, with the actual sign/cap supplied by d_m. This is a graph-filtered residual transformed by a restricted parameter-induced prediction metric. Such kernel/error machinery is established; the proposed difference is where the cotangent is supported. B_m is generally indefinite, and products of a PSD tangent kernel and graph operators need not be symmetric/descent operators. Projection gives the stated local TRAIN derivative, not a general CE, calibration or heldout guarantee. Finite guards remain necessary.

### Exact soft-target distillation reduction

Let p_0 be the warm full-node probabilities, detached. Because the CE residual sums to zero over classes and each H acts identically on class coordinates, every q_m row also sums to zero. Define detached complementary targets

    t_m = p_0 - eta q_m.

Each row of t_m sums to one, and at the warm state the soft-CE logit gradient is exactly p_0-t_m=eta q_m. Its private gradient is eta J_all^T q_m. Thus the pullback alone is **one ordinary gradient of distillation from graph-error-corrected complementary soft targets**. The remasked version has the same reduction and simply leaves off-TRAIN targets equal to p_0. Shared-parameter target detachment matters; differentiating through p_0 or q changes the derivative.

For a proper probability target, a common eta must satisfy eta <= min_(m,v,c:q_m(v,c)>0) p_0(v,c)/q_m(v,c). Nonnegativity plus row sum1 then ensures every target entry is at most1. For exact positive softmax probabilities on a finite graph a sufficiently small positive eta exists, but the bound can be arbitrarily small; numerical zero probabilities can make it zero. Allowing signed targets keeps the algebraic derivative but is not an ordinary proper soft-target CE objective and can be unbounded below. Clipping/renormalization generally changes the exact direction, class/member centering and any inherited support comparison. The current cotangent implementation does not need to construct these targets; the reduction is an attribution result, not a new target-validity restriction on an arbitrary linear cotangent objective.

Projection against g can also be expressed at this one warm point by subtracting a multiple of E r from q before forming t, followed by the common scalar cap. This does not provide a proper target guarantee or a persistent objective. Existing error correction, complementary targets and distillation therefore cover the primitive. The potentially useful scope delta remains the controlled all-output versus TRAIN-remasked initializer on the same private slice and paid continuation.

## Algebraic possibility and null cases

For an illustrative two-node graph with only node 1 labeled, let A=[[0,1],[1,0]], H_1=(I+A)/2, H_2=(I-A)/2, and use M=2 for this witness. These are PSD and sum to I. The contrast operators are B_1=A/2 and B_2=-A/2. A TRAIN residual at node 1 gives q_all,1=(0,r/2), q_all,2=(0,-r/2), while both remasked q are zero.

Let the scalar class-margin output Jacobian rows be J_1=(1,0), J_2=(0,1), so g=(r,0). The full tangents are (0,±r/2) and survive projection against g; the remasked tangents vanish. This proves that the support change can add a private direction. It does not prove useful graph semantics, a trained-rank property, CE classification quality or a predictive gain. The witness uses a simplified two-member scalar margin, not the native four-member degree-3 model.

Conversely, the difference vanishes if propagated q_U=0, if J_U^T q_U=0, or if it lies in the projected-away common-loss direction. It can also change parameter geometry while producing negligible useful logit contrasts. The actual full and remasked tangents need not have any norm/rank ordering, especially after equal radius caps. No “full support is always stronger” claim follows.

## Closest-prior/equivalence assessment

| Saved source/conclusion | Relationship and boundary |
|---|---|
| Correct & Smooth, 2010.13993v2 and pinned operational-sign resolution | Direct TRAIN-error diffusion prior: official code forms Y-p, diffuses and adds a scaled output correction, then uses a separate smoothing step. With the CE residual p-Y this proposal's descent sign must be defined consistently. Pulling related corrected targets back is ordinary distillation by the exact reduction above; a C&S output API alone is too weak an equivalence defense. The bound projected/private-route initializer remains the prospective scope difference. |
| BernNet, 2106.10994v1 | Supplies the degree-3 partition-of-unity graph bank. Spectral bands and error diffusion are attributed; full support changes only the signal boundary. |
| GNTK, 1905.13192v1/v2 saved assessments | Cross-node graph tangent covariance and graph filtering are established. A finite trainable private-factor VJP with CE, projection and finite safeguards is not the infinite-width kernel predictor from the retained scope. No global absence claim is available. |
| Bayesian Deep Ensembles via NTK, 2007.05864v2 | Frozen Jacobian/JVP ensemble construction, function-space scaling and infinite-width squared-loss posterior machinery are prior. This full-J VJP is not its random additive prior function, and no posterior interpretation transfers. |
| GRAND/GraphMix, SEA, label propagation and saved consensus/self-training notes | These cover propagated views, shared training, detached predicted targets and complementary error targets. q_U is represented directly as a signed cotangent, but it has the exact corrected-soft-target CE reduction above when targets are valid. It supplies no unlabeled correctness oracle; avoiding the word pseudolabel does not create novelty. SEA's independent-coordinate/shared-body target-detachment caution remains relevant. |
| Current graph-error initializer and conditional pulse | This is a support amendment to the existing initializer. It must be compared directly with remasking and cannot count as a separate distinct extension discovered independently. Divergent-state pulse safeguards do not transfer automatically. |

No exact matching operation was verified in these retained conclusions. That bounded statement supports a source-specific comparison, not certification that the all-node lift is absent from the literature. The combination may ultimately be attributed as a graph-conditioned tangent correction with broader output support.

## Prospective falsifier for root review

### Core paired comparison

Use complete released graphs, fresh common warm states and seeds 17/29/43. At each seed compare four initializations from the same prospective actual warm state: common-only; existing TRAIN-remasked topology signal; full-node topology signal; full-node node-permuted-topology signal. Use the same private slice, warm schedule, ordinary continuation, graph visibility/normalization, optimizer-state convention, dropout streams, mean-raw-logit pool and selector. Include the already required competent single, size-matched single, ordinary all-layer BE and packed independent ensemble controls before a broad GNNM utility/cost claim; the common-only bank supplies the warm-copy BE reference.

Apply the same contrast Frobenius cap/radius policy across the three contrast arms, and one shared alpha chosen by bounded paired TRAIN-only search. Never choose alpha independently and call the perturbations matched. Require each candidate's member TRAIN-CE and pooled TRAIN-CE finite guards from the same warm baseline. Retain zero signals, projected nulls, finite realization failures and every unmatched search in the denominator. Full support does not relax a finite guard. State whether initial optimizer moments are copied or reset, and preserve the chosen convention in every arm.

Before continuation, report unnormalized q_C/q_U and projected tangent magnitudes, projected full-minus-remask directions, and class-centered true JVP/finite contrasts. These are TRAIN/covariate-only construction diagnostics, not evidence of heldout task improvement. A tiny projected difference or nonconstructible full branch stops the support mechanism at that state. No additional resampling, private slice, filter degree or warm-time search is permitted after outcomes. The full-node signal should be compared at equal parameter radius; that does not isolate orientation at equal output Gram, which would require the separately saved stronger random-control construction.

For the declared sign d_m=-g-lambda t_m, an explicit signed finite extra functional at the **same alpha** is C(alpha)=-sum_m <q_m,z(phi_0+alpha d_m)-z(phi_0-alpha g)>. Its first-order prediction is alpha*lambda*sum_m h_m^T t_m, equal to alpha*lambda*sum_m ||t_m||² before a common cap. After a common cap use the actual h^T t expression. Evaluate the actual common candidate and actual branch with one detached q and the exact same output-row universe/order; never subtract a different-alpha common point. The class sums of q vanish, so this is invariant to per-node scalar logit offsets. A positive finite signed value tests the intended extra direction; it is a construction check and does not certify utility, proper targets or generalization. Full and remasked branches can additionally be evaluated against the same full-q functional to describe the support boundary, without optimizing that post hoc diagnostic.

One predeclared practical quality screen can require mean later pooled VALIDATION NLL improvement at least 0.01 nats over **common-only, remasked and topology-null**, no more than 0.5 percentage point mean accuracy decline, and the same improvement sign in all three seeds. Keep member competence, final versus selected roles, branch failures and complete costs. These are practical planning constants, not a power or significance calculation. More unlabeled signal, larger JVP spread, a successful Armijo guard or a favorable oracle is not a substitute for later pooled utility. A full-node gain without beating topology-null supports broader cotangent work, not a graph-specific effect.

### Representative tasks and scale

The source-bound PolyFormer-Mono/Squirrel complete graph with its 512-coordinate slice is the cleanest compatibility check against the current remasked operation. Its native TRAIN density is not a strong sparse-label benchmark; do not claim it settles the sparse-support motivation. It can provide the paired construction and immediate later-utility falsifier with all existing source bindings preserved. This is a fresh declared followup on an exposed task, not independent confirmation.

For a representative **sparse released TRAIN** quality screen, use complete ogbn-products and a competent modern scalable backbone qualified for identical common warm outputs and the chosen stem.S/head.R private slice. A three-layer residual GraphSAGE h=256 gives a concrete 512-coordinate stem/head factor interface to qualify; a native scalable Polynormer/PolyFormer can instead be elected before the freeze if its complete-model AD and boundary are qualified. None is currently source-qualified by this packet. A backbone switch after outcomes is prohibited. Use native TRAIN/VALIDATION/TEST roles; do not thin labels or use graph subsets as a substitute for the sparse released task. Official TRAIN labels are the only residual information.

For a sampled training recipe, the initialization's J_all must refer to a declared frozen deterministic full-graph output closure, or one fixed explicitly sampled closure shared by every arm. It cannot silently change samples per VJP or replace all-root derivatives by training minibatches. An approximate/subsampled full lift is a different operation with its own estimator/bias/cost contract. Exact rematerialization, streaming or distributed VJP is permissible if source/AD equivalence is qualified. This is a source/resource requirement, not a scientific rejection of larger-scale evaluation.

### Full cost and planning GPU reservation

At a common primal, original full and remasked q reuse the graph filtering. The original and topology-null degree-3 banks require up to six sparse products on N by K error fields. A conservative low-memory account allows 1 common-gradient plus 12 contrast VJP pullbacks, 12 JVP checks, all necessary primals/rematerialization, and up to **96 complete member forwards** for six paired trials over four four-member candidates. Partition-of-unity/reused-primal identities may reduce calls only after exact source qualification. Charge warm fitting, full graph construction, output/error fields, VJP/JVP qualification, every rejected trial, optimizer copies, continuation and all served member passes. No full N*K by parameter Jacobian or explicit NTK is required.

An N by K float32 field costs 4NK bytes. On the published Products size N=2,449,029, K=47, this is approximately **439 MiB** per field, before logits, several filtered buffers and model activations. Four full cotangents need not be held on GPU simultaneously. Unlabeled-root support does not necessarily add a new forward when the base closure already returns all-node logits, but it can make reverse work dense where a sparse TRAIN-root VJP would have pruned branches. Actual autodiff behavior and memory must be measured; neither a full-node output matrix nor the small private parameter count guarantees low cost.

For complete Products, reserve **100–400 A100-80GB GPU-hours** for the three-seed four-arm continuation screen plus competent reference families and approximately 25% qualification/initialization/prediction/failure overhead. A minimal four-arm matrix plus independent/single/size-matched references is roughly 66 single-member full-fit equivalents; a separately initialized native BE reference adds 12, giving approximately 78. At an assumed 1–4 GPU-hours each plus overhead, the reservation is about 100–400 hours. These are planning assumptions, not measured runtime. Provision 2–4x80GB GPUs when exact full-graph VJP/rematerialization requires them, at least 256GB host RAM and sufficient local SSD. A full-data source/throughput/memory preflight must refine the reservation. If the preflight is slower, increase resources or classify the test as resource-deferred; do not call the quality hypothesis scientifically false.

The source-bound Squirrel/Photo comparison has the inherited outcome-free cost evidence below. It is a compatibility and development option, not a resource ceiling or justification for rejecting a larger sparse-label task. No full-support timing has been measured.

**Updated inherited cost evidence:** root supplied an outcome-free timing receipt at `graph_quality_priority_20261002_v1/OUTCOME_FREE_RESOURCE_ESTIMATE.json`, which was read as a permitted scalar resource input. It records 15 completed Squirrel continuations averaging 78.64 seconds and 4 completed Photo continuations averaging 5,325.04 seconds. The four-arm, two-graph, three-seed continuation forecast is **18.01 device-hours** under the inherited timing/device convention. It excludes warm acquisition, full-support initialization/qualification, profiling and contention; it is not an actual full-lift timing or total budget. The underlying run outputs and quality fields were not reopened. This supersedes the generic small-task allowance when planning the source-bound Squirrel/Photo matrix. A final reservation should add measured qualification/acquisition overhead and all new support/control work. The larger-scale reservations above remain planning assumptions for different tasks/backbones, independent of this receipt.

### Heldout confirmation and stopping

Freeze the complete support rule, filters, slice, radius, alpha search, native recipe/selector, failure handling and quality thresholds before opening Products TEST once. For cross-graph sparse-label confirmation use complete ogbn-papers100M, fresh seeds 47/59/71 and native official roles, after separately qualifying a capable deterministic all-root closure. A planning allowance is **500–2,500 A100-80GB GPU-hours**, 8x80GB GPUs available for sharding/rematerialization, and 1–2TB host RAM. This larger reservation is conditional and highly uncertain, not a runtime forecast.

For the public papers100M size N=111,059,956 and K=172, one dense float32 error/logit field alone is about **71.2 GiB**. Multiple fields, hidden states, graph storage, saved activations and backward buffers dominate. Streaming/distribution must retain the same mathematical closure. If complete-source execution is not qualifiable, keep large-scale confirmation resource/source-deferred; no small graph result certifies this scale. Complete arxiv could be a separately declared cross-graph compatibility check, but would not replace sparse-label confirmation under a scale claim.

The saved official OGB documentation was additionally read in exact Products/papers100M resource sections and summary table rows to bind these sizes and class counts (`LARGE_DATASET_SIZE_BINDINGS.json`). Products uses sales-rank TRAIN8%/VALIDATION2%/TEST90%. Papers100M's native prediction target is the approximately 1.5 million arXiv-paper subset; most other papers have no label. The 71.2GiB calculation assumes an explicitly declared **all-111-million-paper output-root closure**, including non-arXiv latent cotangent roots, rather than just the native labeled target subset. Such a broadened root universe requires its own source and semantic qualification. If the closure selects only arXiv targets, specify the corresponding selection/injection around the full-graph filter and charge that exact operation; do not call it the same all-node lift or inherit the all-root formula without the selector. No new labels or correctness targets are supplied for non-arXiv nodes. Runtime dataset version, arrays and actual target coverage remain unopened and unqualified here.

Stop promotion if full support adds negligible feasible private directions, repeated finite guards fail, no later quality improvement meets the locked screen, topology-null matches it, competent controls erase the paid-cost utility, or the source closure is not the declared one. Resource needs alone do not decide scientific validity. Even a positive support result is an amendment of the original initializer and needs honest ingredient attribution and broader confirmation. No novelty or reviewer verdict is implied.

## Binding saved inputs

- `continuous_method_gap_search_v1/round15_graph_route_initialization/REPORT.md` and its `PAPER_CONCLUSIONS.json`: common-closure operation, C&S v2 operational sign, BernNet bank, native modern boundary and finite safeguards.
- `graph_initializer_cache_literature_followup_v1/REPORT.md` and `PAPER_CONCLUSIONS.json`: NTKGP prior, matched output-geometry control, same-alpha and cost boundaries.
- `graph_ensemble_gap_skeptic_v1/ASSESSMENT.md`, scoped lines 45–90: GNTK v2 and true-versus-surrogate forward tangent boundaries.
- `literature_gap_search_root_v5/MOMENT_AND_COUPLING_ASSESSMENT.md`, scoped lines 1–70: GNTK v1 and graph covariance attribution.
- `graph_specific_error_gap_search_v1/REPORT.md`: filtered-error/kernel equivalence risk, graph topology attribution, no persistent or new-pool admission.
- `graph_quality_priority_20261002_v1/OUTCOME_FREE_RESOURCE_ESTIMATE.json`: permitted root outcome-free continuation timing forecast only, no quality inspection or underlying-run revalidation.

Exact hashes/read limits are added to INPUT_BINDINGS.json. This note adds zero primary-paper reads, no model outputs or execution, and no active protocol amendment.
