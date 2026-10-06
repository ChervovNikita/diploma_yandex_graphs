# Graph supervised acquisition and nonlinear branch growth

Two mechanisms remain worth a fixed quality test: positive graph-correlated supervision weights, and new nonlinear message features initialized from graph-error subspaces. They change supervised acquisition or capacity. The consulted literature does not establish originality of either complete recipe; familiar components alone do not settle their empirical usefulness. The LayerNorm gauge proposal is a function-preserving coordinate change with an explicit optimizer equivalent, so it is closed as a separate method direction.

The primary new benchmark for weighted acquisition is full WikiCS with the published Polynormer-r backbone and its existing pinned implementation. The growth mechanism has a smaller first experiment on the already specified full HeaRT Citeseer graph. No training or dataset access is included in this note.

## What the saved ledger already closes

Index72 has260 historical records and207 normalized paper identities; these are not certified full-paper counts. Older packets outside those records matter: MIMO/Packed-Ensembles applicability, Hydra/ensemble-field distillation, and Recon/adaptive sharing were already read. They were reused here, not rediscovered.

| Prior family | Consequence for this proposal |
|---|---|
| TabM, BatchEnsemble, HyperBatchEnsemble, StarSSE | Shared dense weights, first-factor initialization, copied models and varied learning trajectories are established. TabM explicitly permits weak diverse submodels; it supplies no standalone member-competence theorem. |
| MIMO and Packed-Ensembles | Independent-input training and disjoint packed nonlinear subnetworks are established. Repeating one intact graph in every MIMO slot throughout training is a shared multihead port, not the independent-input construction. Packed execution still pays every model's width/work. |
| Hydra and ensemble distribution distillation | A common body with fixed teacher-matched heads already preserves finite member association at exact fitting. Generic teacher copying, aligned hidden/output targets, or edge covariance cannot be advertised as a new preservation principle. |
| Recon, TAG, GDPS, adaptive sharing; PCGrad, OGD and GEM | Choosing what to unshare from conflicts/gradient geometry and locally protecting outputs/losses are prior. Existing saved graph recurrence/noise and finite shared-displacement proposals already cover that direction; another layer selector is not retained here. |
| BernNet, graph-error VJPs, splitting descent, GGN/covariance shallow ensembles | Spectral error transport, centered tangents and useful covariance orientation are established. The Oct4 shallow-ensemble source closes generic centered/covariance initialization claims. Growth must add a different function site, not merely perturb the same fast factors. |
| DICE/decorrelation and deep ensembles | Parameter/feature separation, lower correlation or lower training likelihood loss do not establish better served classification. Invertible feature rescaling does not add information; a capable independent ensemble contains the shared model's parameter restriction. |

## New primary sources and limits

**GradMax**, Evci et al., arXiv2201.05125v1, pp2–4, adds neurons with zero incoming weights and nonzero SVD-selected outgoing weights. Under f(0)=0 and f'(0)=1 the network output and old-parameter gradients initially stay unchanged; incoming new weights have a live gradient while outgoing weights initially do not. This is the direct antecedent of function-preserving gradient-guided growth. Its practical minibatch approximation, activation/normalization conditions and SGD motivation do not certify target accuracy or Adam dynamics.

The printed Eq11 constrains only a Frobenius norm, while the following prose selects multiple singular vectors. Under the literal lone Frobenius budget, concentrating the norm in a leading singular direction can dominate equal-norm top-r directions. This note uses an explicit row-orthonormal constraint instead. No paper proof or author code was audited, and no optimum or efficacy claim is transported from that ambiguous rank/scaling statement.

**On the Initialization of Graph Neural Networks**, Li et al., arXiv2312.02622v1, pp3–5 and method opening p6, derives Virgo graph-aware cold weight variances from message paths and graph powers. Its assumptions include almost perfectly correlated message paths, approximately uncorrelated weight paths/gates, Bernoulli activation probabilities, and uniform initial class predictions. This establishes graph-dependent initialization ancestry. It neither uses learned structural error directions nor preserves a trained predictor; those assumptions do not describe a competent warm nonlinear ensemble automatically.

**GradInit**, Zhu et al., arXiv2102.08098v1, pp3–4, learns nonnegative parameter-block scales for loss after a prescribed first SGD/Adam step, subject to optimizer-specific gradient-norm constraints. It explicitly distinguishes Adam's first sign-like step from a raw-gradient step, uses partially overlapping minibatches, and stops differentiation through the optimizer direction in its approximation. This is close ancestry for scale/optimizer-aware initialization, not an exact reciprocal LayerNorm gauge or a task-quality guarantee.

**The Bootstrap for Network Dependent Processes**, Kojevnikov, arXiv2101.12312v1, pp18–19 and Gaussian construction at p20, uses graph-correlated multipliers for network-dependent statistical observations. Its neighborhood-overlap covariance is PSD, and Gaussian multipliers can be generated from a covariance square root. General graph-distance kernels need not yield a PSD covariance. The paper concerns conditional confidence sets and asymptotics under explicit dependence conditions. It does not establish positive weighted neural-network training, four balanced served routes or a graph-ensemble accuracy benefit. This resolves an Oct2 metadata-only paused lead with a first scoped method read.

## Mechanism one Positive graph correlated supervision

For TRAIN examples v and two fixed bounded public-topology fields z1,z2, use four weights

\[
w_m(v)=1+\delta s_m(v),\quad
s=(z_1,-z_1,z_2,-z_2),\quad\delta=1/2.
\]

The graph fields are generated once from independent seeded Gaussian node vectors using F(P)=(I+P+P²+P³)/4. P uses the same admitted public topology as every predictor. Center each field over the TRAIN mask, standardize TRAIN RMS, then use one common maximum-absolute bound across the graph and IID comparison fields. The IID comparison uses the same white vectors before smoothing, the same centering/RMS procedure and the same common bound. Thus weight ranges, TRAIN mass and RMS match; normalization introduces dependence, so these are IID-draw fields rather than literally independent final weights. Field generation must use a separate RNG so model/dropout initialization is unchanged.

Every weight lies in[.5,1.5], all original TRAIN labels contribute, and mean_m w_m(v)=1 pointwise. Use fixed-count loss reduction

\[
L=\frac1{4|T|}\sum_{m,v\in T}w_m(v)\ell(f_m(v),y_v).
\]

Do not divide each member by its minibatch sum of weights. If functions and shared Jacobians are identical on the same examples and stochastic realization, the shared gradient equals ordinary supervision exactly. Route-dependent inputs, dropout realizations or subsequently diverged functions remove that samplewise identity. Independent dropout remains independent during ordinary training; the identity is a deterministic qualification statement, not a training invariant. For each member, .5 L_own<=L_weighted<=1.5 L_own, hence L_own<=2L_weighted on the same TRAIN population. This finite training-loss bound does not guarantee VALID/TEST competence.

**Hypothesis:** at the shared nonlinear backbone's paid storage/serving point, coherent topology-dependent weighting improves pooled accuracy over both unit weights and equally sized IID-draw weights without reducing mean member accuracy. Coherent graph regions may be a useful axis for acquiring complementary private responses while retaining every example. This is estimation and optimization, with unchanged prediction information and graph operators. It is known bagging/specialist-learning ancestry; no new resampling principle is asserted.

## Concrete full WikiCS protocol

Use WikiCS:11701 nodes,300 public features,10 classes, the full graph and public split0. Pair training seeds17,29,43 on that one split. The author loader combines val_mask OR stopping_mask for VALID; retain this source convention, with TEST closed until the final comparison is frozen. These three seeds are not three independent graph/split replications. Dataset bytes, mask counts/disjointness and cache identities remain to be qualified before execution; none were loaded here.

The published and pinned Polynormer-r code supports WikiCS directly. Use hidden512/head1, seven local layers, two global layers, local100 plus global1000 epochs, Adam lr.001/weight-decay0, and input/local/global dropout.5. The release therefore pays1100 ordinary epochs. The saved paper Table5 lists warmup100/Epochs500; this protocol explicitly chooses the longer released command, uniformly across all arms. No hyperparameter grid, endpoint exclusions, three-pass meta update or extra supervision is added.

Use six complete predictors: strong native single; ordinary shared boundary-GNNM; graph-weight shared GNNM; IID-weight shared GNNM; ordinary independent4; and same-graph-weight independent4. The three shared banks have byte-matched initial parameters/RNG conventions per seed and the existing private stem/head boundary factors; every member executes the full nonlinear local/global trajectory. All shared dense backbone weights learn. There is no common hidden state with only four linear output heads. Independent4 has four complete native encoders/backbones and the same full input/topology/label access. Unit-weight and graph-weight independent4 are initialized with the same four per-member seeds within each block. Field realizations map to the same member identities in the shared and independent banks.

Each epoch has one ordinary own-CE update, with graph/IID weights only in the designated arms. Use original constructor/first-stem factor initialization for all three GNNM arms, rather than silently adding a warm or neutral-factor method. Independent members train with their own ordinary Adam; shared GNNM uses one joint mean-member loss and optimizer. This private-gradient reduction/epsilon and parameter-ownership difference is declared, not a causal sharing isolation. Keep independent dropout per member; no synchronized dropout is imposed.

At the local/global transition restore each bank's selected local model/optimizer state and set the stage flag explicitly, preserving live training RNG. Select a separate best global checkpoint by first maximum complete pooled VALID accuracy, evaluated at the same epoch cadence for every bank; a full native single uses its own accuracy. For independent banks use a common pooled selector over synchronous member states, explicitly a collective-selection adaptation rather than separately picking four different epochs. Store model, all optimizer/RNG states, stage flags and member metrics. Serving and NLL use uniform mean class probabilities, matching the existing allowed Polynormer node-training source. Pooling, schedule and selection are fixed before outcomes.

Charge all18 configuration-by-training-seed cells, shared/independent member work, field construction, acquisition, checkpoint/selection work, memory and serving latency. Ordinary independent4 pays four native learners; no one-forward efficiency claim is made. The existing native model module, boundary family and qualified runtime can be reused, but the Amazon-specific runner's hard-coded node/class counts and roles cannot. A new WikiCS source/custody packet is needed before any fit.

**Decision requirements:** preserve every seed and arm. The graph-weight shared bank must exceed both shared references on equal-seed pooled accuracy, with no lower equal-seed mean member accuracy. A gain supported only by one selected seed, disagreement or NLL fails the quality hypothesis. If IID wins, close topology-coherent weighting; if independent graph weighting gains similarly, attribute ordinary graph-dependent bagging and assess only sharing's measured storage/cost tradeoff. If the strong single or independent controls dominate the proposed operating point, do not claim an ensemble quality advance. Report absolute correct counts, member distributions, pooled gain and inclusive costs, then freeze the decision before a single held-TEST evaluation of every selected seed/arm.

## Mechanism two Nonlinear message growth from graph error subspaces

At a native pre-aggregation interface with common learned H and fixed current support P_s, add

\[
Y_m=P_sH+P_s\tanh(HV_m)B_m,
\quad V_m\in\mathbb R^{d\times r},\ B_m\in\mathbb R^{r\times d'},\quad V_m=0.
\]

Use r=2 and four predictive members. When dimensions differ the native base map remains in its original place; the displayed equal-width case is the representative Citeseer interface. Retain native root/residual paths, decoder and shared learnable weights. Only new growth parameters are private. This is an in-network correction before a potentially lossy aggregation, rather than a frozen independent side encoder.

With tanh(0)=0 and tanh'(0)=1, every member initially has the copied warm function and old-parameter gradients; grad_B L=0, while grad_V L=H^T P_s^T G B^T for the actual interface cotangent G. Setting both matrices to zero would stall the branch. Default ReLU at zero is not a valid substitute for the unit-derivative assumption. Finite precision, native normalization and the exact insertion point still need a numerical copy/gradient check.

Construct four degree3 Bernstein graph bands F_q(P_s)=binom(3,q)(I+P_s)^(3-q)(I-P_s)^q/8, q0..3, already attributed to BernNet, on one fixed target-masked TRAIN calibration support. With G obtained only from its supervised TRAIN episode, form K_q=H^T P_s^T F_q(P_s)G. Set rows of B_q to the top-r right singular vectors of K_q, with B_q B_q^T=I_r; V_q stays exactly zero. The Stiefel constraint makes top-r gradient-energy selection well defined. The unfiltered actual gradient uses K=H^T P_s^T G: the filtered matrices select proxy output subspaces and do not maximize the actual gradient universally or implement a graph band during serving.

This adds new nonlinear feature capacity. If tanh is replaced by identity, the correction is an additive low-rank linear map and can be folded accordingly; that limit is known low-rank adaptation. Later nonlinear pre-aggregation features can distinguish neighborhoods with equal means, as the finite(0,2) versus(1,1) example shows. A capable single with nonlinear incoming messages can do so too. The example does not diagnose the real graph's common errors. No branch can recover information entirely absent from its input H/context.

Larger gradient norm is not automatically useful under Adam: at reset moments the first update is nearly sign(g). Rotating an output basis within an unchanged subspace leaves an isotropic SGD projector response unchanged, while coordinatewise Adam need not be invariant. Distinct basis vectors alone are therefore insufficient evidence; report output projectors and actual parameter/function responses.

**Hypothesis:** at fixed added rank/cost, graph-error growth yields better served quality and retained mean member competence than equally sized unfiltered-gradient growth and ordinary shared GNNM. First use full HeaRT Citeseer, the same three warm seeds0/1/2 and source20/60/eval5 roles already fixed for that task, with only one first pre-aggregation insertion site. Controls are no-growth competent bank, same-rank unfiltered growth, capable same-information native single with eight new message units, and same-growth independent4. Existing strong full-head/single/independent references remain useful, but are not mislabeled as exact new-operation controls. All new sparse/dense work and all member states after divergence are paid.

Stop before scientific fitting if the masked-support insertion cannot preserve the copied native function, incoming gradients are absent, bands have zero usable rank or all output projectors collapse. Stop the graph-specific claim if unfiltered growth matches; stop the capacity claim if a capable single explains the result or member competence/cost fails. A positive result can motivate matched random and linear-activation attribution controls, without retroactively modifying the first fixed screen. The first local growth claim is known GradMax/conditional-message ancestry, not a new expressive ensemble theorem.

## Closed LayerNorm gauge

The simple native valid sites are predictor.lin ops1 to ops4.r and ops5 to ops8.r. Between each normalization and input factor are only Dropout/ReLU. Scale both private LayerNorm affine weight and bias by D_m>0 and divide the following r_m by D_m. Positive diagonal scaling commutes with those homogeneous operations for the same mask, preserving the real function. The terminal xlin/xcnlin/xijlin norms meet residual/additive mixing and do not admit the simple isolated compensation. FP32 bitwise identity is not implied.

For a fixed coordinate scaling z=t theta and reset Adam moments, zero decay and no clipping, the original-coordinate equivalent has learning rate eta/t and epsilon t epsilon. Thus LayerNorm coordinates t=D use eta/D and D epsilon; reciprocal-factor coordinates t=1/D use eta D and epsilon/D. Shared dense coordinates remain unchanged. Transformed carried moments, decay, clipping and numerical arithmetic require their own matching. This is exact coordinate preconditioning in real arithmetic, not new model capacity. Any quality test must compare the mapped optimizer directly; no new gauge strength grid is retained.

## Evidence and next gate

Four first scoped method reads were made, including one earlier metadata-only lead; zero full-paper or empirical-result certifications. The new public PyG2.7 WikiCS loader was read as software source, without importing it or downloading data. Previously read author Polynormer files were revisited only for the named WikiCS contract; those are not new paper identities. Four equation pages were rendered and inspected. All-page PDF extraction and keyword locators are mechanical. A short stdlib arithmetic check verifies the balanced-gradient qualifications, normalization counterexample, gauge/Adam equivalence, zero-growth derivatives and restricted nonlinear-capacity example.

No model/framework import, dataset/label/checkpoint/score access, server, scientific fit, new agent, index mutation or existing-source edit occurred. The two protocols are design artifacts. Root can assign a new source implementation after the structural gate review; existing running work and scores remain untouched.
