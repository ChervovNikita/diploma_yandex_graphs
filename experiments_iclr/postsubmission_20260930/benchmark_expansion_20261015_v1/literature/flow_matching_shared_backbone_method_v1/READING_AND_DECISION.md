# Flow matching with a shared native backbone: method reading and disposition

**Decision: defer a genuine predictive flow experiment; reject calling the existing CE-trained correction “flow matching.”** The current geometry-plus-correction design already addresses the stated goal of improving shared representations while retaining useful variation. A probability-path head is a distinct mechanism, but the papers read here do not show that it improves shared graph ensembles or resolves their shared-gradient interference. No experiment is added to the campaign by this note.

This is a bounded method reading, not a comprehensive novelty review or a full-paper reading count. The original abstract-only record is preserved. Source files, retrieval hashes, exact read scope and short passages are in this folder. No native run, dummy experiment, model fit, GPU action, or Git operation was performed.

## 1. Primary source and scope

### Flow Matching for Generative Modeling

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel and Matt Le. [arXiv:2210.02747v2, 8 February 2023](https://arxiv.org/pdf/2210.02747v2). Read PDF pages 2–6: Sections 2, 3, 3.1, 3.2 and 4/4.1 through the OT construction and its limitation. Theorems 1–3 and equations 1–24 were inspected in those passages; the appendix proofs and experimental results were not reviewed. Page 6 equations and the OT caveat were visually checked.

The method needs a probability path p_t and a velocity generating it. FM regresses a neural velocity v onto that target velocity, not merely onto a desirable endpoint. CFM constructs tractable per-target conditional paths and regresses their velocities. In this terminology, “conditional” can mean conditioning the training path on its endpoint; it does **not** mean the unknown target must be supplied to the learned field during inference.

Theorem 2 gives equality of FM and CFM parameter gradients for the stated fixed path and target distributions, up to a parameter-independent constant and under the positivity assumption. It is not a theorem that every supervised residual objective is FM, that a changing learned prototype path has the same gradient property, or that classification accuracy will improve.

For the paper's linear Gaussian path, with a=1-sigma_min:

    z_t = (1-a*t)*epsilon + t*x_1,
    epsilon ~ N(0,I), t ~ Uniform[0,1],
    target velocity u = x_1 - a*epsilon.

Training this loss requires no ODE trajectory. Generating a sample still requires integrating a learned field. “Simulation-free training” must not be reported as zero-cost inference. The paper also explicitly limits the OT statement: optimal conditional Gaussian transports do not establish that the marginal learned field is the global optimal transport solution.

### CARD: Classification and Regression Diffusion Models

Xizewen Han, Huangjie Zheng and Mingyuan Zhou. NeurIPS 2022; downloaded PDF identifies [arXiv:2206.07275v4, 6 December 2022](https://arxiv.org/pdf/2206.07275v4). Read PDF pages 2–4, Section 2 and its subsections, equations 1–10, Algorithms 1–2; also the first experimental-setup paragraph on page 5. No benchmark outcomes are used here. The inference algorithm and classification decoder on page 4 were visually checked.

This is a genuinely relevant discriminative precedent. CARD models p(y|x), using a pretrained conditional-mean estimator/classifier as a condition and as the center of the noised-response endpoint. For classification it treats one-hot labels as continuous class prototypes. TRAIN uses observed y; inference samples a response using x and the learned predictor, without the unknown label. This demonstrates an existing route from generative dynamics to prediction. It does not demonstrate that stochastic samples equal independently trained ensemble members or that the method transfers to native GNN ranking tasks. CARD's reported setup uses 1000 diffusion timesteps; that cost must not be silently transferred to a cheap four-step FM proposal or used to claim that four steps suffice.

### Contrastive Flow Matching: relevant exclusion, not classifier evidence

George Stoica, Vivek Ramanujan, Xiang Fan, Ali Farhadi, Ranjay Krishna and Judy Hoffman. [ICCV 2025, CVF primary paper](https://openaccess.thecvf.com/content/ICCV2025/html/Stoica_Contrastive_Flow_Matching_ICCV_2025_paper.html), DOI 10.1109/ICCV51701.2025.00118. Read PDF pages 3–4 (printed 1187–1188), Sections 3–4 and Algorithm 1, plus page 5's dataset/metric/sampler setup. Page 4 was visually checked. This was discovered by one bounded Crossref query. It is a conditional image-generation paper; its “discriminative” trajectories are not a discriminative graph-classifier result.

Its field is v(z,t,y), with the generation condition supplied. Section 4 adds a negative squared-distance term between the predicted flow and another example's target flow. The negative example's condition may be the **same** as the positive condition. The implementation uses a target already available in the batch, so it does not require another model forward. Evaluations are image-generation metrics on ImageNet/CC3M, not native classification error complementarity or shared-model ensemble quality. Giving the unknown graph/node label to such a field at inference is inadmissible.

For a fixed positive target u and negative target b, its sample objective satisfies:

    ||v-u||^2 - lambda*||v-b||^2
      = (1-lambda)*||v-(u-lambda*b)/(1-lambda)||^2 + constant,
      0 <= lambda < 1.

This is our algebraic observation, not a new theorem claimed by the paper. It exposes a shifted regression target. Adding it to a predictive flow changes the target relative to ordinary FM; the original fixed-path matching theorem cannot simply be reused for that altered objective. It also does not create independent backbone hypotheses. Its relevance here is to prevent a misleading transfer of “contrastive flow” into unsupported class/route repulsion.

Neural Ordinary Differential Equations was downloaded during discovery and its title/abstract inspected, but its method was not read for this note and it is not counted as a second discriminative method reading. No second eligible ensemble precedent was established in this bounded pass. Search misses or exclusions make no novelty claim.

## 2. Concrete hypothesis, defined so it can be rejected or revisited

**Hypothesis:** a small private field conditioned on each route's native representation could improve a shared bank's predictive distribution by learning a noisy path to label codes. This is a predictive readout experiment, not an input-feature transport experiment or an asserted new method.

### Scope and native ownership

Restrict the thought specification to one observed single-label classification task (categorical node classification or binary molecular classification). Do not automatically port it to link-query ranking, multitask missing-label targets, regression, or the original five datasets.

Keep one native backbone, all native operators, preprocessing, graph topology, source minibatches, optimizer settings and the existing four boundary routes. Each route produces its original native prehead h_m in R^H and logits a_m in R^C. For a binary native scalar logit a, the conditioning vector (-a/2,a/2) is an exact two-class representation of its original sigmoid/BCE. This conversion is only at the wrapper boundary.

A private, bias-free width-r field, with r=8 as an illustrative fixed design, is:

    v_m(t,z,h_m,a_m) = B_m GELU(A_m [z; t; h_m; a_m]),
    A_m: r x (H+2*C+1), B_m: C x r.

Assign A_m and B_m to the original prediction-head anchor's optimizer group without changing that group's settings or existing native parameters. Initialize A_m from isolated route RNG and B_m to zero; record that initial flow serving is therefore noisy, not equal to the native predictor. No neutral-function claim is made. A different native-preserving initialization would be another declared method change, not an unrecorded repair.

### TRAIN target, coupling, path and gradient ownership

Use isolated, checkpointed wrapper RNG for sampled times/noise, so these draws do not consume the native sampler/dropout streams. For each authenticated observed TRAIN record with class y, let c_y be its C-dimensional one-hot code, let sigma_min=0.01, and let a=0.99. Independently draw epsilon from N(0,I_C) and t uniformly from [0,1], then use:

    s(t) = 1-a*t,
    z_t = s(t)*epsilon + t*c_y,
    u_t = c_y - a*epsilon,
    L_CFM = mean_over_routes_records_coordinates ||v_m(t,z_t,h_m,a_m)-u_t||^2.

The label enters target/path construction only. The field receives z_t, t and features/logits obtainable from x; y is not a network condition. No VALID/TEST labels, class statistics, fitted prototypes, teacher model, cross-record nearest-neighbor search, or target cache are needed. The Gaussian-to-observed-code coupling is explicit. Terminal sigma_min>0 gives concentrated code distributions rather than an asserted smooth invertible map to exact point masses. The target/path themselves do not depend on the learned native parameters. If h/a are live conditions, their gradients arise through v, not through a moving label target.

For a concrete accuracy-oriented comparison, retain the original mean native task loss F_native and train the served solver output as well:

    J = 0.5*F_native + 0.5*F_served + 0.05*L_CFM.

Both task losses use the same original observed-label mask and reduction. All declared shared/private parameters receive this one scalar's gradients. The coefficient, rank, terminal noise and solver count above are **illustrative fixed hypothesis choices**, not literature-derived optimal values or an admitted launch preset. No schedule, launch descriptor or pilot is created.

Only L_CFM is simulation-free. F_served differentiates the four-step solver, so this combined training procedure cannot claim the full simulation-free property of the primary paper.

### Inference without labels

Compute h_m/a_m with the existing native route once. Draw one reproducible Gaussian epsilon_m per route using a documented evaluation RNG keyed by record identity and route, independent of labels and selection. Set z_0=epsilon_m and run four fixed Euler steps:

    z_(k+1) = z_k + 0.25*v_m(k/4,z_k,h_m,a_m), k=0,1,2,3.

Use mean route endpoint logits followed by the native task's output transform (binary score endpoint[class 1]-endpoint[class 0] or categorical softmax). This changes the serving head; it is not the original paper's predictor or CARD's decoder. VALID uses the fixed declared RNG and whole native metric; resampling cannot be used to choose a favorable score. More samples or a different solver would be a new cost/selection contract.

A class-code flow is not automatically calibrated. In particular, averaging endpoint codes and applying softmax is not an identity for p(y|x). A stochastic predictive sample is also not a draw of independently fitted backbone parameters. The candidate needs an explicitly chosen prediction-quality or uncertainty question before it has scientific value.

## 3. What the flow objective adds, and what it does not

For the exact Gaussian path above, one can reconstruct the endpoint code from a velocity:

    c_hat = a*z_t + s(t)*v_m(t,z_t,h_m,a_m).

Substituting the sampled z_t and u_t gives **exactly**:

    ||c_hat-c_y||^2 = s(t)^2 * ||v_m-u_t||^2.

Thus this CFM auxiliary is an endpoint denoising/prototype-regression auxiliary with a particular time weight, 1/s(t)^2. It supplies no label information beyond the observed TRAIN target. This does not make the multi-time learned dynamics identical to a one-step static adapter; it does make ordinary endpoint reconstruction an essential control and rules out claiming a new supervision signal solely from the FM name.

For an even simpler latent proposal with frozen h and class prototype mu_y, a straight path z_t=(1-t)h+t*mu_y and a single Euler step at t=0 gives:

    ||v(h,0)-(mu_y-h)||^2 = ||h+v(h,0)-mu_y||^2.

That case is exactly residual prototype regression. Arbitrary class prototypes can compress useful within-class structure, which the existing variance-preservation design explicitly tries to retain. If h/prototypes are jointly learned, one must separately specify stop-gradients and changing targets before appealing to the fixed-path FM theory. Calling a CE-only residual update a “flow” does not settle any of these issues.

All four fields above train toward the same conditional labels. Different route initialization/noise does not itself create complementary errors, resolve shared-parameter conflicts or guarantee stronger members. The same field can generate multiple stochastic samples; independent parameter uncertainty is a separate question.

## 4. Matched comparisons that would be needed if this were reopened

| Comparison | What it isolates | Matching and remaining difference |
|---|---|---|
| Existing native GNNM and its existing geometry-plus-input-correction counterpart | Whether the new head improves the actual campaign question | Same native source, split, route count, selection metric and stopping contract; field capacity/work and placement differ and must be reported |
| Same private field, same four Euler steps, same noise and task losses, no auxiliary | Whether FM supervision helps beyond recurrent residual correction trained by task loss | Exact trainable architecture/inference and native calls; 4 instead of 5 field evaluations per route in TRAIN. Do not add a dummy call to hide that difference |
| Same private field/solver/task losses plus unweighted endpoint loss ||a*z_t+s*v-c_y||^2 | Whether FM's time weighting helps beyond ordinary endpoint denoising | Same one additional sampled time, path, field evaluation, parameters and native work: 5 TRAIN field evaluations per route; difference is the s(t)^2 weighting, not extra labels |
| Same-width one shared field applied to the same four native route contexts/noise samples | Whether four private fields help beyond stochastic sampling of one function | Same per-call field width and number of calls; one quarter of field parameters. A width-32 shared field can match parameter count but costs four times the field MACs, so it is a separate capacity comparison, not a simultaneous compute match |
| Existing genuinely independent native I4 reference | Whether the shared method approaches separately fitted hypotheses | Separate fits/selectors remain charged; stochastic flow samples are not relabeled independent models |

No run of these comparisons is proposed now. If reconsidered, freeze the exact target encoding/decoder, solver, evaluation RNG, selector, stopping contract and compute accounting first. Measure actual selected predictions, member competence and error complementarity; geometry width or flow separation alone is not a success criterion. A predictive-uncertainty study needs proper scoring/calibration/coverage targets and an independent confirmation design, rather than borrowing image-generation FID or the present exploratory VALID curve.

## 5. Cost, without invented measurements

With M=4 routes, field hidden width r, C class-code dimensions and native prehead width H, the added field parameters are:

    P = M*r*(H+3*C+1).

For H=64,C=2,r=8 this is 2,272 parameters; for H=300,C=2,r=8 it is 9,824. These are arithmetic counts, not runtime measurements or a match to the existing input residual's different parameter count/placement.

Each observed minibatch needs the original four native route forwards; h/a are reused for all field calls. The combined proposed J uses 4 solver evaluations plus 1 sampled CFM evaluation per route, **20 small-field evaluations total in TRAIN**, and one scalar reverse traversal through the combined graph. Serving uses **16 field evaluations** plus the original four native route forwards for one sample per route. More S samples per route costs 16*S field evaluations, while native representations can be reused. Training activation storage includes the four-step field trajectory; there is no constant-memory adjoint implementation in this proposal.

A dense field evaluation costs approximately B*r*(H+3*C+1) multiply-accumulates for B records, excluding nonlinearities, concatenation, output transforms and backward. The train/serve field MAC totals are therefore approximately 20 and 16 times that expression. No additional native graph/message-passing forward is required by this definition. End-to-end time, GPU memory, convergence horizon and adequate Euler accuracy remain unmeasured. No numeric speedup or epoch count is inferred from the generative papers.

## 6. Relationship to the current implementation and final disposition

The inspected existing private correction is `portable_private_input_v2/core.py`: one native body, four nonlinear rank-cap-8 residuals, zero initial added output, isolated route RNG, no velocity/path/time/noise target. `interaction_node_molecular_v1/METHODS.json` defines geometry+input as native F+0.05*(alignment+variance). `task_geometry_node_port_v1/objective.py` uses authenticated TRAIN class alignment across route preheads plus a class-conditional variance floor, with no extra native forward and no route repulsion. The separate molecular alignment-only control already distinguishes the variance contribution. Exact local source receipts accompany this note.

These implementations cover supervised shaping and preservation of useful within-class variation together with private nonlinear correction. They do not implement probability-path matching, a time-dependent solver, or conditional label sampling. Adding such mechanisms would therefore be a real change, but the intended accuracy/complementarity benefit is presently unsupported and overlaps the existing goal while adding a new serving contract and solver costs.

**Do not allocate a new native pilot for this lead now.** Retain the precise class-code hypothesis as deferred if the user later prioritizes conditional predictive-distribution modeling and accepts the specified controls/costs. Reject the one-step prototype/CE-residual relabeling as a new FM experiment, and reject transplanting the image generator's true-label condition or trajectory-repulsion claims into inference. This is a scoped mechanism/cost decision, not a proof that flow matching cannot help graph prediction and not a novelty conclusion from search coverage.
