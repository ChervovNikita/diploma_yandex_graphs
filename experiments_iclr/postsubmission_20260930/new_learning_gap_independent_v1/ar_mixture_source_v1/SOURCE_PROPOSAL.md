# Autoregressive feasible-mixture source proposal v1

Status: **unexecuted source only**. No imports, source compilation, numerical routines, models, samples, fits or data generation were run. The sealed `finite_cache_source_v1` packet is unchanged. This proposal changes the source GNNM factor placement and target law; it supplies no novelty, performance or resource-admission claim.

## Architecture and source route

The caller prospectively constructs one fresh authorized source `Model` with output dimension `d`, dropout exactly zero throughout, and audited node-wise LayerNorm or Identity. `make_static_source_encoder` calls that complete source model **once** and returns static node embeddings `H` for the entire supplied graph. All branches and all generation/density trajectories in an objective batch reuse this same H. There are no private GNN message trajectories, random fields or source BE input/output blocks in this encoder.

The source constructor, common initialization, source-graph/conflict-graph binding, dimensions and deterministic module/kernel audit are external required inputs. The wrapper checks the known source version identity, exposed normalization/dropout properties and output shapes; duck-typed injected objects and caller-supplied provenance do not constitute a completed source audit. Training gradients from all generating scores accumulate through the one shared encoder graph. A new encoder forward or optimizer step starts a new declared objective batch.

The factored decoder uses common matrices `W_x:[3h,p]`, `W_h:[3h,h]`, `W_o:[2,h]`, where `p=d+2`. For branch m it supplies private diagonal factors `r_x,s_x,r_h,s_h,r_o,s_o`, private gate/output biases `b_x,b_h,b_o`, a private initial hidden vector `h0_m` and a private recurrent state for every sampled/replayed prefix. Every parameter tensor is initialized externally after admission; this packet freezes no dimensions, initialization, mixture weights or training choices.

At the ordered node v, input `x=[H_v, previous_label, blocked_flag]` uses a zero token before the first node. The explicit GRU-style equations are:

`g_x=(W_x (r_x_m * x)) * s_x_m + b_x_m`

`g_h=(W_h (r_h_m * h_previous)) * s_h_m + b_h_m`

`reset=sigmoid(g_x_reset+g_h_reset)`

`update=sigmoid(g_x_update+g_h_update)`

`candidate=tanh(g_x_candidate+reset*g_h_candidate)`

`h=(1-update)*candidate+update*h_previous`

`logits=(W_o (r_o_m*h))*s_o_m+b_o_m`.

This fixes a readable proposed GRU convention, with the reset applied after the recurrent candidate affine map, as in the corresponding GRU-style formulation. It is a new source prototype over the authorized encoder; it does not preserve the original GNNM's complete private nonlinear graph trajectories or original factor locations. Recurrent BatchEnsemble and variational mixture/boosting ideas are established prior work. The closest-control literature report is maintained separately at `gine_closest_control_v1/structured_generation_gap_v1/REPORT.md`; this packet does not expand that literature review.

## Complete normalized feasible law

The caller supplies the **entire** simple undirected conflict graph, including isolates, and one recorded node permutation, independent of theta. No arbitrary node IDs, coordinates, order generator or fugacity-feature choices are added by this packet. Such choices affect the scientific task and must be declared prospectively.

An emitted active neighbor forces the current node inactive. At a forced step the probabilities are `(1,0)` and the allowed log density is exactly zero. At a free step the two logits have an ordinary binary log-softmax. The hidden state still advances at every step, including forced steps. Thus each branch has the autoregressive law

`q_m(y | G,order)=product_i q_m(y_order[i] | y_order[:i],H,mask_i)`.

Each conditional is normalized. In exact arithmetic with finite parameters, every complete independent set has positive probability under every branch, and every infeasible complete assignment has probability zero. Summing the chain over its allowed decision tree gives total mass one. The common support is the graph's independent-set space `Omega_G`, determined by supplied graph edges and independent of theta. The mask neither tests a learned score nor filters a completed sample.

`generate` accepts a caller-selected branch and exactly N independent uniforms in `[0,1)`, in the recorded order. It consumes the supplied draw slot even when the step is forced; it creates no RNG. Mixture branch selection from fixed pi or stratified branch allocation is also external. `log_density` replays **any supplied complete feasible assignment** under any branch, whether or not that branch generated it. Infeasible assignments return negative infinity. There is no generation-time re-encoding, rejection loop, best-of-samples choice or hidden field refresh.

**Future-lock clamping is not conditioning.** This source has no arbitrary-lock conditional generation interface. A future active lock generally changes probabilities of earlier prefixes, requiring exact inference or a qualified conditional sampler. Teacher-forced full density remains available for such a future inference method; this does not solve its posterior sampling problem.

## Fixed-mixture reverse KL estimator

For immutable supplied positive mixture masses, `FixedMixture` defines fixed normalized pi. The supplied positive fugacities are known theta-independent task inputs. On feasible assignments,

`log gamma(y)=sum_v y_v log lambda_v`, and `p_gamma(y)=gamma(y)/Z_G`.

Define `q_mix(y)=sum_m pi_m q_m(y)`. The objective computed without the theta-independent normalizer is

`F_mix=sum_m pi_m E_(y~q_m)[log q_mix(y)-log gamma(y)]`.

It equals `KL(q_mix || p_gamma)-log Z_G`. The packet supplies no estimate of Z, and does not call the reported F value a normalized KL value.

For every generated sample the full cross-member density matrix contains **all M exact teacher-forced full densities**, including the generating branch. The reward uses `logsumexp(log pi + log q_m(y))`. There is no leave-one-out mixture density. Differentiating F gives a score term plus the explicit density-derivative term

`sum_y q_mix(y) grad log q_mix(y)=sum_y grad q_mix(y)=grad 1=0`.

This cancellation uses normalized common theta-independent support, fixed pi and fixed gamma. It is not a pathwise approximation. The implemented minimization surrogate is

`sum_s w_s stop_gradient(log q_mix(y_s)-log gamma(y_s)-b_generating_s) * log q_generating_s(y_s)`.

Only generating-branch scores require autograd. All other cross densities are computed under the injected backend's `no_grad`; the full reward is detached. Shared encoder and matrix parameters receive their gradient through generating scores across the declared sample allocation. No derivative is taken through sampled binary labels. Trainable pi, theta-dependent masks/orders/fugacities, unnormalized branch laws or clamped-label proposals would require a different derivation and are outside this API.

The API accepts either caller-certified iid mixture samples, weighted `1/K`, or caller-designed stratified samples from every branch, weighted `pi_m/S_m`. The caller supplies the independent uniforms/branch draws; actual independence and distribution cannot be verified from supplied numbers. An `IndependentBaseline` is a graph/branch-specific finite value fixed before the current batch, with a predeclared-constant or previous-independent-batch provenance declaration. An own-sample-including batch mean is forbidden. This contract requires a real external audit; a declaration does not mathematically prove independence. Its subtraction has zero expected score contribution when the contract holds.

The **mean-branch reverse KL** control changes only the detached reward density to `log q_generating(y)`. The full matrix, samples and trajectory/density work stay the same in this objective ablation. Its objective is `sum_m pi_m KL(q_m || p_gamma)` up to `-log Z`. The difference from mixture KL is `sum_m pi_m KL(q_m || q_mix)`, a nonnegative mixture information term, so these are distinct learning objectives.

## Reviewable APIs

| File/API | Future operation |
|---|---|
| `domain.py: ScheduleDomain` | Full graph, original node indices and complete fixed order; deterministic hard-core masks and assignment checks. |
| `FixedMixture`, `IndependentBaseline` | Fixed positive normalized masses; externally certified current-batch-independent baseline provenance. |
| `source_encoder.py: make_static_source_encoder`, `EncodedGraph` | Injected ordinary source Model, one full static encoding and common H; dropout zero and node-wise normalization checks. |
| `ar_decoder.py: make_recurrent_decoder` | Same generation/full-density interface for factored, single and untied decoder arms. All parameter tensors supplied externally. |
| `generate`, `log_density` | Unconditional branch schedule generation from supplied uniforms; exact teacher-forced full density for any feasible complete assignment. |
| `cross_member_log_densities` | Full `[K,M]` densities, with autograd only for generating scores. The generating density is recomputed rather than reused from sampling. |
| `mixture_objective.py: fixed_mixture_reverse_kl_surrogate` | Explicit iid/stratified sample weighting, full mixture reward or work-matched mean-branch reward, detached valid baseline and score surrogate. |
| `work_accounting.py: EncoderAudit`, `WorkLedger` | Audited encoder work and symbolic decoder/cross-matrix counters; no measured timing or memory claim. |

The packet contains no training loop, optimizer, experiment configuration, task generator, executable example, exact oracle or sampler test. Qualification against a small complete hard-core reference can use the separately sealed reference packet after prospective execution admission.

## Controls and unresolved scientific choices

`single` is a conventional one-decoder control (`M=1`) with ordinary shared matrices, private biases/h0 and no diagonal factors. `untied` gives M conventional decoders with private dense matrices, biases and h0, over the same one shared graph encoder. These reuse the same mask/order/sampling/density API. Parameter, initialization and resource matching are **not established** by this API; source tensor construction, widths, shared initialization and independent storage/alias checks are root qualification responsibilities. The single law's natural density cost is lower, and a declared resource comparison must account for that.

Required controls include the same factored architecture with mean-branch reverse KL; matched single and untied recurrent decoders; and appropriate exact/BP/Glauber, faithful structured-generator/variational and order-sensitive controls. Strong-control implementations and exact task/outcome freeze remain outside this minimal source packet. Reverse KL can omit important modes despite full feasible support; feasible generation and normalization do not establish accurate marginals, mixing, conditional inference or scientific benefit. The proposed graph/schedule/configuration uncertainty remains distinct from original node-classification parameter uncertainty.

An **unimplemented exact density-sharing variant** would use a fully shared recurrent transition/h0 and place private factors only in output heads. For the same teacher-forced complete y, graph/order and deterministic transition, every branch would then have identical hidden prefix states: one recurrent density trajectory could serve M heads exactly. During generation, sampled branch prefixes still differ and require their own states. That variant can distinguish the benefit of private transitions from density reuse. It would change the present architecture and requires its own controls and prior-art audit; no such reuse is implemented or claimed here.

Full task/config/outcome/resource choices, all tensor/mixture/baseline provenance, uniform sampling/branch allocation, graph splits, fugacity regimes, orders, precision, optimizer, sample counts, controls and admission gates must be frozen prospectively by the root. Existing phase/diagnostic caps, failed work and the one-A100 limit remain binding. This source packet grants no execution or training permission.
