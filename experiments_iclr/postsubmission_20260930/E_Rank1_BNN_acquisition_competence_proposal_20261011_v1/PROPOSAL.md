# One attributed Rank-1 BNN acquisition comparison for E competence

Status: **SOURCE-LEVEL PROPOSAL ONLY; no execution or automatic continuation.** This instantiates the saved unexecuted `conditional_graph_predictive_family_scout_20261010_v1` utility question on E. It is not a renamed method, a novelty claim or a faithful reproduction of the Rank-1 BNN paper. The existing full studies and manuscript remain unchanged. No teachers, pretrained independent bank, warm donor or student retraining are required. Genuine I4 procedures are independently fitted controls within the comparison.

## Evidence and one failure mechanism

The current evidence map reports that E improves BASE through stronger members and better retention, yet its mean member accuracy remains 0.763178 percentage points below capable equipped I4, with worse member/pooled proper losses. N4 accounts for most of its BASE increment. The exact extra affine-head recipe failed; source-context reassignment establishes reliance of selected functions, not a training cause. Those results motivate acquisition work, not another geometry or decoder capacity claim.

The TRAIN diagnostic source was read before proposing an update. It restores all 15 banks/24 states, takes 51 actual next-dropout own-loss backwards, and compares only shared coordinates within each bank. Root subsequently reports its complete closure in 71.40 seconds. Whole-shared cancellation ratios vary: BASE .270802/.258091/.260580, E .301913/.197579/.261446, N .287516/.252188/.232216; E has 2/5/1 negative pairs of 6. E's sampled TRAIN CE is roughly .16–.22 versus BASE roughly .002–.031, despite better complete E development quality. Selected states, dropout and capacity differ. This is not uniform overwhelming cancellation, an AdamW displacement measurement, a training-history estimate or causal overfitting evidence. No gradient-surgery rescue follows.

**Hypothesis:** fitting local uncertainty around E's existing fast-factor means can yield more competent mean-factor predictors and a better retained probability pool than point acquisition and fixed factor noise. Expected own-loss exposure to perturbations may discourage fragile sharp factor configurations; learning scales may retain useful variation without imposing the same noise everywhere. This is a conjecture about acquisition robustness, not an established explanation of the competence gap. The direct alternative is ordinary noise/anchor regularization; the fixed-scale control decides that distinction.

## Exact prospective acquisition delta

Keep original WikiCS features/edges/TRAIN 580/VALID 5274, native SAGE width 128 / depth 2 / dropout .2, endpoint VALUE maps, residuals, normalization, native head and all current FactorLinear sites. Keep AdamW lr .001 / zero decay, at most 1,000 updates, patience 300, original dropout streams and earliest strict maximum full VALID accuracy. No labels enter factors/builders; there is no pool CE, auxiliary loss, routing, clipping, teacher or partial sharing.

Let mu_m,t be each original registered fast-factor vector at site t, and introduce rho_m,t=log sigma_m,t. For one complete route draw, use

`phi_m,t = mu_m,t + exp(rho_m,t) * epsilon_m,t`, with independent standard-normal epsilon by site/vector and the same sampled vector reused across every node and occurrence of that site in that route. Route/mode identity m persists across all layers. Original live endpoint states still condition E maps. Never assign independent per-node factors or mutate registered means to install a draw.

Use one Gaussian draw per route per update and

`J = mean_m [ CE_TRAIN,m(phi_m) + KL(q_m || p_m)/580 ]`.

CE is the original mean over all 580 TRAIN items. q_m is the product diagonal Gaussian over this route's existing fast coordinates. Define an explicit augmented model with uniform component index m; p(phi|m) has independent Gaussian coordinates centered at that component's original **data-free native factor initialization**, with standard deviation .1. q starts at those same means and standard deviation .1, hence initial conditional KL is zero. First-stem Rademacher centres remain signed as originally initialized. Prior centres are immutable external training constants, not new trainable/mutable model buffers. The exact average conditional KL equals the joint-index KL; it is not claimed to equal the marginal mixture KL. Do not silently replace it with a log-mixture likelihood.

The prior scale .1, initial scale .1 and unit KL/580 normalization are prospective fixed port choices, not author defaults, tuned settings or validated values. No annealing, prior/scale/rate sweep, learned prior or retry is proposed. Keeping native zero weight decay and dropout is a declared adaptation. The initial stochastic function differs from the point bank; equal initial means do not imply equal training forwards.

All original shared slow weights remain one live copy. Original r/s names become means; trainable r_log_std/s_log_std rows accompany them only in the learned-distribution procedures. One native AdamW owns every trainable parameter exactly once, retaining mean-own scaling for shared banks and unscaled own loss for singles/I4. The small functional contract in `acquisition_delta.py` implements only draws, conditional Gaussian KL and reduction. It supplies no runnable acquisition engine or qualified native adapter.

Separate factor RNG streams must be saved/restored strictly with model/optimizer/native dropout streams. Prospectively use factor seed `seed+5000011+1000003*route`; stable sorted registered-site order determines draws. Mean evaluation consumes no factor RNG. Selected-state diagnostics use separate seed `seed+6000011`. No sampled bank is selected for favorable quality.

## One complete representative roster

Use the complete encountered WikiCS graph and original paired optimizer seeds 7301/7403/7507. All procedures are acquired from scratch in one fixed comparison; no control is pretrained to generate targets.

| Procedure | One-seed optimizer fits | Complete TRAIN routes/update ceiling | Purpose |
| --- | ---: | ---: | --- |
| E_point4 |1 |4 | Unchanged original E acquisition/serving anchor |
| E_fixed_noise4 |1 |4 | Same Gaussian/conditional KL law, rho frozen at log(.1); separates learning scales from fixed noise/anchoring |
| E_Rank1_q4 |1 |4 | Candidate: four posterior components and one shared slow body |
| E_point_M1 |1 |1 | Known capable single anchor with the same endpoint/factor opportunity |
| E_point_I4 |4 |4 | Four separately initialized, optimized and own-selected complete point bodies |
| E_Rank1_q_M1 |1 |1 | Same Gaussian/conditional KL opportunity in a capable single body |
| E_Rank1_q_I4 |4 |4 | Four genuinely independent complete stochastic bodies, each with one component |

Worst-case fresh roster: **21 banks / 39 optimizer fits / 39 selected states**. Point controls must retain original native initialization and full selectors. Stochastic M1/I4 use the same prior/scale law and native factor slice as the candidate; I4 has four separate slow bodies and optimizers. Neither a weak stochastic single nor four copied states is a capable reference. Retain point M1/I4 even if a stochastic port weakens them. No seventh seed, selected subset or follow-on panel is available.

## Selector, serving and what can be attributed

Use **mean factors for the original selector and primary serving in every arm**, keeping the original own-body selector for independent fits and probability mean for banks. This deliberately isolates the effect of stochastic acquisition on fitted means and preserves the original deterministic selector/serving budget. It differs from native Rank-1 BNN posterior-predictive deployment. A successful primary contrast supports acquisition robustness; it does not establish the value of predictive integration or Bayesian uncertainty calibration.

After strict restoration, perform one additional declared raw sampled-versus-mean readout for each Gaussian procedure at the same selected state, with dropout off. Use 12 complete graph forwards: three draws per component/body for four-component banks/I4, twelve for q_M1. Average all 12 probability vectors. Retain the three fixed four-forward blocks to disclose Monte Carlo variation; the all 12 mean is the only sampled readout, with no best-block choice. This readout adds no fit/selection/refit and is secondary. Its12-forward cost must be disclosed against the primary 4/1/4-forward procedures. Component means, individual draws and integrated component probabilities remain distinctly labeled.

All 21 primary selected VALID archives receive the unchanged five-fold seed 11709 CPU-float64 Adam calibration (500 updates, lr .01, original moments/epsilon, zero decay, logT 0, fixed endpoint): **105 scalar attempts /52,500 requested updates**. Preserve all failures before interpretation. Sampled diagnostics remain raw; no extra calibration panel or final VALID refit is added. No TEST or unused-confirmation claim follows from this encountered graph.

The specific competence hypothesis requires improved mean-factor mean/worst member accuracy and proper loss beyond both E_point4 and E_fixed_noise4, with whole-population served accuracy/proper-loss improvement and repairs exceeding introduced harms. A mean-member gain of at least .2 percentage points is a prospective practical target, not a significance threshold. A larger spread, lower TRAIN loss, nonzero sigma, common-rival removal or more coverage cannot rescue failure of served quality. Report every paired seed; mixed signs remain mixed evidence.

Capable point and stochastic M1/I4 decide broader utility. If a single or independent counterpart explains the gain, report ordinary stochastic regularization without claiming a sharing-specific benefit. If E improves but retains the I4 competence/confidence gap, report the remaining gap. If fixed noise matches the candidate, learned local uncertainty is unnecessary under this recipe. If sampled deployment matches means, integration adds no support. If the specified candidate fails, close it without a sigma/prior/annealing/temperature or gradient-surgery rescue grid. No component/member/population is omitted.

## Closest primary scope and overlap

[**Efficient and Scalable Bayesian Neural Nets with Rank-1 Factors**](https://arxiv.org/abs/2005.07186), saved §§3.1–3.5, is the direct method source: deterministic shared W, stochastic rank-1 factors, variational learning, nondegenerate/multimodal components and average-versus-log-mixture likelihood. The retained carrier is unversioned ar5iv, so its precise revision/latest status is unqualified. The narrow revisit here concerned factor priors, nondegenerate components and average own likelihood; no published gains or derivations were imported as project evidence. [BatchEnsemble](https://arxiv.org/abs/2002.06715) is the point-factor ancestry. Gaussian perturbation/dropout regularization explains the fixed-scale alternative.

E's native operator still has ECC/GNN-FiLM/NSD ancestry. The newly saved BSNNv2/dependence synthesis remains intact and is not reopened. This proposal changes factor acquisition, not the graph-wide-context primitive. The local saved collision is exact: the 10 October stochastic-factor scout already retained graph-coherent Rank-1 BNN, mean-versus-sampled evaluation and capable stochastic/point references. This packet supplies one finite E placement and variance-learning control, not a new methodological identity. Prior gradient-balancing, AdaTask/GEM and teacher-compression conclusions remain unchanged.

## Work, storage and expected harm

At 1,000 updates per optimizer fit, the complete roster has at most39,000 optimizer transitions, **66,000 TRAIN route trajectories and 66,000 original mean-selector VALID trajectories**. Each trajectory uses both native graph layers. Factor draws and KL are linear in the fast-factor count and add no separate graph/backward pass. Selected primary serving adds 66 trajectories; sampled readouts add144. Calibration adds 105 fixed scalar fits. Saved prior owner metadata records 4,315.771 seconds for the completed original 15-bank / 24-fit E family, whose ceiling was51,000 TRAIN and51,000 selector routes; its actual stopping/counts and contention differ. A deliberately coarse **2–6 GPU-hour** acquisition estimate plus **10–20 CPU minutes** for complete readout is a planning range, not an ETA or measured qualification. Root must separately bind finite resources before any execution; this packet reserves none.

Original E has 221,642 slow parameters and 3,958 fast parameters per route:237,474 shared4,225,600 M1 and902,400 genuine pointI4. Adding one log scale per fast coordinate gives 253,306 q4,229,558 q_M1 and918,232 q_I4. Full scientific states/optimizer and posterior deployment retain scales. A primary mean-only export could omit them only after separate exact source/export qualification; no storage saving is assumed here. Full graph/edge activations and four live training routes remain. New factor noise may enlarge neither tensor shapes nor nominal route count, but actual memory/time still needs one full-input resource qualification.

Likely harms are loss of competence from noise, shrinkage toward data-free centres, overregularization on only 580 labels, confidence deterioration, redundant components, poorly estimated scales, numerical overflow and later stopping. Gaussian perturbations may weaken nonlinear endpoint actions even though the initial zero builders retain identity. Shared-weight capacity is unchanged, so sampling cannot guarantee recovery of information or capacity missing from all routes. The fixed-noise and capable same-information controls make those harms visible.

## Preparation limits

Read pending diagnostic PLAN/source, then current evidence map and saved competence/acquisition conclusions. Root supplied the diagnostic's completed bounded interpretation during preparation; only its closure fields are independently read here. The reference operator is AST parsed/compiled with stdlib and never imported/run. No arrays, checkpoints, current Physics/Penn scores, server, numerical import, model fit, reviewer, manuscript edit or existing sealed-source change occurred. Root owns later source integration, admission, budget, dispatch and interpretation; completion of current work does not auto-activate this proposal.
