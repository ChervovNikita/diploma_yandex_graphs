# One inactive hypothesis for normalized neighbor action diversity

This is one optional, source-unimplemented test after the core and ordinary independent controls are competent. It is an attributed graph-evidence diversity adaptation. It supplies no novelty or predictive guarantee. The earlier saved operator-diversity and learnable-strength conclusions remain binding; no duplicate fit is justified if an existing comparison already answers the same question.

## Exact statistic and restricted update

Prospectively select 32 distinct nonisolated observed node indices by a label-blind fixed hash, then seal their indices and every original neighbor pair `(v,u)` with u in the selected set. Let S be these off-node pairs and B=|S|. Neither labels, feature values, trained factors, correctness nor outcomes select S. No indices are generated here.

Use only the first layer's **actual live normalized/clamped sparse operator** L_m from each complete member path. At evaluation it is conditioned on the same shared stem state; during training states/dropout/jitter can differ and their original draws must be preserved and recorded. For the node-basis probe E_u=delta_u⊗I_d, define

`e_m(v,u) = ||[L_m E_u]_v||_F² / d² = ||L_(m,v,u)||_F²/d²`,

`D_mn = (1/B) sum_((v,u) in S) [e_m(v,u)−e_n(v,u)]²`,

`R_action = (1/6) sum_(m<n) exp(−D_mn)` for M=4.

The current no-LP/HP d=4 scope and native entry clamp give e in [0,1] and D in [0,1]. Compute directly from the selected sparse blocks; no dense N×N matrix or actual dense probe array is necessary. B=0, missing blocks, nonfinite values or unqualified derivatives make the proposed arm unavailable; no replacement probe or epsilon is supplied.

One exact prospective update definition is

`g_theta=partial_theta F`,

`g_eta_m=partial_eta_m F + 0.1*partial_eta_m R_action`.

F is the unchanged mean-own full-input NLL. The coefficient 0.1 is one fixed proposed dose, not fitted or searched. Only private incidence factors receive the extra gradient; gradients through live map/normalization paths into those factors must remain intact. Shared stem/features/head/epsilon/incidence W receive F alone. This restricted recipient is not ordinary joint minimization of F+0.1 R in every parameter, and no implementation is admitted.

## What gauge invariance means here

Under an orthogonal block conjugacy of an already deployed operator,

`L'_(m,v,u)=Q_(m,v) L_(m,v,u) Q_(m,u)^T`,

the Frobenius energy e is unchanged for each pair. Thus R_action is invariant even when members use different orthogonal node frames *at the operator level*. The exact representable edge-row-sign symmetry leaves L unchanged and is also invisible to R_action. This proof compares deployed operators; it does not assume that arbitrary rotations through the actual tanh/SVD-jitter/clamp/ELU/shared-head network implement such a conjugacy or preserve its predictions.

The statistic is only an incomplete energy signature. Equal block energies can conceal different directions, spectra, cycle structure or classifier responses. It is not a holonomy or whole-predictor gauge test. Fixed-coordinate full vector differences `||L_m P−L_n P||` would be edge-basis invariant, but are generally **not** invariant to independently changed node frames with fixed probes; that stronger invariance must not be claimed for them.

## Failure mode

Minimizing R rewards different neighbor transfer energies. Those energies may grow apart on irrelevant/noisy edges while shared feature transforms suppress their action or the shared head ignores the affected state directions. With a zero right feature map, every actual message is zero even when R reports diverse operators. Wrong/noisy messages can also lower R while worsening own and pooled predictions. Stochastic stem/jitter differences can supply an apparent energy spread unrelated to useful learned private geometry.

When all descriptors coincide, the smooth pairwise-distance reward has zero first derivative wherever the native derivative exists. A deterministic identical unit bank therefore does not obtain an asymmetry from this loss alone. This is a property of the kernel-distance objective, not a continuous gauge inferred from the discrete sign witness. A raw row-sign initializer supplies no deployed asymmetry either. Saturated tanh and SVD/cutoff behavior further require native derivative qualification; zero-jitter evaluation SVD gradients are not silently substituted for the original training path.

## Minimal three arms

1. **Diversity off:** unchanged F and unit-factor core initialization; collect the same observation/probe outputs.
2. **Raw-map diversity:** identical architecture/initial state, own losses, recipients, data/update/selector/RNG opportunity and serving, with the same coefficient/kernel but

   `D_raw,mn = [1/(8 B d²)] sum_((v,u) in S) (||F_(m,v,e)−F_(n,v,e)||_F² + ||F_(m,u,e)−F_(n,u,e)||_F²)`,

   `R_raw=(1/6)sum_(m<n)exp(−D_raw,mn)`.

3. **Normalized-action diversity:** the exact R_action above.

Both distances lie in [0,1], which matches their nominal kernel range, not their gradient strength. Record actual auxiliary/private gradient magnitudes and paid normalization/backward work; do not tune dose after seeing quality. Raw-map repulsion can distinguish the exact row-sign witness even though all deployed messages agree; the action statistic cannot.

Use the same fixed source-feature identity intervention and classifier log-odds response report in all three arms, with its extra forwards charged. Native source, original data/role/selection policy and member competence safeguards remain fixed. No learned coefficient, matrix-distance cascade, favorable probe replacement or additional grid is proposed.

A useful result requires better original-input served quality with appropriate member safeguards and classifier-visible source-response differences. A higher action-diversity score alone is failure. Matching the raw-map/off arm, recovery by source-free or node/head capacity, or failure against competent independent/joint alternatives leaves no demonstrated geometry-specific advantage. Even survival does not establish causal mediation or a new principle; it supports only the declared exploratory placement/objective result.
