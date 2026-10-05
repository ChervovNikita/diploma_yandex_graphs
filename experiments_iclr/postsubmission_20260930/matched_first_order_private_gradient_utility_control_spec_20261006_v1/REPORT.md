# Matched first-order private-gradient utility control

6 October 2026. **Prospective specification only; the frozen six-arm pilot is unchanged.** No implementation, numerical import/run, data/checkpoint/scientific-output access, SSH, fitting, manuscript edit or new agent is supplied. This is an attributed control, with no novelty claim.

## The one substitution

The exact operator allocates by a finite **softplus class-competitor margin response to an own-CE private step**; finite CE change is diagnostic. For member m, original private row phi_m and shared theta, define

```text
f_m = mean own_CE over S
b_pim = softplus(z_m[i, competitor_p(i)] − z_m[i, target(i)])
g_m = partial_phi f_m
c_FIN_pim = b_pim(theta, phi_m − eta_probe g_m) − b_pim(theta, phi_m)
c_FO_pim  = −eta_probe <partial_phi b_pim(theta,phi_m), g_m>
```

Use eta_probe=.01 and the Euclidean dot product over every original private R/S/B coordinate, with exact zeros for unused local-head coordinates. S supplies the same all-class mean CE. Preserve both directions of each unordered class pair. Lower signed cost means larger predicted improvement. Do not turn the dot product into cosine similarity, rectify it, normalize either gradient, sum member coordinates into a common expert template, include shared coordinates or use Adam/preconditioned steps. Preserve eta: the fixed epsilon makes its cancellation under normalization inexact.

The proposed distinct arm is `first_order_utility_live`. Replace only the raw allocation cost. Apply the unchanged item/member centering, one pair RMS scale sqrt(epsilon²+mean(centered_cost²)) with epsilon=.001, entropy1, native nonnegative affinity/Laplacian, exact balanced row/member masses and the same eight finite solver steps from uniform Q. Use the same own-CE-plus-extra-margin main loss, coefficient .1 and M/[|S|(C−1)] scaling; Q remains an independent argument for the private partial. Retain the same .01 private step, .001 shared step, disjoint R own/probability-pool CE (.5/.5), original-phi recomputation at theta+, episode detaches and served arithmetic mean of four softmax probabilities. No route is selected at serving.

The first-order cost is the directional derivative of the **whole softplus margin**, equivalently sigmoid(gap) times the directional logit-gap change. Evaluating softplus at linearly perturbed logits is a different cost. Locally smooth, bounded-Hessian segments give c_FIN=c_FO+O(eta_probe²). Native branch crossings or a base kink do not inherit that uniform remainder statement; preserve the previous nonsmooth/coarse-FD findings. The normalization/map can amplify or suppress the remainder, so any predictive difference belongs to the complete substituted learning map.

## Keep the total shared derivative

The original direct main-adapt/query VJP and joint all-member query cotangents remain. Let t_pim be the raw-cost cotangent after differentiating the complete centering/RMS/eight-step map, and hold it fixed in the member cost-credit VJP. Define h_m=partial_phi sum_p,i t_pim b_pim. The replacement credit is

```text
(D_theta c_FO)^T t
  = −eta_probe [ (D_theta h_m)^T g_m + (D_theta g_m)^T h_m ].
```

Both terms are required. Differentiate through the own-CE gradient and weighted-margin private gradient; no stop-gradient is inserted in this live credit helper. Thus “first order” describes the probe-step approximation, not a Hessian-free outer update. It introduces mixed margin and CE Hessian products, while retaining direct mixed credit through the main private SGD. Inspection-only outer-phi credit analogously contains the two private Hessian products and never replaces the private SGD commit. At theta+, recompute g, utility, Q and the main response from the **original** phi, not a cached probe/virtual state.

## Transparent reference schedule and cost

Retain the original paid before/probe-after value passes and actual finite response/CE diagnostics in both comparison arms. Log utility costs separately; never relabel the utility as an observed finite response. This makes diagnostics comparable. In each value bank, reuse the own-CE forward/gradient and compute all utility margins by a matrix-free reverse-over-reverse directional product before releasing that native graph. For margin vector b and independent dummy cotangent a: p=partial_phi<a,b>; then partial_a<p,g_detached>=D_phi b[g]. This needs no item-by-parameter Jacobian. Value detaches are repaired by the explicit live credit above.

The utility credit helper uses one original-state native forward shared by own CE and the t-weighted margins, two differentiable private partials, then one theta/private VJP of their dot product. It replaces the separate negative-before and probe-after finite cost-credit helpers. Graph reuse, native AD support and memory are future qualification obligations.

| Complete M4 live episode, proposed schedule | Finite response | First-order utility |
|---|---:|---:|
| Complete native forward callbacks |48|40|
| Own-CE/main private partials |24|24|
| Direct/cost native member VJPs |12|8|
| Additional utility reverse constructions |0|20|
| Total native reverse constructions, excluding small maps/query |36|52|
| Pair primal maps / map VJPs / small query VJPs |30 /10 /1|30 /10 /1|

The20 new constructions are eight dummy margin-phi VJPs, eight dummy-cotangent reverses across the two value banks, and four weighted-margin private partials for credit. A single H16 utility continuation prospectively costs640 callbacks,384 own/main partials,128 direct/cost VJPs and320 utility reverses, with480 primal maps/160 map VJPs/16 query VJPs. Construction counts do not establish FLOPs or elapsed cost. Forward AD could implement a different exact schedule, but its native support and bill must be qualified separately. No cheaper-cost or memory-success claim follows. Warm16, serving4 per endpoint and the original six-arm5740 bill remain untouched; future comparison costs require separate admission.

## Interpretation and nearest priors

| Saved primary scope | What it establishes and what this control tests |
|---|---|
| GAR2026, 2609.36724v1 §§3/5, D.10, G | Gradient-informed expert allocation is direct ancestry. GAR detaches current gradient observations summed in a common expert template, uses load-normalized signed Gram coherence and directs its auxiliary gradient only to a recomputed router. The matched control uses member-specific utility, native affinity/hard balance and total shared credit; it is an attributed adaptation, not a literal GAR reproduction. |
| Ren,1803.09050 §§3.1–3.3/Eq12 | Gradient alignment values example updates against meta loss. The present S margin versus member S-own-CE alignment occupies that ordinary utility ancestry; no clean-meta-data/robustness guarantee transfers. |
| Meta-Weight-Net,1902.07379 §§2.1–2.3/Eqs3–6 | Independent-coefficient classifier partials, differentiated adaptation, gradient utility and fresh weighted recomputation are established bilevel practice. No new hypergradient principle is claimed. |
| SMCL/MCL,1606.07839 saved method/Eqs1–2 | Loss-dependent specialist assignment and min-loss oracle specialization are established. All-member own CE and complete probability serving require competence/full-pool evidence; winner fields or oracle loss do not establish served quality. |

A future matched comparison must start finite-live and utility-live from the identical approved common state and retain label access, architecture, coefficients, H16, preprocessing, graph/pairs, credit ownership and serving. Independently qualify every utility cost/gradient/commit coordinate, exact dormant constraints, all diagnostics, native restoration and full-FP32 resources before its fit. Freeze comparisons/metrics and resource budgets prospectively; charge failed qualification and all actual work. Same horizon isolates the cost substitution; measured resource tradeoffs need their own prespecified comparison, since these derivative bills differ.

Support for a finite-response contribution would require complete served held-A benefit over this live first-order control with member competence retained, plus benefit beyond competent ordinary own/pool training with identical W/S/R access, independent ensembles and capable singles, followed by untouched confirmation. Assignment disagreement, nonzero finite-minus-utility costs, better S/R diagnostics, balance, response magnitude or lower conditional unanimity alone cannot establish that contribution. A win over uniform/current margins/stop-Q alone does not isolate it. A single screen supplies a conditional empirical result, not novelty or universal superiority.

All literature here reuses existing saved passages/scopes. No new retrieval or full-paper/proof/code/result audit was performed. Unresolved recent methods remain unresolved.
