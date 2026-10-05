# Independent sequential VJP mathematical/specification challenge

6 October 2026. **The proposed decomposition is algebraically coherent under the boundaries below.** Actual implementation, floating-point parity and memory feasibility remain unqualified. This review uses the pinned original operator and the authors' prospective schedule; it does not inspect the unsealed implementation, full-OOM payload, data, masks or SSH, execute/import source, fit, modify predecessors or supply a novelty claim.

## Exact shared-credit identity

Fix the complete original private states phi_m independently of theta. Let

```text
probe_m(theta) = phi_m − eta_probe * partial_phi CE_S(theta, phi_m)
C_m(theta) = margins_S(theta, probe_m(theta)) − margins_S(theta, phi_m)
Q(theta) = F(C(theta))
A_m(theta,Q) = phi_m − eta_private * partial_phi L_main(theta,phi_m,Q)
h_m(theta,Q) = logits_R(theta,A_m(theta,Q)).
```

F includes the original item centering, one epsilon-smoothed RMS per pair and exactly eight balanced sparse assignment steps. In the main private partial, Q is an independent input. The query loss ell(h_1,...,h_M) is the exact original equal own-CE/probability-mean-pool objective over R.

At the fixed current values, compute query-logit leaf cotangents v_m=partial_h_m ell from **all** member logits together. Define

```text
d_m = (partial_theta h_m at independent Q)^T v_m
u_m = (partial_Q h_m)^T v_m
u = sum_m u_m
t = (D_C F)^T u
G_theta = sum_m [d_m + (D_theta C_m)^T t_m].
```

This is the original total shared derivative. d_m includes the direct query-core path and the mixed theta path through the main private gradient; u_m includes its Q dependence. The raw-response term includes the probe-gradient mixed theta derivative. Cotangents are constants during each VJP; detaching stored values is valid because the omitted links are restored by these explicit products. Detaching either main/probe private gradients inside their derivative helpers would delete required shared credit.

The ordinary-autograd main helper must use fresh, independently owned phi and Q inputs, with `create_graph=True` for the private partial. It must not differentiate Q with respect to phi inside that partial. Rebuilding the Q-map VJP from normalized costs instead of raw C would omit centering/RMS credit unless those preceding derivatives are also included. Finite responses must remain actual after-minus-before losses, not a gradient-alignment approximation or exact-optimizer replacement for the eight-step map.

The identity applies as a classical chain rule where the composed map is differentiable. Elsewhere, parity must reproduce the same selected native AD rules; this does not establish global smoothness. Scalar value equality alone does not qualify these derivatives.

## Controls and commit

| Control | Raw input for F | Raw-cost shared credit |
| --- | --- | --- |
| live | finite response | positive probe→after VJP plus negative original-before VJP |
| graph_free | finite response, gamma=0 | same chain; balance remains coupled |
| permuted | finite response, supplied fixed permuted affinity | same chain; native serving graph unchanged |
| margins | current original-before margins | positive before-only VJP |
| uniform | paid response/map, then Q=1/M | zero Q-map credit; retain all direct/mixed main credit |
| stop_q | paid live Q values | zero Q-map credit; retain all direct/mixed main credit |

All controls retain the value probes and prescribed solver work. At theta+, recompute probe values, raw costs, Q and the main private response from the **original** phi, then commit that response and detach the episode. Cached virtual phi at the old theta are not the committed state. Preserve every original diagnostic, including actual response RMS even when margins supply costs.

If outer-phi gradients are reported for inspection, they require the analogous direct identity/Hessian and raw-cost chain. They remain detached diagnostics and do not replace the prescribed private SGD commit.

## Scheduling and counts

The two-native-forward-graph intent is feasible: a main-forward/private-gradient graph plus its adapted query forward, or a probe-own/private-gradient graph plus its after forward. Compute the negative before-response VJP separately. End each helper before the next member, returning only detached caches/cotangents. The pure eval callback permits reusing one forward for before values and own-CE probing.

This bounds intended retained **native forward** graphs. Derivative intermediates, full parameter/gradient tensors, raw/Q banks, temporary buffers and allocator behavior remain outside that bound. `retain_graph=False` alone is insufficient if live outputs or closures retain graphs. Whole-process memory must be measured on the complete native context; OOM remains possible.

Per episode under the agreed schedule:

| Arms | native forwards | private-gradient constructions | native member VJPs | pair primal maps |
| --- | ---: | ---: | ---: | ---: |
| live, graph_free, permuted | 12M | 6M | 3M | 3P |
| margins | 10M | 5M | 2M | 3P |
| uniform, stop_q | 9M | 5M | M | 2P |

Each episode adds one small query-logit VJP. Arms retaining Q credit add P Q-map VJPs. For M4/P10, six H16 continuations charge **4096 native forwards, 2112 private-gradient constructions, 832 native member VJPs, 96 small query VJPs and 640 pair Q-map VJPs**. Adding the unchanged 1600 acquisition forwards, 16 warm-diagnostic forwards and 28 evaluation forwards gives **5740**, with 257 assignment banks/2570 pair maps/20560 solver iterations. Preserve the old5100 protocol; seal the new execution accounting prospectively. A streaming warm-helper substitution would be a separate reviewed change, not an assumed saving. These are construction counts, not FLOPs or equal wall-time claims.

## Required parity and resource evidence

Before native use, freeze all-control parity against the unchanged monolithic operator: pre/post-core raw responses, normalized costs, Q, virtual private states, query values/logits, every shared gradient coordinate, theta+, complete committed private state and diagnostics. Check independent-Q private-partial ownership, live/stop-Q equal forward values, stopped versus fixed-Q derivatives, dormant local heads, and exact supplied permutation orientation. Use fixed tolerances/scales and retain every mismatch; finite arithmetic can alter summation order even when the real-arithmetic chain agrees.

Then qualify the sealed implementation's complete callback counts, native state/input/RNG and gate restoration, higher-order/public episode/recompute path and whole-process resources at full node/feature/graph context. Passing a synthetic parity case or meeting the two-forward scheduling intent does not release W/S/R fitting or A scoring. The competent ordinary all-branch baseline remains a later scientific requirement.
