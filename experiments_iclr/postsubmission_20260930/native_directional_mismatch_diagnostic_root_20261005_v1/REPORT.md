# Native derivative qualification: failure and diagnosis

Updated 2026-10-05T20:32:01.990796+00:00. Engineering evidence only; no data labels, fits, predictive improvement or novelty claim.

The fixed synthetic native qualification failed at its first shared directional finite difference. The original point, 15 artificial nodes, full 512-wide ten-local/one-global architecture and all source bytes were retained. At epsilon0.001, the finite difference was 0.0285137475746, compared with gradient direction 0.00387111865852. The original tolerance was not changed. The failed check prevented public-episode/state-restoration qualification from being reached in that run.

Earlier checks did establish exact full-member callback logit and first-gradient reconstruction, sparse/dense assignment values/gradients and full shared-coordinate dense/sparse outer parity. These cannot by themselves establish the derivative of the served finite-response state map.

A separately frozen diagnosis at the same state compared torch.func with ordinary autograd. The independent main-private derivative used an equal-valued independent private proxy, while responsibilities remained connected to the original private states and shared core. The objective values were identical; maximum difference across every shared derivative coordinate was 1.11022302463e-16. The live responsibility-chain gradient norm was 0.22318048051. This is synthetic sensitivity, not evidence of useful specialization.

The right finite differences converge toward the analytic direction at3e-5 and1e-5, while left differences remain unstable. Smaller central steps do not consistently reduce the discrepancy. This is compatible with crossing a nonsmooth private-gradient response, but the cause is unresolved. A narrowly fixed sign/zero-boundary diagnosis is being prepared; it will preserve the state, graph, sources, original failure and tolerances. No easier backbone/state or qualification pass has been substituted.

The ordinary-autograd diagnosis also checks native parameter/mode/attribute/input/RNG restoration. Qualification still remains incomplete. The six-arm predictive pilot has not started.

The separate safe accessor's missing native self-loop preprocessing was confirmed by source review. A sealed V3 repair restores the original CPU to_undirected/remove_self_loops/add_self_loops recipe and passes independent source-delta review. This clears the source defect, without clearing the native derivative gate or releasing training. No changes to role identities, G0, serving, controls, H16, scoring or original paper scores are authorized by that repair.

Receipts: ../learnability_responsibility_native_synthetic_execution_root_20261005_v1/RESULT.json, RESULT.json, TERMINAL.json. Independent source reviews: ../learnability_responsibility_native_qualifier_wsr_v2_independent_review_20261005_v1/REPORT.md and ../learnability_responsibility_wsr_v3_edge_delta_engineering_interpretation_20261005_v1/REPORT.md.
