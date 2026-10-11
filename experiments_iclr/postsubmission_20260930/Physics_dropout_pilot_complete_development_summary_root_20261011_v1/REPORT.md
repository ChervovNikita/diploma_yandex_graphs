# Complete Physics sign-aware dropout experiment

All42 banks,69 selected optimizer states,42 separate calibration endpoints and21,000 fixed scalar updates completed before one whole development release. Each protocol uses the full34,493-node graph with34,093 development nodes. Three fresh stratified masks and optimizer blocks are paired across every arm. This is development on an encountered graph, not an unused confirmation. Original manuscript scores are unchanged.

## Prediction results

| Procedure | Accuracy % | Mean member accuracy % | Raw NLL | Calibrated NLL |
| --- | ---: | ---: | ---: | ---: |
| O4_stem_IID | 91.0070 | 86.7861 | 0.449750 | 0.283694 |
| O4_stem_sign_pair | 91.1360 | 85.9710 | 0.360242 | 0.277831 |
| O4_graph_input_IID | 90.9835 | 86.6458 | 0.421773 | 0.283108 |
| O4_graph_input_sign_pair | 90.9972 | 86.5622 | 0.434169 | 0.286014 |
| O4_graph_input_uniform_balance | 90.8798 | 86.6197 | 0.438299 | 0.285213 |
| factor_M1_stem_IID | 91.4851 | 91.4851 | 0.452983 | 0.270021 |
| factor_M1_graph_input_IID | 91.3462 | 91.3462 | 0.475240 | 0.274148 |
| genuine_factor_I4_stem_IID | 92.4853 | 91.3707 | 0.364661 | 0.232908 |
| genuine_factor_I4_graph_input_IID | 92.5508 | 91.4565 | 0.373788 | 0.230964 |
| factor_M1_MS4_graph_input_IID | 91.2455 | 91.2455 | 0.478262 | 0.275283 |
| factor_M1_MS4_graph_input_sign_pair | 91.0236 | 91.0236 | 0.472092 | 0.281618 |
| unshared_coupled_I4 | 92.1587 | 90.9879 | 0.390370 | 0.242865 |
| fresh_native_original_sequential_O4 | 90.7567 | 85.0113 | 0.292452 | 0.274336 |
| fresh_source_meanSAGE_M1 | 92.5723 | 92.5723 | 0.949743 | 0.233765 |

## Frozen assessment

The complete development criterion FAILED. The sign-aware first-graph-input candidate gains only0.013688 points versus IID at the same position, with paired deltas0.331446/0.105594/−0.395976 and descriptive95% interval[−0.911201,0.938577]. Its mean members lose0.083595 points. The required0.25-point gain at every positive block is not supported. Against freshly trained original GNNM, its average gain is0.240519 points with two negative blocks; this criterion also fails.

The candidate loses1.553594 points to genuine factor I4 at the same dropout position and1.575103 to the source-derived meanSAGE single. Each of these paired differences is negative at every block. Mean candidate members are4.894211 points weaker than same-position I4 members. Raw-NLL protection versus same-position IID passes, but prescribed calibrated-NLL protections versus capable I4, source single and equally coupled unshared I4 fail. These are practical screens, not noninferiority tests.

Every graph-specific attribution condition fails: candidate versus stem sign coupling has mean−0.138836 points; position×coupling interaction has mean−0.115371; candidate versus uniform balance gains0.117326 but has a negative block. The coupling effect on a four-copy single is−0.221942, and its difference from the shared-bank coupling effect is+0.235630 with an interval including zero. This does not establish useful sharing-specific coupling.

## Error evidence and mechanism

Against same-position I4, across the three complete protocol populations the candidate repairs1,867 decisions but harms3,456. It gains716 member-covered nodes and incurs2,323 extra pooling losses. Candidate-minus-reference net correctness is716−2323+18=−1589. Summed counts span repeated nodes across three protocols, not102,279 independent examples. The complete per-block repair, harm, coverage and pooling counts are retained.

The marginally correct Bernoulli coupling and restricted conditional variance result therefore do not resolve the measured member weakness or final prediction loss. This closes the exact frozen mask-position/sign-matching rule. It does not rule out all ways of obtaining competent and complementary shared members. No coefficient grid, changed serving rule or heldout rescue follows.

## Costs and provenance

Acquisition runtime was2473.53s, with26,233 optimizer updates,53,701 TRAIN route backwards,46,780 SELECT routes and38,800 detached-prefix routes. Peak allocated CUDA was3.59GiB. All three actual acquisition/CAL/score owners exited successfully, waited/reaped their direct children and verified owned CUDA process absence. Qualification overlap is recorded in prior receipts; no isolated speed claim follows.

The complete per-class report remains on the authorized server under the bound receipt in SUMMARY.json. Local/Git files retain every arm, whole-population metric, candidate contrast, frozen criterion and paid count without copying checkpoints, logits or datasets. Three-block intervals/sign flips are descriptive and conditional on the encountered graph.

The distinct source-meanSAGE attribute-peer feedback study was admitted and launched before this dropout result was opened. Its roster, native probability mean and separate calibration recipe are unchanged. It is not a performance-dependent rescue of this study.
