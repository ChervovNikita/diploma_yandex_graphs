# Complete saved-prediction aggregation outcome

The fixed Amazon/Polynormer screen completed all three splits for the shared four-member ensemble, the independent four-member ensemble and a comparable processed single. It used existing selected predictions and the permitted VALID folds. The result retains all 99 configurations, both raw and graph-corrected metrics, fitted-control diagnostics and costs. No backbone training, new base-model inference, final refit or TEST scoring occurred.

## What was tested

The proposed operation combines predictions using local, graph-transported moments of member errors. Its controls include ordinary pools, global temperature/weight calibration, global and diagonal error moments, capable nonlinear score stacking, and projection toward a graph-diffused label posterior. The single has a comparable nonlinear processor. All configurations and three splits were fixed before execution. Setting selection uses the prescribed corrected out-of-fold Brier loss; no accuracy-based replacement was made after inspection.

## Complete processed comparisons

| Prescribed Brier-selected pipeline, mean over three splits | Accuracy | Brier | NLL |
| --- | ---: | ---: | ---: |
| Shared four | 52.5450% | 0.637247 | 1.833855 |
| Independent four | 53.2528% | 0.642339 | 1.784916 |
| Single | 53.1276% | 0.613544 | 1.492955 |

The capable score-context MLP (operator 7) was the selected processor in every ensemble split, with settings 1, 0, 1 for each ensemble family. The processed single selected operator 10, with settings 1, 0, 0. The graph-moment proposal was not selected.

Shared four trails processed independent four by 0.7077 accuracy points, despite a 0.005092 mean Brier advantage; its mean NLL is 0.048940 worse. Shared four trails the processed single by 0.5825 accuracy points, with both worse Brier (+0.023703) and NLL (+0.340901). It fails the fixed quality requirements against both references. Its accuracy differences against independent are +0.5063, -1.1106, -1.5189 points; against the single they are +0.1633, -1.3065, -0.6043 points. No favorable split or metric is substituted for the complete result.

The selected shared processor improves calibration losses over its native pool, but mean accuracy improves by only 0.1415 points and one split loses 0.9309 points. The selected independent processor adds only 0.0762 accuracy points. Both fail their prescribed full quality screens. Improved calibration alone does not establish improved classification or a shared-ensemble advantage.

## Ingredient evidence and research decision

For shared predictions, tuned local-full moments have 0.000714 worse Brier than global-full moments, approximately 0.000020 better Brier than local-diagonal moments, and 0.011447 worse Brier than posterior projection. The complete result also retains same-rho contrasts for both moment controls. There is no supported useful off-diagonal/local-moment contribution under this screen. Stacking and posterior controls must remain in the interpretation.

**Decision: do not launch confirmation or expand the hyperparameter grid for this proposal.** Preserve the complete failure and use it to guide training research. Together with the independent prediction-error diagnosis, the result motivates seeking competent complementary members rather than assuming a new combination rule repairs their repeated errors. This is a hypothesis choice, not causal proof of a representation defect or clearance of methodological novelty. Ordinary NCL/GNCL error-coupled training is an attributed control; its known own/pool loss equivalence prevents treating it as a new method.

The base checkpoints were selected on VALID, and the out-of-fold processor results also choose settings. These are biased retrospective development observations on one graph with overlapping splits. The retained descriptive intervals are not independent confirmation intervals. Final TEST labels remain unopened; original manuscript results are unchanged.

## Costs and preserved failures

The completed screen used 90 small MLP fits, 36 calibration fits and 18,900 optimizer updates. Numerical study time was 165.664 seconds; the physical child including serialization took 169.628 seconds. Peak child RSS was 719,511,552 bytes. There were 144 batched graph propagations, 2,880 sparse steps and 684 logical field propagations. Historical base-model acquisition is an additional separately retained cost.

The first execution failed before fitting/scoring because of projection roundoff. Its original failure, actual row witness, source versions, independent repair review and affected qualification remain preserved. The corrected run changed projection arithmetic, with an explicit numerical rank cutoff; it did not alter scientific settings or spare an unfavorable arm. Neither attempt is omitted from costs or provenance.

- [Recomputed means, gates and retained complete contrasts](SUMMARY.json)
- [Immutable input bindings](SOURCE_BINDINGS.json)
- [All configurations and original terminal result](../amazon_polynormer_logits_graph_moment_cpu_root_preparation_20261005_v2/development_observation_20261005T174207Z/amazon_polynormer_logits_graph_moment_retrospective_cpu_execution_root_20261005_v2/RESULT.json)
- [Independent prediction-error interpretation review](../amazon_polynormer_valid_error_analysis_independent_interpretation_review_20261005_v1/REPORT.md)
