# Stage1 assessment v2 — residual-inclusive training cost

2 October 2026. Source/specification preparation only. All v1 files are preserved. No comparative report, curve, logit, checkpoint outcome or remote state was fetched or read. The validator was not executed, tested or compiled; no tensor contents were loaded.

Root explicitly chose the following analysis precisions while comparative outcomes remained unseen:

- Full training cost is `elapsed_seconds.total_scientific_phase − sum(profiling, operation_breakdown, fresh_checkpoint_deployment_profiling)`. The three disjoint excluded blocks are summed with `math.fsum` before subtraction. This includes scientific setup, complete updates, scheduled validation, all checkpoint/logit I/O, selected reload, deployment preparation, curve/bookkeeping and other scientific residual time. Framework imports preceding the scientific-phase timer and final report/manifest hashing remain outside this measured interval.
- The old six-component training sum and the difference between the primary residual-inclusive timer and that sum are descriptive. Neither replaces the full-training gate.
- Training and warm inference cost gates use the ratio of three-seed arithmetic mean costs separately on each graph. Each fit's warm metric remains its synchronized20-repeat selected-checkpoint all-member median, with full-graph forward, validation gathering and probability pooling.
- Lowest mean primary NLL selects the strongest S/U/H control separately on each graph; exact ties resolve S→U→H. That same comparator supplies the accuracy check.

No threshold, checkpoint rule, horizon, boundary competence rule, NLL aggregation, paired seed identity, graph weighting or evidence gate changed. The v1 rules/requirements remain documented in the preserved v1 README. In particular, each graph's relative NLL gain is1−ratio of its three-seed mean NLLs, and cross-graph P0/B0 summaries average those two graph gains equally. A best selected checkpoint at the final evaluation with strict improvement is inconclusive. Recent earlier improvement does not add a new failure rule.

`SPECIFICATION_v2.json` contains the exact54 prospective cells/18 recipes and unchanged thresholds. `validate_stage1_v2.py` advances specification/adoption/result schemas to v2 and changes only the training timer calculation plus descriptive output fields. `DIFF_from_v1.patch` records the full source delta. `SOURCE_PREPARATION_CHECK_v2.json` records text/hash/JSON checks; it is not a code test or syntax check.

`ROOT_ADOPTION_TEMPLATE_v2.json` remains `root_adopted=false`. Root must freeze a separate adopted v2 file binding this specification/source and the exact rules before outcomes. The validator requires this adopted binding and the separately frozen evidence index hash. There is no remaining analysis-policy ambiguity after root's explicit choices. Current source remains unexecuted and uncompiled.

The future validator retains every one of the54 expected rows and marks missing, corrupt, incomplete, nonfinite, misbound or boundary-improving fits inconclusive. It checks normal zero-exit supervision, exact argv, source/GPU/data/mask/recipe provenance, every artifact's hash/length, complete curve/update schedule, earliest strict selected NLL, byte gates and all original Stage1 effect/cost thresholds. It hashes `.pt` payloads without tensor deserialization, prediction or metric recalculation. Internal checkpoint/logit tensor semantics remain outside this stdlib check.

No optional interval is computed. Any later interval needs a prospectively frozen analysis addendum and must identify three paired model seeds on one core0 partition per graph and two specific graphs; it provides no graph-population significance. Every arm/result and the same-checkpoint sensitivities remain retained, with no metric promotion or favorable subset rescue. A Stage1 pass authorizes no continuation, confirmation/test scoring, novelty claim or paper verdict by itself.
