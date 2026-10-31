# Stage1 assessment specification and validator — source preparation

1 October 2026. Prepared from the frozen staged adoption, exact Stage1 protocol/queue, inherited design definitions and existing calibration report schema. No comparative output was fetched or read, no remote connection was made, and the validator was not executed, tested or compiled. No model/framework import, tensor loading, prediction, metric recalculation or paper review was performed.

`SPECIFICATION_v1.json` records the exact54 identities/18 recipes, fixed thresholds, formulae, provenance requirements and remaining analysis precisions. `CALIBRATION_REPORT_SCHEMA_REFERENCE_v1.json` captures only a retained calibration report's schema. `validate_stage1_v1.py` is standard-library source. `ROOT_ADOPTION_TEMPLATE_v1.json` is deliberately unadopted; the validator refuses to run an assessment without a separately frozen root adoption matching the specification, source hash and exact rules, with comparative outcomes still unseen at adoption.

## Inherited rules

Aliases: PF=`coordinate`, F=`factor`, G0=`original`, P0=`permutation`, B0=`bias_only`, S=`single`, U=`untied`, H=`heads`, T=`gt_sep_single`.

For graph g, arm a, seed s∈{17,29,43}, let N(g,a,s) be **selected-checkpoint arithmetic-probability-pool validation NLL**. Mean NLL is `M(g,a)=sum_s N(g,a,s)/3`. Relative gain against control c is `R(g,c)=1−M(g,PF)/M(g,c)`. This ratio of paired seed means is explicitly defined in the inherited design at PROTOCOL.md:77. Per-seed `1−N(PF)/N(c)` values are retained descriptively; their mean does not substitute for R.

The cross-graph P0/B0 quantity is `(R(Photo,c)+R(CS,c))/2`. Graphs receive equal weight; nodes, classes, member count and graph size do not supply weights. No raw-NLL concatenation across graphs is used.

On each graph, the strongest byte comparator is the member of the fixed S/U/H set with lowest M(g,a). The same NLL-selected comparator is used for the byte NLL and accuracy checks. Choosing the strongest arm anew for every seed or selecting it by accuracy is not admitted. Every S/U/H result remains retained.

Unchanged gates:

1. R(g,F) and R(g,G0)≥0.02 on **each** graph, with PF strictly beating paired F in≥2/3 seeds on each graph. Exact NLL ties are not wins.
2. Equal-graph R(g,P0) and R(g,B0) summaries each≥0.01.
3. Best-byte relative gain≥0.01 on at least one graph and≥−0.01 on both. Mean primary pooled accuracy loss, `100×(mean_accuracy_control−mean_accuracy_PF)`, is≤0.5 percentage point against F, G0 and the same NLL-selected byte control on each graph.
4. PF/F dense parameter bytes match, with only the declared fixed permutation index buffers differing; full training and warm all-member inference ratios each≤1.25 under the prospectively adopted cost precision below.

Threshold comparisons are inclusive and use unrounded retained numerical values without a new comparison tolerance. Curve checkpoint improvements use the runner's strict NLL comparison, with exact ties retaining the earliest epoch. Reported accuracy, macro-F1, same-checkpoint mean-logit sensitivity and disagreement cannot replace the primary NLL gate or select a different checkpoint.

## Precisions root must resolve before outcomes

The source proposes a fixed S→U→H order for exact mean-NLL byte-control ties. That affects only which tied control supplies the accuracy comparison; all tied outcomes are retained.

The source proposes cost ratios as **ratios of three-seed arithmetic mean costs, separately on each graph**. Warm cost is each fit's synchronized20-repeat median from the selected checkpoint, full graph/all actual members, validation-node gathering and probability pooling. P95, fresh deployment, isolated operation timings and favorable individual repeats cannot substitute. Paired seed ratios are descriptive.

The proposed full-training numerator and denominator each sum these disjoint `elapsed_seconds` components:

- `setup_total_including_initial_checkpoint_and_cold_evaluation`
- `training_complete_updates`
- `validation`
- `selected_final_checkpoint_io`
- `selected_checkpoint_reload`
- `deployment_checkpoint_preparation`

This includes measured scientific startup/model/input setup, complete updates, scheduled validation, every selected/final checkpoint-logit write, selected reload and model-only deployment preparation. Setup subcomponents are already nested in setup_total and are not summed again. Framework import time, final report/manifest hashing, warm/fresh profiling and isolated operation profiling are excluded. **Unattributed Python curve/bookkeeping and other residual time is not included in this component sum.** A residual-inclusive alternative would be total scientific phase minus the three non-overlapping profiling blocks; adopting that alternative requires a new immutable source/specification version. Neither invocation time nor the scientific total may silently replace this proposed metric after outcomes. The source freezes no assumption that all three versions are equivalent.

`ROOT_ADOPTION_TEMPLATE_v1.json` exposes these exact proposals. Root must consciously accept them in a new immutable adoption or require a new specification/source, before comparative outcomes are inspected. The template is not execution authorization and must not be edited into an adopted record in place.

## Completion, selection and competence

The future evidence index must be an immutable root-frozen list of phase-relative path/length/SHA256 entries, plus the actual local serial launcher receipt directory. The validator rechecks its supplied hash and the separately frozen specification/adoption hashes. It checks the exact queue/source/protocol/runner/request/protected/allocation controls,54 unique core0 cells in unchanged order, three normal bounded invocations, exact nested argv, one authorized UUID, idle receipts, source/environment/command/log hash links, normal zero exits, unchanged requests and serial batch intervals. Missing supervision makes the whole screen inconclusive.

For each expected cell, it verifies every artifact payload hash/length, the complete expected artifact set, immutable input snapshots/source snapshots, invocation and report identity, exact recipe/optimizer/cell/source/data/graph/mask/target hashes, actual member count, dtype/deployed/index byte gates and serialized checkpoint lengths. `.pt` files are streamed for hashes only. Their tensor contents, internal provenance and prediction metrics are not independently inspected or recalculated.

JSONL must contain every consecutive complete optimizer update with finite training objectives/timings and nonzero-gradient receipts. Exactly every10 updates has validation. The validator reconstructs the earliest strict primary-NLL minimum from reported curve metrics, checks every improvement flag, selected/final metrics and epochs, fixed minimum1000/maximum2000/patience200 stopping at the first eligible evaluation, and timing sums. It does not rerun metrics from logits.

The inherited incompetence rule is **best checkpoint at final evaluation with strict improvement**. At2000, recent improvement before the final evaluation is not automatically classified as incompetence. This preserves the existing rule; it does not certify numerical convergence. Nonfinite, capacity, incomplete, misbound, corrupted or final-improving fits are inconclusive. All54 expected rows remain present; available invalid-row report/failure JSON is retained diagnostically. No favorable complete subset is used for a pass/no-go.

If all54 cells and supervision pass, the script evaluates every fixed gate. A competent failed gate is Stage1 no-go for this composition under the fixed recipe. A pass merely satisfies this exploratory screen. Separate root admission remains required for108 core1/core2 continuation fits; original full gates precede any independent confirmation. Test scoring, original-score recalculation, public claims and novelty/paper verdicts remain outside this source.

## Output and uncertainty

A future separately authorized invocation creates a new confined result directory with `ASSESSMENT.json`, `ALL54_CELLS.csv` and `VERIFIED_EVIDENCE.json`. Every arm, seed and graph is retained, including T and unsuccessful controls. Primary/sensitivity/member metrics, costs, storage, memory, checkpoint/curve identities and fresh deployment summaries remain available; source scope prevents secondary metric promotion.

No interval is computed by this validator. Any optional later descriptive interval needs a separate prospective analysis addendum and must state **three paired model seeds on one core0 partition per graph, two specific graphs**. Nodes and ensemble members are not independent seed replicates. These gates and descriptive summaries do not establish graph-population significance.

## Execution interface

When root has separately authorized assessment and frozen the adopted rules/evidence index, arguments are `--specification`, `--specification-sha256`, `--adoption`, `--adoption-sha256`, `--evidence-index`, `--evidence-index-sha256`, and `--output`. All input/output paths must remain in the allowed phase, and output must be new. Current source preparation makes no runtime or syntax-validation claim.
