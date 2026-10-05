# Original Amazon/Polynormer VALID error analysis

**Source only, unexecuted; root executes after source review.** This packet describes the already selected native predictors retrospectively. It creates no method, fits no parameters, opens no checkpoint and performs no hidden export, graph propagation, oracle LP or final-label evaluation. Preparation accessed safe source text/metadata only.

## Fixed population and scoring

Use the complete original 15-fit/9-family closure and all 6123 selected official VALID nodes in each of the three split/block identities. Reuse the authenticated V2 custody loader unchanged. Its full saved-logit tensors are decoded as required by their storage format, then sliced to VALID before probability/statistical arithmetic. Only compact VALID labels are decoded; features and TRAIN/control/TEST labels are never decoded. Public archive hashing includes compressed bytes but materializes only edge_index and val_mask. The single is independent member0, with no additional fit or load.

Single and individual-member classes use raw FP32 logit argmax. Four-member classes use FP32 softmax followed by arithmetic member mean and argmax. Ties choose the smallest class. Brier uses the exact V2 stable FP64 probability arithmetic; native NLL uses its stable FP64 log-mixture without contamination. FP32/FP64 class discrepancies are reported. Public degree is the number of unique canonical undirected non-self neighbors, before any prediction mask. Bins are exactly 0, 1-2, 3-5, 6-10, 11-20, >20.

## Required complete panels

AGGREGATE.json and AGGREGATE.csv hold the same denominator-rich long metric table. Columns include split, classifier, class/degree scope, pair/member identity, metric, support_n, numerator, denominator, value and status. Every class, requested degree bin and fixed confidence bin is present, including zero-support bins with null ratios. No node-level labels/predictions or favorable subset selection are exported.

- All three native predictors: accuracy, class confusion/recall/precision, Brier, NLL, fixed 15-bin confidence ECE and erroneous confidence/margin summaries. Both native FP32 and stable FP64 error confidence are explicit.
- Shared4 versus independent4 lost/recovered counts, with both-correct and double-fault counts; single comparisons are also retained. The left-only-correct event is recovery of a right error; right-only-correct is a left loss of a right correct prediction. Conditional denominators and full-population fractions are both given.
- Every member accuracy, ensemble rescue/loss against each member, member-correct histogram and ensemble accuracy conditional on that count. Any-member-correct is not a hull-correctability test.
- Every six member pairs per four-member bank, and all native predictor pairs: error overlap/Jaccard, double fault, same wrong-class agreement and error-indicator correlations, unconditional, by true class and in fixed degree bins. Probability and class-error residual correlations are additionally unconditional/by true class for every probability class. Empty or zero-variance correlations are null, never zero by convention.
- Exact accuracy decomposition: shared-minus-independent native accuracy = mean-member raw accuracy difference + difference in pooling gain over mean-member accuracy. Report identity residuals.
- Exact Brier ambiguity: pooled FP64 Brier = mean-member Brier - mean member squared probability deviation from the pool. Report each bank's terms/residual and paired term differences/residuals. This is algebra, not a claim that more diversity improves accuracy.

Each split remains separate. Headline three-split mean/min/max summaries explicitly use three blocks; no node-iid significance, bootstrap or independent-split confirmation is supplied.

## Frozen interpretation and intervention priorities

| Observed descriptive pattern | Next intervention to investigate after diagnosis |
|---|---|
| Shared mean-member accuracy/Brier is worse, while pooling gain is comparable | Base representation or training quality: revisit shared/private capacity, boundary factors or training objective before adding an aggregator. Match budget/base competence in later controls. |
| Mean-member quality is comparable but shared pooling gain is worse, with many members correct on pooled errors | Aggregation/readout or calibration of member influence is a priority. The correct weights may not be learnable from available context; no oracle success is inferred. |
| High double fault or conditional error correlation coincides with weak shared member quality | Investigate target-aligned private representation or training-objective diversity, with matched controls. Correlation does not establish causal sharing collapse or calibrated uncertainty. |
| Accuracy is similar but wrong predictions are more confident and NLL/Brier/ECE are worse | Calibration is the first hypothesis. Any temperature/scaling fit requires a separate frozen development design. |
| Losses recur in a particular class or fixed degree bin | Inspect class imbalance/loss allocation or propagation/context effects using the complete prespecified panels. Sparse bins and overlapping splits limit interpretation; do not promote a selected favorable subgroup. |
| Mean-member quality and pooling benefit are both poor | Investigate both sources separately; a fancier fusion head is not yet justified. |

These rules choose a hypothesis priority, not an automatic intervention or development admission. Stronger score/context readouts may exploit information outside the fixed probability hull. Neither all-member errors nor a separate hull impossibility diagnostic establishes absent score information. No causal claim, novel mechanism, promotion or TEST authority follows from this report. Existing VALID checkpoint selection makes the description retrospective.

## Future root command and outputs

After the independent source review, run with the root-bound complete closure descriptor and the exact V2 protocol. The source-seal hash is provided in SEAL.json's delivery pins. This example contains placeholders for root-owned custody/output choices:

    python run_error_analysis.py --phase /ABS/PHASE \
      --closure-freeze RELATIVE_COMPLETE_COHORT_FREEZE.json \
      --closure-sha256 ROOT_BOUND_CLOSURE_SHA256 \
      --protocol amazon_polynormer_logits_graph_moment_retrospective_protocol_20261005_v2/PROTOCOL.json \
      --protocol-sha256 5d667f15640102995fa998ca6932c53c5b71554273c84d4938a331c0a4eae7ba \
      --source-seal-sha256 ROOT_BOUND_THIS_PACKET_SEAL_SHA256 \
      --output NEW_PHASE_LOCAL_OUTPUT \
      --execute-authorized-valid-error-analysis

The runner authenticates this source and all V2 source payloads before importing V2 custody. Custody authenticates the original full closure and descriptors before numerical imports/decode. No training/study module or original model loader is imported. Scientific dependencies are NumPy and Torch only. No SSH or server launch is part of this preparation.

Outputs: RUN_BINDINGS.json, INPUTS.json, ENVIRONMENT.json, PROGRESS.jsonl, AGGREGATE.json, AGGREGATE.csv, TERMINAL.json. A failed attempt preserves FAILURE.json and is never called complete. Cost records distinguish input/source authentication, descriptive calculation and peak RSS; historical model acquisition is separate. Root can bind output files into its normal result freeze.

## Source V2 bounded correlation repair

The independent auditor identified P2: centering an exactly constant continuous vector by its rounded arithmetic mean can leave a tiny identical residual and yield a spurious defined Pearson correlation. V2 explicitly detects an exact constant column before centering and uses an all-zero centered vector. Its variance and covariance rows are then zero and Pearson is undefined_zero_variance. No variance threshold, diagnostic setting, denominator, output field, scoring or loader changes. V1 is preserved. SOURCE_V1_TO_V2.patch and REPAIR.json authenticate this sole numerical-source change; source-only checks do not establish numerical execution.
