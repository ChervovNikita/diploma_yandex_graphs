# Analysis and convergence protocol for the new study

**Preparation only. No new fit or GO.** This protocol supplements the immutable run roster; it does not change a frozen method, dataset, source recipe or horizon.

## Purpose

The study compares the prediction quality of the original GNNM and its developed extensions with an ordinary single model and a genuine independent ensemble. Every method must retain its actual shared parameters, private parameters, optimizer owners and prediction rule. Four predictions from one body do not constitute four independent learners.

Roman Empire, Amazon Ratings, Minesweeper, Questions and Tolokers are excluded. Original GNNM remains a comparison on compatible new data. Original-paper scores are unchanged. Source author replication is reported separately from standardized comparisons: different splits, normalization, training policies or backbones cannot support a direct score difference.

## Training evidence

Training follows the policy registered for each source procedure. Fixed-budget fits complete their declared horizon and retain selected and terminal states separately. Fits registered with native patience stopping retain that patience, validation cadence and cap, including comparative fits. A coupled four-body procedure retains its own body stopping states. Source author tracks keep their declared author procedure. Record complete training and validation curves, any stopping signal actually computed by the source, selected step, terminal step, every failure and actual cost. An unimplemented diagnostic is not training evidence.

A long registered horizon is not proof of convergence. After complete training, inspect training-loss trends, validation trends and whether the selected checkpoint lies near the budget boundary. Report a budget limitation when these diagnostics do not support convergence. The finite roster's optional 10,000-update alternative is a separate prospective study; no outcome-selected extension is activated by this document. Changes to its fit counts and costs must be materialized and approved before execution.

Hyperparameters are chosen using the declared search seeds and SELECT role. Comparative seeds, masks, coefficients and references are frozen before comparative fits. TEST scores cannot select recipes, continuations, exclusions or a favorable subset. Preserve failed configurations and distinguish a failed search from a completed low-scoring fit.

## Comparisons and uncertainty

For each compatible task, compare each extension with original GNNM, the ordinary single model and its genuine independent ensemble using matched split and comparative seed. Preserve all individual values and report paired differences, including losses and ties. Primary metrics are those bound by the source packet: multiclass node accuracy, binary node ROC-AUC, multilabel node metrics, molecular ROC-AUC or the declared link-ranking metric, as appropriate. Calibration metrics are secondary to the raw task score. A native control with a different width, sampler, information source or posterior sample count remains an explicitly named control; do not silently attribute its difference to parameter sharing alone.

Do not treat overlapping WikiCS masks or repeated Photo masks as independent datasets. For repeated-mask tasks, average paired differences across the complete declared mask set within each seed, then report seed-level uncertainty with the small seed count stated explicitly. Also show variation across masks separately. Three comparative seeds cannot support a precise uncertainty claim or a reliable tail-probability estimate. No zero-variance or few-seed comparison is declared significant automatically.

On single-mask tasks, report every seed's paired difference. Confidence intervals and any tests must state their unit, assumptions and method. A parametric interval with three seed pairs is descriptive and conditional on its assumptions. A three-pair exact sign test cannot attain a two-sided 0.05 threshold. Do not imply statistical support merely because the mean is positive.

If formal tests are used across the prospectively registered comparisons, preserve the entire comparison family and apply Holm correction within the stated family. Report unadjusted and adjusted values together. Do not calculate correction only for the successful methods. Results encountered during method development remain development evidence; this roster does not certify an unseen confirmation set.

## Error analysis

Use the released member logits to separate three effects: individual member quality, agreement between members and the benefit obtained by combining them. For each paired comparison, report mean member accuracy, ensemble accuracy, disagreement, cases where members are correct together, and cases where one model corrects the other's errors. An oracle that chooses the correct member is a diagnostic upper limit, not a deployable predictor.

Graph strata such as degree, feature similarity and neighborhood-label agreement are diagnostic. Define them from the permitted graph and training information before opening comparative evaluation outcomes. A test-label-based stratum must be identified as post-evaluation descriptive analysis; it cannot guide a newly fitted method or a claimed protected confirmation.

## Reproducibility and cost

Keep selected serving states, declared member logits and posterior draws, role custody, configuration, source hashes, complete curves and terminal costs on the servers. Keep complete resumable states while retries or dependent paid stages remain possible. Only the registered, audited retention policy may replace a complete state with a serving state; unsupported source families retain complete states. Git contains source, finite manifests, decision history and compact result metadata. The Mac receives compact research notes rather than the checkpoint archive.

Two slots per GPU permit concurrency when memory is sufficient. Measure actual throughput and wall time; do not divide a serial estimate by two merely because two slots exist. The current TRAIN-pass and serialization measurements use random weights with optimizer actions intercepted. They are resource-planning evidence and contain no scientific score, convergence evidence or whole-study ETA.
