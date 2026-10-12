# Analysis and convergence protocol for the new study

**Preparation only. No new fit or GO.** This protocol supplements the immutable run roster; it does not change a frozen method, dataset, source recipe or horizon.

## Purpose

The study compares the prediction quality of the original GNNM and its developed extensions with an ordinary single model and a genuine independent ensemble. Every method must retain its actual shared parameters, private parameters, optimizer owners and prediction rule. Four predictions from one body do not constitute four independent learners.

Roman Empire, Amazon Ratings, Minesweeper, Questions and Tolokers are excluded. Original GNNM remains a comparison on compatible new data. Original-paper scores are unchanged. Source author replication is reported separately from standardized comparisons: different splits, normalization, training policies or backbones cannot support a direct score difference.

## Training evidence

The standardized comparative fits complete their registered horizons. Validation chooses the checkpoint; it does not terminate these fits early. Record full training and validation curves, the first source stopping signal, selected step, final step, every failure and actual cost. Source author tracks retain their published stopping policy instead.

A long registered horizon is not proof of convergence. After complete training, inspect training-loss trends, validation trends and whether the selected checkpoint lies near the budget boundary. Report a budget limitation when these diagnostics do not support convergence. The finite roster's optional 10,000-update alternative is a separate prospective study; no outcome-selected extension is activated by this document. Changes to its fit counts and costs must be materialized and approved before execution.

Hyperparameters are chosen using the declared search seeds and SELECT role. Comparative seeds, masks, coefficients and references are frozen before comparative fits. TEST scores cannot select recipes, continuations, exclusions or a favorable subset. Preserve failed configurations and distinguish a failed search from a completed low-scoring fit.

## Comparisons and uncertainty

For each compatible task, compare each extension with original GNNM, the ordinary single model and its genuine independent ensemble using matched split and comparative seed. Preserve all individual values and report paired differences, including losses and ties. Primary task metrics remain those bound by the source packet: accuracy for node classification, ROC-AUC for the molecular tasks and the official raw link metric. Calibration metrics are secondary to the raw task score.

Do not treat overlapping WikiCS masks or repeated Photo masks as independent datasets. For repeated-mask tasks, average paired differences across the complete declared mask set within each seed, then report seed-level uncertainty with the small seed count stated explicitly. Also show variation across masks separately. Three comparative seeds cannot support a precise uncertainty claim or a reliable tail-probability estimate. No zero-variance or few-seed comparison is declared significant automatically.

On single-mask tasks, report every seed's paired difference. Confidence intervals and any tests must state their unit, assumptions and method. A parametric interval with three seed pairs is descriptive and conditional on its assumptions. A three-pair exact sign test cannot attain a two-sided 0.05 threshold. Do not imply statistical support merely because the mean is positive.

If formal tests are used across the prospectively registered comparisons, preserve the entire comparison family and apply Holm correction within the stated family. Report unadjusted and adjusted values together. Do not calculate correction only for the successful methods. Results encountered during method development remain development evidence; this roster does not certify an unseen confirmation set.

## Error analysis

Use the released member logits to separate three effects: individual member quality, agreement between members and the benefit obtained by combining them. For each paired comparison, report mean member accuracy, ensemble accuracy, disagreement, cases where members are correct together, and cases where one model corrects the other's errors. An oracle that chooses the correct member is a diagnostic upper limit, not a deployable predictor.

Graph strata such as degree, feature similarity and neighborhood-label agreement are diagnostic. Define them from the permitted graph and training information before opening comparative evaluation outcomes. A test-label-based stratum must be identified as post-evaluation descriptive analysis; it cannot guide a newly fitted method or a claimed protected confirmation.

## Reproducibility and cost

Keep the selected checkpoints, member logits, role custody, configuration, source hashes, complete curves and terminal costs on the servers. Git contains source, finite manifests, decision history and compact result metadata. The Mac receives compact research notes, not the full checkpoint archive.

Two slots per GPU permit concurrency when memory is sufficient. Measure actual throughput and wall time; do not divide a serial estimate by two merely because two slots exist. The current TRAIN-pass and serialization measurements use random weights with optimizer actions intercepted. They are resource-planning evidence and contain no scientific score, convergence evidence or whole-study ETA.
