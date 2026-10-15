# Complete fixed aggregation comparison

Before execution: use all 24 completed parent MolBACE fits and all three completed
private-BN follow-up fits. Reuse their saved selected VALID route logits only.
No new data, model construction, forward, training, selection or test access.

For each seed101/202/303, compare the same two fixed prediction rules for M1,
I4, G0, alignment+variance, private correction, both and privateBN: arithmetic
mean raw logits versus arithmetic mean sigmoid probabilities. Reuse the previous
stable FP64 log-probability BCE and native OGB AUC implementation. I4 concatenates
its four genuinely independent member banks; M1 uses only body0. All151 graph
identities and labels must align exactly. Preserve all21 comparisons and all
failures, with no choice of the better rule after viewing results.

Report each AUC/BCE, paired readout difference, three-seed mean and SD and the
remaining I4 difference under each common rule. Checkpoint selection remains
that of the original mean-logit protocols. These are posthoc selected-VALID
sensitivity estimates, not fresh evidence for selecting a superior method.
No additions or substitutions to6460, no rerun or replacement of original scores.
CPU2 only, CUDA hidden, maximum60seconds, no retries. Tensor inputs remain on
allocation; only source, compact scalar results and costs are published.
