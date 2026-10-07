# Correct interpretation of complete9 error diagnostics

This successor retains the original same-member/all-rivals logic. It adds new common-rival counts over the full development population, alongside counts restricted to the original COMMON obstruction cohort. Candidate correctness on that cohort is recorded separately from strict baseline-error to candidate-correct repair. All actual correctness uses the saved native pool prediction.

Two fields in the sealed closed9 reader have broader semantics than their names suggest. `served_repairs_on_baseline_common_competitor` counts candidate pool correctness on the baseline obstruction cohort. `acquired_and_served_repaired` also requires reversal of its retained rival, but does not explicitly require baseline pool error. Usually a common rival implies a wrong baseline pool. Finite arithmetic can break that implication. This adapter reports the actual baseline/candidate transitions and must supply the strict repair counts used in interpretation. The original fields and sources remain unchanged and visible.

The COMMON and PERMUTED rank comparisons also use different rival identities in the sealed readout. This adapter consistently uses all COMMON rivals for both ROUTE and PERMUTED candidates. These are different diagnostic comparisons, not interchangeable columns.

No scientific arrays were opened in source preparation. The original four synthetic logic checks are rerun. No new model calls or scientific training are required. Root whole9 terminal/collection admission still applies. No frozen source, target, quality gate or study is changed. This is a semantic correction, not a numerical parity gate.
