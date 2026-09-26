# Paired selected partial versus ENS decisions

Run `python analyze_paired_decisions.py --check` with NumPy. The script reads
the frozen validation selection, selected hard class arrays, official test
references, and final score audits from `validation_tuning/`,
`planetoid_confirmation/`, and `cora_preprocessing/`. It verifies their hashes
and selected score arithmetic, then checks the supplied CSV and JSON exactly.
Without `--check`, it regenerates those outputs.

For each graph and optimizer seed, both models use the same test node IDs.
The partial arm is chosen by the stage's locked validation-only family rule,
and each arm's learning rate and weight decay are its own validation selection.
The four counts partition test nodes into both correct, partial only correct,
ENS only correct, and neither correct. Their exact accuracy difference is
`(partial_only_correct - ens_only_correct) / N`.

Three seed rows for one graph reuse the same test nodes, so the setting
summary pools decisions only as descriptive accounting. Cora raw and
normalized use related data, and primary Cora versus the new raw repeat also
reuse the same public graph. No independent-node intervals or p-values are
reported. The selected partial family searched twelve configurations across
two placements, versus six configurations for ENS; these counts do not remove
that selection-budget difference. They do not reveal a causal training
mechanism.

`python conditional_seed_intervals.py --check` verifies an optional table of
three-seed t intervals with critical value 4.3026527299 (two degrees of
freedom). It treats paired optimizer-seed differences as independent and
approximately normal **conditional on the fixed graph split and frozen
selected candidates**. Candidate selection used these seeds' validation
results, so that selection is not covered. The intervals are not across
graphs, splits, or individual nodes, and are not simultaneous intervals.

`python matched_wikics_seed_intervals.py --check` reads the separate audited
54-cell WikiCS storage-control stage and checks paired PRIVATE-LAST minus
ENS92 and BASE198 seed-level accuracy differences, their score hashes, SD,
SE and illustrative t intervals. It keeps the same fixed-split, selected-
configuration, three-seed limitations. The controls were designed after
earlier outcomes and differ in model width and initial function.

`python matched_cora_seed_intervals.py --check` performs the same score-hash
and descriptive three-seed calculation for the separate post hoc normalized-
Cora storage controls. This is the same public Cora split used in the feature
sensitivity study, so it adds optimization repetitions, not independent graph
evidence.
