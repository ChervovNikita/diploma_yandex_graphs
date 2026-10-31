# Paired analysis for the complete BUDDY pilot

Prepared before root access to this family's validation outcomes or final scores, on 2 October 2026. Training is already running. This supplements the existing three contrasts rather than changing the training recipe, checkpoints, pooling or development task. It is not a registration made before training began.

Use every locked cell on complete ogbl-collab. Report the five arms and seeds 0, 1 and 2 without selecting favorable cells. The three comparisons fixed in the existing family README are factorized4 minus single256, independent4 and matched_single. Keep native1024 as the absolute competence reference.

Hits@50 is the fraction of official positive edges ranked above the fiftieth highest score among the same official 100,000 negative edges. Multiply differences by 100 to report percentage points. There are three paired optimizer seeds on one fixed graph and temporal split. Four members of an ensemble do not make four statistical replicates.

For every comparison, report all three differences, their mean, sample standard deviation and range. Add a paired Student t interval with two degrees of freedom at 95% coverage. Its interpretation requires independent seed differences and an approximately normal distribution of differences. Three values cannot check that assumption. Also show Bonferroni simultaneous intervals for the three fixed comparisons, using a per-comparison coverage of 1 - 0.05/3.

Report a two-sided sign reference p-value with zero differences removed, and Holm adjustment across all three comparisons. The sign reference assumes independent seeds and equiprobable signs under a zero-median null. It is not a randomization test of assigned treatments. With three nonzero pairs, its smallest possible two-sided value is 0.25. These values cannot certify superiority at 0.05.

The intervals describe variation across optimizer seeds conditional on this graph, fixed negatives and split. They do not measure uncertainty across graph populations, chronological periods or nodes. Do not resample overlapping edges as if they were independent observations. Do not reinterpret a practical effect as state of the art, calibrated uncertainty or methodological novelty.

Only scalar JSON metadata from completed, audited final evaluation is read. The analyzer verifies all fifteen final-result hashes against the completed evaluation receipt and the complete family lock. It uses no labels, checkpoints, logits, tensor library, extra model forwards or training. Original scores stay unchanged. The report includes input and source hashes. Existing timing and storage evidence must accompany any later efficiency claim, with shared-host contention stated.
