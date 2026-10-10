# Interpreting exploratory screens

This note was written while the PubMed nine-fit and WikiCS six-fit families were incomplete, without reading their quality values. It changes no frozen gate, selector, roster or training source.

## Distinguish the measured question

The first question is whether a change improves factual predictions under the specified training and selection procedure. A smaller gap to an ordinary ensemble can justify more research. Superiority over that ensemble requires a separate comparison with its competent ordinary selection procedure. A gain over a weakened reference does not establish the method's value.

For each complete family, retain every planned seed and condition. Report per-seed differences, pooled quality, mean and worst member quality, class coverage and costs. Identify whether improvements come from repairing shared mistakes, adding correct alternatives or changing how existing alternatives are pooled. Hidden distances and auxiliary loss values are mechanism diagnostics.

## Uncertainty

Optimizer seeds on one graph split measure variability conditional on that split. Connected nodes are not independent replications. Neither node counts nor dependent role-by-node records should enlarge the sample size for a claim across datasets.

With three paired seeds, report all three contrasts and their mean and standard deviation. A Student interval has only two degrees of freedom and needs an approximate normality assumption. An exact two-sided sign-flip test over three pairs cannot yield a probability below0.25 under its exchangeability assumption. Neither approach can turn a three-seed exploratory gain into strong significance evidence. Any intervals or tests are supplementary descriptions and do not alter the frozen continuation rules.

For a supported candidate, freeze the claimed contrasts before unused evaluation, retain all confirmation outcomes and use additional independent splits or tasks. Account for the number of claimed comparisons. Keep validation-selected exploratory numbers separate from final heldout evidence.

## Combining methods

Prefer changes with different measured purposes. One may improve individual prediction quality while another teaches members to use different sources. Compare the common baseline, A, B and A+B. A package is useful only if its final predictions and member quality support it. Preserve failed combinations.

## Next actions

The history researcher reviews complete families and records their remaining implications. The new-method researcher proposes a mechanism with attributed priors and a decisive comparison. The combination researcher tests complementary mechanisms with a bounded roster. Root decides what runs, publishes complete outcomes and updates the paper when the evidence supports a claim. Acceptance remains unmet.
