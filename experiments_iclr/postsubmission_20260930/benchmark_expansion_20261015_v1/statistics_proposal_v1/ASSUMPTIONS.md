# Statistical unit, null and selection assumptions

## Exact sign-flip validity is an assumption, not a consequence of pairing

For a fixed preset and comparator let d_i be the candidate-minus-reference source
metric in paired trial i, after each method's fixed native selection procedure.
The planned statistic is the mean difference, with two-sided tail
`Pr_signs(|mean(s_i*d_i)| >= |mean(d_i)|)` over all sign vectors.

Exactness requires the null distribution to be invariant under the independent
sign flips being used. A sufficient condition is independent symmetric
zero-centered differences, equivalently independent fair signs conditional on
their magnitudes. A mean-zero null alone, arbitrary skewed differences, or a
merely symmetric pooled histogram is insufficient. The same seed labels or
matching split do not establish randomized exchangeability of algorithm labels.
Methods are not randomly assigned as treatments here. Ten differences cannot
reliably verify the assumed symmetry. The design states this assumption; the
audit cannot validate it without inventing outcome evidence.

Missing trials do not become extra sign patterns. Exact zeros yield duplicate
patterns and fewer effective nonzero signs; metric ties can further enlarge a
tail. Do not drop unfavorable or failed trials, impute invented effects or add
unplanned refits to restore n. The current policy requires incomplete presets
and failure reasons to remain visible. This proposal changes no such policy.

## What the ten trials represent

- OGB molecular and Collab use fixed official splits with ten declared optimizer
  seeds. Conditional on the fixed dataset, these concern training randomness,
  subject to the actual RNG/replication assumptions. They do not estimate
  uncertainty over new graphs or scaffold splits.
- PolyFormer has ten declared paired split/seed cohorts, not a ten-by-ten
  Cartesian replication. These share the original graph and draw on its fixed
  labels; claims about new graphs need a different sampling model. An assumption
  about independent random split/training draws conditional on that graph must
  be distinguished from a population of independent datasets.
- Native I4 has four own trained/selected bodies. Its score is one trial-level
  bank result; the four bodies are not four extra paired trials. M1 is included
  body0, inducing dependence between the M1 and I4 contrasts.
- The hundreds or thousands of scored nodes, molecular labels or sampled link
  negatives are not replacements for the ten trial units. Ranking queries share
  endpoints and negative pools; any record-level bootstrap needs its own task
  dependence/cluster assumptions.

## Dependence and multiplicity

Presets reuse datasets, candidate families reuse baselines, and the two reference
contrasts share the candidate. Holm does **not** require these p-values to be
independent: it controls FWER under arbitrary dependence when each input p-value
is valid. That does not solve its discrete first-step threshold at this n.

Simply treating all 226 contrasts as independent evidence, or all 27 presets as
independent datasets, is unjustified. A hierarchical or joint max-statistic
alternative requires its own null/exchangeability model, units, weights and
strong-error-control argument. Marginal sign symmetry alone does not establish
joint vector sign-exchangeability or subset-pivotality for a replacement method.

## Intervals and confirmation

The declared paired percentile bootstrap resamples the ten paired trial rows,
not independently sampled candidate and control scores. Its 95% intervals are
marginal, approximate at this small n, and conditional on the stated unit model.
They are not simultaneous 226-contrast bounds, and 10,000 resamples do not create
10,000 independent trials. Discrete or nearly identical trial effects may give
very narrow or degenerate empirical intervals; that does not remove the p-value
floor or justify a claim of negligible generalization uncertainty.

VALID is used for epoch selection and exploratory method screening. Repeated
VALID analysis is development evidence even when a paired p-value can be
calculated. Untouched TEST can support a prospectively frozen confirmation of
the selected training procedures, provided its role custody and nonuse in design
choices are actually established. More training seeds evaluated on one common
TEST set still do not turn that set into independent population draws.

This audit establishes a mathematical limitation of the **specified** exact test
and Holm family. It is not a finding about algorithm accuracy, equivalence,
manuscript acceptance, or the validity of every possible statistical model.
