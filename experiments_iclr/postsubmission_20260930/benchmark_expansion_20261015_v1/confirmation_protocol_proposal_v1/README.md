# Exploration and independent confirmation

The 6,460-fit matrix is intended to find useful mechanisms. Comparing many
methods and reporting the best result does not establish that the winner will
repeat. This proposal separates those two questions. It does not authorize
training or change the existing roster.

## Broad exploration

Keep all ten declared cohorts, native reproductions, matched singles,
independent ensembles, original GNNM and applicable candidate methods. Use only
TRAIN/VALID to develop methods and select checkpoints. Report failures and
every declared comparison, including unsuccessful combinations. The current
three-seed MolBACE pilot remains a separate development study.

Select at most one method in one reference setting for confirmation. Rank
eligible candidates by their mean VALID improvement over the strongest of the
three required baselines in each paired cohort. This maximum is a ranking
statistic; it does not define a new served baseline. Require a positive mean
and resolve ties by fixed IDs. Freeze the exact candidate, native settings,
source manifests, controller, evaluator, pooling rule and confirmation seeds
before examining confirmation outcomes. No positive candidate means there is
no promotion. A failed or capped training trajectory is disclosed and cannot
silently become a successful converged fit.

## Fresh confirmation, if justified

Propose **20 new paired cohorts** for that one setting. Each cohort trains four
ordinary native bodies, original GNNM and the selected shared method. The
matched single reuses body0. This is **120 additional physical fits**, separate
from 6,460, and needs a reviewed compute budget and campaign GO. Reference-paper
reproductions do not need another rerun.

The primary claim is deliberately specific: the candidate wins in more than
half of training repeats against **each** required baseline on this setting.
Use three one-sided exact sign tests and require all three to pass at .05. The
joint p-value is their maximum. This intersection-union test does not need a
threefold correction for that one joint claim. It cannot justify picking the
easiest comparator or adding more winning settings afterward. Ties and unresolved
outcomes count as non-wins in the fixed denominator. At least 15 of 20 wins are
needed for each component. Independence of repeat RNG streams is required;
multiple graph pairs, routes or molecular observations are not extra repeats.

Also report every paired score, mean difference, SD, median, win/tie/loss count,
and an explicitly approximate paired-bootstrap interval for the mean difference.
The sign test concerns the probability of improvement; it does not test the
mean effect. An interval across seeds describes repeat-training variation on
the declared data protocol. It does not measure uncertainty over new molecules,
graphs or datasets. Twenty cohorts is a fixed proposed budget, not evidence of
adequate power or a promised favorable result.

## Heldout scoring and historical exposure

TEST remains closed during exploration. Release its metrics only after candidate
and comparison-family freeze and after all checkpoint selection is finished.
Audit earlier TEST exposure before calling anything untouched. Old dataset
selection cannot be undone by changing seeds. On fixed official splits, the
confirmation supports repeat-training reliability conditional on that dataset;
it cannot establish universal transfer or a new-data-population effect.

If the user chooses to score all exploratory methods on TEST, preserve that
complete matrix and label it exploratory. TEST results cannot revise the frozen
primary candidate, pooling or checkpoint. Later changes need another genuinely
separate evaluation. Additional primary cells or candidates require a new
prospective multiplicity plan.

## Why replace the old testing rule

With ten paired observations, the smallest two-sided sign-flip p-value is
2/1024. Holm over 226 comparisons begins at .05/226, so even perfect signs cannot
pass. This is a design limitation, not a reason to replace the test after seeing
scores. The proposed small, frozen confirmation family avoids making a broad
exploratory grid masquerade as confirmation.

`PROPOSAL.json` contains the complete proposed contract. `paired_statistics.py`
implements the declared sign test and descriptive summaries without importing
models or reading any result directory. No scientific outcomes are generated
by its semantic checks. This proposal must be integrated with the access ledger,
seed registry and measured budget before it becomes a frozen executable study.
