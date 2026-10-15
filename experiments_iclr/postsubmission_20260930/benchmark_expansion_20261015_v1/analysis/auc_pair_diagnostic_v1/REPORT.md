# Ranking failures and pooling on the first molecular seed

This posthoc diagnostic uses saved selected-VALID predictions only. It adds no
models, forwards, optimizer steps, checkpoint selections or TEST access. The
151 graphs yield 2,730 positive-negative pairs; the pairs reuse graphs and
are not independent experimental repetitions. Native OGB AUC and direct
pair counting agree. The three-seed and I4 comparisons remain incomplete.

| Model | Pool AUC | Reversed pairs | Reversed by every route | Pool reversed despite some correct route | Pool reversed despite a strict correct majority |
| --- | ---: | ---: | ---: | ---: | ---: |
| M1 | 0.73846 | 714 | 714 | 0 | 0 |
| base | 0.62784 | 1016 | 125 | 891 | 167 |
| C1 | 0.71062 | 790 | 218 | 572 | 37 |
| C4 | 0.67729 | 881 | 147 | 734 | 70 |

## What this adds to the error diagnosis

Threshold mistakes were a poor description of this AUC task. A route can make
many threshold mistakes because of a constant class bias while retaining the
same ranking. The relevant unit here is whether a positive graph scores above
a negative graph. Pair margins remove each route's constant logit offset.

Base GNNM reverses 1,016 pairs. For 891 of them at least one route orders the
pair correctly, and for 167 a strict majority does. This identifies a real
pooling problem in addition to weak individual rankings. It does not imply
a realizable gate can recover those pairs simultaneously. Different pairwise
choices may be incompatible with one score per graph.

The alignment/variance model reverses 790 pairs. Relative to base, it repairs
422 and harms 196 pairs. Strict-majority cancellations fall from 167 to 37,
while pairs reversed by every route increase from 125 to 218. Its higher AUC
therefore coexists with less pairwise member coverage. Maximizing disagreement
or member coverage alone would not select the better pool in this example.

The base route with the largest mean absolute pair margin has value 1.907,
versus 0.769–0.826 for the other routes. Alignment/variance has values
0.514–0.780. This is a score-scale diagnostic, not proof that scale caused
the AUC difference: training, selected epochs and learned functions also differ.

The separate fixed mean-probability check changes base AUC by only +0.00366
and alignment/variance AUC by -0.00037, while worsening BCE for both. That
specific readout swap is not a demonstrated solution. It does not exclude
jointly trained aggregation, other cohorts or normalization-state effects.

## Consequences for the prospective studies

Keep member competence, common ranking failures and actual pooled cancellation
as distinct diagnostics. Route-specific factual targets and structural
responsibilities should be judged by task metric and repairs/harms, not by
latent distance or oracle coverage. The existing adapter/geometry interaction,
controlled aggregation and private-BN hypotheses remain worth completing.
This one seed does not identify a successful new method, a cause, statistical
significance or a manuscript claim. No settings are changed from these counts.

In exact arithmetic a raw-score mean cannot repair a pair strictly reversed
by every route, because its pair margin is the mean of member margins.
The diagnostic finds zero such repairs. This is a standard algebraic fact,
not a new ensemble theorem or a claim about multiclass argmax coverage.
