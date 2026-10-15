# Residual-only routes: stronger ranking, little ensemble benefit

All three frozen MolBACE/GIN fits and their selected-state VALID collections
finished successfully. The new method removes the BatchEnsemble factors and
uses four rank-eight additive input corrections with private BN running state.
It keeps one native GIN and adds 19,200 trainable parameters, or 1.0183%.

## Full paired results

AUC is in percent. BCE is the mean binary cross-entropy over all 151 official
VALID molecules; lower is better. These are selected validation results on one
already-exposed split, not independent confirmation.

| Method | Mean AUC ± sample SD | Mean BCE |
| --- | ---: | ---: |
| Native single | 73.321 ± 1.172 | 0.779 |
| Genuine independent four-model ensemble | 74.847 ± 0.500 | 0.890 |
| Original GNNM | 67.778 ± 4.390 | 0.719 |
| GNNM with private BN | 73.614 ± 1.057 | 1.086 |
| GNNM with private BN and residuals | 72.186 ± 1.412 | 0.816 |
| Residual-only routes with private BN | 74.518 ± 2.660 | 1.952 |

Removing the factors improves AUC over factors-plus-residuals in all three
seeds: +4.139, +1.062 and +1.795 points. The mean gain is +2.332 points.
However, BCE worsens in all three. Against I4, AUC changes are +3.114, −1.758
and −2.344 points: a mean −0.330, with worse BCE in every seed. This is not
an improvement over I4, a tie established by equivalence testing, or evidence
of superior probability predictions.

For scale, a conventional paired Student-t 95% interval with three seeds is
[-1.661, 6.325] points for the factors-plus-residuals contrast, and
[-7.773, 7.114] for I4. These descriptive seed intervals assume independent
approximately normal differences, are unadjusted for the developmental
comparisons, and do not capture uncertainty over new molecular scaffolds.
The 2,730 positive-negative pairs share molecules and are not independent
training repetitions.

## What the predictions reveal

Mean individual-route AUC rises to 74.560%, versus 73.410% for I4 members.
Yet pooling changes route-mean AUC by −0.073, −0.009 and −0.046 points in the
three seeds. I4 instead gains 1.438 points on average from pooling.

The shared update also averages losses from four stochastic route forwards.
Consequently, stronger individual-route ranking does not isolate the benefit
of the additive parameters from the effect of shared training with multiple
dropout realizations. The next structural-versus-uniform comparison retains
the same four forwards, parameterization and native dropout setting.

The residual routes share 585/614, 683/720 and 726/753 of their pool's ranking
reversals: 1,994 of 2,087 reversals in total, or 95.54%. Only 31 pool errors
per seed on average have any correctly ranking member. Thus the new routes
have much more similar ranking errors than I4, whose corresponding common
reversal share is 49.71%. These are prediction diagnostics, not proof that
hidden representations are identical or that an input-dependent readout
could never help. In binary ranking, input-dependent convex weights can
sometimes repair a pair reversed by every member.

The useful next question is whether persistent, supervised route
responsibilities can create useful differences without losing this ranking
competence. The prepared structural-versus-permuted comparison tests that
question with the same-capacity uniform-responsibility method as its control.
No embedding-repulsion strength, initialization scale or readout is tuned here.
Shared/private loss separation is attributed to prior work, including ACE;
graph-structural responsibility remains an unproven adaptation.

A subsequent fixed-bank CPU diagnostic bounds the already-proposed convex
readout with a 0.05 minimum weight per route. Even a label-informed choice of
the most favorable permitted score for each molecule has AUC ceilings of
78.681%, 75.421% and 74.505%, respectively: at most +1.172, +1.795 and +2.088
points over the saved pools. These are optimistic endpoint bounds, not
learnable gates or new model scores. They leave some room for aggregation,
but the third seed's ceiling remains below its I4 result (74.762%). This helps
prioritize improving route predictions without ruling out a later controlled
readout experiment. The diagnostic used saved VALID logits only, took 3.77
seconds, and added no forward or optimizer step. Its source and exact outputs
are in `analysis/residual_gate_feasibility_v1` at the expansion root.

## Work and limits

The three fits completed 186, 149 and 149 epochs, selected epochs 136, 9 and
15, and stopped by the frozen patience rule. They used 18,392 optimizer
updates in 4,094.57 summed worker-seconds. No fit reached the 1,000-epoch cap.
Maximum CUDA allocated memory among these fits was 380,173,312 bytes.
Three selected-state collections added 60 native forwards, zero training and
25.33 seconds of supervisor wall time. No controls were retrained or
recollected. The combined report retains 33 banks from 39 unique paid fits.

VALID has 130 positive and 21 negative molecules. Three seed repetitions do
not address split uncertainty, and repeated development on these molecules
limits generalization claims. A later useful recipe needs a separately frozen
transfer or confirmation study. TEST remains closed, original paper scores
are unchanged, and the large campaign has not been launched.
