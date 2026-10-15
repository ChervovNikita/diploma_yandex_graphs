# First-seed diagnostic, not a method comparison result

Source: immutable selected-VALID analysis captured in OBSERVATION_11.json,
native seed101/body0 and base GNNM seed101. No new scoring or training was run
for this note. The other independent native members and full paired trials
are not yet complete. All observations below are exploratory and checkpoint-
selected on the same VALID population.

## Prediction quality and disagreement are different measurements

The151 VALID graphs contain130 positive and21 negative labels. Predicting the
majority label everywhere gives86.09% threshold accuracy. The native single
has ROC-AUC.73846, BCE.92069 and threshold accuracy56.95%; base GNNM has
ROC-AUC.62784, BCE.49848 and threshold accuracy82.12%. These metrics measure
different things. Improved BCE or threshold accuracy here does not imply
improved ranking, the declared source endpoint. Neither fit is compared with
a completed native I4 yet.

Base GNNM's individual route AUCs are.53150,.50366,.58205,.60879. Its threshold
error counts are23,31,98,23, with27 pooled errors. Route2 disagrees with the
other routes on roughly77–79% of graphs and has17 unique threshold-correct
graphs, yet its AUC is only.58205 and BCE1.54004. The union contains150 graphs
on which at least one route is correct. That union can arise largely from
opposing class predictions; it is not evidence of a competent almost-perfect
predictor or a realizable gate. No oracle result is promoted.

The selected prehead norms for route2 are about58.53/54.48 within the two
classes, versus roughly6–8 for the other routes. Its normalized within-class
scatter is also much smaller. These are descriptive source-state observations,
not proof that BatchNorm caused the weakness or that a latent penalty repairs it.
Native/base selected epochs differ98 versus2, so their geometry does not
isolate a causal training effect.

## Consequences for the prepared experiments

- Keep the complete frozen contrastive/input-adapter2×2 and all three trials.
  No coefficient, selector, seed or arm is changed in response to this readout.
- Judge candidates by source ranking and proper loss against complete base/I4
  controls. Record per-route competence, common errors and mean pooling effects.
  Increased latent separation or disagreement alone is not a success criterion.
- The prospectively frozen BN follow-up tests a concrete state-ownership
  explanation using fresh candidate fits and the exact parent controls. It
  neither reconstructs missing private histories nor assumes the cause is proven.
- A conditional aggregator must learn on TRAIN and retain mean/global controls
  under the same loss. The existence of a rare correct route is not sufficient
  to establish that its correctness is predictable from available gate inputs.

No TEST inference, statistical significance, broad transfer, novelty or new
successful method is claimed. Negative outcomes remain part of the study.
