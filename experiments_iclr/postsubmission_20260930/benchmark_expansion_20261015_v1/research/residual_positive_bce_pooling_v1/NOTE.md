# Residual positive BCE and the missing fixed-pool comparison

The completed residual-only bank has similar mean AUC to I4 (74.518% versus 74.847%), but substantially worse positive-class BCE. Its negative-class BCE is lower. This is a class-specific probability failure, not evidence that every wrong prediction is more confident.

| Seed | I4 positive BCE | Residual positive BCE | I4 negative BCE | Residual negative BCE | Residual native-member BCE range |
| --- | ---: | ---: | ---: | ---: | ---: |
| 101 | 1.066595 | 3.919313 | 0.418116 | 0.086143 | 3.352229–3.419872 |
| 202 | 0.800885 | 1.560093 | 0.463946 | 0.184184 | 1.350826–1.383646 |
| 303 | 1.036160 | 1.226475 | 0.345228 | 0.325271 | 1.067660–1.158189 |

The four native routes have similarly poor aggregate BCE within each residual seed. This rules out one conspicuously disastrous whole route as the explanation. It does not establish per-example agreement. The small gap between mean member BCE and logit-pool BCE does not bound the benefit from probability averaging when logits are saturated.

## What is already known

`analysis/complete_fixed_aggregation_v1/REPORT.md` already compares the two fixed pools for seven methods across three seeds. It omits the three remaining private-BN factorial cells and the current residual-only bank. It also does not answer how much of the current positive-class excess survives each pool. Thus the missing comparison is a narrow extension to the fixed current33, not a new aggregation proposal.

The algebra and objective discussion already appears in `research/quality_specialization_synthesis_v1/SYNTHESIS.md`. Mean logits is a normalized geometric opinion pool; mean probabilities is an arithmetic mixture. Neither universally dominates BCE or AUC. Probability averaging may soften common wrong confidence, but it cannot repair a positive-negative ranking reversal shared strictly by every route. The earlier 95.54% common-reversal figure refers to residual pooled reversals, not all VALID pairs.

## What the evidence does not identify

The observed TRAIN/VALID positive fractions (39.67%/86.09%) do not establish pure label shift or explain residual-versus-I4 excess. Shared weights, shared BN affine parameters, route residuals, evaluation normalization and validation selection remain possible contributors. The completed private-BN reassignment and mixture-moment interventions used older private-BN checkpoints; they are not causal evidence for the current residual-only positive loss.

Residual-F trains the mean of the four own BCE losses with ordinary Adam. With ordinary TRAIN BN, changing only ownership of running buffers leaves a matched TRAIN forward/gradient/Adam prefix mathematically unchanged; evaluation normalization and the selected stopping prefix can differ. Private running buffers therefore do not by themselves establish improved TRAIN gradients. A factor of one quarter on private gradients is also not an automatic Adam learning-rate explanation.

## Minimal source-only extension

`run.py` uses the completed probability diagnostic's verified receipt loader and frozen all33 authority. It loads only the 39 saved selected VALID exports and 27 paid saved route collections. It compares the authoritative saved logit pool with uniform mean sigmoid probabilities for every bank, without refitting or selecting a rule. Mixture BCE uses stable log-sigmoid/logsumexp arithmetic; AUC uses stable mixture log odds with half credit for exact numerical ties. Class-only AUC is undefined.

The short report centers on the signed residual-minus-I4 positive BCE excess under both rules and any AUC change. Scalar JSON retains every bank/seed, both class BCE/Brier, overall AUC, fixed-threshold errors and reconstruction discrepancies. No per-example artifact is emitted. This cannot establish a training cause, held-out improvement or new serving choice.

The source is prepared for one root-reviewed CPU pass. No scientific arrays, models or checkpoints were loaded during preparation. Root owns execution and publication. CLI: `python run.py --manifest-sha256 <reviewed digest> --output-dir <fresh direct child of allocation analysis_outputs>`.
