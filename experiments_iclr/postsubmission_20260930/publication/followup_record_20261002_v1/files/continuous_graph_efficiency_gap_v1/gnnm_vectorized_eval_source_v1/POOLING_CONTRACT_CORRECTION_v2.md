# Correction to the prospective output contract

The frozen `QUALIFICATION.md`, section 3, incorrectly identifies averaging member probabilities as the canonical GNNM prediction. The original GNNM and the current correction pilot average **raw member logits**. Multiclass probabilities for that primary predictor are `softmax(mean(member_logits))`.

This addendum supersedes only that pooling statement. The existing evaluator returns all four raw member-logit tensors and requires no code change. Its source hash, numerical equivalence gates, full-trajectory requirement and fair optimized-control comparisons remain unchanged. The earlier packet is preserved.

The primary prediction, covariance correction, calibration and final-set comparison must use the pooling rule declared by the actual frozen study. Mean member probabilities may be computed as a separately named, fixed secondary from the same logits; they cannot silently replace the primary prediction or select a favorable pooling rule. Covariance features still center member probabilities about their mean. Those features do not redefine the canonical point predictor.

No evaluator has been qualified or benchmarked, and no scores, trained models or original results were changed by this correction.
