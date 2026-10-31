# Citeseer initializer source review

No blocking scientific or numerical defect was found in the exact reviewed driver and initializer source. The review is static and does not grant scientific execution authorization.

The five shared F4 variants have the same patched diagonal-factor function class, the same TRAIN-only 20-cycle common warm state, fixed RMS1 initialization, all dense and private parameters learned, and 60 postwarm cycles with complete VALID evaluation every five cycles. Covariance roots, two paired positive and negative directions, and global RMS normalization are correct. The full-TRAIN feature covariance control uses the same graph-derived hidden features.

The independent warm control starts all four routes from the one identical common warm checkpoint. Its returned partition still carries the donor build label `independent_seed_plus5member`; change that metadata to `same_common_warm_checkpoint; independent_postwarm_dropout_and_query_streams`. It is four independent continuations, not four independently trained warm models.

The single uses the union of four inner query banks, while F4 members each use one bank. Each independent single also uses a union with separate query and dropout streams, charging four postwarm fits. The ensemble selects one pooled VALID checkpoint. MRR is rounded to four decimals before strict-greater selection, so rounded ties retain the earlier checkpoint. These details must remain explicit in the comparison.

Exact neutral native copying is asserted by the driver on complete deterministic TRAIN logits before fitting. This static review has not executed that assertion. The bias patch is algebraically the donor factor map but changes floating arithmetic and potentially its gradient trajectory; all five F4 variants share it.

No model, dataset, label, tensor, server, or scientific fit was accessed. AST parsing and source hash verification were the only executed checks. Source bindings and further check detail are in REVIEW.json.
