# Selected molecular VALID analysis

Fit `molecular_input_steering_and_class_geometry_20261015_v1/split-0/seed-202/method-gnnm_private_input_residual`; selected epoch 160; 151 graphs; 4 routes.

These are descriptive results on the validation set used for selection. TEST remained closed.

| Prediction | ROC-AUC | BCE | Brier |
|---|---:|---:|---:|
| Member0 | 0.7065934065934065 | 0.750572 | 0.196975 |
| Member1 | 0.5926739926739927 | 1.638972 | 0.327713 |
| Member2 | 0.7014652014652015 | 0.939480 | 0.224268 |
| Member3 | 0.7450549450549451 | 0.857458 | 0.206667 |
| Mean raw logits | 0.7201465201465201 | 0.753373 | 0.210403 |

All routes wrong on 14 observed graphs. Per-member repairs/harms, pairwise error overlap and class-wise geometry are in RESULT.json.

Hidden-state cosine, variance, rank and CKA describe geometry. They do not establish accuracy or novelty.

Saved pooled-logit sanity check: True; max absolute difference 9.53674e-07. Its disclosed tolerance is diagnostic; no checkpoint or selection changed.

ROUTE_VALID.pt contains actual route logits, source graph IDs/labels and native preheads. No training callback or optimizer step executed.
