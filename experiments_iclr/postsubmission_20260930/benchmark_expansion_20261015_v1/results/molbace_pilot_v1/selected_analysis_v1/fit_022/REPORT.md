# Selected molecular VALID analysis

Fit `molecular_input_steering_and_class_geometry_20261015_v1/split-0/seed-303/method-gnnm_private_input_residual`; selected epoch 94; 151 graphs; 4 routes.

These are descriptive results on the validation set used for selection. TEST remained closed.

| Prediction | ROC-AUC | BCE | Brier |
|---|---:|---:|---:|
| Member0 | 0.6820512820512821 | 1.178280 | 0.300256 |
| Member1 | 0.6714285714285715 | 6.854144 | 0.850532 |
| Member2 | 0.7637362637362638 | 1.117978 | 0.312994 |
| Member3 | 0.5593406593406595 | 0.550172 | 0.158319 |
| Mean raw logits | 0.7553113553113553 | 1.522330 | 0.429181 |

All routes wrong on 8 observed graphs. Per-member repairs/harms, pairwise error overlap and class-wise geometry are in RESULT.json.

Hidden-state cosine, variance, rank and CKA describe geometry. They do not establish accuracy or novelty.

Saved pooled-logit sanity check: True; max absolute difference 7.15256e-07. Its disclosed tolerance is diagnostic; no checkpoint or selection changed.

ROUTE_VALID.pt contains actual route logits, source graph IDs/labels and native preheads. No training callback or optimizer step executed.
