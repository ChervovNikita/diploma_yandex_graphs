# Selected molecular VALID analysis

Fit `molecular_input_steering_and_class_geometry_20261015_v1/split-0/seed-101/method-gnnm_private_input_residual`; selected epoch 38; 151 graphs; 4 routes.

These are descriptive results on the validation set used for selection. TEST remained closed.

| Prediction | ROC-AUC | BCE | Brier |
|---|---:|---:|---:|
| Member0 | 0.5992673992673992 | 0.757999 | 0.258682 |
| Member1 | 0.6241758241758242 | 0.748006 | 0.260815 |
| Member2 | 0.571062271062271 | 0.801327 | 0.281733 |
| Member3 | 0.7117216117216117 | 0.392703 | 0.120581 |
| Mean raw logits | 0.6772893772893774 | 0.546663 | 0.183464 |

All routes wrong on 2 observed graphs. Per-member repairs/harms, pairwise error overlap and class-wise geometry are in RESULT.json.

Hidden-state cosine, variance, rank and CKA describe geometry. They do not establish accuracy or novelty.

Saved pooled-logit sanity check: True; max absolute difference 2.38419e-07. Its disclosed tolerance is diagnostic; no checkpoint or selection changed.

ROUTE_VALID.pt contains actual route logits, source graph IDs/labels and native preheads. No training callback or optimizer step executed.
