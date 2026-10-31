# Selected molecular VALID analysis

Fit `molecular_input_steering_and_class_geometry_20261015_v1/split-0/seed-303/native-body-1`; selected epoch 21; 151 graphs; 1 routes.

These are descriptive results on the validation set used for selection. TEST remained closed.

| Prediction | ROC-AUC | BCE | Brier |
|---|---:|---:|---:|
| Member0 | 0.7333333333333334 | 1.789593 | 0.517537 |
| Mean raw logits | 0.7333333333333334 | 1.789593 | 0.517537 |

All routes wrong on 102 observed graphs. Per-member repairs/harms, pairwise error overlap and class-wise geometry are in RESULT.json.

Hidden-state cosine, variance, rank and CKA describe geometry. They do not establish accuracy or novelty.

Saved pooled-logit sanity check: True; max absolute difference 4.76837e-07. Its disclosed tolerance is diagnostic; no checkpoint or selection changed.

ROUTE_VALID.pt contains actual route logits, source graph IDs/labels and native preheads. No training callback or optimizer step executed.
