# Selected molecular VALID analysis

Fit `molecular_input_steering_and_class_geometry_20261015_v1/split-0/seed-303/method-gnnm_boundary_native`; selected epoch 15; 151 graphs; 4 routes.

These are descriptive results on the validation set used for selection. TEST remained closed.

| Prediction | ROC-AUC | BCE | Brier |
|---|---:|---:|---:|
| Member0 | 0.5992673992673992 | 0.443032 | 0.135874 |
| Member1 | 0.6029304029304029 | 2.180414 | 0.649380 |
| Member2 | 0.743956043956044 | 0.464195 | 0.157304 |
| Member3 | 0.6274725274725276 | 1.228276 | 0.466446 |
| Mean raw logits | 0.7102564102564103 | 0.785984 | 0.285051 |

All routes wrong on 5 observed graphs. Per-member repairs/harms, pairwise error overlap and class-wise geometry are in RESULT.json.

Hidden-state cosine, variance, rank and CKA describe geometry. They do not establish accuracy or novelty.

Saved pooled-logit sanity check: True; max absolute difference 2.38419e-07. Its disclosed tolerance is diagnostic; no checkpoint or selection changed.

ROUTE_VALID.pt contains actual route logits, source graph IDs/labels and native preheads. No training callback or optimizer step executed.
