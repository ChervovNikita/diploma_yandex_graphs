# Selected molecular VALID analysis

Fit `molecular_input_steering_and_class_geometry_20261015_v1/split-0/seed-202/method-gnnm_private_bn_state_v3`; selected epoch 54; 151 graphs; 4 routes.

These are descriptive results on the validation set used for selection. TEST remained closed.

| Prediction | ROC-AUC | BCE | Brier |
|---|---:|---:|---:|
| Member0 | 0.7135531135531136 | 0.855238 | 0.263896 |
| Member1 | 0.7399267399267399 | 0.883025 | 0.251564 |
| Member2 | 0.7120879120879121 | 0.856811 | 0.281669 |
| Member3 | 0.7234432234432234 | 0.858385 | 0.273123 |
| Mean raw logits | 0.7300366300366301 | 0.842212 | 0.262220 |

All routes wrong on 41 observed graphs. Per-member repairs/harms, pairwise error overlap and class-wise geometry are in RESULT.json.

Hidden-state cosine, variance, rank and CKA describe geometry. They do not establish accuracy or novelty.

Saved pooled-logit sanity check: True; max absolute difference 4.76837e-07. Its disclosed tolerance is diagnostic; no checkpoint or selection changed.

ROUTE_VALID.pt contains actual route logits, source graph IDs/labels and native preheads. No training callback or optimizer step executed.
