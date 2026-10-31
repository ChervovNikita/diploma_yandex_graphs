# Selected molecular VALID analysis

Fit `molecular_input_steering_and_class_geometry_20261015_v1/split-0/seed-101/method-gnnm_private_bn_state_v3`; selected epoch 92; 151 graphs; 4 routes.

These are descriptive results on the validation set used for selection. TEST remained closed.

| Prediction | ROC-AUC | BCE | Brier |
|---|---:|---:|---:|
| Member0 | 0.7487179487179487 | 0.588169 | 0.194234 |
| Member1 | 0.7564102564102565 | 0.598046 | 0.196248 |
| Member2 | 0.6908424908424908 | 0.626493 | 0.204844 |
| Member3 | 0.7322344322344323 | 0.551524 | 0.182432 |
| Mean raw logits | 0.7483516483516484 | 0.560477 | 0.184069 |

All routes wrong on 28 observed graphs. Per-member repairs/harms, pairwise error overlap and class-wise geometry are in RESULT.json.

Hidden-state cosine, variance, rank and CKA describe geometry. They do not establish accuracy or novelty.

Saved pooled-logit sanity check: True; max absolute difference 4.76837e-07. Its disclosed tolerance is diagnostic; no checkpoint or selection changed.

ROUTE_VALID.pt contains actual route logits, source graph IDs/labels and native preheads. No training callback or optimizer step executed.
