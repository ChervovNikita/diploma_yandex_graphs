# Selected molecular VALID analysis

Fit `molecular_input_steering_and_class_geometry_20261015_v1/split-0/seed-303/method-gnnm_private_bn_state_v3`; selected epoch 116; 151 graphs; 4 routes.

These are descriptive results on the validation set used for selection. TEST remained closed.

| Prediction | ROC-AUC | BCE | Brier |
|---|---:|---:|---:|
| Member0 | 0.7241758241758242 | 1.521083 | 0.375762 |
| Member1 | 0.7505494505494507 | 2.488505 | 0.525565 |
| Member2 | 0.7175824175824176 | 1.711810 | 0.430832 |
| Member3 | 0.7058608058608059 | 1.918998 | 0.436616 |
| Mean raw logits | 0.7300366300366301 | 1.853881 | 0.448652 |

All routes wrong on 64 observed graphs. Per-member repairs/harms, pairwise error overlap and class-wise geometry are in RESULT.json.

Hidden-state cosine, variance, rank and CKA describe geometry. They do not establish accuracy or novelty.

Saved pooled-logit sanity check: True; max absolute difference 9.53674e-07. Its disclosed tolerance is diagnostic; no checkpoint or selection changed.

ROUTE_VALID.pt contains actual route logits, source graph IDs/labels and native preheads. No training callback or optimizer step executed.
