# Selected molecular VALID analysis

Fit `molecular_input_steering_and_class_geometry_20261015_v1/split-0/seed-101/method-gnnm_input_residual_plus_graph_alignment_variance`; selected epoch 2; 151 graphs; 4 routes.

These are descriptive results on the validation set used for selection. TEST remained closed.

| Prediction | ROC-AUC | BCE | Brier |
|---|---:|---:|---:|
| Member0 | 0.4853479853479854 | 0.475832 | 0.148322 |
| Member1 | 0.5747252747252748 | 0.463380 | 0.143911 |
| Member2 | 0.6014652014652015 | 1.449066 | 0.539549 |
| Member3 | 0.6732600732600733 | 0.391426 | 0.118238 |
| Mean raw logits | 0.6835164835164835 | 0.516817 | 0.165912 |

All routes wrong on 1 observed graphs. Per-member repairs/harms, pairwise error overlap and class-wise geometry are in RESULT.json.

Hidden-state cosine, variance, rank and CKA describe geometry. They do not establish accuracy or novelty.

Saved pooled-logit sanity check: True; max absolute difference 2.38419e-07. Its disclosed tolerance is diagnostic; no checkpoint or selection changed.

ROUTE_VALID.pt contains actual route logits, source graph IDs/labels and native preheads. No training callback or optimizer step executed.
