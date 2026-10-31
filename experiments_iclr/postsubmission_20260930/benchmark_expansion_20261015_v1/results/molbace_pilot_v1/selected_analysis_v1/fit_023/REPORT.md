# Selected molecular VALID analysis

Fit `molecular_input_steering_and_class_geometry_20261015_v1/split-0/seed-303/method-gnnm_input_residual_plus_graph_alignment_variance`; selected epoch 61; 151 graphs; 4 routes.

These are descriptive results on the validation set used for selection. TEST remained closed.

| Prediction | ROC-AUC | BCE | Brier |
|---|---:|---:|---:|
| Member0 | 0.701098901098901 | 1.785093 | 0.514621 |
| Member1 | 0.6945054945054946 | 3.074280 | 0.679535 |
| Member2 | 0.6531135531135531 | 1.392505 | 0.448605 |
| Member3 | 0.5201465201465201 | 0.439736 | 0.125498 |
| Mean raw logits | 0.7135531135531136 | 1.263665 | 0.436449 |

All routes wrong on 0 observed graphs. Per-member repairs/harms, pairwise error overlap and class-wise geometry are in RESULT.json.

Hidden-state cosine, variance, rank and CKA describe geometry. They do not establish accuracy or novelty.

Saved pooled-logit sanity check: True; max absolute difference 4.76837e-07. Its disclosed tolerance is diagnostic; no checkpoint or selection changed.

ROUTE_VALID.pt contains actual route logits, source graph IDs/labels and native preheads. No training callback or optimizer step executed.
