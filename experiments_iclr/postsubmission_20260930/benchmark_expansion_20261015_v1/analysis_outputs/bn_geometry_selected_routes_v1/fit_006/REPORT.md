# Selected molecular VALID analysis

Fit `molbace_bn_geometry_correction_20261015_v1/seed-303/gnnm_private_bn_graph_alignment_variance_v1`; selected epoch 128; 151 graphs; 4 routes.

These are descriptive results on the validation set used for selection. TEST remained closed.

| Prediction | ROC-AUC | BCE | Brier |
|---|---:|---:|---:|
| Member0 | 0.7428571428571429 | 2.113729 | 0.434555 |
| Member1 | 0.7406593406593407 | 2.346215 | 0.475533 |
| Member2 | 0.7479853479853481 | 2.242400 | 0.451568 |
| Member3 | 0.6846153846153846 | 1.341993 | 0.355882 |
| Mean raw logits | 0.7380952380952381 | 1.946824 | 0.432298 |

All routes wrong on 57 observed graphs. Per-member repairs/harms, pairwise error overlap and class-wise geometry are in RESULT.json.

Hidden-state cosine, variance, rank and CKA describe geometry. They do not establish accuracy or novelty.

Saved pooled-logit sanity check: True; max absolute difference 9.53674e-07. Its disclosed tolerance is diagnostic; no checkpoint or selection changed.

ROUTE_VALID.pt contains actual route logits, source graph IDs/labels and native preheads. No training callback or optimizer step executed.
