# Selected molecular VALID analysis

Fit `molbace_residual_only_private_bn_20261015_v1/seed-303/gnnm_residual_only_private_bn_v1`; selected epoch 15; 151 graphs; 4 routes.

These are descriptive results on the validation set used for selection. TEST remained closed.

| Prediction | ROC-AUC | BCE | Brier |
|---|---:|---:|---:|
| Member0 | 0.7263736263736265 | 1.067660 | 0.348704 |
| Member1 | 0.7234432234432234 | 1.158189 | 0.375090 |
| Member2 | 0.726007326007326 | 1.072184 | 0.350555 |
| Member3 | 0.7227106227106227 | 1.109343 | 0.359625 |
| Mean raw logits | 0.7241758241758243 | 1.101142 | 0.358418 |

All routes wrong on 77 observed graphs. Per-member repairs/harms, pairwise error overlap and class-wise geometry are in RESULT.json.

Hidden-state cosine, variance, rank and CKA describe geometry. They do not establish accuracy or novelty.

Saved pooled-logit sanity check: True; max absolute difference 9.53674e-07. Its disclosed tolerance is diagnostic; no checkpoint or selection changed.

ROUTE_VALID.pt contains actual route logits, source graph IDs/labels and native preheads. No training callback or optimizer step executed.
