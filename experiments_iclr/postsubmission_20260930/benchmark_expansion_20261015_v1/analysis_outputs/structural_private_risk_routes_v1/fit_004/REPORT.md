# Selected molecular VALID analysis

Fit `molbace_residual_only_private_bn_20261015_v1/seed-303/gnnm_residual_only_private_bn_v1/residual_private_bn_structural_risk_v2`; selected epoch 24; 151 graphs; 4 routes.

These are descriptive results on the validation set used for selection. TEST remained closed.

| Prediction | ROC-AUC | BCE | Brier |
|---|---:|---:|---:|
| Member0 | 0.7527472527472528 | 1.782163 | 0.538292 |
| Member1 | 0.7366300366300366 | 1.534486 | 0.469731 |
| Member2 | 0.7498168498168498 | 1.741064 | 0.524314 |
| Member3 | 0.7523809523809524 | 2.007939 | 0.577626 |
| Mean raw logits | 0.7509157509157509 | 1.753343 | 0.531119 |

All routes wrong on 100 observed graphs. Per-member repairs/harms, pairwise error overlap and class-wise geometry are in RESULT.json.

Hidden-state cosine, variance, rank and CKA describe geometry. They do not establish accuracy or novelty.

Saved pooled-logit sanity check: True; max absolute difference 4.76837e-07. Its disclosed tolerance is diagnostic; no checkpoint or selection changed.

ROUTE_VALID.pt contains actual route logits, source graph IDs/labels and native preheads. No training callback or optimizer step executed.
