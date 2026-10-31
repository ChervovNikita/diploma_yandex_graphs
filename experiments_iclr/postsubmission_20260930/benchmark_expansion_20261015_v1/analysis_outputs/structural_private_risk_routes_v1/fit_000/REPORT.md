# Selected molecular VALID analysis

Fit `molbace_residual_only_private_bn_20261015_v1/seed-101/gnnm_residual_only_private_bn_v1/residual_private_bn_structural_risk_v2`; selected epoch 18; 151 graphs; 4 routes.

These are descriptive results on the validation set used for selection. TEST remained closed.

| Prediction | ROC-AUC | BCE | Brier |
|---|---:|---:|---:|
| Member0 | 0.7274725274725274 | 0.519739 | 0.174664 |
| Member1 | 0.735897435897436 | 0.547004 | 0.185635 |
| Member2 | 0.7300366300366301 | 0.557128 | 0.189764 |
| Member3 | 0.7278388278388279 | 0.551368 | 0.187742 |
| Mean raw logits | 0.7355311355311355 | 0.539765 | 0.183070 |

All routes wrong on 34 observed graphs. Per-member repairs/harms, pairwise error overlap and class-wise geometry are in RESULT.json.

Hidden-state cosine, variance, rank and CKA describe geometry. They do not establish accuracy or novelty.

Saved pooled-logit sanity check: True; max absolute difference 2.38419e-07. Its disclosed tolerance is diagnostic; no checkpoint or selection changed.

ROUTE_VALID.pt contains actual route logits, source graph IDs/labels and native preheads. No training callback or optimizer step executed.
