# Selected molecular VALID analysis

Fit `molbace_residual_only_private_bn_20261015_v1/seed-202/gnnm_residual_only_private_bn_v1/residual_private_bn_structural_risk_v2`; selected epoch 31; 151 graphs; 4 routes.

These are descriptive results on the validation set used for selection. TEST remained closed.

| Prediction | ROC-AUC | BCE | Brier |
|---|---:|---:|---:|
| Member0 | 0.7282051282051283 | 1.674174 | 0.461689 |
| Member1 | 0.7454212454212454 | 1.665335 | 0.457686 |
| Member2 | 0.7523809523809524 | 1.802876 | 0.475435 |
| Member3 | 0.7428571428571429 | 1.677525 | 0.458181 |
| Mean raw logits | 0.7476190476190476 | 1.696743 | 0.464718 |

All routes wrong on 80 observed graphs. Per-member repairs/harms, pairwise error overlap and class-wise geometry are in RESULT.json.

Hidden-state cosine, variance, rank and CKA describe geometry. They do not establish accuracy or novelty.

Saved pooled-logit sanity check: True; max absolute difference 4.76837e-07. Its disclosed tolerance is diagnostic; no checkpoint or selection changed.

ROUTE_VALID.pt contains actual route logits, source graph IDs/labels and native preheads. No training callback or optimizer step executed.
