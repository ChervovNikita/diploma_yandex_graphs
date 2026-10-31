# Selected molecular VALID analysis

Fit `molbace_residual_only_private_bn_20261015_v1/seed-101/gnnm_residual_only_private_bn_v1/residual_private_bn_permuted_risk_v2`; selected epoch 10; 151 graphs; 4 routes.

These are descriptive results on the validation set used for selection. TEST remained closed.

| Prediction | ROC-AUC | BCE | Brier |
|---|---:|---:|---:|
| Member0 | 0.743956043956044 | 0.827577 | 0.274321 |
| Member1 | 0.7443223443223445 | 0.992402 | 0.315106 |
| Member2 | 0.7501831501831501 | 0.808274 | 0.268827 |
| Member3 | 0.7608058608058608 | 0.800115 | 0.265574 |
| Mean raw logits | 0.7538461538461539 | 0.850496 | 0.281466 |

All routes wrong on 56 observed graphs. Per-member repairs/harms, pairwise error overlap and class-wise geometry are in RESULT.json.

Hidden-state cosine, variance, rank and CKA describe geometry. They do not establish accuracy or novelty.

Saved pooled-logit sanity check: True; max absolute difference 4.76837e-07. Its disclosed tolerance is diagnostic; no checkpoint or selection changed.

ROUTE_VALID.pt contains actual route logits, source graph IDs/labels and native preheads. No training callback or optimizer step executed.
