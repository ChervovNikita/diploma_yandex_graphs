# Selected molecular VALID analysis

Fit `molbace_residual_only_private_bn_20261015_v1/seed-303/gnnm_residual_only_private_bn_v1/residual_private_bn_permuted_risk_v2`; selected epoch 39; 151 graphs; 4 routes.

These are descriptive results on the validation set used for selection. TEST remained closed.

| Prediction | ROC-AUC | BCE | Brier |
|---|---:|---:|---:|
| Member0 | 0.7073260073260074 | 2.524533 | 0.580725 |
| Member1 | 0.7139194139194139 | 2.600184 | 0.594810 |
| Member2 | 0.7084249084249084 | 2.385784 | 0.566302 |
| Member3 | 0.7117216117216117 | 2.532176 | 0.583746 |
| Mean raw logits | 0.710989010989011 | 2.507495 | 0.582008 |

All routes wrong on 99 observed graphs. Per-member repairs/harms, pairwise error overlap and class-wise geometry are in RESULT.json.

Hidden-state cosine, variance, rank and CKA describe geometry. They do not establish accuracy or novelty.

Saved pooled-logit sanity check: True; max absolute difference 9.53674e-07. Its disclosed tolerance is diagnostic; no checkpoint or selection changed.

ROUTE_VALID.pt contains actual route logits, source graph IDs/labels and native preheads. No training callback or optimizer step executed.
