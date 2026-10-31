# Selected molecular VALID analysis

Fit `molbace_residual_only_private_bn_20261015_v1/seed-202/gnnm_residual_only_private_bn_v1/residual_private_bn_permuted_risk_v2`; selected epoch 29; 151 graphs; 4 routes.

These are descriptive results on the validation set used for selection. TEST remained closed.

| Prediction | ROC-AUC | BCE | Brier |
|---|---:|---:|---:|
| Member0 | 0.6813186813186813 | 1.618743 | 0.479827 |
| Member1 | 0.695970695970696 | 1.628733 | 0.485885 |
| Member2 | 0.7183150183150183 | 1.720742 | 0.506009 |
| Member3 | 0.7032967032967034 | 1.690166 | 0.496275 |
| Mean raw logits | 0.6985347985347986 | 1.661890 | 0.491860 |

All routes wrong on 95 observed graphs. Per-member repairs/harms, pairwise error overlap and class-wise geometry are in RESULT.json.

Hidden-state cosine, variance, rank and CKA describe geometry. They do not establish accuracy or novelty.

Saved pooled-logit sanity check: True; max absolute difference 4.76837e-07. Its disclosed tolerance is diagnostic; no checkpoint or selection changed.

ROUTE_VALID.pt contains actual route logits, source graph IDs/labels and native preheads. No training callback or optimizer step executed.
