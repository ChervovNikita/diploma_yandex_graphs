# Selected molecular VALID analysis

Fit `molbace_residual_only_private_bn_20261015_v1/seed-101/gnnm_residual_only_private_bn_v1`; selected epoch 136; 151 graphs; 4 routes.

These are descriptive results on the validation set used for selection. TEST remained closed.

| Prediction | ROC-AUC | BCE | Brier |
|---|---:|---:|---:|
| Member0 | 0.7747252747252747 | 3.371806 | 0.627478 |
| Member1 | 0.7743589743589744 | 3.419872 | 0.630851 |
| Member2 | 0.7813186813186814 | 3.401650 | 0.629348 |
| Member3 | 0.772893772893773 | 3.352229 | 0.626362 |
| Mean raw logits | 0.7750915750915751 | 3.386223 | 0.628557 |

All routes wrong on 105 observed graphs. Per-member repairs/harms, pairwise error overlap and class-wise geometry are in RESULT.json.

Hidden-state cosine, variance, rank and CKA describe geometry. They do not establish accuracy or novelty.

Saved pooled-logit sanity check: True; max absolute difference 9.53674e-07. Its disclosed tolerance is diagnostic; no checkpoint or selection changed.

ROUTE_VALID.pt contains actual route logits, source graph IDs/labels and native preheads. No training callback or optimizer step executed.
