# Selected molecular VALID analysis

Fit `molbace_bn_geometry_correction_20261015_v1/seed-303/gnnm_private_bn_input_residual_v1`; selected epoch 21; 151 graphs; 4 routes.

These are descriptive results on the validation set used for selection. TEST remained closed.

| Prediction | ROC-AUC | BCE | Brier |
|---|---:|---:|---:|
| Member0 | 0.6923076923076923 | 0.858252 | 0.255840 |
| Member1 | 0.702930402930403 | 0.537259 | 0.171014 |
| Member2 | 0.7087912087912088 | 1.019948 | 0.301078 |
| Member3 | 0.6743589743589744 | 0.525362 | 0.159434 |
| Mean raw logits | 0.7062271062271063 | 0.636880 | 0.197363 |

All routes wrong on 21 observed graphs. Per-member repairs/harms, pairwise error overlap and class-wise geometry are in RESULT.json.

Hidden-state cosine, variance, rank and CKA describe geometry. They do not establish accuracy or novelty.

Saved pooled-logit sanity check: True; max absolute difference 9.53674e-07. Its disclosed tolerance is diagnostic; no checkpoint or selection changed.

ROUTE_VALID.pt contains actual route logits, source graph IDs/labels and native preheads. No training callback or optimizer step executed.
