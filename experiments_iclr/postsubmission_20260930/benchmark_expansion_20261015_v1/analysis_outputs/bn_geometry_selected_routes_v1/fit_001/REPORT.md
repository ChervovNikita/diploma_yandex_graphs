# Selected molecular VALID analysis

Fit `molbace_bn_geometry_correction_20261015_v1/seed-101/gnnm_private_bn_input_residual_v1`; selected epoch 102; 151 graphs; 4 routes.

These are descriptive results on the validation set used for selection. TEST remained closed.

| Prediction | ROC-AUC | BCE | Brier |
|---|---:|---:|---:|
| Member0 | 0.7007326007326008 | 1.121865 | 0.315856 |
| Member1 | 0.6886446886446886 | 1.175415 | 0.330374 |
| Member2 | 0.7490842490842491 | 1.263267 | 0.373397 |
| Member3 | 0.7743589743589744 | 1.161437 | 0.326700 |
| Mean raw logits | 0.7336996336996336 | 1.144400 | 0.328263 |

All routes wrong on 56 observed graphs. Per-member repairs/harms, pairwise error overlap and class-wise geometry are in RESULT.json.

Hidden-state cosine, variance, rank and CKA describe geometry. They do not establish accuracy or novelty.

Saved pooled-logit sanity check: True; max absolute difference 4.76837e-07. Its disclosed tolerance is diagnostic; no checkpoint or selection changed.

ROUTE_VALID.pt contains actual route logits, source graph IDs/labels and native preheads. No training callback or optimizer step executed.
