# Selected molecular VALID analysis

Fit `molbace_residual_only_private_bn_20261015_v1/seed-202/gnnm_residual_only_private_bn_v1`; selected epoch 9; 151 graphs; 4 routes.

These are descriptive results on the validation set used for selection. TEST remained closed.

| Prediction | ROC-AUC | BCE | Brier |
|---|---:|---:|---:|
| Member0 | 0.7366300366300366 | 1.380131 | 0.433751 |
| Member1 | 0.7366300366300366 | 1.350826 | 0.426404 |
| Member2 | 0.7366300366300367 | 1.361207 | 0.429400 |
| Member3 | 0.7355311355311355 | 1.383646 | 0.435469 |
| Mean raw logits | 0.7362637362637363 | 1.368741 | 0.431283 |

All routes wrong on 96 observed graphs. Per-member repairs/harms, pairwise error overlap and class-wise geometry are in RESULT.json.

Hidden-state cosine, variance, rank and CKA describe geometry. They do not establish accuracy or novelty.

Saved pooled-logit sanity check: True; max absolute difference 4.76837e-07. Its disclosed tolerance is diagnostic; no checkpoint or selection changed.

ROUTE_VALID.pt contains actual route logits, source graph IDs/labels and native preheads. No training callback or optimizer step executed.
