# Selected molecular VALID analysis

Fit `molbace_bn_geometry_correction_20261015_v1/seed-101/gnnm_private_bn_graph_alignment_variance_v1`; selected epoch 15; 151 graphs; 4 routes.

These are descriptive results on the validation set used for selection. TEST remained closed.

| Prediction | ROC-AUC | BCE | Brier |
|---|---:|---:|---:|
| Member0 | 0.7523809523809524 | 0.727731 | 0.242243 |
| Member1 | 0.7952380952380953 | 0.992979 | 0.327709 |
| Member2 | 0.710989010989011 | 0.857816 | 0.278155 |
| Member3 | 0.6805860805860806 | 0.852434 | 0.287317 |
| Mean raw logits | 0.7520146520146521 | 0.824970 | 0.274272 |

All routes wrong on 43 observed graphs. Per-member repairs/harms, pairwise error overlap and class-wise geometry are in RESULT.json.

Hidden-state cosine, variance, rank and CKA describe geometry. They do not establish accuracy or novelty.

Saved pooled-logit sanity check: True; max absolute difference 4.76837e-07. Its disclosed tolerance is diagnostic; no checkpoint or selection changed.

ROUTE_VALID.pt contains actual route logits, source graph IDs/labels and native preheads. No training callback or optimizer step executed.
