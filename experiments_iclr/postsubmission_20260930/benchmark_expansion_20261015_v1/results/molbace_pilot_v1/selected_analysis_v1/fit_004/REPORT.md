# Selected molecular VALID analysis

Fit `molecular_input_steering_and_class_geometry_20261015_v1/split-0/seed-101/method-gnnm_boundary_native`; selected epoch 2; 151 graphs; 4 routes.

These are descriptive results on the validation set used for selection. TEST remained closed.

| Prediction | ROC-AUC | BCE | Brier |
|---|---:|---:|---:|
| Member0 | 0.5315018315018316 | 0.432378 | 0.127826 |
| Member1 | 0.5036630036630036 | 0.535882 | 0.174762 |
| Member2 | 0.582051282051282 | 1.540038 | 0.484504 |
| Member3 | 0.6087912087912088 | 0.418204 | 0.127087 |
| Mean raw logits | 0.6278388278388278 | 0.498480 | 0.158085 |

All routes wrong on 1 observed graphs. Per-member repairs/harms, pairwise error overlap and class-wise geometry are in RESULT.json.

Hidden-state cosine, variance, rank and CKA describe geometry. They do not establish accuracy or novelty.

Saved pooled-logit sanity check: True; max absolute difference 2.38419e-07. Its disclosed tolerance is diagnostic; no checkpoint or selection changed.

ROUTE_VALID.pt contains actual route logits, source graph IDs/labels and native preheads. No training callback or optimizer step executed.
