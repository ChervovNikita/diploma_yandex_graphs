# Selected molecular VALID analysis

Fit `molbace_bn_geometry_correction_20261015_v1/seed-202/gnnm_private_bn_input_residual_v1`; selected epoch 68; 151 graphs; 4 routes.

These are descriptive results on the validation set used for selection. TEST remained closed.

| Prediction | ROC-AUC | BCE | Brier |
|---|---:|---:|---:|
| Member0 | 0.7201465201465201 | 0.754890 | 0.234709 |
| Member1 | 0.7032967032967032 | 0.577335 | 0.182143 |
| Member2 | 0.7230769230769231 | 0.716203 | 0.227610 |
| Member3 | 0.7457875457875458 | 0.748667 | 0.235731 |
| Mean raw logits | 0.7256410256410256 | 0.666392 | 0.210073 |

All routes wrong on 25 observed graphs. Per-member repairs/harms, pairwise error overlap and class-wise geometry are in RESULT.json.

Hidden-state cosine, variance, rank and CKA describe geometry. They do not establish accuracy or novelty.

Saved pooled-logit sanity check: True; max absolute difference 4.76837e-07. Its disclosed tolerance is diagnostic; no checkpoint or selection changed.

ROUTE_VALID.pt contains actual route logits, source graph IDs/labels and native preheads. No training callback or optimizer step executed.
