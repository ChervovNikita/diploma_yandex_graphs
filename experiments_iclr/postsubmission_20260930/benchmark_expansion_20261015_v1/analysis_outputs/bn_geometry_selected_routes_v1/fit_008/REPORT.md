# Selected molecular VALID analysis

Fit `molbace_bn_geometry_correction_20261015_v1/seed-303/gnnm_private_bn_input_residual_graph_alignment_variance_v1`; selected epoch 78; 151 graphs; 4 routes.

These are descriptive results on the validation set used for selection. TEST remained closed.

| Prediction | ROC-AUC | BCE | Brier |
|---|---:|---:|---:|
| Member0 | 0.717948717948718 | 1.130227 | 0.348175 |
| Member1 | 0.7300366300366301 | 1.383916 | 0.388638 |
| Member2 | 0.712820512820513 | 1.665731 | 0.446446 |
| Member3 | 0.6875457875457877 | 1.157896 | 0.370471 |
| Mean raw logits | 0.7194139194139194 | 1.302814 | 0.390211 |

All routes wrong on 59 observed graphs. Per-member repairs/harms, pairwise error overlap and class-wise geometry are in RESULT.json.

Hidden-state cosine, variance, rank and CKA describe geometry. They do not establish accuracy or novelty.

Saved pooled-logit sanity check: True; max absolute difference 9.53674e-07. Its disclosed tolerance is diagnostic; no checkpoint or selection changed.

ROUTE_VALID.pt contains actual route logits, source graph IDs/labels and native preheads. No training callback or optimizer step executed.
