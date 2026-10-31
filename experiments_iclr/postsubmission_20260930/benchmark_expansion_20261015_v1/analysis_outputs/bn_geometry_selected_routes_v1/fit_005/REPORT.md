# Selected molecular VALID analysis

Fit `molbace_bn_geometry_correction_20261015_v1/seed-202/gnnm_private_bn_input_residual_graph_alignment_variance_v1`; selected epoch 71; 151 graphs; 4 routes.

These are descriptive results on the validation set used for selection. TEST remained closed.

| Prediction | ROC-AUC | BCE | Brier |
|---|---:|---:|---:|
| Member0 | 0.726007326007326 | 1.145567 | 0.362709 |
| Member1 | 0.6989010989010989 | 1.073839 | 0.343687 |
| Member2 | 0.7318681318681319 | 0.863388 | 0.275965 |
| Member3 | 0.6963369963369964 | 0.992658 | 0.301891 |
| Mean raw logits | 0.7227106227106227 | 0.987159 | 0.314963 |

All routes wrong on 53 observed graphs. Per-member repairs/harms, pairwise error overlap and class-wise geometry are in RESULT.json.

Hidden-state cosine, variance, rank and CKA describe geometry. They do not establish accuracy or novelty.

Saved pooled-logit sanity check: True; max absolute difference 9.53674e-07. Its disclosed tolerance is diagnostic; no checkpoint or selection changed.

ROUTE_VALID.pt contains actual route logits, source graph IDs/labels and native preheads. No training callback or optimizer step executed.
