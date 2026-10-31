# Selected molecular VALID analysis

Fit `molbace_bn_geometry_correction_20261015_v1/seed-101/gnnm_private_bn_input_residual_graph_alignment_variance_v1`; selected epoch 19; 151 graphs; 4 routes.

These are descriptive results on the validation set used for selection. TEST remained closed.

| Prediction | ROC-AUC | BCE | Brier |
|---|---:|---:|---:|
| Member0 | 0.76007326007326 | 0.876001 | 0.284301 |
| Member1 | 0.7468864468864469 | 0.889416 | 0.282887 |
| Member2 | 0.7223443223443223 | 0.585305 | 0.191601 |
| Member3 | 0.6952380952380953 | 1.107174 | 0.340124 |
| Mean raw logits | 0.7487179487179488 | 0.799442 | 0.262943 |

All routes wrong on 30 observed graphs. Per-member repairs/harms, pairwise error overlap and class-wise geometry are in RESULT.json.

Hidden-state cosine, variance, rank and CKA describe geometry. They do not establish accuracy or novelty.

Saved pooled-logit sanity check: True; max absolute difference 4.76837e-07. Its disclosed tolerance is diagnostic; no checkpoint or selection changed.

ROUTE_VALID.pt contains actual route logits, source graph IDs/labels and native preheads. No training callback or optimizer step executed.
