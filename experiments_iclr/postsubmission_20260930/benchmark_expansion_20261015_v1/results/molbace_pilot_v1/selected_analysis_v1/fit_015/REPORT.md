# Selected molecular VALID analysis

Fit `molecular_input_steering_and_class_geometry_20261015_v1/split-0/seed-202/method-gnnm_input_residual_plus_graph_alignment_variance`; selected epoch 33; 151 graphs; 4 routes.

These are descriptive results on the validation set used for selection. TEST remained closed.

| Prediction | ROC-AUC | BCE | Brier |
|---|---:|---:|---:|
| Member0 | 0.6131868131868131 | 1.788388 | 0.412917 |
| Member1 | 0.5347985347985349 | 1.633105 | 0.538483 |
| Member2 | 0.6571428571428571 | 0.445977 | 0.138121 |
| Member3 | 0.6117216117216118 | 0.628146 | 0.218846 |
| Mean raw logits | 0.6813186813186813 | 0.864753 | 0.314747 |

All routes wrong on 1 observed graphs. Per-member repairs/harms, pairwise error overlap and class-wise geometry are in RESULT.json.

Hidden-state cosine, variance, rank and CKA describe geometry. They do not establish accuracy or novelty.

Saved pooled-logit sanity check: True; max absolute difference 2.98023e-07. Its disclosed tolerance is diagnostic; no checkpoint or selection changed.

ROUTE_VALID.pt contains actual route logits, source graph IDs/labels and native preheads. No training callback or optimizer step executed.
