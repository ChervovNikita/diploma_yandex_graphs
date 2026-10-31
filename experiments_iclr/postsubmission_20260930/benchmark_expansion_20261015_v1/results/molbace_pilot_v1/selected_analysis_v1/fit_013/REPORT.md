# Selected molecular VALID analysis

Fit `molecular_input_steering_and_class_geometry_20261015_v1/split-0/seed-202/method-gnnm_graph_alignment_variance`; selected epoch 95; 151 graphs; 4 routes.

These are descriptive results on the validation set used for selection. TEST remained closed.

| Prediction | ROC-AUC | BCE | Brier |
|---|---:|---:|---:|
| Member0 | 0.6813186813186813 | 0.470625 | 0.128798 |
| Member1 | 0.5131868131868131 | 0.598263 | 0.135781 |
| Member2 | 0.7117216117216117 | 0.588211 | 0.176549 |
| Member3 | 0.647985347985348 | 1.102621 | 0.312458 |
| Mean raw logits | 0.6857142857142858 | 0.442184 | 0.139485 |

All routes wrong on 7 observed graphs. Per-member repairs/harms, pairwise error overlap and class-wise geometry are in RESULT.json.

Hidden-state cosine, variance, rank and CKA describe geometry. They do not establish accuracy or novelty.

Saved pooled-logit sanity check: True; max absolute difference 7.15256e-07. Its disclosed tolerance is diagnostic; no checkpoint or selection changed.

ROUTE_VALID.pt contains actual route logits, source graph IDs/labels and native preheads. No training callback or optimizer step executed.
