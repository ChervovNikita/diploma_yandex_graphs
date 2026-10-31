# Selected molecular VALID analysis

Fit `molecular_input_steering_and_class_geometry_20261015_v1/split-0/seed-202/method-gnnm_boundary_native`; selected epoch 101; 151 graphs; 4 routes.

These are descriptive results on the validation set used for selection. TEST remained closed.

| Prediction | ROC-AUC | BCE | Brier |
|---|---:|---:|---:|
| Member0 | 0.6732600732600733 | 1.096018 | 0.242017 |
| Member1 | 0.5798534798534799 | 2.737300 | 0.617951 |
| Member2 | 0.6868131868131868 | 0.560879 | 0.167506 |
| Member3 | 0.7010989010989012 | 0.688750 | 0.205696 |
| Mean raw logits | 0.6952380952380953 | 0.873411 | 0.240593 |

All routes wrong on 17 observed graphs. Per-member repairs/harms, pairwise error overlap and class-wise geometry are in RESULT.json.

Hidden-state cosine, variance, rank and CKA describe geometry. They do not establish accuracy or novelty.

Saved pooled-logit sanity check: True; max absolute difference 4.76837e-07. Its disclosed tolerance is diagnostic; no checkpoint or selection changed.

ROUTE_VALID.pt contains actual route logits, source graph IDs/labels and native preheads. No training callback or optimizer step executed.
