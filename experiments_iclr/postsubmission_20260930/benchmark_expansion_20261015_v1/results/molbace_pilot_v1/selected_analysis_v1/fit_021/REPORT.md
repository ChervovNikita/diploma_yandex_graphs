# Selected molecular VALID analysis

Fit `molecular_input_steering_and_class_geometry_20261015_v1/split-0/seed-303/method-gnnm_graph_alignment_variance`; selected epoch 29; 151 graphs; 4 routes.

These are descriptive results on the validation set used for selection. TEST remained closed.

| Prediction | ROC-AUC | BCE | Brier |
|---|---:|---:|---:|
| Member0 | 0.6446886446886447 | 1.231225 | 0.402021 |
| Member1 | 0.6336996336996337 | 1.747818 | 0.576941 |
| Member2 | 0.7058608058608058 | 0.440884 | 0.139695 |
| Member3 | 0.5879120879120879 | 0.556460 | 0.184966 |
| Mean raw logits | 0.7388278388278389 | 0.655194 | 0.224852 |

All routes wrong on 4 observed graphs. Per-member repairs/harms, pairwise error overlap and class-wise geometry are in RESULT.json.

Hidden-state cosine, variance, rank and CKA describe geometry. They do not establish accuracy or novelty.

Saved pooled-logit sanity check: True; max absolute difference 3.57628e-07. Its disclosed tolerance is diagnostic; no checkpoint or selection changed.

ROUTE_VALID.pt contains actual route logits, source graph IDs/labels and native preheads. No training callback or optimizer step executed.
