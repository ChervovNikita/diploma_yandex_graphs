# Selected molecular VALID analysis

Fit `molecular_input_steering_and_class_geometry_20261015_v1/split-0/seed-101/method-gnnm_graph_alignment_variance`; selected epoch 4; 151 graphs; 4 routes.

These are descriptive results on the validation set used for selection. TEST remained closed.

| Prediction | ROC-AUC | BCE | Brier |
|---|---:|---:|---:|
| Member0 | 0.6201465201465202 | 0.468065 | 0.147371 |
| Member1 | 0.6765567765567766 | 0.514366 | 0.165937 |
| Member2 | 0.6241758241758242 | 1.059577 | 0.399647 |
| Member3 | 0.6615384615384615 | 0.461577 | 0.144050 |
| Mean raw logits | 0.7106227106227107 | 0.554208 | 0.183370 |

All routes wrong on 5 observed graphs. Per-member repairs/harms, pairwise error overlap and class-wise geometry are in RESULT.json.

Hidden-state cosine, variance, rank and CKA describe geometry. They do not establish accuracy or novelty.

Saved pooled-logit sanity check: True; max absolute difference 2.38419e-07. Its disclosed tolerance is diagnostic; no checkpoint or selection changed.

ROUTE_VALID.pt contains actual route logits, source graph IDs/labels and native preheads. No training callback or optimizer step executed.
