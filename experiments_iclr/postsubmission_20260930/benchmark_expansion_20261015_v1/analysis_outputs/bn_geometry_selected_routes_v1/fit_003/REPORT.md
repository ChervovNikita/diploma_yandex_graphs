# Selected molecular VALID analysis

Fit `molbace_bn_geometry_correction_20261015_v1/seed-202/gnnm_private_bn_graph_alignment_variance_v1`; selected epoch 38; 151 graphs; 4 routes.

These are descriptive results on the validation set used for selection. TEST remained closed.

| Prediction | ROC-AUC | BCE | Brier |
|---|---:|---:|---:|
| Member0 | 0.6853479853479854 | 0.628606 | 0.213889 |
| Member1 | 0.6824175824175824 | 0.919384 | 0.316963 |
| Member2 | 0.7230769230769231 | 0.589019 | 0.196251 |
| Member3 | 0.7161172161172161 | 0.440665 | 0.142010 |
| Mean raw logits | 0.7194139194139194 | 0.593224 | 0.201093 |

All routes wrong on 18 observed graphs. Per-member repairs/harms, pairwise error overlap and class-wise geometry are in RESULT.json.

Hidden-state cosine, variance, rank and CKA describe geometry. They do not establish accuracy or novelty.

Saved pooled-logit sanity check: True; max absolute difference 2.38419e-07. Its disclosed tolerance is diagnostic; no checkpoint or selection changed.

ROUTE_VALID.pt contains actual route logits, source graph IDs/labels and native preheads. No training callback or optimizer step executed.
