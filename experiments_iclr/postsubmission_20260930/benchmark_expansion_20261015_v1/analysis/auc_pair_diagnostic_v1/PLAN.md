# Pairwise ranking diagnosis on existing molecular predictions

This is posthoc development analysis, using only the four already completed
seed101 selected-VALID banks in fixed_probability_aggregation_v1/INPUTS.json.
No model, new forward, training, selector, fitted combiner or TEST access is
permitted. The original mean-logit rule remains unchanged.

ROC-AUC compares each observed positive graph with each observed negative graph.
We will measure per-route and actual-pool correct/tied/reversed pair orderings,
common reversed pairs, disagreement, and correct route pairs cancelled by
pooling. Compare base versus each candidate's repairs and harms on identical
pairs. Preserve every count and tie; do not choose an aggregation or hypothesis
from only favorable subsets.

Averaging raw scores cannot strictly repair a pair strictly reversed by every
route in exact arithmetic: the pooled pair margin is the average of member
margins. This elementary fact motivates distinguishing competence and pooling
failure for this ranking metric. It is not a novel theorem, a classification
oracle claim or a learnable global ranking guarantee.

All2,730positive-negative pairs reuse the same151graphs. They are diagnostic
comparisons, not2,730independent observations or extra seeds for a significance
test. Selected VALID is already used in model selection. The full three-seed
and genuine I4 cohorts remain incomplete. No method promotion follows here.
