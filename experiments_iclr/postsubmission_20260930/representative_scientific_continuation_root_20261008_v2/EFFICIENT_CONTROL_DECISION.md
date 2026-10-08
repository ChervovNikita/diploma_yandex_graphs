# Capable efficient-ensemble control for the current full graph study

8 October 2026. This is preparation for a future comparison, not a launched experiment. Current Context9/Wiki12/Mol18 gates, recipes and original scores remain unchanged.

## The relevant control

The existing source assessment `efficient_graph_ensemble_baseline_readiness_v1` already inspected canonical MIMO and Packed-Ensembles and their author implementations. This continuation reuses those conclusions. It adds zero primary-paper or implementation-read credits. Two new Crossref title queries returned unrelated records and provide no method evidence.

For the current WikiCS Polynormer model, the applicable published-construction control is four disjoint native subnetworks following Packed-Ensembles with gamma=1. The conventional alpha=2 point changes hidden width from 512 to 256 for each of four subnetworks. Keep the current one attention head, seven local layers, two global layers, input features, graph, own supervised losses, stochastic views, optimizer schedule and probability averaging. Each subnetwork has its own complete learned projections, scorer vectors, normalization, beta and classifier. No hidden states or global node reductions cross subnetworks. This is a declared graph adaptation of Packed-Ensembles, not a native published Polynormer experiment or an exact reproduction of the image results.

The complete actual parameter count must be measured before committing a resource-matching claim. Four half-width models match the leading quadratic dense-map count, but input/output maps, biases, normalization and other linear terms differ. The current model has300 input channels and10 classes. No exact parameter-budget equality follows from width alone. Running four full-width independent models in one batched call is only a different implementation of the existing ordinary control. It adds no new statistical method.

Canonical Packed-Ensembles uses a synchronized pooled checkpoint. Preserve that declared selector rather than quietly mixing own-selected epochs and calling it the original baseline. Ordinary full-width independently own-selected models remain a separate capable quality reference. Every graph port must disclose its selector and full training cost. A future quality claim needs both controls. If a new context regularizer supplies the gain, the frozen full-width untied context reference must also complete.

## Why a simple MIMO port would be misleading here

Canonical MIMO trains on independently paired complete examples and repeated inputs at inference. The current native predictor propagates along the graph at every local layer and performs global reductions across nodes. Independently shuffling node feature rows against an unchanged graph does not create coherent independent examples. Repeating the same graph in every slot throughout training gives a multihead model and removes canonical MIMO's independent-input training. Whole sampled graph components or a decoupled token cache can permit a different graph port, but they change this task or predictor. Do not invent such a baseline merely to place MIMO in a table.

The saved assessment identifies a coherent MIMO boundary for cached PolyFormer polynomial-token rows. That different backbone remains a possible later experiment, not a drop-in current Polynormer control.

## Representative decision

First finish the registered complete families. If a current candidate supplies a useful paired quality result and passes its existing scientific gates, complete its prescribed competent native and loss-matched references. An unused confirmation should freeze the candidate and one appropriate efficient/diverse-ensemble construction before heldout scoring. It must use complete training, representative official splits, and all fixed seed outcomes. The present pilot does not establish generalization, efficient superiority, methodological novelty or acceptance.

Source: `efficient_graph_ensemble_baseline_readiness_v1/REPORT.md`, reused method/applicability conclusions in its Recommendation, published-construction table, native Polynormer boundary and Packed-Ensembles adaptation sections. Original primary/canonical method scopes remain in that packet's READ_SCOPES.json. No primary body was retrieved or reread here.
