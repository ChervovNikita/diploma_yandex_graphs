# Learned graph ensemble steering: user proposal and research decision

The user proposes trainable BatchEnsemble modulation inside a shared GNN, distinct starting factors, a contrastive objective that diversifies member embeddings, possibly a learned diversity coefficient and learned output selection or reinforcement learning. The accuracy goal is useful member complementarity rather than parameters, larger representation distances or disagreement by itself. This is an active proposal; no novelty or predictive improvement is established and no new job is authorized by this note.

## Concrete mechanism worth evaluating

A node enters each member's private multiplicative factors before shared feature mixing and graph propagation. Shared matrices receive learning signals from all members; private factors can change which feature channels and neighbourhood evidence each member uses. Every member retains its own supervised prediction loss. Graph augmentation supplies two views of the same node; a same-member consistency objective can preserve its prediction across declared plausible changes. Across-member diversity should preserve label-relevant information and be assessed in prediction responses and error identities. It should not indiscriminately push every same-node representation away.

Changing an internal representation's coordinates can make it appear far from another representation while its classifier makes exactly the same prediction. Conversely, forcing separation can remove information all members need. The primary question is whether internal modulation changes useful graph evidence and repairs common errors while retaining competent member predictions. Such factors create different member states; one shared set of weights does not imply all graph computations can be performed once.

The existing frozen-backbone residual study tests a different stage of the pipeline: new graph branches act after one trained backbone. Its fixed comparisons remain unchanged. The prior inside-backbone initialization and nonlinear-message studies continue. Their complete results, rather than a new diversity narrative, must determine whether these mechanisms help.

## Closest established methods

- BatchEnsemble and TabM already learn private factors around shared matrices, use deliberate first-factor initialization and own-member supervision. Generic trainable steering is therefore prior-based.
- DICE (ICLR 2021, arXiv2101.05544) explicitly protects class information while penalizing conditional feature redundancy. This is the closest motivation for class-compatible diversity; a simpler centering or cosine penalty is not an equivalent mutual-information estimator.
- CDLG (arXiv2306.11344) already pairs same-node/same-channel graph views positively and same-node/different-channel views negatively. Its inspected printed objective has mathematical ambiguities; pairing ancestry remains clear.
- SuGAr (AAAI 2025, arXiv2410.22228) combines label-aware contrastive graph learning and predicted edge-weight diversity. HGEN (IJCAI 2025, arXiv2509.09843) diversifies graph ensemble representations. Moving a diversity objective to graph evidence is not alone new.
- Saved FoRDE/DICE graph-adaptation sources examine functional sensitivity and conditional redundancy. They are prepared comparators, not proof of completed training or superiority.

These sources were reused from saved scoped conclusions. No new reading count or claim that the complete proposed method has no precedent follows from this note.

## Adaptive strength and aggregation

A free nonnegative coefficient multiplying a nonnegative diversity penalty can approach zero when jointly minimizing that same training loss. A negatively signed separation reward can instead encourage excessive separation. A useful adaptive coefficient needs a specified constrained objective or an outer prediction-quality criterion on a separate training fold; ordinary learned scalar weighting is not methodological novelty. Start by identifying whether a fixed objective improves complete predictive performance, then compare adaptive control to a fixed-strength reference prospectively.

A learned combiner can use node or edge features and member outputs, but it never knows the test answer. Training must provide the same declared supervision opportunity to its controls. If every member gives the same wrong margin sign for a positive-negative pair, no nonnegative convex mean of those margins can repair that pair. The completed Citeseer diagnostic found substantial common wrong-negative overlap, so different useful evidence is a prerequisite for relying on routing alone. It did not prove that sharing caused those errors.

Reinforcement learning is not the first implementation: modulation, contrastive terms and soft combination are differentiable. It becomes a separate justified experiment only if a discrete graph or member action has a clear reward and a meaningful constraint that ordinary gradient training does not address. It must earn its extra variance and compute through a matched comparison.

## Evidence required before promotion

Use the same expressive backbone and prescribed data split for a competent single, ordinary independent four, unchanged shared four, initialization-only modification, contrastive-only modification and their combination. Reuse unchanged valid baselines where data/runtime/selection opportunities match; do not recalculate original manuscript scores. Fix modest pilot recipes and paired seeds before outcome scoring. Treat learned aggregation as a later isolated contrast with the same prediction inputs. Any learned diversity controller requires a fixed-strength control and explicit data role separation.

Judge complete pooled accuracy or the task's predefined ranking metric first. Record member competence, common error identities, net repaired versus introduced errors, predictive responses to declared graph changes and full training/serving costs. Embedding distance, reconstruction loss and disagreement remain diagnostics. A development gain needs unused-population confirmation and closest-prior comparison before a paper claim. Existing initialization/private-message/supervised-correction queues continue; no outcome-guided source change or extra hyperparameter grid is introduced here.
