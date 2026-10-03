# Separate training signals for shared weights and private factors

This is a description of a running experiment. It is not a claim of a new principle, a proven improvement, or a manuscript result.

## The operation being tested

The model receives one graph with node attributes and connections. Four members predict a class for each target node. They use common HGT weights and small, member-specific BatchEnsemble factors. Each member produces one raw score for every class. The model averages the four score vectors and applies softmax to this average when class probabilities are needed.

Training uses labels for TRAIN nodes. For each such node, we calculate two errors from the same forward pass. The pooled error measures the prediction obtained by averaging member scores. The member error measures each member's own prediction, then averages those four errors. Both are cross-entropy losses with the same node labels.

Ordinary BatchEnsemble uses the mean member error to train both the shared weights and the private factors. The candidate uses the pooled error for the shared weights and the mean member error for the private factors. The two gradient blocks are calculated before any parameter changes. The original member averaging factor, optimizer, scheduler and checkpoint-selection rule are retained. There is one optimizer step.

At prediction time, all four members run and their raw scores are averaged as before. No expert-selection mechanism or extra prediction head is introduced. This experiment supplies no inference-speed claim.

## The conditional reason to try it

Shared weights influence every member. Giving them the pooled error trains that common capacity against the prediction actually served. A member can still be wrong when the pool is already confident. Giving its private factors the member error preserves that member's own label signal in such cases.

This is a proposed assignment of training signals to parameter roles. It may fail because the shared weights lose regularization from individual member errors, or because private updates improve individual predictions while making the combined prediction worse. A separate mathematical critique gives an explicit example of the latter under SGD. The native AdamW trajectory and prediction quality must be measured.

The ordinary member loss already contains a consensus term relative to pooled-logit cross entropy. The candidate removes that term's gradient from the shared weights and retains its gradient for private factors. It therefore should not be described as a loss that forces private embeddings or predictions apart.

## The comparisons that determine whether the idea helps

The first term in each policy below gives the shared-weight training signal. The second gives the private-factor signal. Everything else in the model and fitting recipe is fixed.

| Policy | Shared weights learn from | Private factors learn from | Purpose |
|---|---|---|---|
| own/own | Mean member error | Mean member error | Ordinary BatchEnsemble reference |
| pool/pool | Pooled error | Pooled error | Train all parameters against the served prediction |
| pool/own | Pooled error | Mean member error | Proposed role assignment |
| own/pool | Mean member error | Pooled error | Reverse the proposed roles |

The running study uses the complete released DBLP and ACM graphs and five paired seed/split blocks per graph. All four policies are fitted afresh, making forty selected cases in total. The primary comparisons and practical thresholds were fixed before this study's outcomes. Results are opened only after the complete study passes its artifact and selection checks.

Passing the fixed development criteria would justify separate confirmation. It would not by itself establish superiority over the strongest graph models, transfer to other graph tasks, or generalization beyond these graphs. A tie with the reverse policy would leave the claimed parameter-role explanation unsupported.

## Attribution and limits

BatchEnsemble and TabM provide the shared weights, private fast factors and mean-member training ancestry. Joint ensemble training and its possible collusion are established topics. Collaborative Learning, ONE and PCL provide shared-body supervision and partial loss-routing precedents. Tiny Deep Ensemble and LoRA-Ensemble provide related sharing and freezing schedules.

The inspected methods do not establish exact equality to the complete frozen policy. The search also does not establish that the policy is globally novel. An exact short-run comparator already existed in our research history. Full training tests its usefulness rather than discovering block-dependent optimization.

The mathematical critique gives a conditional first-order descent test and a counterexample. These are applications of standard optimization arguments. They supply a limit on claims, rather than a graph-specific generalization theorem.

## Evidence bindings

- Scientific design: `../graph_mixed_block_objectives_execution_root_v1/FROZEN_SCIENTIFIC_DESIGN.json`.
- Executable freeze and actual release: `../graph_mixed_block_execution_root_20261003_v1/`.
- Earlier conceptual and literature assessment: `../graph_mixed_block_objectives_scout_20261003_v1/`.
- Closest ONE/PCL comparison: `../mixed_block_objective_closest_priors_20261003_v1/`.
- Modern ensemble comparison: `../modern_rank_one_block_objective_priors_20261003_v1/`.
- Independent mathematical critique: `../mixed_policy_theory_limits_20261003_v1/`.

No current study outcome was used to write this explanation. Original paper scores remain unchanged.
