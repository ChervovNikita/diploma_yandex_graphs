# Recent graph-evidence ensembles and member-quality training

8 October 2026. One focused search; two new bounded primary method reads.
The current graph-relation fits and pending predictions were not inspected.

## Scientific conclusion

The search supplies **one concrete ICML2026 comparator for member quality**:
*Semi-Supervised Learning for Molecular Graphs via Ensemble Consensus*.
Its members learn from their own labels and a stopped ensemble-consensus target
on unlabeled training inputs. It is relevant when our shared members are weaker
than ordinary ensemble members, but its low-label protocol cannot be silently
transferred to full-label MolHIV.

The other new method, *Persistent Homology-induced Graph Ensembles for Time
Series Regressions*, supplies direct multiscale graph-evidence and learned
fusion ancestry. It constructs different input graphs and jointly learns their
representation fusion. The feature GNN weights are private per graph; shared
fusion maps do not establish a shared GNN backbone.

Neither paper specifies the current labelled own/pool CE restriction to
private local scorers and tied-QK factors in the inspected scopes. That bounded
distinction does not clear novelty. Saved GNCL, shared/private supervision,
CoGNN, DIVE, GRAND and GraphMix already cover the principal ingredients.

## 1. Persistent Homology-induced Graph Ensembles for Time Series Regressions

Nguyen et al., [arXiv2503.14240v2](https://arxiv.org/html/2503.14240v2),
submitted18March2025, revised19March2025. The exact-title Crossref match is
[DOI10.2139/ssrn.5521531](https://doi.org/10.2139/ssrn.5521531), classified
as posted content. No conference/journal publication is verified in this pass.

**Read scope:** Sections3.1–3.2 and3.4–3.5; Eqs6–12,21–23 and graph-generation
prose. Section3.3's information-preservation propositions/proofs and Section4's
experiments were not method-read. Two result tables floated inside the retained
Section3.5 HTML and were incidentally exposed; their scores are not adopted.

The method takes death times from a Vietoris–Rips filtration and constructs a
graph at each selected threshold. Nodes remain the data points; an edge joins
two points when their distance is at most that threshold. The printed graph
construction has no task-loss-trained edge generator. It defines collections
for connected-component and loop scales and their union.

Each graph is processed by its own temporal feature map f^(i) and two-layer
GCN g^(i), with learnable weights Theta^(i) and Psi^(i). The explicit superscripts
and prose distinguish these from shared attention weights W_att and shared
output MLP weights Phi. Attention scores are H^(i)W_att; softmax normalizes
over graph index i; the final representation sums alpha^(i) elementwise times
Z^(i), then the output MLP predicts. This is intermediate graph-branch fusion,
not uniform probability averaging of separately supervised classifiers.

The seismic task uses a final five-output aggregate MSE plus L2 regularization;
traffic uses aggregate L1 loss. No mean-member supervised objective or scorer-only
gradient permissions are stated in the selected method. All branch outputs are
computed before attention fusion; a routing label does not demonstrate skipped
expert computation or speed savings. The H-to-Z naming/shape contract is not
fully executable from these passages, and author code was not audited.

**Relevance:** distinct graph supports and learned graph-branch weighting are
known methods for exposing multiscale evidence. A proposal to give our members
different graph views and learn their representation importance must credit
this ancestry. It is not a native node-classification comparator on WikiCS:
graph construction, temporal inputs, output loss, supervision and parameter
ownership would all require a declared adaptation. No graph-information or
performance guarantee transfers to our private-attention rule.

## 2. Semi-Supervised Learning for Molecular Graphs via Ensemble Consensus

Rasmus Tirsgaard, Laurits Fredsgaard, Marisa Wodrich, Mikkel Jordahn and
Mikkel N. Schmidt, [arXiv2607.28304v1](https://arxiv.org/html/2607.28304v1),
submitted30July2026. **Official ICML2026 publication identity is verified** by
the [conference paper index](https://icml.cc/virtual/2026/papers.html) and
[individual paper/poster63835](https://icml.cc/virtual/2026/poster/63835).
The official poster links OpenReview forum
[TIdOvVntmx](https://openreview.net/forum?id=TIdOvVntmx). The inspected methods
are the exact arXiv v1 body; equality to the proceedings method body and author
implementation is not certified.

**Read scope:** complete Section4/Eqs4–6; AppendixC.2's GNN+ training protocol,
E.1's soft/hard target distinction and E.8's target-gradient discussion.
Appendix tables/figures were excluded from the latter text extraction.
Qualitative performance claims in these passages were incidentally exposed,
not adopted as measured comparative evidence. Section3/AppendixA proofs and
Section6 results were not audited.

The formal method initializes M separate models with different random weights.
It assumes labelled and unlabelled examples come from the same underlying
distribution. At each step it sums each member's own supervised loss on B_L
and gamma times its consistency loss on B_U, then updates all models
simultaneously. The consistency target is the arithmetic mean of the current
members' predictions on the same unlabelled input, with **the mean target
detached**. Task-appropriate distances are suggested, including L2 and KL.
No separately pretrained teacher or mandatory input augmentation is required.

No shared feature backbone, private edge-mask bank or restricted scorer/QK
recipient rule is stated. The complete output/target space and KL orientation
for a classification reproduction remain to be settled from qualified source;
the generic notation cannot certify mean-logit versus mean-probability pooling.
AppendixE.8 compares target detachment with differentiation in two regression
settings; it is not a theorem about shared graphs or a new selective-gradient
principle. No reported equivalence or quality is transferred to our model.

AppendixC.2 uses10% of the original data as labelled GNN+ training data and
shuffles labelled/unlabelled membership within TRAIN for each seed. It tunes
ordinary learning rate/decay on a single uncoupled model and selects SSL settings
on ZINC. Its consensus coupling weight is1.0 in that protocol. These are source
recipe facts, not a recommendation for a new grid or an optimal WikiCS/MolHIV
value. The paper's native labelled budget and selection opportunity matter.

**Relevance:** this is a concrete published way to improve members through
extra task-output supervision on unlabelled input, rather than insisting on
greater member disagreement. It is materially different from our J, which
uses true TRAIN labels to reweight actual pooled risk in designated relation
parameters. Consensus alone cannot fix identical shared wrong predictions:
when predictions coincide, their disagreement target supplies no correction.
Useful effects would rely on input support, regularization or existing useful
member differences; the target is not a correctness oracle.

## Established consistency ancestry

The saved **Graph Random Neural Networks for Semi-Supervised Learning on Graphs
(GRAND)** scope is the official NeurIPS2020 paper, pp3–5/Sections3.1–3.2,
Eqs1–4 and Algorithm1. It uses DropNode views, mixed-order graph propagation,
one shared MLP and supervised CE plus squared consistency against a sharpened
average prediction; serving is one ordinary MLP prediction.

The saved **GraphMix: Regularized Training of Graph Neural Networks for
Semi-Supervised Learning** v1 scope describes shared FCN/GNN training,
alternating manifold-mixup/native GNN losses and dropout-averaged/sharpened
pseudotargets; serving is the GNN. Its latest/final method was not qualified
in that earlier scope. Neither retained primary body was reread here.

These methods precede graph-view consistency and shared-weight predicted-target
training. The new ICML paper applies ensemble consensus to independently
initialized molecular-graph models. Moving that signal into our private routes
would be an attributed adaptation, not discovery of graph consistency or
ensemble self-teaching. Our copied/unit starts and shared parameters also
remove part of its stated independence-based diversity rationale.

## Fair task and data boundaries

**Full-label MolHIV:** the current full-TRAIN-label regime is not the paper's
10%-label GNN+ regime. With every finite TRAIN target already supervised, there
is no corresponding withheld90%-TRAIN pool. Reusing TRAIN graphs as label-blind
consistency inputs while retaining all supervised labels would be a separately
frozen **full-label consistency adaptation**, not faithful native SSL. It may
be a useful comparator, but requires explicit loss/target/reduction and source
qualification before execution.

Alternatively, a representative low-label task can be frozen prospectively:
hide a fixed subset of official TRAIN labels from every compared learner and
use the remaining TRAIN graph inputs without their labels for consistency.
All methods receive the same labelled mask, TRAIN input collection, selection
opportunity and measured budget. This constitutes a new task, not an improvement
to an old full-label score.

For graph-inductive molecule datasets, **official VALID/TEST graph inputs must
not be moved into unlabelled training**, even if their targets are hidden.
Their labels remain solely in their declared selection/evaluation roles.
Additional unlabelled molecules require separately admitted data and overlap
checks; none are acquired or proposed here.

**WikiCS:** the native node task already exposes one transductive graph's
features/edges. A separately frozen consistency adaptation can use permitted
unlabelled node inputs in that existing graph while keeping true VALID/TEST
labels out of training. This convention does not authorize transduction across
held graph-inductive endpoints. Plain, consensus and relation-credit controls
must share the same allowed input population, masks, member/view counts and
selection opportunity before interpreting a gain.

No new fit is admitted. No existing family is extended. The useful conclusion
is a comparator and a labelled/unlabelled mechanism distinction. The current
recipient-placement novelty remains unresolved, and the prior conditional
tying-by-recipient study is not replaced by a coefficient sweep.

## Search and custody

Saved index72, supplement22, closest graph-ensemble/DIVE conclusions and
GRAND/GraphMix scopes were consulted before interpretation. Six bounded arXiv
queries were made: three broad graph/ensemble/evidence queries, then three
targeted title/shared-parameter/diversity queries. Recent-first broad results
contained many unrelated scientific graph ensembles; title constraints improved
relevance. Hits are discovery, not paper reads. No absence claim follows from
query limits, zero shared-phrase hits or unread leads.

Exactly two new primary identities have method scopes. Whole-paper reads,
author-code audits, proof certifications, reproduced results and new scientific
executions are all zero. The already unresolved Information Fusion paper and
previous GNCL/CoGNN/TabM/Fed-GAME/GNNMoE method scopes were not re-fetched.
Raw versioned HTML, metadata, exact section boundaries/extracts and SHA256
custody are retained in this directory. Only project research artifacts were
created; pending outcomes, frozen source, remote jobs and canonical memory
pointers were untouched. Root owns comparator admission and memory integration.
