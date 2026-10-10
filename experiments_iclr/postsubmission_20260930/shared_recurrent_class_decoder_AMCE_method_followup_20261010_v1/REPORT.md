# AMCE full-method resolution and decoder operation mapping

10 October 2026. Source-only successor to the sealed
`shared_recurrent_class_evidence_wrapper_hypothesis_20261010_v1` proposal.
**Resolve the 2011 primary method; retain the existing decoder as an attributed,
untested adaptation.** No recipe, control, gate, implementation or scientific
admission changes are made here.

## Primary source and scope

Hoda Eldardiry and Jennifer Neville, **Across-Model Collective Ensemble
Classification**, AAAI 2011, pp.343–349,
DOI [10.1609/aaai.v25i1.7934](https://doi.org/10.1609/aaai.v25i1.7934).
The method is named **Collective Ensemble Classification (CEC)** in the paper.

The old Purdue author-copy URL returned 404. DOI/OJS retrievals disconnected.
The author's current [VT publications page](https://people.cs.vt.edu/hdardiry/publications_by_year/)
links to a working [official AAAI CDN PDF](https://cdn.aaai.org/ojs/7934/7934-13-11462-1-2-20201228.pdf).
That PDF is preserved as `primary/amce_official.pdf`, SHA256
`961c7f1b3dab55635023de6b95812673b587b699c37e5bca14992b1d54537804`.
Every failed retrieval and non-paper 200 response is retained in the receipts.

All seven pages' extracted text was read. The complete problem formulation,
learning, inference, complexity, baselines and experimental implementation are
on pp.344–346; Algorithms 1–3 on p.344 were checked against rendered pixels.
No author code was read. Plot pixels on pp.347–348 were not inspected, so this
adds one complete primary **method scope**, with no full visual-paper credit.
The authors' performance and variance claims are source claims, not project
accuracy evidence. Existing CPGNN and GBPN complete §2 method scopes were reused
for the requested operation mapping; no additional paper discovery credit.

## What CEC actually does

**Acquisition is independent.** Algorithm 1 learns a relational node classifier
`F_i = P_i(Y_v | X_v, X_neighbors, Y_neighbors)` separately for each link graph.
The same complete training-node set is used for every model, with a different
link type supplying each model's relational view. The training graph is fully
labeled and differs from the partially labeled graph on which collective
inference runs. This is relation-subset ensemble learning, not bootstrap in the
reported recipe. Single-source resampling is future work on p.349.

**The classifiers are fixed during inference.** The paper's implementation uses
relational dependency networks and pseudolikelihood estimation, then collective
inference based on Gibbs sampling (500 samples per reported inference run).
Neither Algorithm 3 nor the training description updates model parameters through
the inference process. There is no described end-to-end unrolled gradient, common
trainable decoder matrix or jointly optimized pooled-ensemble loss.

**Inferred values cross model boundaries.** Algorithm 3 starts a label-state field
`Yhat_i` and history `YT_i` for each model, keeps observed labels, and randomly
initializes unknown values. At a collective-inference update, each model proposes
an inferred value for the same unknown node using its own relation neighborhood
and current inferred neighbor values. Instead of feeding its own proposal into
the next round, it feeds an arithmetic average with the other models' proposals
for that node. The averaged values are also stored in its inference history.
Final per-model marginals are computed from the stored inference values, followed
by a final average over models (Algorithm 3 lines 12–16; p.345 prose).

Thus CEC maintains indexed state containers but **couples their contents through
cross-model inference averaging**. Under an intended simultaneous sweep, the
same-node aggregated update is common to the models. This should not be described
as independent route states with only shared parameters. The RE baseline keeps
collective inference independent and only averages completed outputs; it uses
the same trained classifiers as CEC (p.346).

There is a bounded implementation ambiguity: Algorithm 3 line 9 reuses the node
index as its summation index, and its nested loops do not cleanly specify when
every model's proposal is available. The prose explicitly says simultaneous
inference and averaging other models' predictions for the same node. That
supports the operation above, but not a claim about exact synchronous array
equality, update ordering, burn-in or a multiclass probability-state encoding.
The paper's experiments have binary labels. No corrected author implementation
is assumed here.

**Class interactions are learned inside the separate relational classifiers.**
CEC combines predictions with a fixed arithmetic mean. It does not introduce
the proposal's single `C×C` class-score matrix tied across models and iterations.
Parameter sharing and shared neural bodies are not the operation of Algorithms 1
and3. This is a statement about the inspected method, not an absence claim over
other literature or later implementations.

## Exact sealed candidate and supervision

The candidate retains its sealed definition:

```text
q_m^0 = softmax(Z_m)
S_m^(t+1) = Z_m + (P q_m^t) B; q_m^(t+1) = softmax(S_m^(t+1)), t=0,...,4
L = mean_m [ .5 CE(Z_m, TRAIN) + .5 CE(S_m^5, TRAIN) ]
serve = mean_m q_m^5
```

`P` is the factual incoming nonself row mean, preserving record multiplicity and
giving empty rows zero. One unrestricted `B[C,C]`, initialized zero, is tied
across all routes and steps. Every class state and gradient remains live.
Class states are **separate for each route**; output averaging occurs after the
recurrence. There are no literal label values in the forward messages.

The loss is an average of route-level native and decoded losses. It is **not**
`CE(mean_m q_m^5, y)` and does not directly supervise the pooled distribution.
Routes interact through common body/B parameters and their accumulated gradients,
not through averaging their inference states. Genuine I4 in the fixed control
fits each native+own-B model separately; four untied bodies/common-B are jointly
trained and remain a coupled bank.

## Closest-operation mapping

| Operation | CEC / AMCE | CPGNN, §2 | GBPN, §2 | Sealed candidate |
| --- | --- | --- | --- | --- |
| Native/prior prediction | Independently learned relational conditionals, one per link graph | Any off-the-shelf neural prior estimator; `Bp=softmax(R)` | Neural unary/self potentials, instantiated by MLP softmax | Current native route logits `Z_m`, one native forward each |
| Recurrent state | Unknown inferred values and histories; proposals averaged across models during inference | Centered node beliefs `b0=Bp−1/C`; `b^k=b0+A b^(k−1) Hbar` | Node beliefs plus directed edge messages; reverse-message cavity division | Route-specific `q_m^t`; neighbor class means mapped by common B |
| Class coupling | Encoded in each independently learned relational classifier; fixed arithmetic cross-model mean | One learned compatibility matrix Hbar used across propagation layers | One learned positive symmetric edge coupling H across graph/BP steps | One unconstrained score matrix B across graph, steps **and routes** |
| Acquisition/training | Independent base-model pseudolikelihood learning; fixed inference models | Prior pretraining, label/prior-derived compatibility initialization, joint decoded CE and regularization | Unary/coupling parameters trained through unrolled BP marginal-label NLL | Joint native/decoded route CE, live gradients through all five steps |
| Ensemble supervision | No described pooled ensemble training loss | No ensemble in inspected method | No ensemble in inspected method | Mean route losses; pool used for serving/selection, not CE target |
| Literal known labels in inference | Observed labels retained in the partially labeled inference graph | TRAIN labels enter compatibility initialization; propagation uses predicted priors | GBPN-I uses features only; transductive GBPN conditions on TRAIN labels | No label values in forward recurrence; TRAIN supplies loss targets |

**CPGNN is the closest algebraic primitive.** Both attach learned class-coordinate
propagation to a modular predictor, reuse a native/prior anchor at every step,
tie one class matrix across steps, and jointly learn predictor plus compatibility.
CPGNN's Eq. 6 already supplies the additive neighbor-class matrix operation.
The candidate changes the anchor from centered prior probabilities to logits,
uses row-normalized directed P rather than the stated binary adjacency A, applies
softmax at every step rather than only at the end, starts B at zero, and adds
simultaneous native CE. CPGNN pretrains its prior estimator and initializes Hbar
from TRAIN truths plus other-node predicted priors, with Sinkhorn normalization,
symmetrization/centering and a row-centering penalty. Tying B over ensemble
routes is the candidate's extra estimation constraint; these comparisons do not
establish a new principle or expected gain.

**GBPN is the closest differentiable probabilistic-inference comparator.** In
log space it adds unary log probabilities to sums of log edge messages, while
each message uses `LSE(log H + log p − log reverse_message)`. The candidate uses
`(P q)B` as an expected class-score message and keeps no directed edge/cavity
state. It therefore does not implement GBPN's BP. GBPN-I is the closest
same-information variant because it uses no literal label values in its forward
inference; the transductive variant conditions on half TRAIN during learning and
all TRAIN at serving. The candidate's directed P and unrestricted B supply no
exact undirected-CRF or posterior guarantee. GBPN's degree-related confidence,
degree-adjusted likelihood and edge-message storage also remain source-specific.

**CEC is the closest ensemble-inference collision.** Ensemble collective
prediction with inference-time exchange is established by this primary source.
The sealed candidate shares learned class parameters while retaining separate
states, whereas CEC shares inferred values through averaging while retaining
separately acquired fixed models. The candidate does not reproduce CEC's
consensus mechanism. Conversely, CEC's variance rationale does not establish
that a jointly trained common B will acquire missing correct rankings.

## Disposition

The prior proposal's unresolved AMCE method row is superseded by this source
resolution. All its numerical settings, seven controls, selection conventions,
compute estimates and untested status remain unchanged. CPGNN/GBPN cover the
class compatibility and differentiable inference ingredients; CEC covers the
ensemble/collective-inference intersection. Any useful result must establish
the proposed parameter-sharing benefit against the already fixed capable
single, genuine I4, private-B and untied-body/common-B controls.

Root and the history researcher supplied the complete retrieval-family closure:
the broad live retrieval recipe fails its utility criteria and repairs zero
native common strict wrong rivals across all nine shared banks. No outcome
payload was opened by this researcher. This keeps that exact retrieval recipe
closed; it is not evidence for the decoder's success. Root owns any separate
decoder freeze, execution and interpretation.

Zero fits, model/data imports, TEST access, implementation writes or canonical
state mutations. No novelty, portable gain or acceptance claim.
