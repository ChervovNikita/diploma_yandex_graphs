# Prediction quality in shared-backbone ensembles: focused primary reading

15 October 2026. **Three scoped primary documents: two deeper readings of
retained papers and one source absent from literature index v72. No additional
mechanism is proposed.** This pass strengthens interpretation and controls for
the prepared residual/C1/C6 candidates. It does not close those hypotheses,
establish novelty, transfer a published gain, or alter a frozen pilot.

## 1. What was new in this reading

| Primary document | Exact current scope | Accounting |
| --- | --- | --- |
| [TabM, arXiv:2410.24210v3](https://arxiv.org/html/2410.24210v3) | §§5.2–5.4; Appendix D.6–D.7. Previously read §5.1 supplies competence context only. | Retained paper; additional analysis/protocol scope, not a new paper. |
| [Training Diverse Graph Experts for Ensembles, arXiv:2510.18370v1](https://arxiv.org/html/2510.18370v1) | §§3.4–3.6; Appendix D.4. Revisited §2.3 only to bind EI/oracle metrics and VALID-trained fusion. | Retained index record52; deeper domain-discovery/quality scope beyond its recorded §§2.1–2.3/3.3/B.2/F. |
| [GraphDIVE, IJCAI 2022, DOI:10.24963/ijcai.2022/289](https://www.ijcai.org/proceedings/2022/0289.pdf) | §3.2; §§4.1–4.4, Eqs1–10; §5.1 protocol and §5.4 component-ablation definition. PDF pp2–6; Eq2/Fig2 and Eqs4–10 visually checked on pp3–5. | One primary identity absent from v72 and the consulted expansion notes; no exhaustive project-wide novelty claim. |

No full-paper certification. The recent 2024/2025 papers were prioritized;
the older molecular GraphDIVE is included because it directly defines graph
views, shared representations, experts and conditional fusion. LoRA-Ensemble,
ELREA, TabLoRA, GraphMETRO, Node-MoE and the recent low-loss-curve scout are
retained conclusions, not additional new primary reads. Search metadata for a
2024 knowledge-diverse-expert paper remained discovery only; no method claim
is made from it. `READ_SCOPES.json` and `SEARCH_LIMITS.json` give exact custody.

## 2. Substantial findings

### TabM: collective quality, useful backbone activation, and protocol limits

The already-retained §5.1 observation remains central: individually weaker or
more overfit submodels can make a useful collective predictor. Requiring every
member to outperform native M1 would reject a mechanism explicitly observed in
TabM. Measuring member competence and common-error repair remains necessary;
neither a large pooling gain over poor members nor disagreement alone suffices.

The new §5.2/D.6 scope shows that greedy selection evaluates the **collective
validation effect** of adding a member, rather than sorting individual scores.
It starts with the best member and uses two simultaneous stopping conditions:
no remaining addition improves the current validation metric, and the subset
already beats the original all-member validation predictor. The reported test
comparison is slightly below full TabM. This is a selection/serving tradeoff,
not evidence that pruning repairs weak graph banks. It uses VALID to fit the
subset; it cannot be silently added to our TRAIN-fitted C6 protocol.

§5.3 reports that useful member count depends on backbone width/depth and that
too many members can hurt. **D.7 tunes learning rate separately for each k**,
uses dropout .1, five seeds and 17 datasets. This does not establish a causal
member-count law under our fixed native learning rate or justify a rank/width/k
grid. The current four-member graph bank remains its own declared experiment.

§5.4 defines a dead neuron as one that never activates on 2,048 TRAIN objects
at a selected checkpoint. It reports lower dead-neuron fractions with k=32
than k=1, supporting a possible utilization explanation. This is observational
evidence, not a theorem that weight sharing causes generalization gains. The
exact-zero ReLU statistic does not transfer unchanged to GELU, sigmoid or
normalization-heavy graph operators. No new native activation is introduced.

### Diverse Graph Experts: graph domains need meaningful and deployable signals

§3.4 tests whether a domain expert beats a complete-TRAIN expert **on that same
domain**, using the weighted gain

```text
ΔAcc = sum_d mass(d) [Acc(domain-d expert on d) − Acc(full-data expert on d)].
```

This is more demanding than error inconsistency. In that study, degree,
PageRank and clustering-only domains yield negative gains; feature clustering
is inconsistent. Node homophily and intra-class graph metrics are more useful.
Those are study-specific observations, not a theorem against label-free
structural context or our positive, within-class-mass-controlled C2 rule.

The successful metrics explicitly use `y_v` and neighbors' true classes:
intra-class edges satisfy `y_u=y_v`, and node homophily divides intra-class
degree by total degree. Such a domain partition is a useful diagnostic object.
It is **not a serving input available under our no-unknown-label rule**.
Restricting it to observed TRAIN labels or learning a proxy changes its
information and estimation law; the oracle-domain gains do not establish the
proxy's usefulness. This source does not authorize heldout labels in a gate.

§3.5 also studies full-data pretraining followed by domain fine-tuning.
§3.6 observes a particularly relevant failure mode: increasing partition count
can raise oracle upper-bound accuracy while lowering practical Ensemble/MoE
accuracy. It also reports that adaptive fusion can sometimes exploit weak
experts better than global weights, while labeling its calibration explanation
as a hypothesis. Neither result supplies a competence guarantee.

The source uses independent experts, VALID-tuned global weights and a
VALID-trained adaptive fuser (§2.3). Its domain setting assumes TRAIN and TEST
come from the same mixture (D.4); it is not an OOD-generalization guarantee.
Our live shared backbone, TRAIN-fitted gate, fixed native operators and complete
own losses differ. We do not import its expert pretraining pipeline, its HPO,
its excluded original-paper datasets, or its gain magnitudes. Qualitative
two-dimensional loss-landscape plots are not global convexity proofs.

### GraphDIVE: direct molecular graph/expert ancestry, with a loss-sign limit

GraphDIVE creates diverse node aggregations and graph readouts using
degree-derived weights and generalized p-statistics. It concatenates node
statistics through a learned FC map, forms multiple graph-level views and
applies groups of sigmoid classifiers. Two learned softmax gates combine
experts within a view and then combine views. Its serving rule (Eq5) is

```text
p(y|G) = sum_i a_i(G) sum_m b_im(G) p_im(y|G).
```

This directly precedes broad claims about structure-aware molecular experts,
shared-feature specialization and conditional ensemble fusion. Its views
change neighborhood aggregation/readout operators. Our portable residuals
instead modify an explicitly bound encoded input before the unchanged native
graph computation; they are not a GraphDIVE reproduction. Its classifier/view
organization also differs from four complete native nonlinear trajectories.

The supervised term printed in Eq10 is a gate-weighted **sum of expert
log-probabilities**, not `log(sum gate × probability)` and not our uniform
mean-own risk F. Gate/expert/shared-feature gradients are coupled in the
printed scalar objective; low gate mass reduces an expert's supervised credit.
No uniform per-member competence protection follows. The reported scaffold
protocol uses ten seeds and a substantial validation grid over views, experts,
regularization, statistics and gate temperatures (§5.1); its gains cannot be
attributed to our small fixed adapter or gate alone.

**Unresolved reproducibility point:** Eq9 defines `L_i = −mean_j KL(p_z||p_j)`;
Eq10 prints an outer minus around `[weighted log p + λ L_i]`. Under ordinary
loss minimization, that combination adds positive KL to the loss, contrary to
the surrounding diversity-encouragement prose. The signs/brackets were checked
visually on PDF p5. The printed author-repository URL returned HTTP404. No
implementation audit resolves the tension here. We neither silently reverse
the sign nor adopt this objective as a faithful known repulsion control.
Published experiment rows were incidentally visible during the scoped read;
no result was recomputed or adopted as new graph-benchmark evidence.

## 3. Contrast with the actual prepared mechanisms

Let F denote the mean of the four original native per-route risks, preserving
all original masks and loss groups. Shared native parameters are θ and private
parameters are φ_m. The table describes source preparation, not a new freeze.

| Prepared object | Target/loss and gradient path | Serving and closest literature boundary |
| --- | --- | --- |
| Portable residual with BE, or residual-only | `h_m ← h_m + U_m GELU(V_m h_m)`; rank `min(8,width)`, U starts at zero. All four routes retain F; θ and private residuals receive ordinary native task gradients. No teacher or representation repulsion. | Mean raw logits; all four complete native trajectories. Standard low-rank/adapter and live-shared TabLoRA ancestry; distinct from frozen-base LoRA-Ensemble. |
| C1 molecular alignment plus conditional variation | Existing normalized live graph preheads, observed TRAIN binary labels, same-class nonself positives including same-graph cross-route positives; class-balanced denominator/anchor reduction. `F+.05(A+V)` in the all-credit case; private-credit requires explicit block routing. | Original raw-logit serving unchanged. A is task alignment; V prevents within-class collapse across distinct graphs. It is not cross-member output disagreement. |
| C6 prepared late-gate builder | Native normalized prehead plus signed mean and RMS route-logit deviation feed a shared hidden8 scorer and private route biases. Zero output/scalar biases give a uniform initial gate. Inputs remain live in autograd. **The builder does not define the complete own/pool objective or block-credit policy.** | `z_pool=sum_m softmax(score)_m z_m`, a convex raw-logit mixture. It has no class-label/degree oracle or inference-time input gradient; C6 still needs a task-specific TRAIN objective/controller freeze. |
| TabM mechanism, retained | Mostly shared MLP weights, private efficient adapters/heads, first adapter randomized and later adapters neutral, ordinary mean member loss and collective validation selection. | Native classification probability averaging differs from our declared graph raw-logit mean. New scopes add utilization and pruning observations, not a graph transfer result. |

For the portable residual, `∇V_m=0` at U_m=0 while `∇U_m` depends on the
route's random V_m features. Equal starting functions can therefore have
different private derivative paths. This is already explicit zero-up/random-
down adapter ancestry; it is not a new diversity principle. QR frames are
disjoint only where dimensions permit, and are not kept orthogonal during
learning. BUDDY's width16 branch necessarily has overlapping rank8 frames.

For C1, the actual variance term takes variance **over distinct graph rows
within each class and member**, never over member IDs. One-class batches skip
the alignment term; missing labels are excluded; variance needs at least two
distinct observed graphs in a class. Auxiliary gradients follow the supplied
live prehead graph; downstream output factors receive none if they are outside
that graph. These details prevent a generic “more diverse members” reading.

For C6, the available builder only returns pooled logits/weights. Its gate has
no direct correctness or marginal-utility target. End-to-end CE could learn
confidence scaling or feature corrections without identifying reliable members.
Raw-logit and probability mixtures are different functions, so neither
GraphDIVE nor the retained LoRA-Ensemble calibrates this gate by analogy.

## 4. Decisive comparisons for existing candidates; zero added mechanism

Retain native M1, original boundary GNNM and four genuine separately selected
native models, with exact body0 reuse and prospectively paired protocols.
Do not restore equipped, factorized-single, joint-untied or capacity-control
fits. No new arm/fit count or HPO grid is introduced by this note.

- **Residual question:** compare the two already-prepared residual variants
  against base GNNM and native references using the same native F, raw-logit
  serving and fixed rank. Report added private parameters and measured cost;
  BE dependence is tested, parameter matching is not claimed.
- **C1 question:** retain its already-declared no-auxiliary/alignment/full
  alignment-plus-variation controls. Evaluate served source quality and proper
  loss, each/mean/worst member quality, and repairs/harms. Sample-axis variance
  or latent separation is not a substitute for prediction quality.
- **C6 question:** before any fit, freeze its task-native pool/own loss and
  shared/private/gate credit; fit the gate only on TRAIN. Compare same-bank mean,
  TRAIN-fitted global weights and the conditional gate with meaningful input
  ablations. Never fit gate weights/subsets on the VALID set already used to
  choose that bank and call this the same TRAIN-fitted protocol. An oracle
  best-member readout is a diagnostic upper bound, not served quality.

These are refinements of existing falsifiable questions. The reading supplies
no additional target or gradient rule with a justified advantage over those
questions. Label-oracle graph domains are unavailable at serving; changed
graph operators would broaden scope; full-data expert pretraining is an
already-known separate pipeline; fixing GraphDIVE's ambiguous printed sign
would invent a reproduction choice. **Additional mechanism: none.** This is
not a claim that every possible shared-ensemble improvement has been exhausted.

No model imports, fits, scientific server actions, TEST access, frozen source
edits or manuscript/reviewer verdicts occurred. Source artifacts and exact
read/delivery hashes are recorded in `MANIFEST.json`.
