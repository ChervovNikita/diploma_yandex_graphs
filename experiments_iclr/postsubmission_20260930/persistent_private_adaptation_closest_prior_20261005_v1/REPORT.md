# Closest prior for persistent private adaptation

## Assessment

The broad rationale is already online meta-representation learning. **OML explicitly trains representations so head-only online updates on correlated data transfer and interfere less. C-MAML/La-MAML explicitly carries that objective into ongoing online learning with persistent model parameters.** Combined with the retained BMAML/EMAML shared-feature/private-classifier ensemble prior, persistence does not establish a new learning principle.

The inspected methods do not specify the candidate's exact update commitment: update shared parameters from a virtual private step, then replay the same private optimizer transformation from the original private parameters and moments at the new shared parameters, committing it once. That is a concrete state-transition difference. It is a consistency convention and an algorithmic variant whose novelty and usefulness remain unestablished. Differentiating the current Adam transformation is an implementation requirement for the chosen objective; it is not itself a new meta-learning idea.

The defensible remaining graph difference is a **testable task-construction choice**: inner labels exclude both endpoints of every positive and negative outer query, while the graph still supplies non-target observed context around those endpoints. It supports an endpoint-transfer hypothesis. It does not create independent tasks, erase historical outer-label influence, or establish new-node generalization. A positive result against generic random episodic meta training would be needed to show that this is more than an OML-like update applied to graph queries.

## Retained evidence first; bounded new reading

Retained v63 index SHA256: `b9e7ebe7bec178b57d9fece610baa3efc37824aa72e90208e284cc400948b4c4`. Its conclusions already attribute ANIL's head-only inner/shared-feature outer updates; BMAML/EMAML's shared-feature meta ensemble; Meta-Graph's graph-task link meta learning; MLDG's train-only transfer/no-adaptation deployment; and MetaReg's persistent source heads/virtual private updates. These were reused without reopening their papers or adding primary-read credit. `REUSED_CONCLUSIONS.json` preserves the selected provenance.

Exactly three new paper identities received targeted method scopes. Versioned HTML sources, selected passages, algorithm lines, retrieval hashes, and actual scope limits are retained. No author implementation was read, no proof audited, and no whole-paper read is claimed. The OML source is **1905.12588v1**: its prose calls the objective OML and the implementation MRCL. No claim about a later implementation follows from that naming difference. Metadata-only discovery of a Bayesian-online title added zero method reads and supplies no finding.

No local training/geometry/heldout outcomes or execution receipts were opened. No source implementation was modified, no canonical literature index was edited, and no model, server or GPU work was performed. Incidental published qualitative result prose and a parameter table were exposed in the OML appendix selection; a faulty author locator also returned a truncated overbroad preview. These are recorded as incidental exposure, not adopted as empirical evidence or credited as complete reading.

## The exact transition being compared

For a current episode, keep starting private weights \(\phi_m\), previous private optimizer state \(s_m\), inner data \(I_m\), support \(G\), and inner randomness \(\xi_m\) fixed. Let \(U_m\) return both new weights and new optimizer state:

\[
(\widetilde\phi_m,\widetilde s_m)
 =U_m(\theta,\phi_m,s_m;I_m,G,\xi_m),
\]

\[
F(\theta)=\tfrac12\operatorname{BCE}(y,\operatorname{mean}_m z_m)
 +\tfrac12\operatorname{mean}_m\operatorname{BCE}(y,z_m),
\quad z_m=f_{\rm eval}(\theta,\widetilde\phi_m;O,G).
\]

The shared optimizer advances once using the complete current-step derivative of \(F\). After obtaining \(\theta^+\), discard all virtual private state and commit

\[
(\phi_m^+,s_m^+)=U_m(\theta^+,\phi_m,s_m;I_m,G,\xi_m).
\]

The committed private step uses its own inner loss only. There is no direct outer-loss private parameter update and no inference adaptation. Historical parameters and moments carry forward numerically, but their earlier update histories are **not differentiated through**. This is a current-step conditional derivative, not a full online trajectory gradient or an equilibrium bilevel derivative. Adam's current moments affect the transformation, while their historical dependence remains held constant.

Recomputation makes the committed private weights and moments correspond to the new shared representation and the same episode update map. It does not guarantee monotone improvement under Adam. It also does not equate episode support and dropout conditions with every eventual evaluation support. The mean-logit/member-loss scalar tradeoff has retained own/pool objective ancestry.

## Three targeted primary comparisons

| Primary method scope | Established overlap | Exact difference and limits |
| --- | --- | --- |
| **Javed & White, OML/MRCL**, [1905.12588v1](https://arxiv.org/html/1905.12588v1), §2–3, Eq3, AppendixA Algorithm1, AppendixB | §2 allows correlated sequences. §3 holds representation parameters fixed during online head updates and learns the representation through the loss after those updates. It expressly permits inner learning from a subset while scoring the larger visited population. This directly precedes the transfer/interference rationale and head-only adaptation credit. | Algorithm1 starts each inner trajectory at current \(W\), makes temporary \(W_j\) updates, then meta-updates both initial \(W\) and \(\theta\). It does not set \(W^+=U_W(\theta^+,W,s)\) as an inner-only commitment. The candidate excludes current outer endpoints from inner labels and uses a competence-anchored ensemble objective. OML's offline representation-pretraining use is narrower than its generic objective; offline use does not justify claiming that online transfer credit is new. |
| **Gupta, Yadav & Paull, C-MAML/La-MAML**, [2007.13904v1](https://arxiv.org/html/2007.13904v1), §4, Eqs5–8, Algorithm1 and AppendixC Algorithm2 | §4 expressly optimizes the OML objective online. Persistent base parameters undergo fast current-stream updates for a meta loss on current plus replay samples, followed by a meta update. Anticipatory ongoing training and persistent state are already explicit. | C-MAML commits the base meta update, rather than an inner-only fast private block. La-MAML learns per-parameter rates and updates rates before using them to scale the base meta gradient; its Eq8 uses the gradient from the existing lookahead trajectory. That asynchronous rate/weight update is not the candidate's replay of private Adam at \(\theta^+\). No shared/private graph ensemble, endpoint-positive-and-negative exclusion, or recomputed private Adam moments is specified in this scope. No claimed equivalence proof was audited. |
| **Finn, Rajeswaran, Kakade & Levine, FTML**, [1902.08438v1](https://arxiv.org/html/1902.08438v1), §3, §4.1, §5, Algorithms1–2 | Meta parameters persist and improve as tasks/data arrive. Local updates are explicitly included in the objective; practical meta gradients use separate inner/outer minibatches and Adam for outer optimization. Recomputing local adaptation from the current base for evaluation is already present. | Algorithm1 carries \(w_{t+1}\leftarrow w_t\), after base meta updates. Its evaluation copy \(\widetilde w_t\) is made by `Update-Procedure` from current \(w_t\); it is not the base carried to the next task. Candidate private adapted weights and their recomputed moments are carried into the next graph episode, and inference uses the final committed frozen predictor. FTML's regret setting is task-based with local deployment adaptation; its assumptions/theory do not transfer to this graph recipe. |

OML's objective is already written for a generic predictor and generic online update. Replacing that predictor with an ensemble and prescribing a graph-conditioned inner/outer sampling distribution fits that broad template. The exact sampled distribution, objective, parameter partition and persistent commitment can still be an empirical method contribution. Naming the same mechanism “private transfer” does not by itself supply one.

### Persistent meta ensembles: what the retained scope establishes

BMAML/EMAML already makes an ensemble initialization persistent across outer meta updates, with shared features and private classifiers. Ordinary MAML also persists a meta initialization. This form of persistence must be separated from **carrying the fast inner-updated private learner and its moments** between episodes.

The retained BMAML/EMAML scope does not qualify the latter exact commitment rule or Adam moment custody. That unresolved detail is not evidence that no earlier method used it. No additional ensemble paper was read merely to enlarge the citation list. An exact earlier shared-representation/meta-ensemble algorithm with inner-only fast-state commitment after representation updates would remove the candidate's proposed state-schedule distinction; the graph episode construction would then carry all remaining specificity.

## What endpoint conditioning could support

For outer queries \(O\), define \(V_O\) from **both query classes**, and require every inner pair to have neither endpoint in \(V_O\). The criterion then asks whether changing representations changes what private updates elsewhere learn, as measured on these outer pairs. This is more specific than merely drawing two random minibatches. Target positives are masked from the episode support; other observed TRAIN edges incident to \(V_O\) can remain context.

The inspected scopes do not prescribe that exact graph sampling-and-context contract. That bounded absence is a recipe difference, not global novelty clearance. The strong reason to test it is repeated endpoint/neighborhood supervision: a current private update that avoids the outer endpoints may favor transferable corrections over local endpoint fitting. The premise is unproved, and conventional transductive link evaluation does not automatically represent new-node or sparsely supervised endpoint deployment.

The method still retains features, neighboring observed edges, shared parameters, private history and historical labels associated with those nodes. Endpoint separation changes current **supervision**, not information independence. Degree/CN filtering and hub eligibility also change label exposure. Any advantage may arise from context removal, altered degree distribution, exposure, extra work or generic meta regularization.

## Falsifiers and required attribution

| Claim | Decisive comparison or finding | Interpretation if it fails |
| --- | --- | --- |
| Endpoint-conditioned supervision contributes | Same outer queries, per-route class counts, full-TRAIN degree/CN strata, paired common target mask, update rule and paid schedule; endpoint exclusion versus random inner supervision. Measure post-mask strata and per-ID/node exposure as remaining confounds. | Equal performance leaves endpoint-specific attribution unsupported. Material exposure/support differences leave the attribution unresolved even after an apparent gain. |
| Adaptation credit contributes | Same virtual private updates and recomputation, with live versus detached \(D_\theta U\). | Equal performance supports a schedule/masking explanation rather than shared credit through private learning. |
| Separate private learners contribute | A capable single with the same four inner streams concatenated, repetitions retained, same sampling/support and adaptation schedule; compare served quality under the declared budget. | A matched single removes ensemble-specific evidence; improvement over an ordinary single alone is insufficient. |
| Sharing contributes | Untied four using the same ensemble objective and persistent update rule. | If untied four matches or wins, generic ensemble meta training may explain the result; ordinary-independent comparisons alone do not isolate sharing. |
| Recomputation is useful beyond consistency | A separately reviewed commitment comparison: retain the virtual private step made at old \(\theta\), versus recompute from original private weights/moments at \(\theta^+\), accounting for the extra pass. | No meaningful gain makes replay a consistency/cost choice, not an established scientific mechanism. This comparison is proposed, not authorized or adopted here. |
| Recipe improves practical prediction | Replicated served ranking quality against competent ordinary and adapted controls, with complete compute/memory/rejection accounting. | Training BCE, diversity, a valid derivative, and optimizer-state parity cannot substitute for predictive evidence. |

A primary method implementing the same persistent private commitment and endpoint-conditioned graph supervision would falsify an exact-method novelty claim immediately. A generic OML implementation winning under matched graph geometry would favor the interpretation that this is an existing online meta method with a useful task construction. No evidence in this report settles either outcome.

## Recommended claim boundary

Position the candidate as **an endpoint-conditioned graph instantiation of online meta-representation learning, with a declared persistent private-state commitment and a competent served-ensemble objective**. Attribute OML and online meta learning directly alongside the already retained ANIL/meta-ensemble/domain-generalization sources.

There is a precise, falsifiable graph recipe worth evaluating. The present evidence does not support a new online-learning principle, a general persistence principle, an optimizer-state novelty claim, independent graph tasks, or established quality improvement. Whether endpoint conditioning supplies a defensible contribution depends on its matched graph controls and relevant evaluation—not on renaming the learning operator.
