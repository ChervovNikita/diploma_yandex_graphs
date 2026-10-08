# Recent graph contrastive prior: FairACE

## Scoped conclusion

FairACE is a useful new related-work scope, not a newly supported direction for
GNNM. Its inspected method provides a graph-neighborhood prediction objective
and an online/EMA-target contrastive pair. It does not specify four simultaneously
served predictors with a shared backbone and private BE factors. No result,
novelty clearance, implementation equivalence, or scientific fit is adopted.

The relevant research question remains whether different supervised graph
targets can improve the combined prediction while preserving member accuracy.
Making embeddings different is insufficient. The previously closed Wiki24
diagnosis found that old unit+contrast members were individually competent but
still shared errors; the ordinary independent pool remained stronger. This new
reading supplies no evidence that degree fairness or node repulsion resolves
that common-error problem. The distributed error profile also does not support
a low-degree-only explanation.

## Discovery and reading scope

Three discovery queries were executed: recent graph-ensemble complementarity,
recent supervised graph contrastive/graph ensemble contrastive methods, and
recent shared-parameter ensemble diversity. Exact queries and response bytes
are saved in `DISCOVERY.json` and its three named response files. The date
restriction was 2025-01-01 through 2026-10-08. Crossref results were often
application papers or review/decision-letter records; that discovery quality
limit is retained. The arXiv response exposed FairACE as an accessible method.

One previously unread primary method scope was selected:

- **Identity:** `arxiv:2504.09210v2`.
- **Title:** FairACE: Achieving Degree Fairness in Graph Neural Networks via
  Contrastive and Adversarial Group-Balanced Training.
- **Source:** <https://arxiv.org/html/2504.09210v2>.
- **Version metadata:** first submission 2025-04-12; selected v2 update
  2025-04-15, from the saved arXiv API response. No DOI or journal reference
  appeared in that selected metadata entry; no publication status is inferred.
- **Actual read:** primary HTML Section IV-A–G, Equations 11–21 and complete
  Algorithm 1. The extracted passage retains HTML paragraph/equation IDs and
  MathML `alttext` formulas.
- **Not read:** full paper, experimental tables, proof appendix, author code,
  runtime dependencies or checkpoints. Abstract discovery was a locator, not
  independent full-paper credit. No published scores are borrowed.

The exact ID/title was absent from the inspected v72 index and active supplement
ID files. This is one new scoped primary-method read and zero full-paper reads.
Other discovery hits remain metadata/abstract-only, including SupGCL, SGCL,
Grad, MolFM-Lite and multiple-interpretation ensemble distillation. They are not
credited as method reads, recommended as validated baselines, or described as
having an inspected operation.

## Actual FairACE operation

Section IV-B specifies a two-layer graph convolution encoder followed by
PairNorm. In IV-C, a predictor maps a node's identity representation toward its
neighbors' representations. Neighbor targets come from an EMA target encoder;
negative node representations come from the online encoder. Equation 16 is the
stated asymmetric contrastive objective. Equation 15 separately describes
repulsion between **different nodes**, not between different predictors of the
same object.

Section IV-D adds a degree-regression discriminator and gradient reversal.
Section IV-E uses supervised CE averaged within degree groups and additionally
weights the group means by inverse group size. Equation 21 combines the
asymmetric contrastive loss, degree loss and supervised group loss. Algorithm 1
alternates discriminator and encoder/projector/classifier updates and updates
the target encoder by EMA. Section IV-G states that the final encoder and
classifier are used downstream. The inspected deployment rule does not pool
four independent or shared-backbone predictions.

The online/target pair serves a temporal training role. It is not evidence of
diverse served ensemble members. The word “ensemble” in the acronym's abstract
expansion does not change the specified parameter roles and deployment rule.

## Complete-rule comparison

| Rule component | FairACE inspected scope | Retained GNNM context-target hypothesis |
|---|---|---|
| Predictor parameters | One online GCN encoder, one EMA target copy, predictor, classifier and degree discriminator | Four private BE factor rows inside one shared native backbone; no teacher or discriminator |
| Supervised labels | Group-weighted classifier CE | Ordinary own CE on every TRAIN label for every route |
| Positive targets | Immediate neighbors represented by the EMA encoder | Fixed TRAIN-only same-class positives selected using one of four feature/graph signatures |
| Contrastive roles | Online identity, predicted neighbor context and target encoder | Symmetric stochastic cross-view representations from each persistent route |
| Diversity pressure | Node-level uniformity/repulsion is described; degree invariance is encouraged | Different label-compatible targets provide route-specific cotangents; no unconditional inter-route repulsion |
| Graph role | Graph supplies propagation and neighbor prediction targets; degree is a training attribute | Native predictor graph is unchanged; fixed graph-filter signatures define auxiliary positive relations |
| Training updates | Alternating discriminator/model updates and EMA | One scalar own-CE plus fixed-coefficient alignment objective with native optimizer and selector |
| Deployment | One trained encoder/classifier | Unchanged mean-probability pool of four persistent routes |
| Decisive comparison | No relevant four-route shared/private control specified in the read scope | Matched common targets, class-preserving shuffled targets, then capable single and ordinary/context-matched untied ensembles if the frozen screen passes |

FairACE therefore owns useful neighborhood-context prediction and degree-fair
contrastive training ingredients. It does not specify the full retained
route-to-context supervised BE assignment rule in the inspected section.
That bounded difference is **not** novelty clearance: saved SupCon, HLCL,
MA-GCL, AMCL, CGCL and ensemble priors remain direct ancestry.

## Source ambiguities and limits

The text introduces Equation 15 as a uniformity loss, but Equation 21 includes
L1, L2 and L3, without a separate LUNI summand. The prose says gradient reversal
trains degree-invariant embeddings. Algorithm 1 explicitly applies GRL while
the encoder is frozen in the discriminator update, but does not explicitly
repeat that operation in its encoder-update bullet list. An author-code read
would be needed before claiming a fully reproduced optimizer or silently
choosing one interpretation. These are source-level specification limits, not
experimental findings about FairACE's quality.

Neither the source's stated fairness intention nor its node-uniformity term
guarantees that our four routes will preserve member competence, acquire correct
rankings, or improve mean-probability serving. No extrapolation to graph-level
or link-prediction performance is made.

## Decision

Preserve FairACE as an attributed related-work scope. Do not add a new training
arm, revive the inactive degree-allocation proposal, inspect pending outcomes,
or tune a coefficient from the closed error profile. The existing frozen
context-target study already asks the stronger relevant question: whether
persistent label-compatible target assignment changes useful route decisions.

Any later elected alternative must be specified before scoring and include
capable single and ordinary independent ensembles, plus a matched shared-route
control. Its required diagnostics are paired net repairs, member mean/worst
accuracy, pooled loss, any-correct coverage and strict common wrong-rival
support across the full selected population. A gain in hidden-space diversity
alone is a failure of the intended scientific claim.

This packet is a bounded literature addition. Canonical pointer adoption,
scientific admission, manuscript use and publication remain parent-owned.
