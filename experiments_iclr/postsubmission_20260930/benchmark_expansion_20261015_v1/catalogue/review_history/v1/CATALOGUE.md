# Finite benchmark expansion catalogue

15 October 2026. Preparation only. This catalogue proposes scientific questions;
it is not a run matrix, source admission or new launch. It leaves the frozen
24-fit MolBACE pilot and all molecular source unchanged. No benchmark fit total
is computed: task recipes, representative settings, same-operation controls,
stages and reuse must be specified before such a total has meaning.

## References and decision rules that stay in the design

- Retain the **270 separate native source-reference reproductions**, with source
  architectures/horizons and declared protocol limits. Extensions never enter
  that track. Their reported paper numbers remain historical context.
- At every compatible matched setting retain native M1, native-preserving
  boundary GNNM, and **genuine independent4**: four separately trained and
  independently selected complete native bodies. Matched M1 is its exact body0.
  GNNM selects one complete shared bank collectively. A jointly regularized
  untied bank is an additional control and cannot be renamed independent4.
- Start candidate comparisons from those controls; add an equipped M1/I4 or
  joint-untied control when needed to separate extra capacity, exposure, task
  information or coupled learning from sharing. Count these only after their
  actual recipes are frozen. Preserve native parameter identity, optimizer
  groups and source reductions; disclose each new adapter, state or loss rule.
- Compare pooled source task quality and proper task loss, costs, errors repaired
  and introduced, and alternatives retained or cancelled by pooling. Report
  individual/mean/worst member strength. **Do not require every member to beat
  native M1.** Collectively useful submodels can be weaker or individually
  overfit; this is a question to measure, not an automatic failure criterion.
  Existing frozen pilot criteria are not revised retrospectively.
- Use task-meaningful targets. Native node classification, binary molecular
  graph classification and sampled-query link ranking are not interchangeable
  CE populations. A missing task port is an implementation gap, not a scientific
  exclusion. A genuinely absent object—such as BN state in a model with no BN,
  or a wrong-class simplex for a scalar binary output—is a semantic limit.
- Keep fixed rules, no HPO or outcome-selected seed/site/rank/coefficient grid.
  TRAIN learns; whole VALID selects under the frozen policy; TEST stays closed
  in preparation/pilots. Prior original-paper datasets are history, not newly
  admitted experiments. No old recipe is rerun merely to rescue a closed result.

## Meaning of status

**Retained**: an obligatory reference, already declared training rule, or named
diagnostic control. **Test-next**: a finite scientific hypothesis worth preparing
with its task targets and falsifiers; this does not assert executable coverage.
**Deferred**: a specific scientific/design/evidence issue must be resolved first.
**Redundant**: an alias, algebraic duplicate, or exact already-closed recipe,
retained in history rather than counted as another candidate. Negative evidence
is scoped to its actual data/model/controller and does not exclude a different
task-specific hypothesis.

## Catalogue

The machine-readable [CATALOGUE.json](CATALOGUE.json) gives each rule, ancestry,
task adaptations, controls, readiness and disposition. IDs are mechanisms, not
fits, paper counts, or all possible combinations.

| ID | Status | Concrete question | Distinguishing controls |
| --- | --- | --- | --- |
| REF0 | Retained | Does the source-preserving native implementation reproduce its declared reference setting? | Fixed source horizon/evaluator; no new wrapper. |
| REF1 | Retained | What does shared boundary GNNM achieve relative to a capable native single and independently selected native4? | Exact body0 reuse; collective shared-bank selection versus four own selectors. |
| L1 | Retained | Does the served-pool loss help all parameters or specifically private credit? | Own F, global half-mixture J, private J/shared F; native task-specific F/L. |
| L2 | Retained | Does fixed positive private responsibility change useful predictions while the common body retains ordinary risk? | Uniform weights, matched IID weights, fixed TRAIN table; preserve native reduction groups. |
| C1 | Test-next | Can task-aware alignment with an explicit variation penalty improve pooled prediction quality? | No auxiliary, alignment only, alignment+variance; task-specific pairs and coverage. |
| C2 | Test-next | Does persistent, interpretable responsibility provide useful specialization beyond generic reweighting? | Class/group-mass-matched shuffled/IID assignment, all-block credit, capable equipped controls. |
| C3 | Test-next | Do different meaningful context targets teach complementary information? | Common target mixture, uniform target, mass-preserving shuffled target, assigned untied controls. |
| C4 | Test-next | Can small nonlinear private adaptations overcome a diagonal-factor limitation? | Zero start; linear residual, parameter-matched shared adapter, capable equipped M1/I4. |
| C5 | Test-next | Is a fixed task-relevant internal factor site more useful than boundaries alone? | Boundary GNNM, one predeclared interior site, factorized single and coupled untied control. |
| C6 | Test-next | Can TRAIN-learned conditional aggregation retain useful member alternatives that a fixed mean loses? | Same-bank mean, global weights/temperature, equally capable nonlinear concatenation gate. |
| C7 | Test-next | Does joint structural-pattern supervision improve native link ranking beyond independent bits or counts? | No pattern auxiliary, joint versus per-bit mixture, capable exact-count control. |
| C8 | Test-next | Are eval errors partly caused by sharing running BN statistics across distinct route distributions? | Original shared state; private route mean/variance/counters with shared affine; same-weight readout when prospectively possible. |
| D1 | Deferred | Does measured actual native shared-step interference justify an established gradient-conflict method? | Native Adam displacement diagnostic first; faithful PCGrad/CAGrad-family comparator, not a renamed projection. |
| D2 | Deferred | Does TRAIN-label retrieval add useful information beyond a stronger head? | Query exclusion, same-information equipped single/I4, detached/live retrieval; no VALID labels. |
| D3 | Deferred | Do learned private factor distributions add predictive utility beyond point factors or fixed noise? | Point, fixed noise, learned posterior; exact KL/draw/serving and equal compute. |
| D4 | Deferred | Would a different propagation/context/memory operator improve quality, and does sharing matter? | Same-operation native/equipped single and I4, support/parameter/view controls; separately labelled architecture programme. |
| D5 | Deferred | Is learning-response credit or TRAIN-guided initialization more useful than its matched paid updates/random disturbance? | Strong single/untied controls, matched work and initialization radius; no restart or selection grid. |
| D6 | Deferred | Can task-function diversity improve common errors rather than hidden coordinates alone? | Known functional-diversity prior, competence/proper-loss controls; valid intervention or output object required. |
| D7 | Deferred | What complete predictive task/target does a synthetic/source-only prototype actually solve? | Define provider, native comparator, source loss, selector and cost before allocating fits. |
| DIAG | Retained | Is an apparent representation/ensemble effect explained by head capacity, exposure, or postfit serving? | Own selected donors, unrestricted/affine head control, multi-view single, honest TRAIN-only fitted combiner. |
| X1 | Redundant | Generic member repulsion is already a known, tested ingredient; latent spread does not establish useful diversity. | Preserve it as a named ablation when scientifically needed, not a new umbrella candidate. |
| X2 | Redundant | Exact closed recipes retain their outcomes; no coefficient/site/normalization rescue is inferred. | New domains require a new explicit hypothesis; earlier negative evidence stays scoped. |
| X3 | Redundant | Aliases, execution-equivalent buckets and renamed half-pool credit are not new scientific methods. | Keep recipe differences when they change objective, information, selection or serving. |

## Why the cross-task versions differ

**All-task scientific candidates: C1–C6.** Each has a meaningful node, molecular
graph and link-query version in CATALOGUE.json. This is design compatibility,
not a claim that every native frontend is already implemented or qualified.
C7 currently has a defined link-pattern object and needs a new explicit node/
molecule target. C8 applies where the native architecture actually has tracked
BN: current OGB/BUDDY, not the inspected BN-free PolyFormer/NCN/NCNC. These limits
are distinct from ordinary implementation gaps. No expanded fit count is fixed.

For C1, nodes use observed class-compatible TRAIN pairs; molecules use pooled
graph vectors, binary class balancing, cross-route same-graph positives, a
two-class contrast gate and distinct-graph variance support. Links use the same
authenticated query across routes and endpoint/relation-conditioned native
alternatives, preserving sampled-negative meaning and source group reductions.
An absent molecular positive class is not repaired with instance negatives.
Different positive edges are not automatically interchangeable positives.

For C2, a candidate node assignment is fixed within-TRAIN-class degree quantiles;
molecules can use within-class graph-size/scaffold-context assignment; links can
use the prepared endpoint-correlated identity weights with a query-IID control.
All maintain positive ordinary task supervision and declared mass normalization.
These targets require independent task freezes. The same node code is not the
scientific definition of all three versions.

For C3, node context signatures, molecule TRAIN chemical-context signatures and
link endpoint/witness context are possible target objects. A teacher fingerprint
or observed structural target belongs to auxiliary TRAIN construction, not a
silent new inference input. Fixed-key, affinity-weighted and positive-exclusion
losses remain distinct variants; prepare one justified contrast before combining
them. No extra molecule augmentation is assumed label-preserving.

For C7, conditional joint observation patterns are directly meaningful for native
link completion. A molecular/ordinary node counterpart needs a new observed
pattern target and native-compatible reduction. It is not scientifically excluded
because only the link implementation currently exists, nor counted as a port
that is already defined.

For C8 see [BN_STATE.md](BN_STATE.md). Current PolyFormer and NCN/NCNC sources have
no inspected BN target; OGB and BUDDY do. Do not add BN to create compatibility.

## How the audit constrains the next work

[HISTORY_AUDIT.json](HISTORY_AUDIT.json) accounts for every record of the supplied
prior-result index. [V29_FAMILY_CROSSWALK.csv](V29_FAMILY_CROSSWALK.csv) maps every
canonical v29 family into this catalogue without importing its huge matrix.
The compact roster's many raw labels are not treated as extra mechanisms;
unmatched exact aliases are reported rather than guessed into a recipe.

Known closures include weakened members under private bootstrap/missing-hop,
failed normalization competence, redundant private pooled CE, Q/K/recurrent
decoder failures, and exchange failing its fixed gate. They motivate stronger
controls and diagnosis; they neither imply that all task-specific adaptations
fail nor justify silently repeating the same closed candidate. Engineering-only
shared-message/cost checks supply no predictive result.

[INTERACTIONS.md](INTERACTIONS.md) selects a few defined interactions. The first
is already the frozen MolBACE pilot. There is no all-family Cartesian product.
Missing target definitions, adapter integration, native state/replay qualification
and cost measurement remain separate fields from scientific merit. Source hashes
and precise reading limits are in [READ_SCOPE.json](READ_SCOPE.json).
