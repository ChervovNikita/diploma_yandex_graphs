# Private CMCL credit on a live PolyFormer core

**Disabled representative source.** No provider/model/tensor/data import or call,
fixture, fit, server action, teacher acquisition, active-source edit or launch was
performed. All capabilities are false. The first required next action is a fresh
source review, then a separately admitted complete-input numerical qualification.

## Learning mechanism

All four routes see the same complete PubMed graph and the same fixed native K2
monomial channels. The exact published mono PolyFormer has two 256-wide layers,
eight heads, native token MLPs/FFNs and three categorical logits. The existing
23-site BE adapter retains every original weight and supplies private factor rows.
There are no feature masks, extra label fields, reconstruction decoder, CORE loss,
teacher or initial fitted router. The failed masked study is not combined here.

The shared core receives the ordinary mean of four complete own cross entropies.
Each private factor row additionally receives Confident Multiple Choice Learning
credit: K3 owner CE and nonowner beta=.75 KL(uniform || predictive), with lambda=1.
Every member retains all TRAIN supervision through the own-loss floor.

For each incoming full TRAIN pass, owner ranking is `CE - beta*KL(U || P)`.
That is the complete CMCL assignment cost up to a common constant, not minimum CE
alone. Ownership is stopped, with stable member-index ties. The CMCL component is
summed over members and averaged over nodes, following the saved published overlap
objective; the own floor retains its 1/M normalization. No silent M/K division is
added. K3/beta .75 come from the earlier authenticated CMCL close design. Lambda 1
is an untested design constant, fixed equally for the matched controls.

The current shared **data gradient** excludes extra CMCL credit. Adam moments and
weight decay remain live. Private updates change later shared own gradients through
changed predictions; this is not a claim of identical shared trajectories or a
descent guarantee. The mixed field generally is not the gradient of one scalar
loss over all parameters. Block-selective backward itself is established prior.

## Exact controls and the gradient-scale issue

| Condition | Shared data gradient | Private factor data gradient |
| --- | --- | --- |
| own_floor | E | E |
| private_cmcl | E | E+C |
| all_block_cmcl | E+C | E+C |
| vanilla_cmcl | C | C |
| private_uniform | E | E plus mass-matched uniform penalty |
| private_constant_credit | E | E plus constant K/M CE and beta(M−K)/M KL credit |

E is mean own CE; C is the exact member-summed CMCL overlap component. The
constant-credit control keeps the owner/nonowner aggregate gradient mass while
removing state-dependent assignment. Otherwise an improvement could be explained
by stronger private CE updates or confidence regularization. These are fixed
attribution controls, not a coefficient/hypothesis search. Three fixed optimizer
seeds 9101/9203/9307 share the already declared split seed 190111.

Vanilla CMCL reproduces the attributed objective on this shared graph model, not
the original independent image architectures or their optimizer. Private rows are
not independently initialized bodies. Proper native single and genuinely
independently initialized, individually selected I4 references remain mandatory
external comparisons. Existing completed scores are not recalculated or admitted
as matched references by this packet.

## Source and representative training

Native class ASTs and the literal mono preprocessing helper are hash-bound from
the saved author commit. Author module imports/global seed42 and training/TEST
entry points are omitted; complete class/function bodies remain unchanged. BE
factors use unit initialization, bias outside output scaling, and every affine
site. No model reset follows installation. The symbolic native count is 2,069,875
parameters; shared BE4 adds 55,772 factors, totaling 2,125,647. These are symbolic
source counts, unmeasured.

The public feature/edge provider has separate TRAIN and VALID bundles and no full-y
or TEST file interface. Its full 19717x500 graph is preprocessed once without labels.
The fixed class-stratified floor60/20/20 split is a disclosed amendment: the author
launcher uses equal-per-class training budgets and a global validation count.
Source shape checks do not qualify normalization, graph identity or role custody.

CMCL scoring first streams four complete native TRAIN-mode forwards without tapes.
The gradient pass replays those same incoming parameters and member-owned dropout
draws, with explicit finite FP32 drift tolerance. Each replay then frees its tape.
Private-only conditions compute own gradients on all parameters and extra VJPs only
on private factors. Every extra VJP must leave existing grad fields intact and have
zero entries in other members' factor rows. One deduplicated native Adam commit
follows all four members. No optimizer step occurs during assignment or replay.

All original optimizer groups are retained: attnmodule lr .0005/wd 1e-8 and remaining
parameters lr .005/wd .001, native Adam defaults. Training is FP32, not a new AMP
recipe. No graph, width, population, classes, TRAIN targets or horizon can shrink.
fit.py implements max 2000 epochs and 250 consecutive VALID nonimprovements. Strict
complete pooled VALID accuracy selects the earliest tie. MacroF1, NLL and member
competence are reported at that one checkpoint; they do not shop for new epochs.
The selector starts below zero so the first finite checkpoint always exists.

Serving computes four complete full-graph predictions and averages member
softmax probabilities; stable logsumexp gives pool NLL. This primary-consistent
choice is frozen across all controls and differs from typed-context's new raw-logit
rule. Fresh construction restores exact native weights, factors, buffers, Adam
and global/member RNG; complete outputs require drift<=.001 and no prediction
changes, without bitwise floating-output equality. State/role/RNG guards stay exact.

CMCL conditions cost eight full-graph TRAIN forwards per update, versus four for
own/uniform/constant controls. Private-only credit uses two VJPs per member; all-block
credit uses one combined VJP. Scores, replay, extra gradients, preparation, complete
validation, snapshot, serialization and selected reconstruction must count. Parameter
sharing alone does not establish latency, memory or training-time superiority.

## Competence and common-rival criteria

The completed PubMed error synthesis indicates common false rivals: on the important
shared failures, every member places the same false class above truth. Any unchanged
nonnegative convex probability pool—and positive member temperature scaling—keeps
that strict rival above truth. Confidence reduction alone is insufficient. The new
learning rule must create useful alternative competence and clear that ordering.

diagnostics.py compares a root-bound frozen baseline VALID population and its fixed
rival identities. It reports any-member rival clearance, any-member correct
coverage, actual pool repairs/new errors and net repairs. Current common-rival
counts, overall accuracy, macroF1, class risk, NLL and own member competence remain
separate metrics. Clearing one rival is not necessarily a correct prediction.
Neither entropy, geometry nor agreement is a success surrogate. These diagnostic
labels never enter the assignment/training API.

SOURCE_COLLISION_ASSESSMENT.md and CANONICAL_ALIAS_CORRECTION.json preserve the
earlier CMCL/block-learning ancestry and correct the October10 reread credit.
No exact full published equality is established in the bounded saved scopes;
no novelty/absence, utility, calibration, causal or acceptance claim is made.

## Handoff API

`Caps()` is closed. A separate root-reviewed successor must bind the exact source
seal, review, qualified input custody and runtime before opening capabilities.
There is no numerical entry point or launcher in this packet.

1. Supply original full public `x`/`edges` and isolated `Role(ids, targets,
   custody_sha256)` bundles. `prepare(...)` performs the literal native full-graph
   monomial preprocessing and produces the `Inputs` interface.
2. Construct `Costs` in a fresh phase-owned output folder and `Session(condition,
   seed, inputs, device, costs, identity, caps)`. Identity contains the source seal,
   root review and actual feature/edge, TRAIN and VALID custody digests.
3. `fit(..., numerical_qualification=True)` permits one complete update under
   engineering authority; `fit(...)` requires separate scientific authority and
   the complete declared horizon. Qualification outcomes are not paper evidence.
4. Release the fitted session, then use `load_and_reconstruct(...)` for fresh
   selected-state reconstruction. Log all output drift, state/RNG guards and costs.
5. `compare_fixed_common_rival(...)` consumes separately prebound baseline and
   candidate VALID outputs for diagnosis only. No diagnostic labels feed learning.

Native single, four-loss/dropout single and genuinely independent, individually
selected I4 references need compatible full PubMed inputs, split and native recipe.
A four-loss single advantage can reflect established multi-sample dropout or
gradient averaging; it is not ensemble novelty. Source-level parameter sharing,
coverage or common-rival clearance cannot establish pooled predictive improvement.

`python3 -B verify_static.py` runs only stdlib AST/source/JSON/hash checks and writes
`STATIC_CHECKS.json`. It never imports candidate modules or invokes model/data code.
