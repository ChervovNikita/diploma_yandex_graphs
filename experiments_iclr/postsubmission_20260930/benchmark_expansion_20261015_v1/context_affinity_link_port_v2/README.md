# Factual context affinity for native NCN

This fixed port compares a common factual-context target with a split target
on the actual NCN / ogbl-collab source model. It is a task port of the shared
context objective; it makes no novelty, convergence or predictive claim.

Every update keeps all 65,536 observed positive and 65,536 source-sampled
negative queries, the native negative sampler, minibatch graph mask, four
native route forwards and positive-mean-plus-negative-mean task reduction.
Only its scalar loss gains the predeclared auxiliary `0.05 A`. No source
model class, Parameter, optimizer group, predictor operator or serving rule
changes. The final prediction remains the mean of four raw native logits.

The auxiliary captures the original 64-wide prehead before the final boundary
input factors. It admits up to 512 distinct nonself, unambiguous semantic
queries per source group from those already in the update, selected by the
frozen link hash law. Conflicts and same-group duplicate handling are defined
in LINK_SUPPORT_ADDENDUM.md. All omitted AUX rows remain in F.

Factual targets use a cached simple undirected loopless CSR graph from all
native TRAIN-positive records. CSR construction sorts and deduplicates signed
int64 coordinates directly. It uses no dense adjacency, count accumulator or
unbounded query cache. For each selected query, A is its sorted endpoint
log-degrees after deleting only that pair; B is the log1p common-neighbor,
Adamic-Adar and resource-allocation vector from that same graph. Distinct
witness degrees do not change when the queried pair is removed. The full
TRAIN target graph is auxiliary supervision and differs from the predictor's
unchanged whole-minibatch edge mask.

Targets are detached donor softmaxes of context cosines at temperature 0.2,
over distinct objects sharing the observed source group and native relation.
Common gives every route the average A/B target. Split assigns A to routes
0/1 and B to routes 2/3. Both sum the two source-group means and then average
the four routes, matching NCN's separate-positive/negative convention.
Zero context gives a uniform target. Unsupported anchors are reported and
omitted only from AUX. No fallback target or sampled query is invented.

The wrapper observes actual source calls and authenticates the complete
four-route query order. Checkpoints include semantic cohorts, actual bounded
target tensors, target statistics, source hashes and the original native
sampler/RNG/optimizer/epoch state. Restore reconstructs the saved contexts from
the saved native batch and immutable TRAIN graph before loading that state.

Engineering qualification is explicitly finite: common and split run
sequentially on CPU with two native updates each, full official VALID before
and after a fresh checkpoint restore, then a resumed update. A passive
retained-graph AUX gradient collection on each update is counted separately.
It must show active shared/input-factor gradients and zero direct final-bank
AUX gradient. Four total optimizer transitions, four extra gradient
collections, six full VALID calls and four native constructions are the
maximum. No TEST, GPU fit, convergence or accuracy evidence is authorized by
these checks. CUDA pair admission, full-worker integration and final TEST
reconstruction remain separate gates. NCNC and BUDDY are unsupported by this
specific port.

## Device-safe checkpoint repair v2

The numerical target law, selection, update and inference are identical to v1.
The only execution repair explicitly gathers each endpoint tensor with an index
on that tensor's own device, then authenticates both on CPU. Method/state
identities differ because this is an immutable source generation. The v1 CPU
qualification is preserved; this generation needs its own declared device gate.

The CPU procedure above describes the preserved v1 lineage, not a v2 device
qualification. The declared v2 common CUDA gate allows three actual transitions
(step1, uninterrupted step2, fresh replay step2), four whole VALID calls and two
native constructions. It compares original/Adam/sampler/global-and-private-RNG
states and outputs exactly. Split, NCNC/BUDDY, pair admission and full scientific
worker remain separate gates. No third simultaneous own GPU worker is allowed.
