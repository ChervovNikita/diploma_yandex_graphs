# Two recent sources: useful prediction alternatives and belief heads

8 October 2026. This is a bounded primary-source review, not a new experiment,
author-code reproduction, manuscript review or novelty certificate. The running
graph-relation and molecular protocols are unchanged. Original scores are unchanged.

## Prediction complementarity in link prediction

**Méroué, Gandon and Monnin, _Oracle, will I ever learn? A study of prediction
convergence and complementarity across link prediction models_,
arXiv:2609.02638v1, submitted 2 September 2026.** No accepted venue was verified.

The inspected Section 3 defines an instance by architecture, configuration and
training seed. Its oracle takes the best ground-truth entity rank among a group
for each query. This measures the potential of selecting an existing member's
ranking. Section 4's inspected narrative describes knowledge-graph completion,
filtered candidate ranks, pessimistic ties, independently trained seed groups,
and validation configuration selection. It also discloses exclusion of runs
below validation MRR 0.05. Its numerical result tables, code and complete
configuration files were not audited or adopted here.

The oracle is an upper bound **for choosing one existing ranking per query**.
It is not a universal ceiling for score fusion or a new predictor learned from
member representations. For example, two class-probability vectors, ordered as
truth/a/b, can be `(0.4, 0.5, 0.1)` and `(0.4, 0.1, 0.5)`. Both rank truth second;
their average `(0.4, 0.3, 0.3)` ranks truth first. This is an elementary
counterexample supplied in this note, not a trained-model result or a new theorem.

That distinction supports the current GNNM diagnosis. Its stronger convex-pool
barrier comes from a **common rival that outranks truth in every member**, not
merely from every member being wrong. With that common rival, any convex mixture
of the same probabilities remains wrong. Existing collected Wiki15 summaries
already measure common rivals and actual pool-only rescues; no predictions were
reloaded or scores recalculated for this source review.

**Decision:** retain this source as recent ancestry for prediction-level
complementarity analysis. Preserve run failures rather than copy the source's
exclusion rule into our comparisons. Knowledge-graph ranking is a different task
from untyped Collab link prediction and WikiCS node classification. No new GNNM
method, dataset admission, oracle-based model selection or empirical claim follows.

## Random-set graph prediction heads

**Woodley, Manchingal, Tolloso, Bacciu and Cuzzolin, _Random-Set Graph Neural
Networks_, arXiv:2605.11987v1, submitted 12 May 2026.** No accepted venue was verified.

The inspected Section 3 retains a graph backbone and predicts node-level beliefs
over subsets of classes. A Möbius transform yields masses; a pignistic transform
yields class scores. Belief-space binary cross-entropy supervises which subsets
contain the true class. Penalties discourage negative masses and a mass sum
different from one. The proposed uncertainty signals are pignistic entropy and
singleton interval width. The full power set is the default; an optional focal
budget is described for larger class spaces.

This is a different prediction-head and uncertainty mechanism. It does not
provide four independently supervised graph-evidence routes or demonstrate
superiority of a shared-backbone ensemble. Its stated soft penalties do not by
themselves prove exact nonnegativity and normalization at every served node.
Author implementation and optional-budget inversion semantics remain unverified.
The experimental tables and accuracy/OOD claims were not adopted here.

**Decision:** save the mechanism as attributed uncertainty-head prior. Do not
replace the current accuracy hypothesis with belief-head complexity or count
this as a demonstrated remedy for common GNNM errors. A future uncertainty claim
would require a separately frozen task, valid serving distributions and capable
uncertainty baselines. No implementation or pilot is admitted by this review.

## Scope and custody

Three recent-work metadata searches found these two sources among numerous
irrelevant applications and already indexed methods. Metadata does not establish
a method or priority. Both exact versioned HTML bodies and arXiv identity records
were acquired. The inspected passages are identified in `READ_SCOPES.json` and
the compact method excerpts. Two new bounded source scopes, zero full-paper
reads, zero author-code/proof audits and zero adopted numerical results are
credited. Uninspected sections in acquired bodies receive no reading credit.

The useful next scientific action remains the complete registered graph-relation
and molecular comparisons, followed by their required controls if the fixed
candidate succeeds. These papers supply interpretation and prior-work boundaries;
they do not change a running recipe or authorize a rescue grid.
