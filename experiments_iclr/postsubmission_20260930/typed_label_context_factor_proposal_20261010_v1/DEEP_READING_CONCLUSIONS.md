# Two deeper primary readings and their limits

## Rank-1 BNN:arXiv2005.07186

Read the complete main method (§3.1–3.6), background definitions used by that
method, the full AppendixA variance argument, AppendixB implementation/experimental
design, AppendixC initialization/scale/sampling ablations, AppendixD limitations
and future work, and AppendixE loss definitions/Jensen distinction. Exact retained
block ranges are in READ_SCOPES.json. Results-section numerical tables, figure
pixels, all references and author model code were not audited; this is a deep
bounded reading, not a full-paper or benchmark-reproduction claim.

The paper treats common W as deterministic and r/s as variational distributions.
It learns mixture components jointly and evaluates marginal class probabilities.
Expected own-component NLL differs from NLL of the mixture; mixture training
couples component credit. Its experiments use real-valued Gaussian/Cauchy factor
families, four components, sampling/initialization/prior choices and KL annealing.
Those image/EHR choices are not a qualified GNN recipe. Deterministic initial
scales, heavy-tail instability and inference sample costs matter. Its public
claims of improved uncertainty/diversity are not project evidence.

The AppendixA argument matches local second-order expected score fluctuations
for a fully connected network under a specific multiplicative covariance form.
It does not show universal equality with arbitrary full-weight posterior laws,
calibration, finite-step competence, graph-dependent factor behavior or useful
ensemble disagreement. Printed covariance indexing and some derivative notation
are not silently repaired into a new graph theorem. The structural chain-rule
idea was inspected; the proposal imports **no variance guarantee**.

Its useful design lesson here is compact multiplicative conditional paths and
separate member competence. The first proposal does not use its Bayesian
posterior or claim its theorem for node-dependent factors. Ordinary supervised
member losses make the simplest graph mechanism test interpretable.

## MPNP:arXiv2009.13895

Read the complete main method, its background/related-method distinctions,
experiments/baseline rationale and discussion, AppendixA generative description,
the full encoder permutation argument and ELBO derivation, Cora source/task
definition, common model/optimizer details, Cora design and the arbitrary-label
chance proof. Algorithm/figure pixels, every other task's appendix, full numeric
tables and author implementation were not audited. No full-paper claim is made.

The model receives labels for a context subset, includes graph neighborhoods in
the context encoder and target decoder, and aggregates to a global Gaussian
latent. The training posterior sees context and target TRAIN labels, while the
serving distribution sees context only. Its output convention uses Gaussian
likelihood over softmax/one-hot labels; this is not ordinary Bernoulli/categorical
classification likelihood. A categorical or multilabel port must be declared.

The permutation proof establishes equivariance of node representations under
reindexing and, after invariant aggregation, invariance of the context summary.
The text calls intermediate node transforms invariant; the substantive distinction
is important. The ELBO derivation uses a learned approximation to the context
prior; the standard lower bound is valid for a declared conditional generative
model, but it is not proof of a context-consistent exact Bayesian process or
calibrated predictions on one graph. The arbitrary-label chance proof concerns
averaging uniformly over label permutations without context. It does not make
the ordinary fixed-label native GNN weak by construction.

The Cora experiment is based on CitationFull/Cora-Branched with70 topics and
specific discipline/class partitions. It is not the familiar small Cora split.
Context/target rates10–50%, a two-step MP architecture, class-aware aggregation,
and task-specific training are empirical design details, not portable native
hyperparameters. The source uses one stated torch seed in the experiments;
repeated matched seeds/splits are needed in our screen.

The paper explicitly notes ordinary GNN advantages on some fixed-label tasks
and label-propagation advantages at high context density. Its principal strength
in arbitrary-label, variable-rule and few-shot tasks need not transfer to one
fixed-label transductive IMDB graph. The proposal therefore changes the model's
conditional evidence transparently, uses query-local typed fields rather than
claiming a new global latent process, and requires capable same-information
single/additive controls. No artificial relabeling or induced-class task is used
to make a context-free baseline lose.

## Revised triage

The earlier scout's statement that no new principle was established remains true.
It no longer serves as a reason to reject a useful graph-specific extension.
The present proposal isolates a concrete architecture/input hypothesis and
preserves its direct priors. Whether it improves competent predictions, depends
on local graph evidence and benefits from ensembling is left to the complete
predeclared experiment. No manuscript acceptance verdict follows from reading.
