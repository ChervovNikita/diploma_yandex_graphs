# Prospective link support clarification

This addendum fixes the interpretation before native execution. It does not
change the already pinned cross-task target specification.

The factual B vector is `[log1p(CN), log1p(AA_sum), log1p(RA_sum)]`.
All three coordinates use `log1p`. The AA and RA sums use the degrees of
distinct common witnesses after deletion of the current queried pair. For a
witness distinct from both endpoints, deleting that pair leaves its degree
unchanged. Only the endpoint degrees can decrease.

The target graph is a cached simple undirected loopless graph made from all
native TRAIN-positive records. Delete only the currently queried pair for its
target calculation, equally for observed positives and sampled negatives.
This is auxiliary supervision. The predictor still uses its original
whole-minibatch edge mask, native operators and sampler.

Semantic identity is `(min(u,v), max(u,v), native_relation)`. Exclude self
queries from AUX, avoiding an invented AA convention for a degree-one
self-query witness. If an identity occurs in both source groups, exclude that
conflicting identity from AUX. Within one group keep the earliest native row
for a duplicate identity. All rows remain in the original native task loss F.

Rank remaining unique rows separately within each source group using the
existing frozen link law `(_hash(seed, 'anchor', semantic_key), native_row)`.
Keep up to 512 per group and restore source row order. No fill queries or
alternative targets are introduced. Report omissions, zero contexts, donor
support and all actual work. No target, coefficient or source-group choice
depends on a validation result.
