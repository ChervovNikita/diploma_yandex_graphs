# Norm-matched SYNC: planned direction-versus-global-magnitude ablation

This design was fixed before the 96-cell SYNC mechanism study had any test
scores. It is a diagnostic adaptation of separate Adam moments, not a new
optimizer claim. AdaTask already uses task-specific Adam moments; gradient
normalization and update balancing have prior work. The question here is
whether changing only the *global length* of SYNC's graph-parameter update
can account for differences from TIED under this one recipe.

## Complete fixed matrix

Run exactly 12 fresh cells: Cora, WikiCS, Actor, official filtered
Chameleon, seeds 0, 1, and 2. Use the same frozen split-0 graph tensors,
width-128 two-block SAGE model, four boundary-projector members, LayerNorm,
dropout 0.2, full-batch propagation, 1,000 finite epochs, and default
AdamW learning rate 0.001, betas (0.9, 0.999), epsilon 1e-8, zero weight
decay. The loss is the mean of four member training cross-entropies.
At every forward, all four graph stacks have exactly equal weights. Their
AdamW moments stay separate. Shared boundary and final-normalization
parameters receive one AdamW update from the ordinary mean-loss gradient.

## Per-step graph update

At current common graph weights `theta_t`, mean-loss backward gives private
graph-copy gradient `g_m/4`. Multiply only each private graph gradient by
four so its AdamW state sees the full member gradient `g_m`. The private
AdamW optimizer makes four provisional graph updates. Average their graph
weights, yielding candidate update `d_S` relative to `theta_t`.

In parallel, maintain one *virtual reference* AdamW first/second-moment
state fed the current mean graph gradient `mean(g_m)`. Before its step, copy
`theta_t` into its dummy graph parameters. Its proposed update `d_R` is
computed on the current NORM-SYNC trajectory, without changing the real
model. The reference moment state persists across epochs. It is therefore
not the actual separately trained TIED arm's history.

If both update norms are positive and finite, use
`theta_{t+1} = theta_t + d_S * (||d_R||_2 / ||d_S||_2)`, then copy this graph
weight vector into all four stacks. This preserves the candidate direction
and matches the reference's one global graph-update L2 norm. If
`||d_R||_2 = 0`, set the graph update to zero. If `||d_S||_2 = 0` while
`||d_R||_2 > 0`, or either norm is nonfinite, mark the cell invalid rather
than inventing a direction or substituting a fallback. Require the applied
graph-update norm to match `||d_R||_2` within relative tolerance 1e-5
(with absolute floor 1e-8) every finite step. Decoupled decay is zero in
this matrix, so the norm comparison concerns the full graph update without
a decay term.

## Validation and test firewall

Before any training, freeze the exact source, this protocol, source/data
manifests, 12 cells, and numeric gates. An independent source review and
CPU/CUDA preflight must pass. Preflight should check matched canonical
initialization/RNG, equal graph weights, separate member moments, a
persisting virtual mean-gradient moment state, and the norm equality after
several steps. Keep a 1,000-row validation trace for every finite cell,
including validation accuracy, validation CE, selected flag, reference and
candidate graph-update norms, applied norm, and norm error. Select one
whole-model checkpoint by highest pooled validation accuracy, then lowest
pooled validation CE, then earliest epoch. Check that its four graph stacks
collapse to one TIED inference stack with matching member logits and class
decisions. Audit all 12 cells, traces, checkpoints, source/data hashes,
initialization, and norm gates before writing a validation-only lock.

Test inference is permitted only after that complete 12-cell lock, the
separate complete 96-cell SYNC lock, and the separate complete 432-cell
selection lock match their source/data freezes. Score all 12 cells exactly
once, then independently replay and audit those scores. Report all graphs
and seeds, paired with the already frozen default TIED and SYNC arms. No
candidate search is used. The complete audit cutoff is 06:00 UTC on
26 September 2026.

This controls only a *single global update norm*. It does not equalize
coordinate-wise updates, accumulated moment histories, boundary-projector
trajectories, model capacity, or later graph weights. It cannot establish a
general selection rule across unseen graphs. Three seeds on one split are
optimizer replicates, not independent graph draws.
