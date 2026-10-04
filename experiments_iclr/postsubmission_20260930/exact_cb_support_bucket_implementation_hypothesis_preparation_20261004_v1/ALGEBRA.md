# Exact conditional-Bernoulli support buckets: implementation hypothesis

This is training feasibility work. It proposes no new scientific objective,
tuning grid, inference route, methodological novelty or measured speedup.
Candidate code and QA have not been imported, compiled or executed.

## 1. Preserve real support and count exactly

For side/query q, keep its original ordered n_q logits eta_mqi and detached
binary TRAIN pattern z_qi. Derive k_q=sum_i z_qi and r_q=min(k_q,n_q-k_q).
The one shared teacher vector, side split, denominator max(n_L+n_R,1), member
mixtures, all-query means, positive/negative reduction and coefficient1 remain
the declared core's. Counts never enter native completion or link serving.

As in core-v2, subtract the first real logit differentiably and complement
logits and bits when k_q>n_q-k_q. Call the resulting real logits a_mqi and
bits b_qi. Then sum_i b_qi=r_q and

NLL_mq = log e_(r_q)(exp a_mq) - sum_i a_mqi b_qi.

r0 has one feasible pattern and uses an empty-slice connected zero without
gathering or summing large logits. A nonconstant query always has n_q>=2r_q.

## 2. Pack different lengths at an exact reduced degree

An active bucket contains only queries with one identical r. Its padded length
W is the next power of two for its member support lengths, so n_q<=W<2n_q.
Each query's real slots remain in positions0..n_q-1 in their original order.
Padding is a suffix. Gather padding from that same query's first real slot,
which exists because n_q>=2r>=2. Thus there is no out-of-range or cross-query
address; its finite centered value is0. Set padded observed bits to0 after
complementing the real teacher.

Use two arrays with distinct arithmetic roles:

- `centered` is finite (subject to the inherited finite-working-precision
  guard) and is used for the observed dot product. Padded bits are0, so this
  never evaluates0*(-inf).
- `weighted` equals `centered` on real positions and -inf on padding. A padded
  position therefore has ESP weight exp(-inf)=0.

Every size-r subset containing a padded position has zero weight. Every subset
of the original real support retains its original weight, exactly once. Hence
e_r(real weights plus zero weights)=e_r(real weights). No real slot/subset is
sampled, removed, relabeled, averaged with a dummy query or assigned epsilon.

For r1, masked logsumexp is the exact categorical/complement normalizer. Its
selected score is a real slot because the oriented real pattern contains one1.

## 3. Reachability and gradients

The inherited vector recurrence starts with e0=1 (log0), grows only reachable
counts, and caps its table at r. All first r positions are real in every query,
because n_q>=2r. Therefore every retained count0..r is finite before any query
reaches its padded suffix, apart from the already disclosed working-precision
overflow/accuracy limitations of finite real log-weight accumulation.

For a padding step and retained count j>0, inclusion is -inf+previous[j-1],
exclusion is previous[j]. The update is logaddexp(finite,-inf)=previous[j].
The newly reachable pure-inclusion branch is used only during the first r
real positions. No padding step forms logaddexp(-inf,-inf); the recurrence
does not add an unreachable count axis. Its forward table after all W slots
is the table after the query's real n_q slots.

In real arithmetic, padded-weight derivatives are0 and real-logit derivatives
remain the original inclusion probabilities. masked_fill blocks the repeated
anchor gather's padded derivatives. The finite observed dot product has
padded bits0, so it also contributes no padded derivative. Differentiable
centering still removes the common-offset direction; complement still maps
back to pi-z on the original real logits. No neural input is detached.

J_K, J_K_sep, W_K and rho use the exact same mixture arithmetic as core-v2.
Thus their logit gradients keep the same joint, side-specific or uniform
responsibility weights, divided by the same denominator and external all-query
mean. All empty/extreme-query losses and gradients remain connected zeros.

These are algebraic statements. Actual PyTorch float32/float64 behavior,
padding backward, reordered reductions, source integration, native gradients
and inclusive runtime/memory require separately admitted qualification. There
is no arbitrary-finite-input or large-support floating-point guarantee.

## 4. Reduce recurrence dispatch with a deterministic fallback

First form the existing unique integer(n,r) descriptors. Plan using that small
detached descriptor list; there is no per-query .item count. Group descriptors
by(r,next-power-of-two(n)). Merge an r>=2 bucket only when W is strictly less
than the sum of its distinct original n values. Otherwise retain its direct
(n,r) groups with no padding. r1 merges multiple supports within one bucket;
it has no recurrence slot loop. r0 descriptors share one zero group.

Let L_direct=sum_(distinct n,r>=2) n. The new loop count is the sum of one W
per merged genuine bucket plus n per retained direct genuine group. Every
merge strictly lowers that sum; every fallback preserves it. Therefore
L_bucket<=L_direct for all valid descriptor sets. Genuine/categorical group
count does not increase. This is a Python recurrence-loop bound, not a bound
on elapsed time, CUDA launches in the complete adapter or total native work.

For a fixed r with several populated power-of-two buckets, sum W<2W_max,
where W_max<2n_max. Dense distinct lengths can therefore collapse many slot
loops. An illustrative exact-r bucket with n33..64 uses64 loops rather than
1552 (24.25x fewer recurrence Python iterations). It is fabricated arithmetic,
not an observed native histogram or timing result. Sparse supports fall back
when padding would consume as many or more loops.

## 5. Cost and feasibility limits

Each merged packed array has sum_q W<2sum_q n_q. Its leading M*g*W*r work
scale is less than twice the corresponding M*sum_q n_q*r scale. This does not
assert a2x bound on exact reachable-cell updates, saved tensors, allocated or
reserved memory, peaks, kernels or wall time. A merged group's peak can greatly
exceed any one old direct group's peak because it combines their queries.

Extra costs include a descriptor plan, a small codebook host-to-device transfer,
mask/address arrays, masked weights and padded recurrence arithmetic. unique,
sort, gather/scatter, group finite-check synchronizations and O(Q) diagnostic
transfers remain. Autograd retains recurrence inputs/states plus native neural
graphs. Fewer iterations can still run slower or exceed a memory cap.

Root reported121k-140k direct recurrence slot loops per positive full batch.
That report motivates this source hypothesis but supplies no exact-r/support
histogram here. The retained compact census describes support sparsity and
separate per-side maxima; it cannot determine this bucket plan's native loop
count or peak. No server, histogram array, data/state/score payload or native
runtime was accessed. No numeric speedup or new full-batch feasibility claim
is made. A future exact histogram projection and inclusive full65536 native
forward/backward gate must resolve these costs without changing precision,
batch, degree, splitsize, epoch, schedule or auxiliary law to fit a cap.

All native neural calls remain mandatory, including negative populations and
teacher r0. This hypothesis changes auxiliary normalization packing only.
