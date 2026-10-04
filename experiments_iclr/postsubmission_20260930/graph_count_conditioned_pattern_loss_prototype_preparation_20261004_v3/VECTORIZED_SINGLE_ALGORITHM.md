# Exact teacher-forced S_K batching

This is a disabled source implementation successor of grouped core-v2. It
changes neither the law/head nor J_K/J_K_sep/W_K. Exact means real-arithmetic
law and differentiable feature identities; floating operation order differs
and remains subject to the existing fixed tolerances. No numerical code was
imported, compiled or executed during preparation.

## Visible context

For side s, sorted native query rows give counts n_s and segment starts a_s by
integer bincount/cumsum. Every native unary slot i is centered by
`unary[i] - unary[a_s[rows[i]]]`. This is the same label-free first-slot anchor
as the sequential context builder, without one Python slice per query. Endpoint
orientation, ascending unique candidate order, own/other/candidate h64, visible
support means, endpoint product and log1p support features remain unchanged.

## Teacher-forced prefixes

Let z_(q,s,i) be the detached shared TRAIN bit, and x_(q,s,i) its candidate h64.
Before slot i, the selected count and feature sum are

- c_(q,s,i) = sum_(j<i) z_(q,s,j)
- v_(q,s,i) = sum_(j<i) z_(q,s,j) x_(q,s,j)
- k_(q,s) = sum_j z_(q,s,j)

Integer counts use an exclusive int64 global cumulative sum minus the exact
segment-start integer value. Float candidate sums never subtract global totals
of unrelated queries. Nonempty queries are grouped by visible support length n,
real addresses are gathered as [g,n] without padding, and weighted candidate
features have a local `cumsum(dim=1)`. A leading zero and shifted cumulative
values implement the exclusive prefix, without subtracting the current slot.
A single differentiable index_copy at unique original addresses restores native
slot order; a separate unique query index_copy restores complete side summaries.
There is no Python query or slot loop, no per-query/slot `.item`, and one dispatch
per distinct visible support size. Both metadata .tolist transfers per side,
grouping, gather/scatter, cumulative values and retained graphs remain charged.

Before a left slot, selected=(c_L,0), remaining=(k_L-c_L,k_R),
unseen=(n_L-i,n_R), and prefix=(v_L/max(c_L,1),0). Before a right slot,
selected=(k_L,c_R), remaining=(0,k_R-c_R), unseen=(0,n_R-i), and
prefix=(V_L/max(k_L,1),v_R/max(c_R,1)), where V_L is the completed left sum.
These equal the sequential state immediately before the same slot. Current or
future identity bits cannot enter the selected prefixes. Future membership
enters only through the original permitted count budgets. The visible context
builder has no label/count arguments; teacher forcing stays training-only.

## Unchanged head and law

All original slot features, their query context, the128 prefix features and8
progress ratios are assembled in left-then-right flat slot order. One call to
the unchanged Linear523->64/ReLU/Linear64->1 head evaluates all real rows,
including forced choices. A zero-row call is also retained for all-empty support.
The initialization constructor and33601parameter count are AST-identical v2.
The no-dropout/no-normalization head has no dependence between batched rows, so
this evaluates the same mathematical logits as independent scalar head calls.

For active remaining budget b and unseen t, a choice is forced precisely when
b=0 or b=t; its bit must be the unique legal choice. Its term is `score*0`,
including graph connections. Otherwise the term is `-logsigmoid((2*z-1)*score)`.
Segment sums on each side add these terms for every query and divide by
max(n_L+n_R,1), retaining empty and extreme queries and the caller's all-query
mean. Parameter empty-slice sums and visible-slot empty-slice sums preserve the
original connected zero convention. There is no probability clipping, slot
omission, support truncation, parameter addition or serving/count inference API.

By induction over the canonical left-then-right order, the vectorized features
equal each sequential pre-choice state. Consequently each legal conditional
Bernoulli choice/forced probability, normalized product law and exact
real-arithmetic head/h/unary derivatives equal core-v2. Gather/cumsum/index_copy
and segment sum remain differentiable in candidate/context tensors; detached
bits/counts/indexing carry no trainable path. Floating summation, mean ratios,
GEMM and reduction order are different; source algebra supplies no universal
finite-precision or bitwise equality guarantee.

## Unexecuted assurance and resources

Inherited v1 and v2 law/ragged QA sources are byte-identical. One future freshly
admitted CPU attempt runs those then the new supplement, under unchanged900s/
4GiB caps and float32(atol=rtol=1.52587890625e-5)/float64(1e-10,1e-9) tolerances.
The supplement compares exact sealed core-v2 NLL and per-query/all-query mean
gradients for every head parameter,h,both unaries, requiring nonzero regular
fixtures. It reconstructs every past-only feature via explicit selected sets
and all affine/ReLU/BCE head derivatives, and verifies one all-row head call,
forced/empty exact zeros. New exhaustive CB references include right/both
r>=2 and mixed categorical/genuine sides. No predecessor run is repeated.

The head MAC count remains total_slots*(523*64+64). Explicit batched head input
is total_slots*523*dtype_bytes, hidden output is total_slots*64*dtype_bytes,
prefix buffers include two128/64-wide side arrays and grouped gathers/cumulative
values, and saved tensors/native graphs add costs. There is no claimed measured
peak or resource bound from these formulas. The actual scoped TRAIN census shows
substantial positive slot/group work; this source is not full65536 feasible or
native-gradient/state/optimizer/serving qualified. Exact core/physical supervisor,
root/runtime admission and inclusive native resource qualification remain gates.
