# Inactive CoGNN-style external-message prototype

2026-10-07. Source readiness only: no adoption, training, model import, data or
scientific-server access. No novelty, performance or efficiency claim.

## Files and status

- `prototype.py`: numerical API behind `RUNTIME_ENABLED=False`. It accepts an
  already constructed native model and a separate policy bank; no native class,
  parameter, constructor or factor installation is changed.
- `runtime_fixture.py`: bounded disabled CPU qualification source. Its first
  runtime call raises before Torch/model imports. No loader, runner or enable CLI.
- `static_checks.py`: the only executed code; stdlib AST/compile/hash checks.
  These checks do not establish numerical identity or gradients.

## Native path

The retained PyG2.7 GAT SHA is
`a7b2353003394ab433f909c0dbd7e5a06b7ae1289939857245ae3062f1eb5901`.
Supported scope is homogeneous `[N,D]` states, plain long `[2,E]` COO support,
source-to-target additive GAT, no edge attributes, prepared loops and
`add_self_loops=False`. Sparse/bipartite/EdgeIndex, explain/decomposed execution,
custom GAT subclasses and GAT/top-level forward hooks are rejected.

The function calls the original `lin`, attention parameters and `edge_updater`.
Original scores, softmax and attention dropout use unchanged support. It then
multiplies alpha by a new live `receive[v]*send[u]` tensor on nonself edges and
constant1 on self edges, and calls original `propagate(x,alpha,size)`. No custom
PyG kwargs, edge deletion/coalescing, second softmax, log gate or gate cache is
used. An off sender still participates in the original attention denominator.

The complete functional Polynormer entry retains the original dense/attention
parameter objects, stem/dropout order, local own/root `lins(x)` route, optional
pre-LN, nonlinearities, beta mixtures, normalizations, local accumulator, global
attention and prediction heads. All-keep bypasses policy/noise evaluation and
preserves native dropout draws. Numerical parity is pending; no bitwise claim.

## Parameters, graphs and RNG

Policies stay outside the native factor traversal. Install existing factors on
the native model first, then construct the separate policy bank. Keep native
common/factor parameters and private policy parameters disjoint; the prototype
does not implement loss routing, checkpoint selection or a training loop.

One private receive/send MeanGNN-style pair plus learned linear temperature is
reused across local depth per member. The policy reads `[state,mean_original_neighbors]`;
temperature is `1/(softplus(linear(state))+tau0)` with required positive `tau0`.
Policy initialization and fixture temperature are scaffolding, not a scientific
recipe. The factory isolates CPU constructor draws and restores the live CPU RNG
in `finally`; it does not reset native modules or seed CUDA.

Each call receives explicit member/view and constant Gumbels indexed by
`(member,view,layer)`. It returns `ForwardRecord(logits,gates)`; use `.logits` where
the native bare-logit return was expected. The native forward API is untouched.
Every record retains its own live tensors; no module current-gate/member/view
state is written. Caller uses the existing serial factor member context and must
not mutate graph inputs/returned tensors before backward. Concurrent safety of
the preexisting factor facade is not extended.

Sampling uses author class0 keep and `hard-soft.detach()+soft`, with cloned
explicit noise and no hidden draw. Product gradients use the other bit's hard
value; both off gives zero direct edge credit. Empty rows, zero messages and
downstream nullspaces can also give zero credit.

## Sampling and serving ownership

Caller owns independent training policy noise streams per member/view/layer,
separate from native dropout, plus their checkpointed generator states and
model/policy/optimizer states. No production noise/checkpoint callback is supplied.
The existing native local/global restoration and RNG custody remain caller-owned.

For the plan's fixed-panel serving adaptation, caller supplies one label-blind
panel per member/layer and a fixed serving-view key at every selection/replay.
`.eval()` does not silently change sampling to argmax/probabilities or create a
fresh draw. This differs from official CoGNN's fresh eval samples. Serving choice
and panel identity remain unadopted; all-keep is a qualification mode.

## Disabled fixtures and attribution

Authored fixtures cover directed action combinations, duplicate self edges and
empty rows; hard-on/off ST product derivatives; tiny native all-keep outputs,
original gradients and one Adam displacement/state in local/global modes; and
four members/two views retaining distinct gate graphs. The last fixture compares
joint gradients with the sum over separately retained graphs; its direct gate
terms test retention only. Policy ownership and constructor RNG are also checked.
No fixture has run, and tiny parity would not certify a production graph or fit.

The implementation follows the saved `IMPLEMENTATION_PLAN.md`, native
[Polynormer](https://github.com/cornell-zhang/Polynormer/tree/fc8c276c9c5dfbd616d83f65338a3392188a5e08)
forward, pinned PyG2.7 GAT/MessagePassing, and official
[CoGNN](https://github.com/benfinkelshtein/CoGNN/tree/79e930861043b39b31c49ac918ff545395811cc1)
receive/send/ST scopes. This adapts those mechanics to native attention and
retained self/root paths; it is not a native CoGNN/GAT reproduction. Saved primary
scopes were reused, with no re-fetch. Bindings/static evidence and the manifest
cover only this new folder; existing sources, jobs and ledgers are unchanged.
