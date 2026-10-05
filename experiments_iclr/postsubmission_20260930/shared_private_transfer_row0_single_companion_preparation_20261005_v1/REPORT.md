# Disabled fixed-row0 F1 companion source

## Scope

Separate source preparation for **nine fits only**: endpoint live transfer,
detached transfer and ordinary joint, each at original b0/b1/b2. All fit jobs,
source review, new training gate, external hard-bound confirmation, VALID
access and plan adoption remain disabled. No companion qualification, cost
cycle, model construction, numerical import or fit has run. No server source
was edited. This packet adds no queue implementation or acceptance gate.

The original 30-fit study, rich `capable_single`, sources, running queues and
promotion gate are retained. The original canonical plan is bound in
`COMPANION_PLAN_DISABLED.json` and `DEPENDENCY_BINDINGS.json`. Nine proposed
cells preserve their corresponding original F4 endpoint cell's seed, factor
seed, 64 outer / 256 inner sizes, 60-cycle horizon, evaluation every five cycles,
eleven-miss stop and first maximum rounded complete VALID MRR selector. Their
execution order follows those original cells' canonical relative order.

## Minimal source changes

- `models.py`: build the complete unchanged unframed F4 bank under the original
  seed/factor seed, then retain **fixed row0** through the existing route slicer.
  Initial parameter and logit parity is to F4 **route0**; no equality to pooled
  four-member logits is claimed. No outcome chooses the row.
- `transfer_step.py`: the flagged F1 performs four explicit forwards on the
  original four query/dropout streams, all through route0. It averages the four
  normalized own losses' private gradients and makes one virtual private Adam
  update. After the outer update, it recomputes all four with replayed masks and
  advances each stream once. The ordinary path uses mean loss on each inner
  pass and retains the existing inner / outer / repeated-inner three native
  commits. Transfer and ordinary episodes each pay nine functional forwards.
- `run.py`: admit only this F1 arm, endpoint geometry and the three stated rules;
  require its fresh exact-source qualifier and label its donor head correctly.
  The training loop, full VALID serving, metric, schedule freeze and selector
  are retained.
- `qualify_training_step.py`: qualify only the F1 branch. Existing F4 is built
  solely as the initial row0 reference. The new checks use the actual first
  TRAIN-only episode, unchanged tolerances and native Adam constants. Unchanged
  original architecture numerical gates and scalar Adam tests are not rerun.

`COMPANION_DIFF.patch` is the exact four-file diff. `private_adam.py`, singleton
`custody.py` and reviewed `custody77.py` are unchanged bytes. The two planned
source manifests differ only in the previously reviewed custody provider;
eventual staging uses the selected custody bytes as `custody.py` and the
selected manifest as `SOURCE_MANIFEST.json`.

## Parameter roles

| Block | Outer coordinates | Private coordinates |
|---|---:|---:|
| Encoder (`jkparams`, `xemb.1.weight`, `xemb.1.bias`) | 948,225 | 0 |
| Dense head bases and dense biases | 460,801 | 0 |
| Fixed-row0 r/s factors | 0 | 3,841 |
| Fixed-row0 LayerNorm affine weights and biases | 0 | 2,560 |
| Fixed-row0 beta | 0 | 1 |
| Total | **1,409,026** | **6,402** |

Factors use `xlin.ops.{0,3}`, `xcnlin.ops.{0,3}`, `xijlin.ops.0` and
`lin.ops.{0,4,8}`. Private norms use `xlin.ops.4`, `xcnlin.ops.4`,
`xijlin.ops.1` and `lin.ops.{1,5}`. Dense biases stay outer; norm biases stay
private. This removes the rich single's dense-head role relocation from this
companion comparison while retaining the rich single as a quality baseline.
Four member-private states versus one, pooled versus one-route serving and
four-stream aggregation remain the intended member-count changes.

## Minimal numerical gate proposed for root review

1. One discarded F1 qualification at the selected first provider: exact initial
   every-parameter role/row0 equality and all initial outer-fixture logits;
   an independent objective-first mean-of-four gradient oracle; zero plus two
   reachable prior histories with all-coordinate native private/shared Adam
   parameter and moment parity; independent direct-plus-mixed chain-rule parity;
   recomputed committed serving; unchanged global RNG/training modes; replay
   and exactly one advancement of each of the four streams.
2. In that same qualifier, an explicit ordinary inner / outer / repeated-inner
   native-Adam reference verifies all parameters and moments, nine forwards,
   three commits, committed serving/donor parity and RNG/mode/stream guards.
3. After root accepts the gate, measure discarded complete-cycle F1 TRAIN costs
   for the three proposed rules through the existing runner and supervisor.
   Root must adopt costs, fairness and concrete bounds before any fit. Existing
   exact-provider custody/runtime admission still applies to every fit provider.

Costs and hard bounds are **pending**, with corresponding original F4 limits
carried as proposed bounds only. No numeric job has been released by this packet.
No tuning, extra random-geometry cells, retries or score-based rescue is added.

## Static verification and evidence

`STATIC_VERIFICATION.json` records 129 passing stdlib-only checks. It proves
unchanged ASTs for dropout streams, losses, functional forward, virtual state,
transfer episode step, metric, schedule freeze and native qualifier helpers;
unchanged nested model operations; exact preservation of all original model
branches after removing the declared insertion; and exact original transfer
source under the false F1 flag. Runner `main` matches after normalizing only
its descriptive head label. Original source/dependency bytes and the canonical
plan hash were checked, plus every disabled job identity, binding and provider.

The saved original b1/b2 metadata observation was sealed separately at
`shared_private_transfer_gpu77_block_observation_20261005_v2`; it read no scores
and made no launches or signals. The original launch-count clarification is
bound in `DEPENDENCY_BINDINGS.json`: its count zero came from the prefit freeze;
the preserved initial live observation authenticated one first fit child per
block. The original launch packet was not rewritten.

Root review is the next step. This seal records source preparation completion,
not numerical qualification or authorization to run fits.
