# Link-task specialization and contrastive extensions

**Prepared standalone code; no model, fit, server operation, or TEST access was
performed here.** Frozen smoke ports and the 270 separate paper-reproduction
baselines remain unchanged. Root owns integration, pilot freeze and launch.

## Accepted interface

From the phase's `benchmark_expansion_20261015_v1` directory:

```python
from link import TaskPreheadCapture, build_objectives, step_blocks
```

The root's new orchestration supplies four aligned raw-logit rows `[4,Q]`,
live original task-prehead tensors `[4,Q,H]`, actual source query endpoint IDs,
source observed-positive/sampled-negative groups, and the four unchanged
native scalar losses. `TaskPreheadCapture` authenticates endpoint/relation/
group order across all four routes. `build_objectives` requires those four
signatures; the source loss still uses every native query.

`TaskPreheadCapture(native_final_leaf, top_level_task_gate)` is a passive
prehook. Use one `capture.task(member, endpoints, groups)` context for each
original task predictor call: positive then negative for NCN/NCNC; one mixed
batch for BUDDY. Its `aligned(torch)` output supplies preheads and authenticated
query metadata. NCNC must use the explicit depth-one task gate from the
corrected port, including no-grad execution. Do not infer scope from gradient
enabled status. Call `clear()` after the update to release old graphs; `close()`
removes its hook. No source method or session is modified by package import.

```python
aligned = capture.aligned(torch)
objectives = build_objectives(
    torch, family="NCNC", logits=aligned_native_logits,
    native_member_losses=original_four_source_losses,
    native_pool_loss=original_source_loss_of_mean_logits,
    seed=declared_seed, contrast="endpoint", responsibility="none",
    aux_credit="private", **aligned,
)
counts = step_blocks(torch, original_optimizer,
    explicit_shared_parameters, explicit_private_parameters, objectives)
capture.clear()
```

The caller preserves original graph construction, negative sampling, query
masking, native dropout/BN route order, source-sized minibatches, tail policy,
optimizer groups, RNG, full-epoch controller and selected-state evaluation.
The helper verifies disjoint complete optimizer ownership, collects all block
gradients before one original optimizer transition, and retains None gradients
of inactive native Parameters. Caller owns native `zero_grad` and clipping.

## Loss definitions

Let `F` be the mean of the four caller-supplied unchanged source losses.

- **NCN/NCNC:** each source loss is mean positive negative-log-sigmoid plus mean
  negative negative-log-sigmoid. Positive and negative groups remain separate.
- **BUDDY:** the source BCE is one mean over the original shuffled mixed-label
  minibatch. No equal positive/negative reweighting is silently introduced.

The private responsibility criterion `W` uses the same source per-record loss
and reductions, with positive weights normalized to mean one **within each
original reduction group of this minibatch**. Its unit-weight reduction is
checked against the caller's live source scalars. This is not a replacement of
the shared native task criterion `F`.

### Endpoint-correlated private responsibility

A fixed BLAKE2b seed maps each `(member, relation, node)` to an Exp(1) value.
For undirected query `{u,v}`, raw weight is
`0.25 + 0.5 * (Exp_m(u) + Exp_m(v))`. The two endpoints enter symmetrically;
neighboring queries have correlated responsibilities. The source query label
is absent from the hash. All weights remain positive, so no task row is dropped.
The same-query IID control uses `0.25 + Exp_m({u,v},relation)` instead.
Shared native Parameters receive `grad F`; private factors receive `grad W`.
Raw weights are fixed by identity; minibatch mass normalization is explicit.

### Query-conditioned contrast

For anchor query `q` in route `m`, its positive is **the same authenticated
query q in another existing route n**. Different observed edges are not declared
interchangeable positives. There is no added graph augmentation or source pass.

Endpoint-conditioned alternatives are already-present native query rows with
the opposite observed/sampled binary label, the same relation ID and a shared
endpoint. Their role is conditional sampled discrimination at that endpoint.
They are not claimed to be verified absent or latent-negative relations.
Duplicate semantic queries and relation mismatches are excluded.

The auxiliary is cosine InfoNCE with temperature `0.2`, averaged over all 12
ordered route pairs. Anchor groups use the same NCN separate-group / BUDDY
joint reduction convention. At most 1024 auxiliary anchors (up to 512 per
binary group) and 16 existing alternatives per anchor participate. Deterministic
hash-cyclic selection changes auxiliary work only. Zero-coverage anchors are
reported and omitted; if no eligible anchor exists, `A=0`. There is no fallback
that invents a query, asserts a nonlink, or changes the native task population.

`cosine` control uses same-query `1-cosine` alignment with no negatives.
`global` control admits opposite-label alternatives from the same relation
without endpoint conditioning. Both retain the same budgets and native task
loss. By default both controls use the endpoint candidate's exact eligible
anchor set; the global control also matches each anchor's number of alternatives.
This distinguishes alignment pressure from the conditional denominator without
silently giving the global control more active anchors or negatives.

With fixed coefficient `0.05`, private-auxiliary credit uses shared `F`, private
`F+0.05A`; the shared-auxiliary control gives both blocks `F+0.05A`. The combined
candidate uses shared `F`, private `W+0.05A`. These are explicit block rules,
not necessarily a single scalar potential. Passive preheads precede final
boundary factors, so `A` does not directly train those final factors; native
task and responsibility losses still do.

## Competence and complementarity

`link_bank_diagnostics` accepts all 60,084/100,000 official VALID records and
the native evaluator. It reports individual, mean-member and served-pool
Hits@50, member-hit disagreement, common misses, member hits cancelled by
pooling and common misses rescued by pooling. Each predictor has its own
native negative-pool top-50 threshold; union-of-member hits is not a pooling
upper bound. Positive/negative logit variance is descriptive only.

Pilot comparison must retain member strength, complete VALID pooling, native
task losses, coverage and real wall/memory/optimizer work. Reduced latent
distance or larger disagreement alone is not evidence of a useful ensemble.
Explicit expected failures include alignment collapsing member variety,
endpoint weights rewarding degree patterns without ranking benefit, and sparse
compatible alternatives leaving the auxiliary inactive in BUDDY minibatches.

## Small optional cross-task hypotheses

- `make_hidden_residual`: `h + B_m GELU(A_m h)` at the fixed **task** prehead
  before output-boundary R/native affine. Zero B preserves the existing wrapped
  function initially. Four rank-8 private hidden adapters add `64H` Parameters:
  4096 for NCN/NCNC H64; 66,560 for BUDDY H1040. A shared rank-32 adapter has the
  identical parameter count. `nonlinear=False` is a known linear-head-capacity
  control on scalar readouts, not expressive rank-8 readout novelty. Root must
  retain explicit NCNC top-level gating and native Parameter identity.
- `make_late_gate`: a small shared GELU gate uses native normalized preheads and
  mean/RMS route-logit deviations, with no labels at serving. Zero output/bias
  initializes a convex mean-raw-logit pool. Train on TRAIN only with the declared
  half-own/half-served native task objective, selecting the whole bank on VALID.
  Compare against the existing mean-logit half-pool objective under identical
  source forwards and against a parameter-matched nonlinear concatenation gate.
  Its parameter count is `8H+28`; learned aggregation is a separately labelled
  hypothesis, not the unchanged baseline aggregation rule.

These modules are isolated builders, not integrated sessions or LoRA-Ensemble /
ELREA reproductions. Their shared native body stays shared. Literature provides
ancestry and testable questions, not a graph-utility or novelty guarantee.

## Checks and readiness

`python3 -B -m link.check_planning` runs pure-Python query/role/pairing/weight
checks and imports no Torch. These checks passed locally.
`allocation_checks.run(torch)` contains root-invoked synthetic checks for source
reductions, one-old-state block gradients and adapter/gate initialization; it
has not been executed here. Real NCN/NCNC/BUDDY forward/gradient, query-capture,
source loss parity, RNG/resume and resource qualification are still required.
Existing full-run epoch/selection/TEST gaps are not solved by this package.

Conceptual cross-task applicability is explicit in `CANDIDATES.json`: same-row
alignment and query-IID responsibility can be specialized to authenticated
node/graph IDs, with their own native reductions. Endpoint-conditioned query
denominators and endpoint responsibilities are link-specific; this package
does not substitute them for node class positives or molecular graph targets.
The optional task-prehead adapter and late gate have node/molecular/link
interfaces, but each native task still needs its own capture/loss integration.

`CANDIDATES.json` lists fixed candidate/control definitions, all inactive until
root freezes an actual pilot. `READING_NOTES.md`, `READ_SCOPES.json` and
`PRIOR_REUSE.json` separate new method scopes from retained conclusions.
