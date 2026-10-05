# Shared core with private transfer: source release

This packet contains **unexecuted source**, not a result. No numerical library was imported, model constructed, gradient evaluated, fit made, server accessed, score recomputed or paper edited during preparation. Root must review the sealed bytes, execute the finite FP32 gate, measure complete-cycle TRAIN cost, and freeze the cohort before any representative fit. Disabled templates are not authorization.

## What the proposed rule does

The shared core contains the NCN graph encoder and the dense matrices of its nonlinear predictor. Four private routes contain BatchEnsemble input/output factors, LayerNorm affine parameters and the predictor's scalar mixture coefficient. The unchanged unframed F4 donor supplies these operations. There is no Householder frame, new graph operator or inference adaptation.

Each TRAIN episode has four inner edge streams and one outer edge batch. Inner streams are drawn without either endpoint of any outer positive **or negative** query. Route m evaluates its own normalized native positive/negative log-sigmoid loss and makes a differentiable virtual Adam update of its private parameters. Previous moments are frozen episode state. The outer objective is the equally weighted sum of the BCE of mean raw logits and mean individual-route BCE. Its derivative updates the shared core, including credit through the virtual private updates. The virtual state is discarded. The private losses are then recomputed at the updated shared core from the original private parameters/moments with the same dropout masks; exactly one resulting private Adam step is committed. Outer labels never directly update committed private parameters. Serving uses the committed state and mean raw logits.

This is a one-step, truncated derivative through **current** adaptation. It does not differentiate through optimizer history, previous episodes or graph structure. All Adam constants stay native: learning rate .001, betas (.9,.999), epsilon 1e-8 and no decay. The backend is explicitly single-tensor native Adam (`foreach=False`, `fused=False`) for shared/joint commits. `private_adam.py` preserves the same update with a stable exact zero-history square-root identity; it does not smooth, clip or substitute SGD. Nonzero squared-gradient underflow and inconsistent zero-second/nonzero-first moments are rejected, rather than silently assigned a different update or derivative.

## What is inherited and what remains a hypothesis

BatchEnsemble and TabM already supply shared dense matrices with small private fast factors. MLDG explicitly supplies source-only virtual inner/outer learning, fixed inference and gradient alignment; ANIL and BMAML supply restricted adaptation and ensemble/meta-learning antecedents. OML/MRCL, La-MAML and FTML also overlap with online correlated head learning, persistent representations and adaptation credit. **Persistence, a shared backbone and meta-gradients are not claimed as novel.** The scoped literature conclusions are in the preceding `private_neighborhood_quantile_quality_hypothesis_20261005_v1` packet and the independent reviewer’s `shared_backbone_private_transfer_source_independent_review_20261005_v1` packet. No new full-paper reading count is asserted here.

The narrow hypothesis is whether training a graph ensemble's shared core for endpoint-separated private learning, with a private commitment recomputed after the core changes, improves committed-state link prediction. Its exact novelty remains unresolved. A favorable comparison alone cannot establish novelty. The stale-commit ablation is needed if recomputation/state alignment is later highlighted: it pays the identical private recomputation, then discards it and commits the earlier virtual private parameters/moments. It receives no different examples, dropout advancement or outer objective. Root decides inclusion before outcomes.

## Controls and the questions they separate

| Comparison | What it can test | Limits |
|---|---|---|
| `shared_f4`: live versus detached transfer | Credit through private adaptation | Identical adapted values, losses, forward work and recomputed commitment; only the shared derivative differs |
| `shared_f4`: endpoint versus matched random | Endpoint-separation training rule | Full-TRAIN degree/CN strata are matched exactly; post-mask strata are measured and can differ |
| Live versus `live_stale_transfer` | Recomputed versus pre-core private commitment | Identical paid recomputation/exposure; not a claim that this commitment rule is new |
| `capable_single`: live versus detached; ordinary joint reference | Generic transfer benefit | Entire native nonlinear predictor adapts; encoder alone is outer. This is a richer private block than F4, not a restricted factor-only single |
| `shared_f4` versus `capable_single` | Four-route ensemble contribution | Architectures and private capacities differ; no ensemble-specific conclusion from candidate-only improvement |
| `shared_f4` versus `untied4` | Effect of tying the core | Same operations/private factors; untied encoders/base matrices begin identical to the shared route's base and use the corresponding factor row. Full initial parameter/logit parity is gated |
| Ordinary native nonlinear single/sharedF4/native independent4 | Strong ordinary quality references | Paid three-update schedules are explicitly new controls; preserved original native fits remain separate references |

For F4/untied inner updates, each private block gets its own normalized route loss; gradients across disjoint private blocks are summed without dividing by four. The capable single uses the full union of the four streams in one normalized loss, retaining sampled repeats. All transfer architectures optimize the same outer mean-logit/own objective. Each untied encoder therefore receives the objective's 1/M branch scaling, while tied core gradients accumulate contributions. This convention is fixed before outcomes. Finite epsilon, optimizer history and stochastic gradients mean Adam is **not presumed** to cancel rescaling effects.

The ordinary paid-exposure control performs three committed joint native Adam updates in the order inner, outer, repeated inner. The two inner passes replay the same examples/masks; only the last advances each dropout stream. Its outer pass is dropout-off and uses the same mean-logit/own objective. Independent ordinary routes sum normalized own inner losses and scale the outer objective by M; shared F4/single use mean/union-normalized losses. The independently initialized ordinary native4 retains seed+5×member initializations as a strong quality reference. These are comparable exposure/paid-work controls, **not identical optimization or wall time**.

## Graph visibility and exposure

Both episode arms share the union support mask of their outer positives and **both** sets of inner positives. A random arm's target edges never remain in the endpoint arm's support, or vice versa. Support retains all 3327 nodes and other TRAIN context edges, including incident edges. Inner/outer endpoints are disjoint only in their supervised queries; neighborhoods and graph context can overlap. This is a fixed transductive Citeseer/HeaRT split, not independent tasks, new graphs or an inductive claim. Native TRAIN negatives exclude TRAIN facts; unknown heldout facts are not consulted.

Root’s preceding one-cycle TRAIN geometry measurement found all 61 proposed outer64/inner256 pairs feasible in 10.91 seconds, with no redraw/skip/fallback. All 3870 outer positives were covered once, including the 30-row tail. Minimum eligible populations were 2913 positive and 6519 negative queries. The paired common support removed 1541–1610 positive facts (about 40–42%); each route received 15,616 inner positives (mean exposure 4.035 per TRAIN positive), while 91–103 positives were unseen by that route within that cycle. All were seen by outer supervision. This receipt clears one geometry cycle only, **not future cycles, predictive quality or complete-cycle compute**. It is not a score in this packet.

`run.py` recreates and preserves both full episode recipes and full before/after strata in compressed server-side history. Infeasibility fails and preserves the failed draw; there is no redraw, dropped tail, fallback, outcome-adaptive recipe or reuse of a wrong-allocation result. Full-cycle exposure and measured neural/geometry/wall costs are recorded separately. There are 61 episodes for the proposed sizes, versus three full native 1024-edge batches with a dropped 798-edge tail. A nominal cycle is not a native epoch or a cheap unit of work. Virtual and recomputed inner exposure is counted twice, despite one private commit.

## Finite FP32 gate

`qualify_training_step.py` requires all three transfer architectures. It runs two reachable Adam-history steps and one additional discarded stale-commit step per architecture on one prospectively chosen first TRAIN episode. No VALID values or metric are read. It checks:

1. All declared private names receive a non-None inner derivative across their own-route calls; unrelated untied route blocks may be None within other calls.
2. Functional Adam parameters and first/second moments against independent `torch.optim.Adam` commits, including scalar zero-gradient state and explicit underflow/inconsistent-state rejection.
3. Every shared FP32 derivative coordinate against a separate direct-plus-mixed chain rule. The independent outer point treats adapted private parameters as leaves; an analytic scalar Adam gradient Jacobian supplies the cotangent for the inner mixed-Hessian VJP. No finite differences or omitted coordinates are admitted.
4. Identical live/detached adapted outer logits. Missing mixed derivatives are explicitly reported as zero contributions; declared active private derivatives cannot disappear silently.
5. Complete native shared updates, same-mask recomputation from original moments, committed parameters/moments, explicit committed-state functional serving, and original unchanged donor serving parity.
6. Untied versus shared initial parameter/base/factor-row and complete-logit parity; global RNG, private stream advancement, moments, modes and native sparse alias restoration.

Prospective symmetric coordinate tolerances are parameter `atol=2e-7, rtol=2e-6`; moment `atol=1e-10, rtol=2e-6`; derivative `atol=1e-7, rtol=5e-5`, with each derivative name also requiring relative L2≤2e-5. Serving versus the original donor and initial-sharing parity require exact equality. They will not be loosened after outcomes. The constant-adjacency recursive adjoint’s forward stays native; only its recorded dense-input transpose pullback repairs higher-order differentiation. Sparse structure/values receive no derivatives. Earlier private-frame SGD/float64 diagnostics and inconclusive FP32 finite differences remain limited to their own recipes and do not qualify this one.

## Release and selection

Each entry point verifies Linux repository containment, independently bound hostname, exact singleton physical GPU UUID, fresh output, source/manifest/dependency/input hashes, review and runtime authority. Seven-GPU science, alternative hosts and TEST roles are rejected. No system mount/namespace changes or sudo exist. Source/dependencies are pinned by bytes; numerical imports occur only after authorization.

After the finite gate, cost mode runs exactly one complete TRAIN cycle, discards states, reads no VALID values and writes actual costs/failures. Fit mode requires root’s immutable cohort, all cell identities and numeric horizons fixed after measured TRAIN cost and before scores, explicit comparator paid/selection budgets, complete VALID every five cycles and eleven consecutive misses. It preserves the first maximum rounded-four-decimal complete VALID MRR, model/optimizer/private moments/RNG/partition and every member logit in the selected checkpoint. All failed/partial attempts stay recorded, with no implicit retry. There is no TEST evaluation or paper update in this source.

A passed implementation gate is not quality evidence. Root must assess representative paired fits against the capable single and independently initialized ordinary native ensemble, separate generic transfer from ensemble/sharing effects, and resolve closest-prior overlap before claiming a methodological contribution. Original paper scores stay unchanged.


## Immutable v2: diagnostic JSON correction only

v1 is preserved at `shared_backbone_private_transfer_training_source_20261005_v1`; its manifest SHA is `81c36df0ccad682f06cd27d5b4f5b59ea70bb074f908b38f519f147d58712337`. v2 changes only the qualifier’s diagnostic reporting of difference/tolerance ratios: an exact zero-difference/zero-limit coordinate reports zero; a nonzero error at zero tolerance still fails and reports a null ratio with an explicit status. Any nonfinite diagnostic scalar is represented as null plus a status, so the exception metadata does not serialize NaN/Infinity. This does not change any criterion, tolerance, optimizer, derivative, model, training rule, schedule or source admission.

The exact qualifier diff is `QUALIFIER_DIAGNOSTIC_DIFF.patch`, SHA `2de57bbdb3dc236ab0b0af25949a27a5a2c0d5ef30a6ad32dac9f528727cf64a`. `difference`, `limit`, `relative` and `ok` AST statements and all top-level numeric assignments are unchanged. `private_adam.py`, `transfer_step.py`, `models.py`, `run.py` and `custody.py` remain byte-identical to v1. Disabled job templates merely rebind the qualifier program SHA; static hashes and this manifest are refreshed. Only stdlib source/AST/hash operations ran. There were no numerical imports, executions, server actions, new results or literature searches.
