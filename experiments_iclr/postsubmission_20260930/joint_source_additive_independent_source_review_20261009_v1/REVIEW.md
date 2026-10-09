# Independent source review of immutable V1

9 October 2026. Reviewed source seal: `b147739a9af48afbdba32cdf706a2b91373a1daebc37724feefb3c91fb333656`. The original packet and dependencies were not edited. Review used source, AST/JSON and hashes only; no numerical providers, model objects, inputs, checkpoints, outcome arrays or servers were used. `python3 -B verify_static.py` passes: four Python files and 43 bound dependency files. That checker does not establish runtime correctness.

## Current source findings

**F1 — P1, construction blocker.** `joint_training.py:147` rejects every native buffer, with a repeated rejection at `:213`. The exact bound `engine.py:191–194` constructs SeHGNN with four task layers; the IMDB branch in `model.py:163–167` creates three `nn.BatchNorm1d(hidden, affine=False)` layers. Their default `track_running_stats=True` registers running mean, variance and batch counter buffers. Every declared condition therefore fails construction. The stateless premise in `IMPLEMENTATION_LIMITS.md:15` and `ROOT_CLARIFICATIONS.json` conflicts with the pinned source. A revised source must specify member-private native BN state, reference/replay advancement and terminal-peer behavior while retaining the architecture. Removing the assertions alone is insufficient: bound `bank_training.py:158–172` restores original buffers after every scratch forward.

**F2 — P2, exported role guard gap.** At `joint_training.py:203–205`, the token role is checked but an explicit `ids` override is accepted unchecked. A token claiming TRAIN can therefore request VALID or unassigned rows, including an ablated view; `:215` validates only output length. Require the exact complete canonical role rows or their allowed permutation. The internal TRAIN loop already checks its batch at `:231`, and factual evaluate uses canonical rows; this finding does not establish leakage in those internal flows. Source-ablated VALID readout should remain an explicitly authorized diagnostic path.

## Source checks with no additional concrete defect found

- P is own four-view BCE with reduction `1/(4*count)`; independent scale4 gives each body's four-view mean. S uses recipient factual output plus three detached absent peers in stable Bernoulli event probability pooling, then recipient-private VJP with shared coefficient1/3 or independent4/3; neutral averages three contexts. All P/S contributions precede Adam (`joint_training.py:236–295`, helper `source_supply.py:161–194`).
- The additive grouped and semantic corrections have the native axes and preserve biases, affine boundaries and nonlinear placement; shared slow identity and private storage checks are present (`joint_additive_adapter.py:26–87`, adapter `grouped_member_factors.py:90–125,220–263`). Native zero gamma and empty-support channels are preserved by the source.
- Each independent body owns a disjoint native body and Adam, selects and stops on its own factual BCE, skips gradients/Adam after stopping and restores its own best only in fresh final serving (`joint_training.py:127–191,287–295,437–474`). All factual serving members are pooled in FP32 probabilities with strict >0.5; original current4×3 diagnostics are reused.
- The bound role loader opens only named node/link/development-label members, and both full and ablated propagation construct label inputs from TRAIN labels only (`role_loader.py:36–47,209–229`, engine `engine.py:155–166`, views `native_family_views.py:185–225`).

## Later full-input qualification obligations

These are qualification work after correcting source findings, not additional demonstrated code failures:

1. Exercise the complete native architecture and chosen BN policy through reference/replay, active and stopped peers, exact selected model/buffer/Adam/RNG restore and fresh serving.
2. Verify zero-start additive materiality and real P/S gradients through all nonlinear paths, including zero-gamma and empty-support nulls; physically measure counts/storage and actual Adam ownership/update. Confirm the declared reductions and private-only S contribution on admitted full inputs.
3. Set and freeze numerical replay acceptance criteria for direct FP32 before fitting; source currently records drift without an acceptance gate (`joint_training.py:256,304–306`). Existing BE or AMP site qualification does not qualify this loop.
4. Verify complete raw-family reconstruction, role/target and source custody, retained enabled release and qualification/freeze bindings, checkpoints/outputs/failures, setup and full lifecycle costs. Admit native and independent reference competence and freeze quality/resource/custody criteria prospectively. No runtime, quality, resource advantage or novelty conclusion is established here.

All findings are preserved in `FINDINGS.json`. Review applies to immutable V1 only; any separately authored revision needs its own source review and full-input qualification.
