# Independent source review: graph relation learning field

Date: 2026-10-08. Scope: mathematical source semantics and selected-state serving reconstruction. This is not manuscript acceptance, numerical equivalence, utility, novelty, runtime admission, or permission to open comparative outcomes.

Reviewed source: `graph_relation_private_credit_source_20261008_v1`, with `SOURCE_MANIFEST.json` SHA256 `7b4e72b5d1ba4168fc854fefdac404a78078e63beaa7afbda486359ae97c3b5c`. The manifest hash, all 21 current payload hashes/sizes, ten directly bound source/dependency records, four core source hashes, and the inspected scorer/pool source hashes matched. I inspected `permissions.py`, `replay_adapter.py`, `train.py`, `qualify.py`, `PROTOCOL.json`, and only the pinned source dependencies needed to trace their semantics within this phase directory. I did not rely on the prior `SOURCE_REVIEW.md` assessment.

## Finding

**No concrete blocker was found in the reviewed static loss, recipient, replay, constructor, or serving reconstruction semantics.** This finding supports proceeding to separately authorized native engineering qualification, subject to the root's runtime/resource/custody review. It does not certify a successful execution or gradient equality on the actual backend. The known v1 original12 launch gate is outside this finding; a separate gate-only v2 requires its own exact source identity and gate review. Anchor numerical reuse and comparative opening must remain prohibited until the required complete unions close.

## Mathematical field

For view `v`, member `m`, and `N=580` TRAIN rows, let `c_vm` be the mean categorical CE and `p_vm` its class probabilities. The source implements

`F_v = (1/4) sum_m c_vm`, `L_v = -(1/N) sum_n log[(1/4) sum_m p_vmn,y_n]`,

`F = (F_a+F_b)/2`, `L = (L_a+L_b)/2`, `J = (F+L)/2`.

`own_supervision` returns one mean CE per member (`core/objectives.py:6–10`); `replay_adapter.py:16–21` averages the four members and two views with the stated coefficients. The pinned `served_pool_supervision` uses `log_softmax` and `logsumexp` on true-class probabilities (`adapter.py:6–18`), so it computes CE of the mean probabilities separately within each view, without probability clamping or averaging logits. No additional member or view coefficient appears in the VJP accumulation.

The parameter field is exactly the declared partial-gradient assignment:

| Policy | J recipients | F recipients |
| --- | --- | --- |
| alphaF | None | All parameters |
| allJ | All parameters | None |
| phiJ | 56 internal dense r/s tensors | Exhaustive complement, including local scorers and six boundary r/s tensors |
| relationJ | 14 local scorer banks and four tied-QK r/s tensors | Exhaustive complement |

The scalar CE mixture supplies positive own-CE reweighting at each member's logits. For one row/view, with `a_m = p_m,y / sum_j p_j,y`, the J logit gradient is `(0.5 + 2 a_m)` times the corresponding F logit gradient. This is not a direct disagreement objective or a competence guarantee. The mixed recipient fields need not be gradients of a common scalar; the source explicitly treats its reported scalars as diagnostics (`replay_adapter.py:35–38`).

## Recipients and differentiation

`permissions.py:19–84` binds the core/model/native profile, demands one shared native M4 body, checks the exact factor-map inventory and bank shapes, rejects parameter aliases and unknown roles, and checks that every parameter appears exactly once in the original Adam bank. The target/complement identity sets are disjoint and exhaustive. Counts agree with the protocol: 56 internal factors, six boundary factors, 18 relation tensors, and four tensors shared between the phi and relation recipient sets.

There are two physical `k_lins` maps. Native `GlobalAttn.forward` sets `q=k` (`native_polynormer.py:58–71`); `permissions.py:28–35,66–68` rejects an independent-query fallback and selects one r/s bank per physical map. Differentiation includes both query and key uses of that same tensor, with no duplicated recipient entry. Value-map factors receive J under allJ/phiJ; their shared weights receive J only under allJ, as the descriptor states.

`output_credit` differentiates F and J with respect to detached output leaves; `collect` applies the selected cotangents to the connected member graph for each disjoint parameter group (`replay_adapter.py:27–58`). Detaching these cotangents is the ordinary first-order chain rule and does not detach upstream parameter paths. Representations receive zero cotangents because the reviewed loss has no representation auxiliary. Inactive native global/head parameters retain `grad=None` rather than receiving fabricated zero gradients.

## Replay, initialization, and selection

The adapter hash-pins the public replay and replaces only its output-loss fragment and member reverse site (`replay_adapter.py:61–110`). The retained engine zeros gradients once, records parameter versions, saves member streams, collects two shadow views, restores the starting streams, replays all eight member/view forwards, checks exact stream endpoints and unchanged parameter versions, and calls the sole Adam after complete accumulation (`context_recompute.py:27–35,63–101`). The stateless model/buffer prohibition is retained. Two group VJPs per member/view produce 16 reverse collections and one Adam update.

The pinned constructor inserts scorer installation before the original optimizer factory; copying native scorer rows adds no random draw (`constructor_adapter.py:75–97`; `private_local_attention.py:25–32,133–141`). Every shadow/replay/serving member call uses the explicit scorer-row wrapper and native BE context. `make_session` requires a fresh unit constructor, disables the contrastive flag, sets both legacy auxiliary weights to zero, verifies unit r/s and identical copied scorer rows, then installs the revised replay (`train.py:115–142`). The replaced replay loss contains no alignment/residual call.

The original complete 1100-epoch loop, strict-first joint VALID selector, and selected-local model/Adam restore with live end-local streams remain in use. No per-member checkpoint splicing is introduced. The full-run completion checks require the declared actual work counts (`train.py:194–202`).

## Reconstruction and limits

`reconstruct_selected` rebuilds the same fresh bank/wrapper construction, checks the saved train/source/partition/constructor identity, requires an explicit native stage and matching model recipe, strictly loads the state dictionary, restores local/global mode, enters evaluation mode, and disables its training entry (`train.py:232–251`). This is a coherent serving reconstruction of a trusted selected-state dictionary; it neither reselects a checkpoint nor loads optimizer/RNG history. Actual selected artifact integrity and reconstructed prediction equality were not tested or read in this review.

`qualify.py:52–124` specifies eight fresh complete TRAIN updates, one per policy and native local/global stage. Its observer checks unchanged parameter versions before Adam, group finite non-None gradients, inactive global paths, active tied-QK pullbacks, work counts, and one optimizer call. These are useful execution checks, **not an independent gradient oracle**. Exact RNG endpoint equality does not establish identical shadow/replay numerical outputs on a backend with nondeterministic operations; prediction-difference diagnostics are disabled. Numerical gradient equivalence and serving prediction equality remain unverified and must not be inferred from a successful qualifier.

No Torch/numerical module was imported, no fixture or source module was executed, and no data array, checkpoint, outcome, server, or long historical custody manifest was read. No frozen source was changed. The sole written artifact is this independent review.
