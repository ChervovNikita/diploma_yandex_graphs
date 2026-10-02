# Independent Squirrel driver v2 delta review

Date: 2026-10-03  
Verdict: **B1 and B2 closed. No outstanding material source blocker found in the focused delta.**

Verified the exact v2 script `60fb7675318252ac43a00d847e5f43fa6f20521868cb6ca1610f75a9e39186ee`, manifest `3f2d8faa81039fd6e5ceb5fecf7ea4b46bf725b516dbb040827926a92ddde452`, and seal `d2813df117b8cb59b2d5de08871671e26f28f412c675aa4ac6533905fd5ba9cf`. The seal binds that manifest; all seven payload sizes/hashes match.

## Closure of findings

- **B1 — Closed.** The sole executable-source change at `prototype/continue_squirrel.py:257` passes `ctx['source_labels']['validation']` to the unchanged pinned label loader. In all three cells, this descriptor is exactly the prior kind-bearing descriptor minus `kind`, retaining the same path/hash/bytes. The real pinned verifier's extracted stdlib functions accepted the descriptor shape for seeds 17, 29, and 43 using synthetic text bytes. The scope/preservation record and late admission point remain intact.
- **B2 — Closed.** Frozen original records add exactly `graph_full_node_cotangent_paired_alpha_v1/base/graph_band_route_initializer.py`, with 31,778 bytes and SHA-256 `25b5e55b5150a7ee00a10d28d02b97fa689276ffd1f4326011a13209becfbabb`. The descriptor matches the sealed paired manifest and current source copy. The existing `verify_inputs(original_records)` call precedes helper import and also runs after execution, so this actual transitive import now participates in admission and preservation.

The frozen metadata delta comprises that added record and a v2 schema identifier; no record was removed. Mode, all donor identities, schedules, arms, alpha policy, optimizer transport, RNG handling, installed-output qualification, selector/replay, label boundaries, and twelve-fit development gating are unchanged. No numerical gate or new science was introduced. The v1 review's conclusions about those unchanged paths continue to apply.

`DELTA_EVIDENCE.json` records the focused checks. No original numeric inputs or labels, native/Torch/GPU computation, remote operations, or broad test suite were used. V1 and all source inputs remain preserved. This is a source verdict for the exact v2 seal, without a native success or predictive-quality claim.
