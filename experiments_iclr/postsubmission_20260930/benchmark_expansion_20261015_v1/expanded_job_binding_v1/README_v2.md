# BUDDY interaction source overlay v2

This successor consumes the frozen `qualification/interaction_BUDDY_v1_COMPILER_BINDING.json` handoff (SHA256 `91b8e73d2de761ffbad2a2b416199defe6f37341d6a5bcfd58c8f9206c941185`). It adds the four authentic BUDDY interaction source bindings with exactly the declared manifest/context/context-hash settings. Source support is seed 0 only; CPU gate and runtime qualification remain separate.

The expected effective inventory is 6,370 concrete descriptors and 90 blocked native seed contexts, retaining all 6,460 fit IDs. The allocation holds four new full descriptors. The 6,366 existing descriptors are referenced by immutable receipts. Thirty-six previously missing-source interaction intentions are reclassified as unsupported seed contexts with their own blocked metadata; 6,420 index rows are unchanged. The 5,920 bank IDs and 5,110 paired control rows keep the same topology, with current status references updated.

`compile_overlay_v2.py` authenticates the parent READINESS and independent metadata verification, source/package/dependency pins and existing descriptor receipts. It validates only the four new jobs with the pinned runtime metadata validator. It writes a fresh `generated_v2`; it never changes `generated_v1`, the original source registry, old descriptors or ancestor contracts.

The effective source registry is `REGISTRY_v1.json` plus `OVERLAY_v2.json`. V1 descriptors keep their original registry provenance. Changed intentions and four new jobs explicitly bind the overlay and frozen handoff. The overlay verifies 84 unique source/metadata files (86 source pin occurrences), including every declared package file and relevant portable native source interface.

No task models or transforms are imported. There are no dataset/tensor reads, data-hash refreshes, fits, GPU compute, TEST access, resource bindings, runtime lifecycle qualifications or GO. Runtime v4's known link selected-state/export and outer source-authority gaps remain explicit. The 90 blocks cover 45 C6/responsibility, 36 interaction and nine private-BN nonzero-seed contexts.

The first independent v1 verifier's incorrect assumption that old shared fits use `body=None` was preserved in `VERIFICATION_ATTEMPT01.json`. The corrected immutable verifier v2 respects the preserved original `body=0` convention and passed all 6,460 intentions. No compiler output changed to satisfy that verifier.
