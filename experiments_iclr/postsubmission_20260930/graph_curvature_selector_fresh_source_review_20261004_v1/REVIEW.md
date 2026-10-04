# Fresh technical source review

Reviewed the sealed v2 selector/driver and their eight explicitly bound runtime source dependencies. This is an independent source review, with no manuscript acceptance verdict and no numerical or predictive qualification.

Before and after inspection, manifest `09e41d6ca83f0c4516eb2b1229465fad24c4ebbe994128379d4190e13ded103b` and seal `da140ff5e8019b6c3e92057f985b7efcb124399843252d8d81cf05dc94fb0998` matched. All eight payload lengths/hashes, directory mode 0555, file modes 0444, and eight runtime-source pins matched. Sources were unchanged.

## Confirmed finding

**F1 — medium: cached same-path modules bypass executed-byte verification.** In `driver.py:39–41`, `load_bound_source` verifies current file bytes but reuses `sys.modules[name]` after checking only `__file__`. Its definitions may come from previously loaded code or bytecode. The native API check at `driver.py:135–145` also verifies the file rather than the executed module. This can make a future qualification run execute an unverified adapter/dependency while all current disk pins pass.

A stdlib-only temporary stub reproduced the defect: the pinned file contained `VALUE = "verified_current_source"`; a preexisting `types.ModuleType` with that same name/path held `VALUE = "previously_loaded_unverified_code"`. Calling the actual loader returned the preexisting module. Observed result:

```json
{"file_pin_correct": true, "loaded_VALUE": "previously_loaded_unverified_code", "returned_preexisting_module": true, "numerical_imports_present": []}
```

The fixture restored `driver.HERE`, removed its module entry and deleted its temporary tree. `python3 -B` avoided bytecode writes. No numerical donor module was imported.

Before qualified execution, reused modules need trusted evidence of the exact bytes compiled by this loader, or unverified cached entries must be rejected and loaded fresh from verified bytes. `native_api` should be the resulting verified module identity. The clean module-table branch already compiles verified bytes; this finding is conditional on module reuse and does not imply every import is stale.

## Source-consistent behavior

- **Common center/antithetic algebra:** the closure varies only final `head.R`/`global_head.R`; copied body, identity S and copied B make logits affine in that slice. One common-only center is shared by all arms. Mean logits agree in exact arithmetic; actual casts/forwards remain unqualified.
- **Span controls:** TRAIN injection/remasking, the same common gradient, ordered rank-three construction and exactly three pair slots are preserved. Permutation changes topology alignment with fixed labels/features. Random construction uses its fixed independent CPU generator; no redraw appears.
- **D/finite safeguards:** affine antithetic CE gives an even, nondecreasing exact Jensen gap on nonnegative radius, supporting capped bisection. Original member/pooled TRAIN bounds, offset geometry, mean logits and six class-centered response differences are guarded. Native precision and FP64 D measurement need numerical verification.
- **Trials/return state:** independent complete K4 clones receive the original named coupled Adam state; one dropout-off mean-own-CE step updates live shared/private parameters. Trial state is discarded and native RNG restored. Returned models contain selected initialization slices, pre-trial transported optimizer state and copied native RNG.
- **Warm/continuation:** source follows Squirrel fixed-last 50 native updates and Photo 200-local strict-best model/Adam handoff with post-all200 RNG plus fixed-last 50 global updates. Declared native continuation schedules and pooled validation-NLL selector remain present.
- **Portability:** relative paths, dependency preloading and verified direct compilation work by source for fresh modules; F1 is the exception.

## Remaining execution limits

The missing prospective constants, actual fresh-warm provenance/affinity checks, live-factor Adam/RNG/buffer checks, and trial cost/memory are acknowledged qualification limits. They are not accuracy negatives. No numerical imports, graph runs, real data/checkpoints/current outcomes, source edits, server/Desktop actions or PDF compilation occurred. This review supplies no gain or novelty claim.
