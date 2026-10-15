# Explicit node/molecular interaction readiness

All four recipes passed on both actual CiteSeer-Cheb and MolBACE GIN/dropout.5 CPU cells.
Source manifest: `d5928d216cd09c03bad5244bed87e3c0b92d169cb308bbffc87ee099aa6d7998`.
Qualification manifest: `d7d53f0d8b5e07f1b58e4215ca9ff13bd15b99dd3be6abebebd74fc301740bfb`.

| Recipe | Node | Molecular |
| --- | --- | --- |
| I1 geometry + portablev2 BE input residual | passed | passed |
| I2 structured responsibility + private half-pool | passed | passed |
| I3 BE input + meanJ | passed | passed |
| I3 BE input + conditionalJ | passed | passed |

16 actual optimizer attempts/completions,16 native constructions,16 complete VALID calls and16 same-call mean diagnostics; fresh restores used zero optimizer steps. Every node update retained all1929 TRAIN nodes; I1 auxiliary retained its frozen512 cohort. Molecular updates used the first two original32-graph shuffled source batches atseed101, both with observed mixed-class support; exact IDs/counts are retained. VALID covered665 nodes or151 graphs.

Native source Parameter identities, complete disjoint parameter/optimizer ownership, original group objects/LR/WD/options and native reductions were preserved. I2 used sharedF/private(.5W+.5L), two old-state reverse collections and one transition. Other recipes used one scalar reverse and transition. Residual down/up and all conditional scorer blocks had live gradients after two updates.

Both actual serving and fixed same-bank mean outputs/losses/metrics were saved from each VALID call. IDs/labels/metrics matched exactly after fresh restore; maximum actual and mean logit differences were0. Model parameters/buffers remained unchanged during VALID. Modified component/table checkpoint bindings and stale/missing mean getters were refused.

## Boundaries

These are new combined engineering states with exact component/source bindings, not aliases of legacy molecular pilot or standalone component checkpoints. No selector/full scientific run_fit, GPU, TEST or accuracy/convergence claim was added. The full wrapped checkpoint was preserved.

The native getter supplies a zero-forward same-call sidecar. It intentionally does not fabricate a selected-checkpoint authority. Frozen runtime_v3 drops extra native mean fields; the future production finalizer must persist the sidecar after loading/validating its own selected checkpoint and bind job/hash/epoch/IDs/labels. Runtime_v4 source work is separate.

Observed cost under concurrent NCNC CPU work: loop 67.535922s; full process 71.132901s; CPU user/system 106.141468/15.062738s; peak RSS 3,499,436 KiB. Two CPU threads, CUDA hidden; no CUDA context. All seven source packages rechecked unchanged locally; the allocation qualifier reverified the combined bundle after completion. No automatic retries.
