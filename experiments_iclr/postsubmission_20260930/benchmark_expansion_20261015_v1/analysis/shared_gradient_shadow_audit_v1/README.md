# Native shared gradient shadow audit

Source prepared for root review. The finite command in `COMMAND.json` has not been executed. It uses the original three genuine MolBACE/GIN private-BN selected states, seeds 101/202/303 and epochs 92/54/116. This implements only the approved TRAIN audit; it contains no PCGrad or QP training arm and makes no campaign recipe change.

## Scientific context

The complete nine-fit result in `benchmark_expansion_20261015_v1/BN_GEOMETRY_COMPLETE_16.json` reports private-BN AUC 73.614%, +geometry 73.651%, +input correction 72.186%, and +both 73.028%, versus I4 74.847%. Geometry's +0.0366-point mean has mixed seed signs; correction harms AUC in all three seeds. The positive private interaction (+0.806 versus shared -5.873) does not make both ingredients better than private BN alone. The audit therefore asks whether the actual shared Adam displacement causes repeated own-risk harm after accounting for private progress. The selected-state AUC recovery, adverse BCE, and earlier gradient diagnostics do not answer that question.

The separate `analysis/shared_gradient_private_progress_proposal_v1/PROPOSAL.md` is frozen at 17,231 bytes, SHA256 `5038408f2519d8783fa59419b8d76cb5c65b68ec486b127a045154b884e8d479`. Its intervention comparison remains a proposal. This package supplies the approved audit and its explicit work accounting.

## Native execution and restoration

`audit.py` restores the authentic original runtime, JOB, selected checkpoint, optimizer, global Python/NumPy/Torch CPU/CUDA RNG, and private loader state. It verifies the saved producer sources and constructor/role bindings without rewriting a JOB, checkpoint or runtime binding. It then calls the original `adapter.train_epoch()` and original `session.train_update()` for one complete 1,210-graph TRAIN epoch per seed. Three models run serially in one process.

The instrumentation captures the existing four route outputs, adds four unscaled own-loss gradient collections, and allows the original mean-own backward and original Adam step exactly once. The native gradient must agree with the route mean at fixed rtol=1e-4, atol=1e-6. Shared parameters are precisely the original native parameter references, including BN affine parameters; private learned boundary banks form the disjoint remainder. Other routes' private-row gradients must be zero. Private BN banks are buffer state, rather than gradient coordinates.

After each actual temporary Adam proposal, it measures O/B/D losses on the same batch. B combines old shared parameters with proposed private parameters; D is the joint native proposal. B/D use TRAIN BN and the original route order under no_grad, restoring pre-forward global RNG and all buffer values for each replay. Their consumed RNG progression must equal the native forward progression. The native post-step parameters, bank values/live pointers, optimizer, loader RNG and global RNG must then be unchanged by the measurements. One additional O replay on the first batch of each seed verifies original loss reconstruction at the proposal's fixed tolerance. No retry or tolerance tuning occurs.

At completion or a caught failure, the disposable model is restored to its original selected parameters, optimizer, banks, modes, loader state and RNG, then released. Only scalar gradient/loss records, IDs/order metadata and operation receipts are written. No weights, predictions or new selected model are exported. Source construction/checkpoint deserialization carries original role IDs and historical score metadata; no VALID/TEST prediction or score is computed. VALID and shuffled TRAIN-eval entry points are blocked.

## Fixed gate and costs

For each route-batch event, record old-gradient products with actual shared/private displacements, their sum, and finite private/shared-after-private/joint loss changes. The gate is unchanged: at least 16 of 152 events in at least two of three seeds must have predicted joint change, finite shared-after-private change, and finite joint change all greater than `tau = 1e-4 + 1e-4*abs(old_loss)`. All three seeds and every batch enter the denominator. Negative pair cosine is secondary. A failed seed makes the aggregate gate unavailable. The audit never launches an intervention, whether the gate opens or closes.

| Operation | All three seeds |
| --- | ---: |
| Native constructions and selected loads | 3 each |
| Original full TRAIN epochs on disposable branches | 3 |
| Temporary native Adam attempts/completions at success | 114/114 |
| Original mean-own reverse collections | 114 |
| Additional route-own reverse collections | 456 |
| Total reverse collections | 570 |
| Native route forwards | 456 |
| B/D replay route forwards | 912 |
| First-batch O reconstruction route forwards | 12 |
| Total route forwards | **1,380** |
| Route backward equivalents | 912 |
| VALID/TEST calls and retained trained models | 0 |

The 12 O forwards are explicit additional reconstruction work beyond the proposal's basic 1,368-forward audit arithmetic. Each operation has a before-attempt and after-completion ledger entry; failures retain partial counts and partial scalar rows. Epoch coverage requires all 1,210 TRAIN positions exactly once, 38 native raw/eligible batches and zero skips. Model/bank copying, FP64 dot products, hashes and state restoration are additional measured wall/memory work, rather than free operations.

The command fixes two Torch threads, one interop thread, 600 seconds per seed and an absent fresh output directory. Root should enforce its whole-process 1,900-second limit and existing GPU/RAM/peer review before running. The original three BN fits took 1.317 hours for 464 epochs; those timings do not qualify this instrumented path. Actual per-seed CUDA peaks, lifetime process RSS, case CPU time and wall time are saved; overall wall includes preflight and construction. The same process means later RSS peaks are lifetime peaks, not independent case measurements.

## Source checks and review

`check_source.py` uses only the standard library. Six checks cover disjoint/exhaustive parameter ownership, exact work ceilings, attempted versus completed failure charges, the full gate denominator, one native step/original epoch-loop call, no exports/heldout calls, allocation preflight before Torch import, and unresolved globals. `SOURCE_CHECKS.json` records their result. They do not qualify native autograd/RNG/BN replay; the actual admitted audit must verify those properties and retain failures.

`INPUTS.json` binds the three saved RESULT authorities, original runtime and fixed gate. `COMMAND.json` supplies the exact finite argv with the original project interpreter. `MANIFEST.json` lists this package's reviewed source bytes. No generic launcher or new admission framework is introduced.
