# Frozen structural private-risk pilot

## Question and readout

Does assigning private residual learning by native TRAIN structure improve the served mean-logit VALID AUC over **both** the paid uniform residual bank and a matched within-class permutation, while strengthening usable member differences?

The paid uniform residual pilot gave pooled AUC 74.518 ± 2.660%, genuine I4 74.847 ± 0.500%, mean-member AUC 74.560%, and pool-minus-mean-member AUC −0.043 points. Its pooled uniform-graph BCE was 1.952 versus I4 0.890. Common member ranking reversals averaged 664.7, while only 31 pooled losses had a correct member alternative. These are descriptive results on the same already-exposed VALID split. They motivate testing route learning; they do not establish that this intervention works.

After all six fits and collections succeed, report all three signed seed differences, their means and sample SDs. A positive three-seed mean for **both** structured-minus-uniform and structured-minus-permuted AUC is the bounded primary AUC signal. Report permuted-minus-uniform separately. Keep M1, genuine I4, original G0 and paid private-BN controls visible. No significance, independent generalization or novelty claim follows.

Report the usable-difference signature separately: individual/mean/worst member AUC and BCE, pool-minus-mean-member AUC, common member ranking reversals, losses despite a correct member, exact repairs of uniform-common errors, acquired correct-member alternatives still lost in pooling, new common errors, and strict harms over all aligned pairs. Preserve tied pairs with half AUC credit. Pairs and routes are correlated diagnostics, not independent repetitions. Increased disagreement or member coverage alone is insufficient. An AUC gain without the signature remains a result for the whole procedure.

Retain every signed native-reduction BCE and uniform-graph BCE contrast. There is no new loss veto, calibration, clipping, alternative pool, checkpoint rule or posthoc subset.

## Frozen methods and native settings

Six new fits only, in this order:

| Index | Seed | Private risk |
| ---: | ---: | --- |
| 0 | 101 | structural |
| 1 | 101 | permuted |
| 2 | 202 | structural |
| 3 | 202 | permuted |
| 4 | 303 | structural |
| 5 | 303 | permuted |

The immutable v2 producer supplies the exact six JOBs. Native TRAIN atom counts define four within-class quartiles with the retained fractional tie-overlap rule. Weights are `w = .5 + 2q` in [.5, 2.5], with exact original route/class mass and per-object mean one. The control permutes the complete four-route weight vector within class using the fixed isolated seed+7030031. There is no redraw or minibatch renormalization.

One unchanged native GIN backbone has five layers, width 300, full nine categorical molecular features, dropout .5, mean graph pooling and batch 32. Each update uses the original unweighted native objective F for all shared parameters, including BN affine and head, and weighted W only for the original rank-eight residual down/up coordinates. Four route forwards, two reverse collections and one original Adam transition are used. Native BCE, optimizer options/defaults, TRAIN order, route-private BN buffers, source preprocessing, split and mean raw-logit serving remain fixed. No additional learned parameters are introduced by assignment.

The 19,200 private learned coordinates use 76,800 FP32 parameter bytes and 153,600 dense Adam moment bytes if all receive gradients. Private BN buffers use 144,320 bytes and add no learned parameters. Actual memory peaks, serialized checkpoint/export sizes, maps and walls remain separate costs.

The unchanged controller evaluates complete VALID each epoch, maximizes AUC, retains the first strict maximum, requires at least 100 epochs, uses patience 50 and caps at 1000 epochs. Fresh native construction and zero-up residual initialization are required. No engineering checkpoint is a scientific starting point. No HPO, new control fit, extra seed, TEST or full-campaign admission is included.

## Engineering authority

Both actual v2 CPU qualifiers completed two constructions, two Adam updates, 73 native forwards, eight reverses including checks and three full VALID calls. They establish initial native parity, shared-F/private-W credit, BN counters, changed-state rejection and a fresh-restored bit-exact VALID result. The repaired restore contract preserves ordered live parameter IDs, group boundaries, every group option and defaults while allowing genuine parent restore to replace group dictionary objects. The v2 manifest and both actual authority files are pinned in SPEC.

The old v1 failure remains pinned: two constructions, 29 forwards, four reverses, one Adam transition and one VALID call before rejection of replaced group dictionaries. Its receipt and costs are preserved. CPU lifecycle checks establish engineering behavior, not GPU memory admission, speed or accuracy.

## Finite execution and publication gates

`run.py` defaults to preflight. It verifies the dedicated hostname/GPU before project reads, the pilot manifest, exact v2 descriptors, original runner validation, native/data/role pins, both actual v2 qualifications and all 39 paid control RESULT receipts. Allocation data and installed OGB checks remain actual remote preflight gates; source checks never open local tensors or data payloads.

Root reviews and publishes this source, records the actual publication commit and package/SPEC hashes in a fresh copy of ADMISSION_TEMPLATE, and admits explicit positive GPU/host reservation budgets, the shared slot lock and at most one older peer by PID/start ticks. The template itself grants nothing. The launcher requires the original repository `.venv`, the admitted checkout commit, one new serial GPU worker and a 14,400-second child limit per fit. The maximum serial training wall allowance is 86,400 seconds. Unknown GPU processes block admission. Any failure, timeout or close error stops later fits; no automatic retry or resume occurs. START, OUTCOMES, END, child logs and genuine RESULT receipts retain completed, failed and unstarted outcomes.

Maximum production work is 6000 epochs, 228,000 Adam maps, 456,000 reverse collections, 912,000 TRAIN route forwards and 120,120 fit VALID route forwards including finalization. These are caps, not expected work or resource guarantees.

## Selected VALID handoff

After the all-six terminal gate, root fills a fresh COLLECTION_ADMISSION_TEMPLATE with the actual END/OUTCOMES hashes, current pilot manifest, plan/script hashes, fresh collection/analysis roots and one-collector resource admission. Actual per-fit RESULT/checkpoint/export pins are taken from the terminal outcomes, never assigned in advance.

`selected_VALID.py --mode commands` displays six exact invocations of the unchanged selected-molecular collector. `--mode collect` runs them serially, for at most 600 seconds each, under the shared slot lock; failure stops later collectors without retry. Six collections require 120 native VALID route forwards and zero TRAIN/optimizer work. `--mode analyze` requires `CUDA_VISIBLE_DEVICES` empty and all six complete collections before importing tensor libraries. It loads only saved prediction/route tensors and constructs no model.

The saved selected mean raw logits remain the scoring authority. Reconstructed route means are disclosed sanity checks only; warnings never replace saved predictions or trigger retries. The three already-paid uniform residual route/export banks are receipted and reused for aligned pair analysis, without old-bank recollection.

The final addition contains 39 banks (33 paid + six new) and 45 unique training fits (39 paid + six new). It retains all old training and collection outcomes, six new training outcomes, six new collection costs, 33 total route collection outcomes and the historical original collection receipt. Do not add concurrent walls or peaks. No partial scientific summary is produced after a missing/failed fit or collection.

## Root next steps

1. Review this manifest, SPEC, six JOBs, source checks and unchanged producer.
2. Publish through the established source workflow; run allocation preflight.
3. Fill and hash a fresh training admission after reviewing live resources, then launch the finite serial runner if admitted.
4. Wait for the all-six terminal result; bind its genuine hashes in a fresh collection admission.
5. Collect six banks, then run CPU saved-prediction analysis and review all signed results and costs against the frozen readout.

This source preparation performed no remote action, Git operation, fit, model construction, native forward, optimizer update or TEST access.
