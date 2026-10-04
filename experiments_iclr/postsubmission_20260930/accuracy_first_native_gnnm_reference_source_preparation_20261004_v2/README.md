# Ordinary native GNNM with all TRAIN labels

**Separate source-only quality/augmentation reference; no scientific execution or novelty claim.** This supplies the missing normal native GNNM at the same full official TRAIN budget. The sealed four-bank and native-reference packages remain unchanged; no fifth bank condition is added to either protocol. V6's FIT-only GNNM fits are ineligible as this all-TRAIN control.

For all three fixed blocks `(split,seed)=(0,17),(1,29),(2,43)`, the caller uses byte-bound bank v2 `build_bank('tied_persistent',...)`. The modulations, backbone, constructor/reset/wrap order, optimizer, private R/S/B, trainable shared parameters and initialized effective functions are identical. The only new training operation is `native_train_step`: four complete native member forwards in member0..3 order, each TRAIN CE/4 backpropagated immediately, followed by one Adam step. Its objective is the arithmetic mean of four native CEs, with no probes, learned gate or other method.

`bank.py`, `runtime.py`, `schedule.py`, `views.py` and `bank_driver.py` are exact bound reuse copies from sealed bank v2; `SOURCE_BINDINGS.json` records predecessor/current equality. Existing build, gradient/ownership, snapshot, restore, transition, evaluation, probability-pool selector, reporting and image-writing helpers are called directly. Neural bodies still load from immutable v6 by exact hash. Portable paths resolve against `ROOT.parent`; no original source is edited or given bytecode-cache writes.

The new thin driver keeps one pooled native VALIDATION strict correct-count selector, earliest ties, update1 as first candidate and one best across 200 local +2500 global updates. At actual200 it restores selected-local model+Adam while retaining live RNG and actual clock; final restoration preserves the selected stage, including a local winner. Input roles must expose all12246 official TRAIN labels, raw features/native graph must pair with the banks, and VALIDATION labels are selector/reporting inputs only. No TEST-label reader or acquisition exists. A caller-owned paired protocol must bind both exact source manifests; that protocol remains pending here.

`python3 -B native_driver.py` prints source-only metadata and imports no numerics. Deliberate runtime/scientific calls require `execute=True` with exact source/protocol and paired input context. No queue, approval ladder, host restriction or scientific launch is created.

The four retained stdlib checks cover fixed blocks/stages/pass accounting, pooled earliest ties, import exclusion and portable descriptors. Deliberate fixtures stay inside project ROOT. `test_numerical.py` contains two **unrun** synthetic CPU objective/state checks: both stages' four streamed CEs versus the summed mean4 loss/gradient/one-step parameters and RNG/buffer semantics; full state/function restore, exact next-native-update replay and selected-local transition/local final restoration. Small fixture dimensions verify implementation, not full-shape performance or predictive utility. Bank v2 numerical verification does not qualify this new objective/driver.

Charge10800 training forwards/backwards and10800 native selection forwards per block, plus preparation/checkpoint/restoration/reporting/host costs. The augmented banks use eight training passes and a native-plus-probe coefficient total2; this reference uses four passes and total1. Additional compute, dropout-stream consumption and probe supervision therefore differ, and native versus augmented accuracy alone cannot isolate semantic-view roles. Initialization and dropout policies are matched; ambient streams are not claimed identical after different pass counts. The untied/shuffled/random-null controls and competent same-budget singles/conventional ensembles remain necessary.

Exact numerical qualification, joint paired protocol and worthwhile-gain rule, full-shape/resource measurements, verified input provenance and confirmation/consumed split/TEST-history rules remain pending. TRAIN metrics are fit diagnostics; selected VALIDATION accuracy/NLL/macro-F1/member competence are exploratory development evidence. Parameter reduction is secondary to cleaner ensemble prediction accuracy.

## V2 admission repairs

This preserved v2 successor repairs the fresh source review and applies the same paired-input requirement to both quality-reference packages. V1 sources and their review/results remain preserved. The mathematical objectives, model operations, seeds, optimizers, clocks, selectors, restoration operations and numerical test code are unchanged.

Created output directories and checkpoint paths are resolved to absolute paths at creation. A checkpoint descriptor therefore reopens the same file after the caller changes working directory. Bound source descriptors retain their portable phase-relative encoding.

Fitting admits CPU and selected-device CUDA only. Unsupported device types, including MPS, are rejected before feature movement, model construction, RNG seeding or fitting. The ordinary native builder has the same guard. The unchanged bank helper's broader parser is not a claim of reference support for other backends.

The future caller-owned paired master has the exact source fields `bank_manifest_sha256`, `reference_manifest_sha256`, and `native_gnnm_manifest_sha256`. Each package verifies its own exact manifest and the bank manifest. It also requires `paired["input_bindings_by_split"][str(split)]` to be an object containing these four fields:

| Field | Required value |
| --- | --- |
| `role` | The live full-TRAIN `role.identity()` |
| `native_edges_sha256` | The live verified bundle's native-edge digest |
| `feature_identity` | The caller context's feature identity, already checked against the live features |
| `validation_labels_sha256` | The caller context's label digest, already checked against the live VALIDATION labels |

A missing split mapping/entry/field or any unequal field is rejected. The ordinary reference also retains exact eligible official-TRAIN coverage-file verification. No actual input bindings or future paired master are produced here; test identities are explicitly synthetic.

Exact v2 numerical checks are **unrun**. Root reported the unchanged v1 small CPU objective/state suites passed (four ordinary checks and two native-GNNM checks). Root also reported that default full-shape CUDA kernels failed exact function/next-update replay despite restored state, with evaluation differences around 1e-5; deterministic CUDA runtime qualification is pending. Those reports do not qualify this v2 source or prove exact GPU replay. No deterministic policy is introduced by this source repair; the prospective all-arm policy remains root-owned.

V2 stdlib verification passed 8 checks, including integration checks for all three split entries, absent/conflicting input/source bindings, relative-path round trips from another working directory, duplicate/missing/conflicting paths and unsupported backends. Deliberate temporary fixtures remain inside this package and are removed. No numerical imports, datasets, servers or predictive results were accessed.
