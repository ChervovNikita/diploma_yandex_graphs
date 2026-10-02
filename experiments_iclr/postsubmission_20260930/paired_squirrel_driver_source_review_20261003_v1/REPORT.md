# Independent Squirrel continuation driver source review

Date: 2026-10-03  
Verdict: **Two material findings. Preserve this v1 seal; fix and reseal before execution.**

Reviewed script SHA-256 `0886faf04a858e35a2c22888d7c9e3391103e30aa939dc2525df0e062a69b307`, under manifest `65815e691679e7edef1dd27e259416d9a3ff7bd4a016c0fea43d409b975445ce`. All seven payload sizes/hashes verified. Review used the packet's source/metadata plus the specific reused source primitives needed to trace its calls. No numeric originals, label values, checkpoints, fitted outcomes, native/Torch/GPU execution, or remote access were used. No broad check suite was copied or run.

## Findings

### B1 — P1: Every admitted block fails at the validation descriptor boundary

`prototype/continue_squirrel.py:257` passes `cell['validation_labels']` to the pinned `driver.load_labels`. All three sealed cell descriptors include `kind: "validation_labels"`. The pinned `graph_init_driver.py:50–52` verifier accepts only `path`, `sha256`, and optional `bytes`, and `load_labels` calls that verifier before loading the compact pack.

Consequently, after successful TRAIN qualification and all four installed-state checks, each block deterministically raises `ValueError: Path/hash descriptor with only optional bytes required`. No continuation fit can begin. The exception becomes a block qualification failure, making the twelve-fit trigger unevaluable after paying the preparation costs.

**Fix:** Keep the kind-bearing record for the new driver's scope/preservation checks, but pass the exact already-bound `cell['context']['source_labels']['validation']` descriptor, or an explicit `path`/`sha256`/`bytes` projection, to the unchanged pinned loader. Preserve the late admission point and exact row binding.

A focused stdlib check extracted only `require`/`verified` from the pinned source and applied the three sealed descriptors. All three produced the exact rejection, before any input hash/open callback was reached. No label bytes were accessed.

### B2 — P2: The paired helper's actual imported base source is outside custody

`prototype/continue_squirrel.py:365` loads the paired helper after verifying its own descriptor. That helper immediately imports `graph_full_node_cotangent_paired_alpha_v1/base/graph_band_route_initializer.py` at its lines 13–16. `FROZEN_STUDY.json.original_records` omits this exact base path. The main driver hashes the paired manifest but never verifies its payload, and the pinned `source_guard` covers the Round17 runtime rather than this separate paired base.

The imported base therefore participates in scientific construction without a before-import fingerprint check or the driver's before/after preservation check. The bound support prototype is a different path and does not establish this copy's bytes. A changed base could be executed while every declared input check still passes.

**Fix:** Add the exact paired base descriptor from its already sealed manifest to the frozen source/preservation records, so it is verified before helper import and after execution. Verifying the paired manifest's source payload before import is also sufficient for admission, with this actual base retained in preservation accounting. No new numerical qualification gate is needed.

The current local base matches its paired-manifest hash, `25b5e55b5150a7ee00a10d28d02b97fa689276ffd1f4326011a13209becfbabb`; no modification was observed. The finding concerns the missing execution binding and preservation path.

## Other requested paths

- **Admission and freeze:** The earlier design clarification is implemented. The amended policy and root-choice hash bind staged mode/all six donors before prospective native work. The native branch checks the manifest, choice, independent release, and fresh run name, then writes the admission marker before loading the scientific runtime. Existing markers refuse another execution. Fingerprint/resource preflight paths precede the native release branch as documented.
- **TRAIN qualification/resource semantics:** Each ordered block constructs a TRAIN-only context, checks its exact warm specification/preprocessing, runs full-output identity/precision AD and exact permutation correspondence, and invokes the paired helper once. The persistent observed allocator sentinel overrides returned acceptance/failure; cause/context classification and geometry-abort reports are retained.
- **Optimizer/state handling:** The disposable actual-warm Adam audit freezes every new R/S and checks the one-step transport on copies. Useful arm copies have R/S trainable. Each arm transports the same named warm optimizer history before installing its slice, verifies warm/installed outputs, restores post-warm RNG, and serializes its initialization. Fresh fit copies restore that exact state, named optimizer, and RNG after cloning.
- **Selector and labels:** Validation is admitted only after all four installed arms pass. Once B1 is fixed, the reused continuation selects epoch 0 or the earliest strict predictor-NLL minimum, trains mean member CE, pools mean raw logits, retains selected optimizer/model state, and verifies selected-state NLL replay. No final-label loader is called; full-node saved logits contain predictions only.
- **Terminals and preservation:** All three block returns precede `development_gate`; it scores only twelve selected fits from completed blocks. Failed/deferred blocks yield an unevaluable gate, with no subset average or Photo launch. Fit exceptions retain per-arm costs/errors, block exceptions fill pending terminals, and original-preservation failure closes comparison eligibility. Writes target the fresh packet run root; old phase/registry/cohort mutation entrypoints are absent. B2 is the identified preservation omission.

After these two fixes, review the new seal's focused delta and descriptor boundary. This source review supplies no native correctness or predictive-quality result.
