# Synthetic implementation qualification harness

This packet tests the preserved curvature-selector v3 implementation and the saved prospective constants. GNNM's broader objective is predictive accuracy through ensembles; storage is secondary. These implementation checks establish no accuracy, predictive gain, novelty, actual-warm compatibility or cost result.

## Run

From the research phase directory on a host whose existing Python has the requirements:

```sh
python3 -B graph_curvature_selector_numerical_harness_preparation_20261004_v2/launch.py --report HOST_SYNTHETIC_RESULT.json
```

The exact absolute local command is:

```sh
/usr/bin/python3 -B '/Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/graph_curvature_selector_numerical_harness_preparation_20261004_v2/launch.py' --report SYNTHETIC_RESULT.json
```

For source pins, frozen-constant validation and cache checks only:

```sh
python3 -B graph_curvature_selector_numerical_harness_preparation_20261004_v2/launch.py --stdlib-only --report HOST_STDLIB_RESULT.json
```

Use a fresh report filename for every run. Existing results are rejected before work begins. Exit 0 means the requested checks passed; `--stdlib-only` still labels numerical checks unexecuted. Exit 2 means numerical requirements are missing. Exit 1 records an import, source-pin or synthetic-test failure. The report always distinguishes numerical execution from preparation.

## Requirements and source custody

- CPython 3.9+; NumPy; PyTorch 2.0+ with `torch.func`. Tests run on CPU. CUDA RNG inventories are additionally checked if CUDA is available.
- PyTorch Geometric is unnecessary for these synthetic checks. Native modern-adapter/model modules are pinned and read for integrity, but are not imported or exercised.
- Preserve the phase-relative layout: this harness, `graph_curvature_selector_source_preparation_20261004_v3`, `graph_curvature_selector_prospective_constants_root_20261004_v1`, and the bound donor files listed in `SOURCE_PINS.json`.
- Source manifest: `d53d6963b3c66cb59061c4f5be6a81108adf8e9792a3f3cf93daea56fe35ec70`. Constants SHA256: `633ea814bfc82c094ecb2d98699e77258f0f644d13cf17fa7f8d65691cf637dc`.

The launcher verifies the v3 manifest/seal, all nine v3 payloads, eight runtime source pins and the complete frozen-constants file before and after a run. Selector/driver entry modules and numerical test code are compiled directly without bytecode caches; the preserved v3 loader loads only the integration, initializer and boundary roles for numerical tests.

Temporary cache/state fixtures are created only under `<harness>/fixtures/cache_*` and `<harness>/fixtures/state_*`, then removed. No global temporary directory is used. Synthetic serialization files are named `synthetic_state.pt`; they contain toy tensors, never native warm states or real checkpoints.

## Implemented checks

The locally executed stdlib checks verify unverified same-path cache rejection, exact executed-digest reuse, wrong-digest rejection, cache cleanup after failed execution and rejection of a different native API before acquisition. They also validate the frozen constants and all pins.

The twelve numerical cases cover exact Python/NumPy/Torch RNG roundtrips; named/aliased nonzero Adam-history transport and synthetic checkpoint restoration; discard/restoration after an intentionally partial failing trial; the 512-coordinate Photo final-head closure; and FP32/FP64 checks of independent one-step trials, ordered rank/null behavior, D matching, caps/finite guards, topology-permutation identities and all five returned arm states.

The trial oracle independently forms the mean of four member losses and evaluates coupled-decay Adam/AMSGrad equations. It checks the private gradient factor 1/4, live shared/private gradients and the source trial's updated tensors, buffers and Adam state. A test observer holds transient trial clones solely for comparison; production receipts are checked to contain no tensor state. Prototype state, gradients, optimizer snapshot and native RNG remain unchanged.

D tests use a declared synthetic affine class-logit map with three independent response directions, the original frozen radius/cap/tolerances, and no substitutions. A small fixed radius list measures synthetic monotonicity only; it does not change scientific candidate enumeration. End-to-end control tests invoke the actual selector once on a ten-node toy boundary family, retain all null/abstaining arms, check three slots per span and at most ten trial starts, reconstruct selected pre-trial slices, and compare original optimizer/RNG return custody.

The PolyFormer-shaped toy has 10 nodes, 2 tokens, 4 inputs, 256 hidden coordinates and 3 classes. Its body uses a vector gain, an alias and a buffer. The Photo closure uses 4 synthetic hidden rows of width 512 and 3 classes. These fixtures preserve the admitted head dimensions while avoiding real backbones, data acquisition, warm updates, continuation or predictive training. Nonzero optimizer history is assigned analytically.

## Preserved predecessor and corrections

The original v1 packet and its server result remain unchanged: seven cases passed and three failed in 3.440172s supervisor wall time, with 0.418682s summed failed-case wall time. Its two full control tests never reached the five-arm return assertions.

V2 corrects only the restore factory dtype and the default connected-TRAIN fixture. The checkpoint factory now constructs the saved fixture dtype before state loading. The default toy topology adds the declared TRAIN chain (0,2),(2,4),(4,7),(7,9), enabling the first graph-filter term. All weights, labels, seeds, scientific constants and selector/protocol bytes stay fixed.

The exact v1 independent-TRAIN topology is retained by `connected_train=False` and exercised in two separate FP32/FP64 regressions. They require projected rank at most two, retain all nine candidate nulls and check all five common-init model/Adam/RNG returns. This counterexample is mathematically valid: TRAIN remasking annihilates S, so cubic rows span only g plus the S²/S³ images; projection removes g.

The v2 Python files were parsed/compiled locally with stdlib. Root has authorized one distinct CPU execution on the previously verified anogena-2 route; results belong in the v2 execution packet. Neither an engineering pass nor real-warm/predictive qualification is claimed by this preparation packet.
