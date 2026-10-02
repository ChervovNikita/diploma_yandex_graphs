# Full-node cotangent support initializer v1

Source-only amendment of the pinned Round17 v3 precision initializer. This packet contains an initializer, source fixtures and documentation. It supplies no training driver or study orchestration. It is a one-time operation at a common identity-factor warm boundary.

## Source status

- `base/graph_band_route_initializer.py` is the exact original source, SHA-256 `1a8036c7bbf9f2b831747f99d3aa2dfdabc31cb636cd41ccad9457206a70cbcf`.
- `prototype/graph_band_route_initializer.py` adds a support choice and full-output construction diagnostics.
- All 11 files in the active Round17 v3 packet are bound in `ACTIVE_SOURCE_BINDINGS.json` and verified unchanged before and after the stdlib fixture.
- **97 stdlib checks passed.** The candidate was parsed, not imported. Torch numerical fixtures were authored and syntax-checked, **not run** by this packet's Mac author. No native model, scientific data, quality score or GPU execution was used.

## API

```python
slices, stats = initialize_four_routes(
    logits_fn, theta0, S, target_nodes, train_rows, train_labels,
    tangent_mode='graph', control_seed=None,
    cotangent_support='full_node',
    homogeneous_full_node_outputs=True,
)
```

Use `cotangent_support='train_remasked'` for the pinned support comparator. Both modes require the same deterministic full-node class-logit closure and an explicit semantic assertion that its outputs are homogeneous predictions. `target_nodes` must contain every graph node exactly once; row permutations are allowed. `train_rows` indexes that output-row order, and `train_labels` contains only those TRAIN labels. A tensor shape alone cannot certify homogeneous output semantics.

The task gradient remains exact TRAIN mean cross-entropy. Projection, centering, common cap, factor radius, branch search, TRAIN functional separation, member TRAIN CE and pooled TRAIN CE acceptance expressions are preserved. A full-support direction visible only at unlabeled output roots can therefore fail the existing TRAIN separation guard. Such a fallback does not disprove the broader support hypothesis.

The retained signed diagnostic uses **shared full-node q in both support arms**. Its provenance fields identify the remasked arm as a cross-support diagnostic. No support-matched remask signed functional is computed here. Neither signed value nor full-output Gram/pair RMS is an acceptance gate or a claim of predictive gain.

Each invocation compares its finite candidate with common descent at that invocation's exact same alpha. The source does **not** choose a shared alpha across support arms. A paired study needs a separately reviewed integration that freezes the common warm state and uses a bounded shared-alpha procedure; independently accepted invocations cannot be called matched.

## Verification commands

Executed, stdlib only:

```sh
'/Users/alex/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3' -B '/Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/graph_full_node_cotangent_support_v1/fixtures/stdlib_checks.py'
```

Authored future CPU fixture, requiring a qualified Torch runtime; **not executed here**:

```sh
python -B '/Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/graph_full_node_cotangent_support_v1/fixtures/torch_synthetic_checks.py'
```

The Torch fixture uses a synthetic 7-node nonlinear closure and a deterministic 4-node margin closure. The latter compares the returned accepted steps and output Grams against an explicit full-output `jacrev`/dense-band reference, with a guaranteed nonzero support delta. A future pass would check derivative implementation and call accounting, without qualifying a native graph model or scientific outcome. Re-running the stdlib fixture rewrites `STDLIB_RESULTS.json` with deterministic contents; verify the manifest afterward:

```sh
'/Users/alex/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3' -B '/Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/graph_full_node_cotangent_support_v1/fixtures/verify_manifest.py'
```

## Packet map

`API_DIFF.md` gives the exact interface/output changes; `initializer.diff` is the complete unified source diff. `MATH_AND_LIMITS.md` states the support algebra and attribution limits. `COSTS_AND_FAILURES.md` accounts for calls and memory. `CHANGED_OPERATIONS.json` records machine-readable changes. `SOURCE_BINDINGS.json` binds the saved mathematical inputs. `MANIFEST.json` hashes the payload; `SEAL.json` hashes the manifest. The seal describes this source snapshot, not execution admission.
