# Independent NCNC selected-state replay source review

Verdict: **CHANGES_REQUIRED for this exact packet**. This is a scientific-code source review, not a numerical replay result, runtime release, or manuscript acceptance review. No author-requested verdict was used.

Reviewed packet: `graph_ncNC_complete_family_selected_state_replay_preparation_20261004_v1`, MANIFEST SHA256 `1bb39193e8f9e33c4995d3567d12ea2e4fe5e886e93daad30716ee7f2e66b95f`.

## Material correction

**B1 — The final custody recheck omits source/release/authority inputs.** `replay_run.py:88–90` calls only `family_gate(context)`. That function reauthenticates the lock and unit artifacts, but does not repeat the source-manifest checks, release hash, data/runtime authority descriptors, or review/qualification/admission receipt bindings from `preflight` (`replay_gate.py:72–108`). It also does not rehash the actual TRAIN/raw/VALID files or runtime source/binary pins authenticated earlier by the original APIs. Consequently, the final step labelled `immutable_source_input_custody_recheck` can publish PASS and summaries without establishing the README's condition that immutable input custody still matches.

Smallest correction: factor a read-only custody verification helper out of preflight and invoke it before public success disclosure, alongside the existing family gate. Compare against the original admitted release hash and fixed descriptors; cover sidecar/four source manifests and payloads, original authority and three independent receipts, and the bound data/runtime input bytes. Keep the fresh-output check solely in initial preflight. A changed binding must leave all25 slots present and public metrics/contrasts null. No fitting, selector, tolerance, or endpoint change is needed.

## Qualification and execution scope

**Numerical qualification remains an unmet execution prerequisite.** The only supplied QA program is `stdlib_check.py`. It uses scalar snapshot trees, and does not invoke `trusted_load`, actual Torch/Adam/flags/RNG restoration, `replay_numeric.run`, real scoring routes, CUDA accounting, or the production failure/disclosure flow. The original `pilot_synthetic.py` qualifies the original driver and does not exercise this new sidecar. The new preflight accepts a separately bound PASS qualification receipt by schema/status/identity/hash (`replay_gate.py:100–104`); it does not itself establish those numerical checks.

The production CLI cannot be used to bootstrap fabricated-state qualification: it requires a qualification PASS receipt and the actual immutable family lock, and its data loader requires the original project data. This is not proof that qualification is impossible through a separate harness, but such a harness/source/invocation is absent here.

Before any actual replay invocation, supply and review a separately sealed ordinary-runtime qualification harness that exercises this exact sidecar core on trusted fabricated journal/selected states and fabricated queries. It must use the actual restore/digest/scoring/pooling/Hits50 APIs, cover all25 slots/40 nominal calls and the N64 versus I4 differing-selection case, and inject custody/state/raw-score/selected-metric/scoring/accounting failures to verify failure retention and null public summaries. Bind its exact source and invocation, ordinary-runtime identity, and check results in the qualification receipt. It must have an explicitly admitted synthetic entry path that does not require or read the study lock/data. The current stdlib PASS receipt cannot satisfy that prerequisite.

## Source findings that pass

- All20 unit identities, own outputs, closures, COMPLETE or explicitly retired FAILED terminals, completed-unit integer exit0 physical receipts, closed attempt ledgers, journals, and all25 selected artifact bytes are authenticated before the first Torch deserialization. Each checkpoint is hashed again on the file descriptor later passed to `torch.load`.
- Journal checks preserve all100 epochs,17 full TRAIN batches,60084 positives/100000 shared negatives,101 I4 candidates and four extra passes, original member seeds and five F4 paired initialization/stream/RNG histories. N64 is checked against native member0's individual best, independently of the served I4 state.
- The numeric loop reuses the original constructors, strict state/Adam/flags/RNG restore, complete TRAIN graph, canonical VALID queries, private or pooled-after-clamp routes, equal mean raw logits, and exact official strict-tie Hits50. It checks full restored state and unchanged model/optimizer/RNG during scoring. Exact raw-score and selected-Hits50 mismatches fail without tolerance, retry, retraining, or reselection.
- Every admitted cell is attempted; numerical failures preserve the25-slot denominator and suppress public metrics and family contrasts. Terminal-failed families take the metadata-only path. Original35 fits/3500 epochs/59500 updates, N64 fit reuse, closed-attempt costs, interrupted observed lower bounds, unknown remainders,40 nominal replay calls, CUDA peak allocation/reservation, and the disclosed unmeasured write tail are retained. CPU/resource capacity and actual CUDA accounting remain qualification/admission facts.
- No replay path trains or opens TEST. Original complete-family descriptive endpoints are returned only after all25 pass; no significance, novelty, continuation, or superiority threshold is created. Exact reproduction with deterministic_algorithms=False remains unverified.

## Evidence and limits

Independently verified SHA256 and byte lengths for all12 replay payloads and all76 payloads in the four pinned original source packets (driver21/design9/prototype25/resource21). Verified the data/runtime authority metadata and sealed minimal-plan descriptors. Parsed and compiled all6 replay Python sources with stdlib; no source was executed for numerical QA. Read executable replay and original state/model/data/scoring/fit/lock source, including factor routing and native evaluation iteration.

The expected lock descriptor is SHA256 `ab1d6a4b3a0fd02bcbb2db0a50764362cff7b20206eda6a805e89f8880393f50`. Its actual saved contents/bytes, selected values, checkpoints, arrays and data were deliberately not opened. No SSH, installs, numerical libraries, scientific executions, or canonical edits were performed. This review supplies no runtime or predictive verdict and no execution authorization.
