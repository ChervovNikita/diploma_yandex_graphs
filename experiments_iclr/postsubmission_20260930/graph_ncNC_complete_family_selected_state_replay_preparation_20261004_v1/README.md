# NCNC complete-family selected-state replay source

This packet adds the missing numerical selected-checkpoint replay pass to the original NCNC driver. It does not fit, resume, choose a checkpoint, create a family lock, or authorize execution. The supplied root release template is disabled.

`replay_run.py` accepts only `--root-release` and `--output`. A separately reviewed, qualified, root-authorized release must bind one fresh output and one GPU UUID. There is no resume or retry option. Execution outputs are private, external to every sealed packet and fit directory.

## Gate and replay

1. The stdlib gate verifies this exact manifest, the four original source manifests, original TRAIN/raw/VALID and ordinary-runtime authorities, separate sidecar review/qualification/resource receipts, the immutable family lock and all20 original unit identities. It checks closure markers, exact COMPLETE or explicitly retired FAILED receipts, physical zero exits for completed units, every closed attempt ledger, own journal JSON and selected-state bytes. No Torch or project data is opened here.
2. Complete units must contain all100 ordered epochs, full17-batch TRAIN and official60084/100000 VALID coverage, correct fit seeds,35unique fits,25cells and I4 candidate101/four extra passes. Five F4 pairs must match initial snapshot/RNG and every100epoch stream/start/end RNG. Pending, duplicate, borrowed or missing slots fail closed.
3. A terminal-failed immutable family produces only a metadata/cost failure receipt with all25 slots and null contrasts. It never imports the numerical runtime or replays a subset.
4. Complete-family replay imports only the pinned original `pilot_common`, `pilot_model`, `pilot_data`, `pilot_state` and `pilot_evaluate`. It reauthenticates each open checkpoint file before deserializing that same file descriptor. All20 journal payloads and all25 selected snapshot contracts must pass before any scoring or data load.
5. Journal selectors are rechecked with the original strict-improvement/first-tie selector. N64 is the individually selected member0 snapshot in its own native-bank journal; it is permitted to differ from member0 in the served I4 checkpoint. I4 checks its separate101-candidate selector and its four selected states.
6. Original model constructors and state helpers restore strict model/Adam/flags/RNG trees. No optimizer update is called. Full snapshot digests must match before scoring; model/optimizer and RNG must remain unchanged by evaluation. Original `score_valid`, `mean_native_scores`, `evaluator` and `hits50` retain complete TRAIN graphs, raw-logit pooling, official query order/tails and strict ties.
7. Original per-model raw score digests, where retained in the journal, must match exactly. The pooled I4 digest was not exported by the driver and is not invented. Every selected Hits50 must equal the original selected value exactly. A mismatch remains a failure; no tolerance, replacement, retraining or calibration exists.
8. All25 slots are attempted without retry after successful snapshot admission. Nominally40 `score_valid` calls serve5N64+20I4+5privateF4+5pooledF4+5N70; each F4 call serves four members. An execution failure can leave fewer calls, which are retained against the full denominator.
9. Private intermediate details remain in the mode0600 output receipt. The public result exposes metrics and original complete-family summaries only if all25 pass and immutable input custody still matches. Otherwise all public metrics and contrasts are null. Progress stdout/STATUS contains only counts and terminal status.

## Endpoints and costs

Successful replay returns the original lock's five paired private-minus-pooled development differences, mean/sampleSD/range/sign counts and allfive per-arm summaries/selectors. Baseline comparisons retain every seed for N64/I4/N70 and both F4 arms and are exploratory descriptive comparisons. There is no margin, significance, continuation or novelty rule. VALID was used for checkpoint selection and these contrasts; the unit is training randomness conditional on one graph/time split. No TEST accessor is present.

Original training denominator is35unique fits,3500fit epochs and59500updates. N64 reuses the exact I4 member0 fit and does not add five fits. All closed original attempt inclusive wall times are retained; interrupted observed lower bounds and unknown remainders are reported separately. Candidate101 work is charged once to native-bank fitting. Replay custody hashing, loading, construction, restoration, transfer and complete scoring are separately timed, with fresh CUDA peak allocated/reserved memory. The final accounting write tail is disclosed as unmeasured.

## Verification and release status

`stdlib_check.py` uses fabricated metadata and scalar snapshot trees only. It checks full/failed family gates, missing/pending/duplicate/borrowed units, coverage, candidate101, F4 RNG, byte custody, closure, selectors, N64 alias, complete snapshot digest mismatches and summary suppression. Python sources are parsed/compiled; Torch, NumPy, project data, checkpoints and current outcomes are not imported or loaded.

This is source preparation. It requires independent source review and admitted ordinary-runtime synthetic replay before root release. Numerical QA must cover trusted fabricated Torch snapshots, strict restore/digests, full routing, private versus pooled-after-clamp, equal raw-logit mean, official strict Hits50/ties, coverage/tails, exact mismatch retention, memory/cost accounting and all25 failure suppression. The frozen runtime has `deterministic_algorithms=False`; exact raw-score reproduction and numerical replay success remain unqualified runtime facts.

Root executed the original family lock once at2026-10-03T21:14:47UTC. The recorded immutable lock descriptor is included as provenance, without copying its outcomes into this packet. Do not execute the original lock again. Root must obtain and bind all20 postclosure journal/attempt/closure/physical-terminal receipts and fresh resource admission before one numerical invocation.
