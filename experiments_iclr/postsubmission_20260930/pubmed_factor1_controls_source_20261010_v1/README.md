# Disabled PubMed M1 factorized controls

This is a new exploratory check of a competing explanation for the completed shared-own result: a single predictor may benefit from redundant factor-coordinate optimization. Two controls×seeds9101/9203/9307 use the unchanged native full graph, split190111, slow initialization, original Adam groups and own raw-logit strict-first2000/250 selector.

- `factor1_native`: one factual dropout CE and one Adam step.
- `factor1_mean4_dropout`: mean of four factual dropout CEs at the same parameter state, followed by one Adam step.

Both use the existing factor adapter at all23 native affine sites, one unit R/S row per site, bias outside output scaling. Every original slow parameter object is retained, with no reset after wrapping. There are2069875 original coordinates plus13943 added coordinates, totaling2083818. Fixed M1 factors can fold into native affine matrices in real arithmetic; this tests a reparameterization and training explanation. Added coordinates also receive the original name-based weight decay. It is not a pure optimization-only causal claim, a native parameter-count match, or a full M4 parameter match: M4 adds55772 factor coordinates.

## Small source delta and reused routines

`adapter.py` binds the exact strong-reference ordinary/four-loss factory, installs M1 factors before any update and rebuilds the same native name-based Adam groups to include them. The original factory's empty Adam is unused and discarded; both constructions are charged. All slow and factor coordinates receive the ordinary scalar loss gradient. No block routing, masks, CORE, reconstruction loss, CMCL, teacher, router or HPO is added.

`run.py` uses a scoped factory substitution for the exact existing strong-reference `fit_body`, full-array/provider loader, metrics, selected restore and3-warmup/10-measured serving benchmark. Its internal aliases (`single_native`/`single_mean4_dropout`) are explicit; actual completion/roster IDs identify M1 controls. Initialization seeds and four persistent factual dropout streams match the references. Full TRAIN11829, VALID3942,19717x500 graph and88648 ordered edges are hash/fingerprint bound by the existing receipts.

## Minimal root interface

1. Stage this packet alongside exact strong-reference source V2 and its existing dependencies. Root reviews `adapter.py` and the factory substitution in `run.reference_surface`/`run.execute`, using `SOURCE_REVIEW.json`, `SOURCE_BINDINGS.json` and `PROTOCOL.json`; no new audit packet is required. Create actual root source/delta approval from its disabled template, exact queue approval and fresh observed readiness with32GiB free. Root will schedule these controls after the active allocation family to keep acquisition costs separate.
2. Invoke the existing interpreter from the allocation repository cwd with `render.py --purpose engineering`. Optional source-review/owner-review/readiness path arguments select the actual receipts. It renders two seed9101 TRAIN-only releases and starts nothing. Run the resulting plan with `queue.py --plan ABSOLUTE_PLAN --plan-sha256 EXACT_SHA`. Each interface performs one complete TRAIN update, one full factual prediction, then a fresh M1 factorized construction/weight restore and prediction replay. Exact TRAIN predictions/correct masks and2e-6 restore-only float checks are required; VALID and TEST stay closed.600 active seconds plus10 shared cleanup seconds per interface.
3. After both actual completions and raw direct waits/owned absence close, invoke `render.py --purpose science`, then the same finite owner with its generated science plan. All six source fits remain max2000/patience250,9000 active seconds plus10 shared cleanup seconds each. Workers use Python3.11 `-B -P`; the existing resource/direct-wait/shared-cleanup/absence logic is reused. Renderers retain sibling imports. Outputs stay in this packet's separate engineering/science namespaces and never overwrite a previous attempt.

The direct worker API is `run.py --mode engineering|science --release ABSOLUTE_RELEASE --release-sha256 EXACT_SHA`. No renderer, model, array, server or fit call was made in preparing this packet.

## Compute and interpretation

The scientific envelope is12000 epochs,30000 TRAIN forwards/backwards,12000 Adam steps,12000 regular VALID forwards and84 selected/benchmark forwards. Six models/preprocessing banks and twelve Adam constructions are charged. The two qualification interfaces add four model/preprocessing constructions, eight Adam constructions, five TRAIN forwards/backwards, two steps and four serving/replay calls. Checkpoints, input/imports, readouts, owner/cleanup/transport and every failure remain additional costs.

Using the historical shared-own cycle time for every M1 epoch gives a rough all2000 planning proxy of about89 minutes. It is not qualified M1 throughput or an ETA. The actual two complete-input qualifiers provide update/serving rates; native preprocessing, validation, I/O and ownership remain separate. The full science active-plus-cleanup cap is54060 seconds; qualifier cap1220 seconds. Observed GPU/RSS peaks remain unmeasured here.

Close all six new records and actual owner custody before comparing quality. Compare M1-own against the completed native single, M1-mean4 against the completed four-loss single, retained M4 shared-own against each M1 control, and M1-mean4 against M1-own. Preserve all paired seeds, accuracy/NLL, classes, macro and repair/coverage readouts, plus costs. No confirmation, pure coupling causality, paper-score recalculation or new grid is claimed. Current CMCL18/77 partial quality was not read.
