# Private GRU whole-family watcher — source only

Prepared by adapting `peer_readout_source_20261011_v1/watcher.py`. This package is disabled and unrun. Root reviews, admits and deploys it on the authorized allocation. The exact sealed GRU reader and acquisition source remain unchanged. No model, training, numerical qualification, array, partial outcome or remote call was used in preparation.

## Producer binding and completion order

The watcher pins the actual acquisition CONFIG receipt and saved launch metadata. Its copied `ACQUISITION_ACTUAL_LAUNCH.json` is deployed with this package. The live acquisition `owner_v1/OWNER_START.json` must match the pinned receipt and that launch's nested start: owner **640426 / start_ticks 6055039154 / pgid 640426**, child **640428 / start_ticks 6055039165 / pgid 640428**. Exact stage, worker, CONFIG, limits, output and owner-output paths are checked. Acquisition and reader manifests are distinct and separately sealed.

1. Wait only for that owner's end receipt; verify its successful exit, matching birth identities/command, direct child reaping and worker/CUDA absence. While waiting, read no partial panel, trace, child log or prediction output. A missing/failed owner end stops the watcher without modifying acquisition.
2. Load `runtime.py`, `arithmetic.py` and `reader.py` from the exact `native_meanSAGE_private_GRU_readout_source_20261011_v1` directory. Their top-level imports are stdlib only. Call its metadata-only `close_panel` after owner closure. Verify the panel's acquisition custody and stage1-before-stage2 provenance against the actual producer CONFIG.
3. Only now fill the reader's sealed disabled CAL template with immutable current receipts and fresh watcher-local outputs. Launch the sealed reader's finite owner. Record the produced CONFIG receipt and match actual owner/worker start/end identities, exact command and output paths; wait/reap the owner.
4. Verify every one of 39 bound temperatures: 33 fresh 500-update fits (16,500 updates) and six unchanged inherited endpoints (3,000 bound updates). Bind the temperatures artifact to this CAL CONFIG, reader source, source custody and closed CAL owner. Write the original literal fitted-CAL confidence guard from all complete `CAL_minimum_NLL` records. Only then fill the sealed score template and launch its finite owner; the guard never chooses whether score proceeds.
5. The unchanged reader makes one whole DEV release after all CAL closure. Wait/reap score and bind its complete 39-row/156-contrast report to the owned score producer. The watcher checks closure metadata and receipts; scientific gate results never decide whether handoff proceeds or watcher completion succeeds.

The inventory remains **51 cohort trajectories, 48 new trajectories and 12 additional external WI4 trajectories**. GA4 has no fit; nine extra P1 ancestors remain paid diagnostics. No inherited endpoint is refit or re-exported. No training, rescue, retry or additional qualification is offered.

Root's frozen `private_GRU_confidence_population_clarification_root_20261011_v1` preserves two confidence populations. `CAL_CONFIDENCE_GUARD.json` reports the specification's original fitted-CAL NLL rule (not independent evidence). The sealed reader's calibrated DEVELOPMENT NLL verdict is separately labelled added heldout confidence protection. Both use each-block +0.005 nat and candidate mean no worse than the best reference mean. Both remain visible, neither replaces the other or affects handoff/raw/member continuation.

## Root admission

Deploy this complete package and the separately sealed reader source. Copy `CONFIG.disabled.json` to the fresh root execution directory `private_GRU_readout_after_whole_acquisition_root_20261011_v1`, fill the root-decision and watcher-manifest receipts, and enable only that external CONFIG. Keep package templates and sources sealed.

The root decision must have `admitted_stages` containing `watch`, `calibrate` and `score`, `v2_spec_sha256` equal to the fixed v2 hash, and exact copies of these CONFIG fields: `watcher_manifest`, `acquisition_config`, `acquisition_owner_start`, `acquisition_launch`, `reader_source_manifest`, `confidence_clarification`, `output_directory`, `wait_seconds`, `poll_seconds`, `whole_watch_wall_seconds`, `calibrate_limits`, `score_limits`. The watcher verifies each field before waiting or launching a reader.

From the already verified allocation repository, root launches:

```sh
.venv/bin/python -B -u /absolute/deployed/private_GRU_whole_readout_watcher_source_20261011_v1/watcher.py /absolute/readout-root/CONFIG.json
```

## Finite envelopes

Wait **604,800 seconds**, poll every 30 seconds, CAL worker **7,200 seconds**, score worker **7,200 seconds**, whole watcher **619,380 seconds**. Each CPU reader carries **16 GiB RSS / 32 GiB CUDA** caps and its sealed owner allows worker+60 seconds; the watcher waits worker+65 seconds. These are finite limits, not runtime forecasts.

Twenty seconds of bounded cleanup are reserved inside the whole envelope: the alarm fires at 619,360 seconds and exceptional TERM/KILL/wait cleanup targets only this watcher's own readout owner and identified worker. A stage is not launched without enough remaining inclusive budget. Acquisition processes are never killed by this watcher. Failed/partial artifacts remain labelled and no retry follows.

Key outputs: `WHOLE_FAMILY_CLOSURE.json`, `ALL_CAL_CLOSURE.json`, `CAL_CONFIDENCE_GUARD.json`, generated stage CONFIGs, stage owner transport/start/end receipts, `WATCH_END.json`, and the complete `score_v1/DEVELOPMENT_REPORT.json`. Root remains responsible for actual admission and execution; this source packet certifies no numerical result.
