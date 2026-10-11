# Peer whole-family readout watcher — source only

This package prepares automatic readout for the admitted fixed peer V3 family. It was not launched or numerically executed locally. No sealed source, completed qualification, or running acquisition was changed, and no partial predictions or dropout outcomes were read.

## Root launch

Root creates the execution directory `meanSAGE_attribute_peer_feedback_readout_after_whole_acquisition_root_20261011_v1`, saves the readout decision there, binds this package's manifest and decision receipts in `CONFIG.disabled.json`, and enables that config. The template already binds the actual acquisition CONFIG and OWNER_START receipts and derives its output paths, custody, runtime versions and V3 source manifest from that CONFIG. The watcher output directory must be fresh.

From the already verified authorized allocation repository:

```sh
.venv/bin/python -B -u /absolute/deployed/peer_readout_source_20261011_v1/watcher.py /absolute/readout-root/CONFIG.json
```

The program verifies hostname and sole GPU before project/config access. It is one finite watcher; no model job or qualification is launched. It only launches the existing V3 finite `owner.py` for the two readout stages.

## Completion order

1. Wait for the exact admitted acquisition owner's `OWNER_END.json`. A dead owner without that receipt or any failed exit stops readout. No partial panel, bank, trace or child log is opened while waiting.
2. Require owner exit0, direct worker wait/reap, worker/CUDA absence and matching start identity. Call the sealed reader's stdlib-only `close_panel`: all 30 roster banks, 48 selected optimizer states, 57 physical bodies, both role exports, exact custody and source bytes must close.
3. Launch one finite CAL owner for all 30 banks. CPU float64 Adam .05 takes exactly 500 updates per bank (15,000 total); clamp logT to [-5,5] and retain the latest tied minimum over steps 0–500. The positive common scalar is applied inside each member's softmax before the native probability mean.
4. Wait/reap the CAL owner process and require that it reaped its worker successfully. Require all 30 endpoint records and all 15,000 updates. One CAL owner covers the entire family; its complete closure is the all-CAL-owner gate.
5. Launch the finite score owner. The existing V3 reader releases all three complete DEV roles together only after those gates, then reports raw and calibrated accuracy, proper losses and the frozen screens. Raw native FP32 probability-mean accuracy is primary. No gate result is used to choose whether readout completes.
6. Wait/reap the score owner and retain a receipt of the complete 30-bank report. No additional study or manuscript action follows.

## Finite envelopes and failures

The disabled template proposes a 43,500-second acquisition closure wait, 1,800-second CAL worker and 900-second score worker, 30-second polling, and a 47,000-second total watch alarm. Each stage owner has its existing worker+60-second envelope; the watcher waits up to worker+65 seconds and has up to 20 seconds of bounded cleanup for its own stage owner/worker. Exceptional cleanup can extend beyond the total alarm by those 20 seconds. These are finite admission envelopes, not timing predictions. CPU reader workers enforce wall/RSS caps; no new CUDA model allocation occurs in these stages.

On any failure, `WATCH_END.json` and stage transport metadata retain the reason, owner identities, receipts and cleanup status. Existing partial files and logs remain. No retry, endpoint replacement or survivor subset is admitted. Any predecessor failure blocks DEV; a failure after the whole DEV release is retained without another launch. Cleanup only targets the watcher's own readout owner and its identified worker; an acquisition failure is retained for root rather than modifying that job.

Key outputs are `WHOLE_FAMILY_CLOSURE.json`, `ALL_CAL_CLOSURE.json`, both stage configs and owner transport receipts, `WATCH_END.json`, and, after complete success, `score_v1/DEVELOPMENT_REPORT.json`. `INTERFACE_INSPECTION.json` records the source audit; `SOURCE_BINDINGS.json` and this package's manifest preserve provenance.
