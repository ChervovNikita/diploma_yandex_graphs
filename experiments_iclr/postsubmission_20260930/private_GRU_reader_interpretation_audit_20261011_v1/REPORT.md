# Private GRU reader: interpretation and admission audit

11 October 2026. Read-only source audit of `native_meanSAGE_private_GRU_readout_source_20261011_v1`, its fixed v2 specification and prospective boundary, and the acquisition export producer. No remote calls, scientific execution, numerical imports, prediction arrays or partial outcomes were used. The sealed reader was not edited.

## Scientific definitions verified

- Raw bank decisions use the exported native FP32 probability mean. GM4/GN4 competence and coverage use four private route decisions; GI16/GFI16 competence and coverage use four independently selected native body-average decisions. Their sixteen view metrics remain separately labelled. G1/GF1/B4 views are labelled tied views, not independent members (`reader.py:21–29,291–335,350–356`).
- CAL applies one scalar temperature inside all four or sixteen view distributions before their mean. It does not temperature-scale the already pooled distribution. Legacy P1/WI4 decisions and temperatures are preserved; FP64 reconstruction is labelled numerical sensitivity (`reader.py:163–191,211–253,395–405`).
- The exact inventory is 33 cohort fit banks/51 optimizer trajectories, including three inherited P1 banks/trajectories; 30 banks/48 trajectories are new. External WI4 adds three old banks/twelve old trajectories. GA4 adds three serving/CAL rows and no fit. Nine extra P1 ancestor diagnostics remain paid records. Whole serving/CAL has 39 rows: 33 new and six inherited; 16,500 new plus 3,000 inherited scalar updates (`runtime.py:17–21`, `reader.py:75–140,193–264`).
- Frozen gates match v2: GM4 beats each of nine raw references by at least 0.20 pp mean and strictly in all three blocks; route competence versus each GI16/GFI16 body mean is at least −0.25 pp mean and −0.50 pp in each block. Calibrated DEV NLL protection is separate: at most +0.005 nat in each block against every reference and no worse than the best reference mean. P1 means P1_b0. No exploratory contrast substitutes for a failed gate (`reader.py:337–370`).
- Exact 39-row acquisition and all 39 CAL endpoints are checked before DEV truth opens. Three-block intervals/sign assignments, inherited hard cohorts and indexwise route/body differences are descriptive. A pass does not establish topology/order causality, unused confirmation, novelty or paper promotion (`reader.py:382–483`, `PROSPECTIVE_BOUNDARY.json`).

## Concrete admission condition: owner-to-producer linkage

`arithmetic.py:10–13` verifies successful exit/reaping but does not itself verify the owner's stage, command or producing artifact. Its current acquisition/CAL calls (`reader.py:76,385`) rely on root supplying the correct immutable receipts. This is an admission linkage gap, not an aggregation or numerical fault.

Root's immutable configs and launch/watcher preflight can fully satisfy the boundary without changing the sealed reader, provided they bind and verify:

1. **Acquisition:** exact CONFIG receipt; `stage=acquire`; exact acquisition worker and CONFIG path in owner `command`; CONFIG `source_manifest`, `output`, `owner_output`; the panel receipt at `output/COMPLETE_PANEL.json`; matching panel source/custody/roles/roster/accounting.
2. **CAL:** exact CONFIG receipt; `stage=calibrate`; exact sealed reader worker and CONFIG path in owner `command`; CONFIG reader manifest, panel/acquisition-owner/roles/inherited receipts, `output`, `owner_output`; temperatures receipt at `output/TEMPERATURES.json`; matching temperatures panel/roles/custody and complete 39-endpoint/19,500-bound-update inventory.
3. **Both owners:** launch and end `child` and `owner` birth identities agree (`pid`, `start_ticks`, `pgid`); `command`, stage and finite limits agree; end is `COMPLETE_EXIT_0`, exit code 0, no timeout/retry, `direct_child_wait=true`, child absent and owned CUDA PID absent. Each end receipt comes from that CONFIG's owner output.
4. **Score admission:** its panel/acquisition-owner and temperatures/CAL-owner receipts are exactly those verified above; the root decision and score CONFIG are immutable before launch. DEV release follows the actual CAL producer's successful owner closure.

With this explicit producer linkage enforced at root admission/preflight, no reader correction is required for this issue. If that external gate is omitted, the reader's generic successful-owner check alone is insufficient to certify the intended producer closure. No other material scientific-definition issue was found in this bounded source audit; numerical execution remains unverified.
