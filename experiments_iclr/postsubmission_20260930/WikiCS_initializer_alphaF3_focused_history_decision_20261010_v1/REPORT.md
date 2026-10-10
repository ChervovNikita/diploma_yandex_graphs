# WikiCS native initializer F-only: completed failed utility screen

2026-10-10. Focused author history audit of the whole completed three-seed readout. Original scores, frozen criteria, selected modes, source failures and prior seals remain preserved. This adds scalar interpretation and metadata bindings; it launches no jobs and opens no raw arrays, datasets or checkpoints.

## Decision

**Close the original initializer utility screen as failed.** Native independent local-scorer starts improve served accuracy on all three seeds, averaging **+0.278094 pp**, and improve mean member accuracy on all three. Served NLL worsens on all three, averaging **+0.065817**. The frozen NLL condition fails; the accuracy and member-floor conditions pass. These results do not advance the ingredient to confirmation, calibration or another grid.

| Seed | Copied F accuracy % | Native F accuracy % | Accuracy delta pp | NLL delta | Mean member delta pp | Worst member delta pp |
|---|---:|---:|---:|---:|---:|---:|
| 6101 | 81.266591 | 81.759575 | +0.492984 | +0.073826 | +0.237012 | +0.189609 |
| 6203 | 81.475161 | 81.513083 | +0.037922 | +0.109650 | +0.037922 | −0.113766 |
| 6307 | 81.266591 | 81.569966 | +0.303375 | +0.013974 | +0.222791 | +0.227531 |
| Mean | 81.336114 | 81.614208 | +0.278094 | +0.065817 | +0.165908 | +0.101125 |

Mean copied/native NLL is 1.059889/1.125705. The original screen requires positive paired accuracy on all three, mean gain at least 0.2 pp, mean NLL no worse, and mean/worst member degradation limits of 0.1/0.2 pp. The worst member loses 0.113766 pp at 6203 within its allowed floor. Same-development optimizer repeats remain exploratory rather than confirmation. The completed comparison records `passes:false`, `confirmation:false`, novelty `none`, and no execution extension or TEST access. [E01]

## Exact prediction flows

All counts use the same 5,274-node development population. Correct-member coverage means at least one member is correct. Pool loss means a correct member exists but the mean-probability pool is wrong. No old or new bank rescues an all-member-wrong node through pooling.

| Seed | Coverage old → new | Net coverage | Pool loss old → new | Extra loss | Pool correct old → new | Net served |
|---|---:|---:|---:|---:|---:|---:|
| 6101 | 4318 → 4373 | +55 | 32 → 61 | +29 | 4286 → 4312 | +26 |
| 6203 | 4317 → 4371 | +54 | 20 → 72 | +52 | 4297 → 4299 | +2 |
| 6307 | 4311 → 4375 | +64 | 25 → 73 | +48 | 4286 → 4302 | +16 |

The marginal identities are 55−29=26, 54−52=2 and 64−48=16. **Net coverage gains are not exact acquisitions.** The complete paired output identifies both acquisitions and coverage losses:

| Seed | Coverage gained / lost / retained / neither | Pool-loss appeared / cleared / persisted | Pool repairs / harms / net | Both pool correct / wrong |
|---|---|---|---|---|
| 6101 | 157 / 102 / 4216 / 799 | 61 / 32 / 0 | 140 / 114 / +26 | 4172 / 848 |
| 6203 | 143 / 89 / 4228 / 814 | 71 / 19 / 1 | 129 / 127 / +2 | 4170 / 848 |
| 6307 | 193 / 129 / 4182 / 770 | 70 / 22 / 3 | 177 / 161 / +16 | 4125 / 811 |

Among copied-F all-member-wrong nodes, native starts acquire 157/143/193 correct alternatives, serve 123/116/162 and lose 34/27/31 during pooling. Full-population repairs include other old pool errors, so those cohort repairs must not be substituted for the total repairs. Strict common wrong competitors clear/appear on 158/101, 143/89 and 193/130 nodes. Truth outranking one fixed old rival is a separate diagnostic from actually predicting the truth. [E01]

This closes the pending acquisition/serving question: useful alternatives and improved mean competence coexist with more pool losses and adverse NLL. Signs previously traded away competence; their failed recipe does not establish that all asymmetry has the same effect. Native starts are an established initialization ingredient, with no novelty claim. The failed frozen gate determines disposition even though accuracy improves.

## Endpoint and reference scope

Original copied F selected epoch/mode is 65/local, 125/global and 50/local. Native F selected epoch/mode is 132/global, 130/global and 125/global. Preserve the original earliest strict complete-VALID maximum over 1,100 epochs, native local/global transition and local selected restoration. The prior metadata audit passes common source, recipe, selector, data, parameter-role and paired-training-GPU checks. It does not force endpoint mode equality or establish bitwise parity or a clean population causal effect. [E09]

Historical ordinary independent4 remains stronger in accuracy at **82.0439875%** versus native F **81.6142081%**. Its provider and independent selector opportunities differ; it is contextual rather than a matched causal reference. Its original result and all prior scores remain unchanged. [E12]

## Cost, custody and source failures

The completed worker attempted and completed **12 new member forwards**, with zero old forwards, backward calls, Adam steps or training updates. Worker inclusive wall is **12.805690475 s**; CPU user/system is 11.681780/3.465445 s. Peak CUDA allocated/reserved is 1,998,874,112/2,910,846,976 bytes and peak worker RSS 1,277,763,584 bytes. The original finite owner retained 2,990 active plus 10 cleanup seconds, 32 GiB GPU and 16 GiB RSS caps. [E02,E07,E08]

The actual child direct wait observes exit 0, absent child and no child CUDA rows. Owner wait wall **15.100778867 s** contains the worker; do not add it to worker wall. The local completed observation reports owner null. Detached parent OS exit remains null and its direct wait remains false; no parent exit is fabricated. Root retains external wait/cleanup/transfer cost, and old training/reader costs are preserved rather than re-added. [E03–E08]

The first two reader attempts remain source failures (`UnboundLocalError importlib`, then `KeyError path`) with `calls:null`; they are neither measured-zero-call runs nor adverse prediction outcomes. V4's manifest and seal, the complete comparison/cost files, clean exit records and both failed observations are hash-bound in the evidence index. [E10,E11,E13,E14]

## Evidence-map consequence

The new assumption-map v2 supersedes only the pending initializer NLL/coverage/readout status. The two remaining actions are the sole private-Adam normalization control and, after full hop18 closure, the existing small simultaneous stored-error join. Native receiver exchange stays a separate attributed source proposal with no admitted exchange quality or automatic continuation. No calibration, grid, new confirmation, methodological novelty or acceptance conclusion follows from this closure.
