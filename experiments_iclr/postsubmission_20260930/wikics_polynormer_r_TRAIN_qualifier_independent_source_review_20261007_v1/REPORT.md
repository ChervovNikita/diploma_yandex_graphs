# WikiCS TRAIN qualifier independent source review

**Decision: approve a root-owned bounded qualification attempt only. No concrete source bug or launch blocker found.** Full fits remain unapproved.

Reviewed sealed qualifierb3026f9f, trainer/native boundary adapter/weight generator, TRAIN projection source, selector v2 amendment and source consistency review. All8 qualifier,13 trainer,3 amendment and2 consistency manifest entries agree with byte counts/hashes. Constructor,core androute_rng ASTs match the actual trainer exactly. No numerical work, dataset/checkpoint/score values, held outcomes, server access or source modifications performed.

The qualifier faithfully checks native parameter copies and complete local/global forwards; neutral deterministic mean-weight/shared-gradient identity; stage-active finite gradients and stage-inactivegradNone; actual non-neutral route dropout and serial-backward versus per-route gradient accumulation; resulting native Adam moments/parameters/dropout states; and disjoint independent model/optimizer ownership. All six costs include complete580-node TRAIN losses and one update per model on full11701-node/442907-edge graphs. Family routes release their graph before the next route.

The trainer matches selector v2: first strict maximum spans local andglobal, selected local model/optimizer restoration with live RNG retention at epoch100, explicit selected_global restoration, separate native member selectors and fixed mean-probability reporting. This qualifier does not run that held selector.

## Bounds

The proposed24GiB owned GPU,32GiB RSS,28GiB fresh GPU minimum and900/1200-second soft/hard bounds are reasonable for one bounded discarded attempt. The full qualifier performs approximately58 complete forward/backward trajectories plus10 dropout-off copy forwards; the six timed cost epochs account for18 of those backward trajectories. Only one route training graph is live at a time. Static review establishes no guarantee that the workload completes within the caps. Preserve cap/gate failures; no automatic expansion, retry or partial-cost success.

Fresh global-mode epoch costs do not certify the local100/global1000 fit horizon, future selected-stage serving, per-epoch validation/checkpoint cost or full-fit limits. Root must separately adopt those after actual qualification/cost receipts.

Exact source pins and bounded approval are inFINDINGS.json.
