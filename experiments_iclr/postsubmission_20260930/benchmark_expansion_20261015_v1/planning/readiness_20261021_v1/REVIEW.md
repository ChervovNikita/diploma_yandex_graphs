# Full-training source readiness review — 21 October 2026

Scope: read-only local review of execution7, port6 native training/reporting/I4, smoke6, and retained lifecycle/resource evidence. No model execution, tests, remote calls, source changes, or launch occurred. Paths below are relative to `benchmark_expansion_20261015_v1`. The latest user request requires a readiness report before a new full-training decision; historical conditional-GO language in CONTINUATION is not a new launch instruction.

## Assessment

No deterministic fresh-start training failure was established in the reviewed code. The all-fit smoke is strong evidence for each exact descriptor's construction, two real optimizer updates, complete VALID, finite values and exact model/optimizer reload. It is not a full-lifecycle guarantee. Existing completed lifecycle evidence covers important omitted branches, so this review does not request repeated smoke grids or new experiments.

One concrete export-integrity defect remains in the queue, and the long-run storage policy is less complete than the earlier planning proposal. Neither establishes that the first full-training update will fail. Whole-campaign readiness must also use the root's current receipt/host verification; this local review does not establish current server51 availability or a global PASS.

## Concrete finding

**I4 completion/recovery does not verify the aggregate file.** `campaign_execution_20261016_v7/worker.py:99` validates the I4 schema and four member IDs/job hashes, then returns `complete` without verifying `bank_receipt`. The same method explicitly verifies the selected export for fit tasks at line 106. `load_states()` reuses this validation when adopting existing successful results. Consequently a missing or changed `I4_selected_VALID.pt` can remain classified as complete on restart even though the aggregator originally recorded its SHA256. The producer does create the receipt in `server51_campaign_port_v6/runtime_I4/aggregate_selected.py:78`.

This is a persisted-output integrity defect, outside the 6,460 fit smokes; it affects the 270 export tasks. A narrow repair should require the I4 `bank_receipt` to name the task's own `output/I4_selected_VALID.pt` and verify its bytes/hash before accepting completion. This review made no repair and observed no corrupted I4 output.

## What smoke omits, and what already supports it

| Path | Current implementation and evidence | Remaining limit |
| --- | --- | --- |
| Complete TRAIN epochs | Node updates are whole-graph epochs. OGB's adapter exhausts the loader and verifies every TRAIN position exactly once, including native skipped batches. Link sessions enforce the exact native epoch boundary. | Two-update molecular/link smokes stop before later minibatches and the tail; they do not prove all shuffled batches for every method. |
| Repeated VALID and selection | Full runtimes validate after every completed epoch, apply the frozen selector, save selected/recovery state, and record curves. Link policy explicitly requires interval 1, constant native optimizer groups and zero warmup; a missing learning-rate scheduler is not a demonstrated defect. | Smoke calls VALID once and does not advance selector patience, cap, or source diagnostic bookkeeping. Source-reference OGB full TRAIN diagnostics and omitted-TEST RNG bookkeeping are additional full-run paths. |
| Resume and selected export | Full recovery binds job hash, source binding, optimizer state, selector and RNG, and requires adapter/selector epoch clocks to match. A stopped resume finalizes without an extra epoch. Full finalization loads selected state and evaluates/export VALID. | Smoke deliberately reports `TRAIN_updates_after_reload: 0` and `complete_epoch_or_resume_claim: false`; it does not execute a resumed update or the final selected-export/reporting path. |
| I4 | The aggregator requires four different seeds/bodies, matching cohort/settings/IDs/labels, finite aligned logits and exact link query roles. It performs no training and preserves body-0/M1 reuse. Historical PubMed pilot metadata contains real I4 bank receipts. | No I4 task runs in the all-fit smoke. The queue receipt omission above remains. |
| Long-lived execution | tmux owners, inherited flock leases, separate guardian/child identities and preserved failed attempts protect ownership across supervisor loss. Full outputs are fresh and never initialized from smoke. | Full phase intentionally stops new tasks after an ordinary failure, drains active children, and needs explicit retry handling. This is not an automatically self-recovering campaign. |

Relevant source locations: `qualification/all_fit_smoke_20261016_v6/smoke_control.py:28`; `run_one.py:121`; `server51_campaign_port_v6/runtime_node_molecular/adapters.py:313`; `runtime_node_molecular/run_fit.py:181`, `:211`, `:244`; `runtime_link/run_fit.py:283`; `link_policy_v1/policy.py:112`; `campaign_execution_20261016_v7/worker.py:188`, `:300`; `guardian.py:17`.

Retained evidence already covers more than two updates:

- `BUDDY_LIFECYCLE_TERMINAL_22.json`: passed 2 complete epochs / 3,170 updates, including the 794-record tail, four VALID calls, exact restored states/RNG and selected exports. Its producer generation differs from port6; use it as lifecycle evidence with source applicability, not a claim that every port6 method completed two epochs.
- `qualification/runtime_link_v6_native_lifecycle_driver_v1/fetched_attempt01/RESULT.json`: passed NCNC 2 epochs / 34 updates and exact restore checks. `qualification/campaign_resources_v1/MEASURED_PROFILES.json` also pins passed NCN 2 epochs / 34 updates.
- The same measured profiles pin MolHIV GIN-VN 1 complete epoch / 1,029 updates, two VALID calls and selected/recovery/export. `qualification/campaign_resources_v1/MOLECULAR_SIZE_BOUNDS.json` already provides any-shuffle batch-32 bounds: MolHIV 5,222 nodes / 11,114 directed edges; MolBACE 1,740 / 3,674. No repeated raw-data scan is indicated.
- `qualification/runtime_v4_molecular_selected_mean_lifecycle_v1/fetched_attempt01/RESULT.json`: passed CPU MolBACE 2 complete epochs / 76 updates, exact state/RNG restores and exact selected/mean exports after fresh stopped resume. CPU timing is not GPU timing.
- `planning/all_fit_launch_20261016_v1/ALLOCATION_PREREQUISITES.json`: retained PASS for the 38-update MolBACE observer lifecycle and repaired zero-update BUDDY selected replay. `PUBMED24_COMPLETED_METADATA_AUDIT_20261016.json` records 24 completed historical fits and their I4/selected bank metadata.
- Existing source checks were read, not rerun: port6 26, smoke6 8, execution7 7, allocation memory-admission 12; link policy `LOCAL_CHECKS.json` includes source horizon, BUDDY zero sentinel/ties, minimum/patience/cap and strict resume/stop-on-entry checks.

## Memory, concurrency and disk

Execution7 admits at most two owned jobs per GPU, with frozen profile compatibility, actual GPU free space, reserved remaining growth, host RAM/CPU and disk checks under the host ledger lock. Heavy Physics Mono shared jobs have a 69 GiB reservation and stay serial. The allocation's additive memory-admission wrapper credits bounded clean active/inactive file cache; it preserves the rest of execution7. These are conservative engineering budgets, not measured upper bounds for every full fit. The Oct19 Physics OOM and successful isolated retry demonstrate that external GPU growth can still invalidate a previously adequate admission snapshot.

**Disk is checked incrementally, not reserved for the whole campaign.** Frozen profiles reserve 2 GiB per node fit and 4 GiB per molecular/link fit or export. `capacity.py:175` requires current free bytes to cover the candidate plus all active disk reservations and an 8 GiB floor. Native training replaces `selected.pt`/`recovery.pt` atomically, so checkpoint storage does not grow by one retained checkpoint per epoch; curves and provider ledgers do grow. Successful smoke checkpoints are removed, so smoke output size cannot estimate full-training retention.

The earlier `planning/campaign_dependencies_concurrency_20261015_v1/resource_scenarios_v2/STORAGE_BUDGET.json` proposed a 20 GiB floor, a reviewed retained-data cap, peer/transfer allowances and checks at checkpoint boundaries. Execution7 does not implement those additional controls. This is a concrete difference from that proposal, not evidence that the current free space is insufficient. Root reports approximately 698 GiB currently free on allocation; server51 must use a fresh observation when reachable. Do not claim all retained campaign outputs fit based only on active reservations. Existing approximate payload scenarios explicitly disclaim being a campaign estimate.

The evidence supports a bounded-risk readiness assessment after current host/receipt checks and a deliberate launch decision. It cannot promise no later numerical, memory, disk, host or export failures, and it supplies no reliable campaign ETA.
