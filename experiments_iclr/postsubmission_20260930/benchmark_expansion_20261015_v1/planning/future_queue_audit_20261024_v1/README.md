# Queue audit and metadata collector, 24 October 2026

This package reads execution7 queue state and preserved training metadata. It never
loads tensors, constructs models, reads TEST labels, changes queues, or changes
scientific settings. The frozen collectors, runtime sources and historical
observations remain unchanged.

## Findings

- The three exact plan files cover **6,460 unique fits and 270 I4 export tasks**.
  All 1,080 I4 dependency edges refer to four earlier native bodies on the same
  worker. No task overlap, missing dependency, cycle, smoke identity mismatch or
  resource reservation mismatch was found.
- The old composed reporting witness rejects documentation files in genuine
  dependency closures. The previous one-README repair handles only one provider.
  Actual source-only binding calls expose further Markdown, HTML, text and XML
  dependencies for the BUDDY interaction and context methods. This is a final
  reporting failure after successful training, not a model failure. The additive
  dependency-byte repair v2 addresses it while retaining confined byte/hash
  checks. `QUEUE_AUDIT.json` records the exact files and hashes.
- The old campaign collector looks in canonical output folders. Execution7 stores
  results in per-task attempt folders instead, so the old collector would miss
  the campaign outputs. `collect_queue.py` uses frozen plans and verified terminal
  STATE/ATTEMPT_RESULT/NATIVE_RESULT custody to locate the actual outputs and
  their scalar observations.
- Old and new repaired witnesses carry different producer receipts. The queue
  collector verifies the original, v1 and v2 rules explicitly and retains each
  original witness unchanged. It does not relabel v1 evidence as v2 evidence.

## Verification

Run local metadata checks from this folder:

```sh
python3 -B audit_queue.py
python3 -B -m unittest -v test_queue_support
```

The tests cover terminal and active task paths, the different task/reporting hash
formats, missing native failure records, duplicate ownership, incomplete registry
coverage, output escapes, changed results and states, original/v1/v2 witness
compatibility, altered receipts, changed dependency hashes and changed runtime
identities. `CPU_TESTS.json` stores the outcome. These are metadata tests, not
additional training or prediction evidence.

## Collection on a verified host

Use the existing host guard environment and pass the host's frozen registry and
every plan for that host. Allocation requires its one plan; server51 requires
both plans. The destination must be a fresh repository directory.

```sh
python3 -B collect_queue.py --registry-index /absolute/project/registry/INDEX.json --plan /absolute/project/PLAN-allocation_gpu0.json --output /absolute/project/fresh-report
```

The two repair modules are pinned dependencies. Stage both
`campaign_reporting_repair_20261023_v1/reporting_patch.py` and
`campaign_reporting_repair_20261024_v2/reporting_patch.py` on server51, even if no
v1-repaired sidecar exists there. No other files from v1 are imported by the
collector. The frozen port6 reporting package and statistical reporter must be
present and retain their registered hashes.

The result preserves all retained terminal attempt receipts, including failures
without a native RESULT. Historical costs can overlap observer costs; they are
marked and must not be summed twice. Pending and active tasks are omitted from
scientific comparisons and listed explicitly. Incomplete contrasts remain
withheld under the original reporter's rules.

## Limits

All-fit two-update smoke tests establish training startup and checkpoint reload,
not every full-fit finalization branch. This audit does not promise all future
fits will finish. Full selected-state finalization remains to be observed for
later runtime families. The launch disk floor is per-task admission protection;
it does not reserve enough space to retain the complete campaign. The sum of
per-task reservations is not a storage forecast. Live retained-size and remaining
space audits are handled separately.

Large fetched plan copies under `plans/` are audit inputs, not source release
contents. Publish the source, compact JSON audit and test receipts only.
