# Trained-weight deployment companion — source only

This packet implements the frozen `trained_shared_message_benchmark_v1/PROTOCOL.md`. Source preparation does not authorize a launch. The request, prior and approval templates are deliberately unapproved, and no inputs, targets, checkpoints, predictions, quality outcomes, Torch, GPU or remote runtime were accessed during preparation.

## Frozen computation

Six fresh children run complete Reddit2 then ogbn-products, each with seeds 17,29,43 in order. Every child retains final then selected checkpoint roles, including separate work when checkpoint bytes coincide. Exact packed and serial modes use one resident exact-member model loaded with the same checkpoint. Shared packed uses its own trained shared-mean checkpoint and a second resident model. Both reuse one full source graph/operator and complete input lineage. Constructor checkpoint flags are preserved; eval/no-grad makes checkpointing inactive. No optimizer is constructed and no training or target scoring is called.

Each role first checks complete-node exact pooled logits with atol=rtol=1e-5, reporting maximum absolute discrepancy and class-decision disagreement. The failure condition is the frozen logit tolerance; no extra unprescribed class-decision threshold is introduced. Shared predictions are not equated to exact predictions. Comparison tensors remain temporary CPU objects and are never saved.

Five warmups per mode, all 50 paired rounds and the exact seeded permutation schedule are retained, as are additional isolated-forward memory samples. Medians use all 50 timing samples; p90 is nearest-rank `ceil(0.9*n)`. Signed shared reductions are reported against packed exact, serial exact and whichever exact median is smaller. Warmup times are saved separately and excluded from that summary. Timing brackets the full-node forward and raw-logit pooling with CUDA synchronization. CPU copies, numerical comparison, audits, source/checkpoint hashes, loading, setup, writes and supervision are excluded from that bracket and charged through complete supervised wall evidence.

The isolated-forward peak keeps graph, inputs and both models resident, clears the cache and resets counters, and leaves no other output live. Baseline allocated/reserved bytes and total forward peak allocated/reserved bytes are recorded. A 60 GiB reserved ceiling is checked throughout, including setup and equivalence. Graph storage, both models' parameter bytes, input/operator/model setup, checkpoint hash/load/transfer time, comparison/CPU-copy time, source/admission time, host RSS and hash work are separate fields.

## Guard identity and complete-family gate

`guard.py` first checks this packet's request-bound manifest. It verifies the immutable scientific-v3 packet with its own original guard, then requires `CALLER_IMPORT_POLICY_v1.json` to equal the original scientific policy plus this packet's exact Python source files. Scientific package roots and generated-source rules are unchanged. Own caller files are loaded explicitly from sealed real paths, including the real `__main__`; no `__file__` substitution, scientific source edit, old-import bypass or parent-folder import exposure is used. Runtime receipts use the augmented caller policy, while saved scientific receipts retain their original policy during the inherited consumer audit.

`binding()` and `audit_family_lock()` must pass before any benchmark input or checkpoint loading. This audits every fit and replay and the complete family, rather than trusting a lock flag. Existing score metadata and prediction/checkpoint hashes can be parsed for integrity by that consumer; no new quality scoring is performed. The companion reads validation-freeze JSON identity only. It never invokes `validation_freeze()` or `load_validation()`, which would open validation bundles. The existing input loader can materialize its already exported training targets; the benchmark does not index or score them.

## Actual root artifacts required

Create a distinct final request from `REQUEST_TEMPLATE_v1.json` only after the complete family passes. Keep `fixed_policy`, prescribed paths and six-child order unchanged. Fill exact `{path,sha256}` references for the actual scientific binding, freeze JSON and `FAMILY_LOCK.json`; all benchmark checkpoints come solely from its fully audited rows. The request also binds this source manifest, the prospective protocol freeze, the actual GPU UUID and these root-owned artifacts:

- `source_audit`: independently reviewed/adopted JSON with `qualified=true` and `audited_source_manifest_sha256` equal to this final source manifest hash.
- `prior_accounting`: schema `gnnm-trained-inference-prior-accounting-root-v1`, `qualified=true`, `updated_after_full_family_and_source_preparation=true`, `declared_children_already_included=false`. Its four numeric fields are in `PRIOR_TEMPLATE_v1.json`. Combined phase wall must cover the scientific binding's prior plus all completed scientific/export/replay/family-lock wall, failures, source publication and coordination/query/audit costs. Root's separately charged coordination reserve must cover prelaunch and final consumer work. Previous companion failures outside the six declared jobs also require explicit prior companion wall/artifact charges.
- `root_approval`: schema `gnnm-trained-inference-root-approval-v1`, `qualified=true`, `complete_family_and_source_and_resources_qualified=true`; first finalize every request payload field, including `root_authorized=true` and `scientific_queue_completed=true`. `approved_payload_sha256` hashes canonical sorted JSON excluding only `root_approval`. Issue that approval, then insert its exact reference. Do not change the request between children.

The templates issue no permission. They do not point at live or unfinished scientific outputs.

## Prospective launch procedure

Use the existing repository wrapper, exact `.venv/bin/python`, corrected sole authorized allocation and idle-GPU authorization supervisor. Do not overlap the scientific queue. No SSH command is supplied by this packet.

Before each launch, invoke `prelaunch.py --request R --task T --seed S` under the root's source-bound read-only coordination/accounting context. It performs the full family consumer audit and all remaining-cap checks, refuses any existing current inner/outer/output directory, and prints one exact `bounded_run_v1.py` argv. It does not execute that argv. The printed command binds the same request, uses `--cap-seconds 600 --grace-seconds 5`, and enters `run_authorized_v2.py --require-idle-gpu` before the worker:

```text
runner.py --request R --supervisor S --task T --seed SEED
```

The worker repeats the family/resource checks before its live GPU query and input loading, verifies the exact outer/inner argv and the 600/5 cap/grace, and charges elapsed time from the outer start. Every previously started companion terminal is charged, including failed jobs. A previous failed job or current terminal refuses continuation/retry. All remaining full 600-second child caps must fit the 3,600-second companion budget and the combined 21,600-second phase wall. Allocated file bytes (`st_blocks*512`) from all declared inner/outer/output directories, root prior bytes and a 1 MiB metadata reserve must fit 64 MiB new companion artifacts and 200 GiB combined phase disk. Root coordination costs are separately reserved/charged in the combined prior. There is no automatic retry or resource amendment.

Failed timeout/cleanup wall remains charged in full even if it exceeds the 600-second cap; such a terminal cannot qualify as a successful child or permit continuation.

Each output retains `START.json`, `final.json`, `selected.json`, `RESULT.json` and an exact artifact manifest. Partial failures remain, with a failure receipt when its budget permits; outer supervision always retains its terminal evidence. `RESULT.json` cannot include an outer terminal that has not happened yet: it explicitly marks complete supervision pending. After all six successful terminals, use `prelaunch.py --request R --consume-completed-family` under the separately charged root consumer context. It replays every sample/order/summary, checkpoint binding, caller import receipt and successful inner/outer chain and returns all full supervised wall charges. It rejects a subset or failed/extra/changed artifacts. Save/publish that returned report only through root's accounted finalization.

This benchmark supports a trained-weight deployment cost comparison. It does not establish novelty, quality retention, predictive superiority, calibration or training gains.

## Preparation verification

`verify_stdlib.py` checks seeded balanced orders, all 12 roles, sample statistics, signed reductions, frozen tolerance/caps, draft refusal, full remaining-cap refusals, prior failure charges, path confinement, source-only imports and lack of training/target/prediction-write calls. `seal_source.py` produces the separate policy, source bindings, check receipt and manifest. These checks do not qualify actual packed inference, GPU feasibility, completed scientific family or resource admission.
