# Inactive fixed public-author SAGE context

Arxiv is **encountered**, following history's affirmative prior predictive-result finding and root's correction. This packet supplies a fixed prospective replication/control source; it supplies no unused-input, unused-TEST or confirmation admission. It does not choose the next dataset or alter any E/F/sequential packet. No data, tensor, prediction, model, remote process, qualifier or reviewer was used in its preparation.

`author_model.py` extracts only the exact `SAGE` class from the pinned OGB author `gnn.py`, commit `61e9784ca76edeaa6e259ba0f836099608ff0586`, SHA256 `050b6b7a0fc86ef99b237438f1506f2868a442833d963cfe03efa3419e60d365`. It never imports the stock provider or per-epoch TEST driver. The peer audit and fixed spec are under `ogbn_arxiv_official_provider_SAGE_confirmation_source_audit_20261011_v1`; root supplies that retained primary-source path.

The fixed model is ordinary SAGEConv128→256→256→40, BatchNorm/ReLU/dropout0.5 after the first two layers, Adam0.01/zero decay, own full-TRAIN NLL and **exactly500** updates. No patience, scheduler or grid exists. Ten independent M1 bodies use seeds8101,8219,8363,8501,8663,8819,9001,9199,9403,9623. Each fresh constructor is moved to CUDA and reset under its seed, matching the author's constructor/reset lifecycle with explicit seeded repetitions. The source checks expected218664 parameters and4112 bytes of BN running mean/variance/batch-counter buffers during future construction; none was constructed here.

## Root callable interface

Call `consumer.run(cfg, fresh_output, root_validate)` only if root elects to run this encountered control. `cfg` must include:

- `execution_enabled=True`, exact `seeds` and `point` from `author_model.py`;
- `author_source`, `device="cuda:0"`, and `support_policy="symmetric_coalesced_no_added_loops"`;
- finite positive integer `limits={wall_seconds,cuda_bytes,rss_bytes}`;
- `profile={edges,self_edges}` from root's full provider audit;
- exactly `paths={train,select,cal_ids,confirm_ids}` pointing to history's `train.npz`, `select.npz`, `cal_ids.npy`, `confirm_ids.npy`;
- `runtime_sources={sage_conv,sparse_tensor}`, each an exact absolute-path/bytes/SHA256 receipt for the actual imported PyG SAGEConv and torch-sparse SparseTensor source;
- root's remaining source/provider/official-membership/configuration custody fields.

The callback signature is `root_validate(cfg, *, action, source_receipts)`; it binds all sources, actual host/GPU/runtime, resource policy, original features, full official roles/support and prospective control disposition before numerical imports. It returns primitive checkpoint custody with `status="ROOT_AUTHOR_CONTEXT_FITS_ADMITTED"`. Existing finite owner/watchdog conventions remain root's responsibility; this packet creates no launcher or admission. Missing torch-sparse or failed source/runtime/cap checks are retained failures, not a COO/model fallback.

TRAIN NPZ keys are exactly `{x,edge_index,ids,y}`; SELECT keys are `{ids,y}`. Only TRAIN and SELECT truth is loaded. CAL/CONFIRM inputs are label-free NPY membership arrays. Expected official prepared counts are90941/14900/14899/48603; unexpected counts stop as provider discrepancies. COO is transposed into SparseTensor `adj_t` on the common symmetric/coalesced support, retaining raw diagonals and adding no loops or symmetry. Root binds the underlying original-feature/support hashes; this consumer does not build a different graph.

## Outputs and selection

Every update performs TRAIN NLL/backward/Adam, then SELECT-only accuracy. Integer correct counts implement the earliest strict maximum. The loop always completes500 updates. Each selected checkpoint includes parameters, all BN buffers, Adam moments, Python/NumPy/Torch CPU/CUDA RNG, seed/step, SELECT criterion and custody. After500, strict restoration checks those states and BN batch counters, then performs one selected full-graph forward.

Per seed, `selected.pt` retains the full selected state. `selected_SELECT.npz`, `selected_CAL.npz` and `selected_CONFIRM.npz` contain exactly `ids`, native float32 `member_log_probs` shaped `[1,rows,40]`, and int64 `native_member_argmax` shaped `[1,rows]`. No CAL/CONFIRM truth, loss or accuracy is read or computed. Root's later common temperature fitting and single scorer-only truth release remain external. This follows the donor Penn selected-prediction archive convention; it is not the stock author's reported-score replication.

`trace.jsonl`, `PROGRESS.json`, `RESULT.json` and `COMPLETE_AUTHOR_CONTEXT.json` record the fixed loop, selection, source custody, full-state restore and costs. Ten successful bodies mean5000 TRAIN,5000 SELECT,5000 backward/Adam calls and10 selected forwards (30030 declared SAGEConv calls). Constructor/snapshot/restore/archive work, actual CUDA peaks and process lifetime RSS are included; no timing or memory claim is made here. A failure remains terminal with its actual completed step counts and no substituted seed, automatic retry, calibration or TEST release.

The only verification performed here is source syntax/structural inspection. There is no numerical PASS, qualification suite or scientific outcome. Future factual-profile work waits for root's concrete dataset choice.
