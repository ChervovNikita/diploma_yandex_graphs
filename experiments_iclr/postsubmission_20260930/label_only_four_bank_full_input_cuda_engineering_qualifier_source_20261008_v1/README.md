# Disabled full-input four-bank CUDA engineering qualifier

This packet contains a callable, owned structural qualifier for the separately
sealed four-bank first-screen source. It was prepared with source/static checks
only. No numerical imports, datasets, CUDA, training, server access or launch were
used during preparation. All predecessor sources and canonical artifacts remain
unchanged.

## Fixed discarded work

One native seed, **6101**, executes four complete updates on the original WikiCS
graph: 11701 nodes, 442907 ordered edges, 300 input features, all 580 TRAIN labels,
and the original two native own-CE views per update. Each existing view B supplies
all four correction banks. There is no extra native TRAIN forward or replay.

1. Complete local update at engineered epoch1 and take an unscored snapshot.
2. Complete local update at engineered epoch2, then serve all 5274 development IDs.
3. Restore the predetermined epoch1 native model/Adam and every bank's learned
   parameters/Adam using the bank driver's own-local restoration path. Retain
   the live end-local mask streams, native RNG, buffers and logical work counters.
4. Complete two global updates labeled engineered epochs101/102; serve the full
   development ID population, reconstruct the final unscored engineering state,
   and compare its serving outputs at `atol=rtol=2e-5`.

This explicitly engineered transition exercises both native capture paths. It
does **not** execute 100 local epochs, reproduce the 1100-epoch schedule, establish
scientific resume, or demonstrate scientific accuracy. The original native
1100/100/512 recipe remains unchanged.

| Bank | TRAIN head calls | Backwards | Corrector Adam calls | Logical updates |
|---|---:|---:|---:|---:|
| C4 |16|16|4|4|
| S_joint4head |16|4|4|4|
| U4_sharedB |16|16|16|4|
| S_one_path |4|4|4|4|

Native work is four backwards/Adam updates and eight TRAIN head captures.
Original-session serving adds two native captures; reconstruction adds one.
VALID evaluation events and selectors are zero. All learned checkpoints stay in
memory and are discarded; no model checkpoint is written.

The four-bank source fixes candidate/route0 initializer to the native seed,
subsequent control heads to `seed+1009/+2018/+3027`, and every private mask seed to
`seed+1900001`. The old one-bank integration's `seed+1700001` initializer is not
used by this qualifier.

## What is checked

- Actual route calls receive the same common Q and exact visible TRAIN complement.
- Embedding hooks read only the permitted TRAIN label vector in its bound order.
- All fixed nonself neighbor records retain their original order/multiplicity.
- Every actual route/head has nonvacuous real zero-context rows and exact zero
  correction/message, including after learned output maps have been updated.
- The existing full view-B values are supplied as gradient-requiring leaves to
  the unchanged bank executor; neither H nor base logits receives a gradient.
  The executor also checks the native own-gradient bank and parameter versions.
- Mask draws, bank updates, serving and coherent restoration preserve native
  private and ambient RNG custody.
- Restoration copies exact native and all-bank learned/Adam states from a
  genuinely changed epoch2 state, while logical counters and masks stay live.
- Reconstruction copies the same source/role/model/Adam state and rejects training.
  Prediction consistency uses the declared float32 tolerance; no bitwise native
  trajectory claim is made.

`audit.py` wraps existing corrector route methods and adds read-only Embedding
hooks. It does not implement another training executor or perform an extra
corrector forward/backward. All numerical work uses the complete real graph.

The unchanged public `load_train_valid` and source validators **deserialize and
validate VALID y**. The qualifier then keeps development IDs only. It performs
no VALID truth scoring, model input, objective, checkpoint selection or scientific
accuracy endpoint. Every engineering snapshot uses metrics `None`,
`selector_performed=False`, and `snapshot_purpose='engineering_qualification'`.

## Ownership and execution gate

`owned.run_owned(release=..., later_execution_authorized=False)` and
`qualify.run_fixed(owner=..., later_execution_authorized=False)` reject by default.
An adopted run requires a separate root-authored release copied from
`ROOT_RELEASE_TEMPLATE_DISABLED.json`, with only these fields changed:

- `enabled=true`
- `engineering_execution_authorized=true`
- `source_manifest_sha256=<this packet's final manifest hash>`

The release fixes the sole allocation `anogena-2-0`/GPU `44039938`, authorized Mol18
coexecution, existing runtime/overlay, exact SAFE/native/TRAIN/VALID payload hashes,
seed, fresh output and resource bounds. Metadata JSON formatting is not a launch
gate: its official schema/role/source-custody fields are checked and its observed
hash is recorded. Exact numerical payload hashes remain mandatory.

The normal-host owner starts one new worker process group with the existing
runtime. It records parent/worker PID and Linux birth ticks, waits at most 900
active seconds, and allows 15 cleanup seconds for its own child only. Own CUDA
peak allocated/reserved memory is bounded at 70 GiB. There is no retry, resume, remote access,
installer, service change or settings change.

After root review/deployment, the intended normal-host command is:

```sh
/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/native_ncn_runtime_20261005_v1/.venv/bin/python \
  /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/label_only_four_bank_full_input_cuda_engineering_qualifier_source_20261008_v1/owned.py \
  --release /absolute/path/to/root-enabled-engineering-release.json \
  --later-execution-authorized
```

The owner sets the declared dependency overlay for its worker. The sealed
disabled template itself cannot launch a worker.

## Receipts and limits

The fresh owned output contains `OWNER.json`, `WORKER.log`, `PROGRESS.json`,
`ENGINEERING_RESULT.json`, and `TERMINAL.json`. Failed partial work is retained,
including the underlying executor's actual attempted/completed counts. Timeout
and worker failures always receive a terminal owner receipt; a forced termination
may leave the last progress receipt without a final worker result.

A future `engineering_passed` result establishes only the declared fixed
structural checks on that input/runtime/source. It does not admit a scientific
launch, establish native author parity, qualify the complete schedule, resolve
comparative accuracy, or implement the broader scientific comparator panel.
