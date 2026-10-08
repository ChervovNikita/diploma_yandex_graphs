# Sequential single8 own-CE control

8 October 2026. **Disabled source preparation only.** This successor implements the required supervised-view attribution control with one native single predictor. It adds no GNNM method, pooled loss, contrastive loss, teacher, untied ensemble or dataset recipe. No model, dataset, checkpoint, scientific outcome or server was accessed, and no runtime qualification or cell was launched.

`single8.py` is a narrow successor of the exact V2 single8 scaffold. Its model construction, four persistent TRAIN streams, single serving, snapshot metadata and complete public V2 driver are preserved. Only WikiCS and `single8_own` are accepted. All numerical entry points retain `later_execution_authorized=False` as the default and reject calls before loading numerical source unless a later explicit adoption supplies `True`. This package supplies no CLI, owner, release, queue or automatic launch.

## Update and tape lifetime

At each complete TRAIN update, zero the one native Adam gradient bank once. Visit the predecessor's views in order **A0,A1,A2,A3,B0,B1,B2,B3**. For each stream, restore its CPU/CUDA RNG, run source member0, save that stream's forward RNG endpoint, and leave its `fork_rng` context. Compute the original complete native CE on all 580 labels, divide it by eight, and call backward with default graph release. Delete attached logits, representations and losses before the next forward. Retain only a detached scalar loss diagnostic and accumulated parameter gradients.

After all eight backwards, check that parameter versions equal the old versions and that active gradients are finite. Then perform exactly one native Adam step. Inactive native global/head parameters retain `grad=None`. The complete gradient is

`sum_(v,m) grad_theta CE(f_theta(x; stream_m, view_v), y) / 8`

at the same old parameters. This is the predecessor's `mean_m [.5 CE_A,m + .5 CE_B,m]` by linearity. Floating point accumulation order changes; no bitwise gradient, trajectory or author-runtime parity is asserted. The bound concerns the number of live backbone autograd tapes by source lifetime. Peak allocated/reserved memory and actual full-input feasibility remain unqualified.

The original source gives one native body, one Adam, dropout stream seeds `seed + 1009*m + 300001`, stateless WikiCS computation, and deterministic one-function evaluation. No new initialization draw, factor installation or optimizer construction is added. Source recipe contrastive fields remain inherited dormant metadata; effective auxiliary weights are explicitly zero and the training code contains no auxiliary call.

## Full driver and callable interface

The scaffold-compatible interfaces remain:

```python
make_session(
    task='wikics', control='single8_own', seed=6101, device='cpu',
    polynormer=bound_native_path, public_root=bound_public_v2_root,
    later_execution_authorized=False,
)

run_complete(
    task='wikics', control='single8_own',
    train=bound_train_path, valid=bound_valid_path, output=fresh_output_path,
    seed=6101, device='cpu', polynormer=bound_native_path,
    public_root=bound_public_v2_root,
    later_execution_authorized=False,
)
```

These examples are intentionally disabled. NCN arguments remain signature-compatible but are rejected when non-null. A future caller must separately bind roles/source/runtime and authorize/qualify the actual complete path. The public V2 manifest and all its payloads are verified by the unchanged scaffold loader.

`run_complete` still calls the unmodified public V2 `train.main`: 1,100 complete epochs, 100 local then 1,000 global, strict-first complete development maximum, original joint local model/Adam restore at epoch101 with **live end-local TRAIN streams**, and deterministic single-body serving. It has no shortened horizon, early stop, checkpoint reselection, view-specific selector or TEST reader. The original driver may save unused native own-best diagnostic files for the underlying single body; this successor serves its original coherent joint selected checkpoint. Exact resume remains unsupported.

## Work and custody

Per successful update: eight native TRAIN body forwards, eight complete CE views/backward calls, 4,640 label-view presentations, one old-parameter version check and one Adam step. Full-fit expected counts are 8,800 TRAIN forwards/backwards, 5,104,000 label-view presentations, 1,100 updates/Adam calls and 1,100 validation body forwards. Shadow/replay and auxiliary work are zero. This does not equal graph12's replay/reverse work or establish equal runtime, capacity or inference opportunity.

Attempted and successful forward/backward counters are distinct; partially completed work is retained in original failure artifacts. `maximum_live_TRAIN_backbone_tapes` is a source structural bound recorded when a view is constructed, not an allocator measurement. Selected metadata and RUN/PROGRESS/COMPLETE/FAILURE annotations carry the successor source hash, predecessor identity, execution mode and counters. COMPLETE requires the full count contract.

`SOURCE_DIFF.patch` records the exact source change. `SOURCE_BINDINGS.json` pins the authoritative predecessor, unchanged public package and relevant saved attribution principle. `STATIC_CHECKS.json` records AST/JSON/manifest inspection only: no adapter import, numerical dependency import, synthetic training, model call, runtime qualification, fit, or outcome opening. `PROTOCOL_DISABLED.json` leaves all execution/adoption/opening flags false. The existing scaffold, reference packet, public package, graph12 gates, canonical records and jobs remain unchanged.
