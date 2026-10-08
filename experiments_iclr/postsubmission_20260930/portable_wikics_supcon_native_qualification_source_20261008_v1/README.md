# Two-update native SupCon engineering qualifier

This source runs exactly two discarded `supcon_eq2` TRAIN updates using the
sealed public successor: one local, then one global. It uses the full native
Polynormer geometry, shared unit BE, all 580 labels and the original fixed
512-node auxiliary panel. Root executes it under a finite process owner in the
existing normal GPU77 environment after inspection. No remote execution,
model construction or numerical test occurred during this preparation.

The script verifies the qualifier/public/SupCon manifests, existing TRAIN NPZ
and pinned model hashes, normal host/repository/executable, six provider
versions, physical GPU inventory and one mapped GPU. The owned allocator cap
is within the previously reviewed 24–32 GiB range, with 4 GiB fresh headroom.
Root supplies the physical GPU and a fresh engineering output directory.

Only TRAIN is loaded through the public core's numeric NPZ loader. Its full
projection and four ordered-array fingerprints use the same public WikiCS
pins as the successor. No development/TEST input is accepted. The unchanged
`make_session` and replay perform both updates. The facade is observed without
changing its numerical arguments; both live views, native `[4,512,512]`
representations and panel labels are checked. A CPU/CUDA audit of the original
linspace operator checks identity without replacing the replay's panel.

The first update starts in native local mode. The discarded model then calls
`set_global(True)` for global-path coverage. It creates no checkpoint and does
not invoke or qualify the complete schedule's selected-local restoration.
There are no extra inference calls, predictive metrics or selector decisions.
Existing finite-prediction checks inside the TRAIN update remain in place.

Expected total accounting is 32 training member forwards (16 shadow +16
replay), 16 member VJPs, two output-cotangent collections, two Adam steps and
two exact member-RNG endpoint checks. The public development-call count must
stay zero. Outputs are JSON only: per-stage `PROGRESS`, `QUALIFIED`, or
`FAILURE`; they record actual call counters, finite/nonzero accumulated
gradients, replay diagnostics, losses as engineering diagnostics, full source/
runtime/data identity and CUDA allocated/reserved peaks. No snapshots are
written. Both updates and their model are discarded when the owned process
exits.

Example inside the finite root-owned process, from the registered repository:

```sh
python portable_wikics_supcon_native_qualification_source_20261008_v1/qualify.py \
  --source-manifest-sha256 QUALIFIER_MANIFEST_SHA \
  --supcon-source portable_wikics_supcon_loss_comparison_20261008_v1 \
  --public-interface portable_internal_be_public_interface_20261007_v2 \
  --train wikics_unit_contrastive_attribution_gpu77_data_20261007_v1/train.npz \
  --polynormer wikics_staged_private_graph_residual_method_preparation_20261007_v4/vendor/native_polynormer.py \
  --physical-gpu-uuid ROOT_SELECTED_GPU_UUID \
  --owned-gpu-memory-cap-bytes 34359738368 \
  --output ROOT_FRESH_ENGINEERING_OUTPUT
```

Use the exact normal77 interpreter from `SOURCE_BINDINGS.json`, empty
`PYTHONPATH`, and `CUDA_VISIBLE_DEVICES` equal to the selected physical UUID.
The fixed engineering seed is 901337, separate from the paired scientific
seeds. There is no automatic retry, fit launch, schedule shortcut, data
download or installation. Root's external owner supplies the finite active/
cleanup/hard envelope and preserves other jobs.

The prior CPU semantic fixture passed, bound by its root receipt. This native
probe remains unexecuted. A pass would establish engineering eligibility of
these two paths only. Reuse execution gates, whole-family closure, original
scores and scientific method-quality criteria remain unchanged.
