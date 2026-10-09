# Inactive authentic full-input NSD qualifier

`qualifier.py` is a small optional callable/CLI beside the sealed native source and baseline runner. It performs **one full-graph own-TRAIN update for each of the four frozen native presets**, with the original NLL, backward and Adam; finite output/gradient/parameter/optimizer checks; fixed post-update state save; and fresh native/Adam/RNG reconstruction. It does not launch the competence screen or a synthetic audit campaign.

No numerical module, graph, label or checkpoint was imported/read/executed by the preparing agent. No server was contacted. Existing source and runner seals are unchanged.

## Root release and inputs

The public callable `qualify_roles(..., execute=False)` and CLI default inactive. Activation requires an enabled root release bound to both source seals, authentic role archive/metadata hashes, a fresh output directory, final device, permitted dependency overlay, actual package versions and explicit determinism/TF32 policy. `RELEASE_TEMPLATE_DISABLED.json` is not an authorization. Native qualification is not required to have passed beforehand; this is the engineering step that records its evidence.

Future authorized invocation:

```sh
python qualifier.py --execute --release ROOT_QUALIFICATION_RELEASE.json --roles /authorized/roles.npz --output /fresh/qualification --device cuda:0 --dependency-overlay /authorized/new_dependency_overlay
```

Transfer this folder beside `private_sheaf_train_valid_runner_20261009_v1` and `private_sheaf_native_source_design_20261009_v1`. The qualifier verifies the existing source chain and calls the sealed adapter/CPU topology placement helper. Root reports Torch2.1.2+cu118, NumPy1.26.4, PyG2.7.0 and real torch-householder1.0.1 in a new dependency overlay; those remain target facts supplied by root until the activated process records versions and the genuine eager dependency origin. No package installation, downgrade or numerical source change is included.

The six-key official TRAIN/VALID numeric role archive is authenticated before decoding. The qualifier loads only `x`, `edge_index`, `train_index`, `train_y`. VALID arrays stay unopened. It checks finite full features, every TRAIN row, binary TRAIN truth and actual canonical sorted unique reverse-paired support counts. No edge count is guessed. Complete graph features/support are the requested transductive input; no TEST truth, mask, metric or TEST-selected predictions are read or emitted.

## One placement comparison, four updates

The first preset (`d2_f32_L2`, initialization seed1103) optionally builds one unchanged author model directly on the final device and compares every static graph/index/degree tensor and seeded native initial state exactly against the CPU-built/transferred model. This is one **authentic complete graph** comparison. Original direct CUDA construction can perform many scalar host-device reads; if root judges it impractical, `--direct-index-check skip` requires a bound root reason and records placement parity pending. A failed direct comparison remains a failure; it does not silently certify placement or prevent the four prescribed engineering update attempts.

Each preset uses own-update seed `1103+1000003`, isolating dropout and original CPU SVD jitter from optional comparator RNG consumption. It performs one unscaled all-TRAIN NLL backward and Adam update with the sealed `.01/.0005` settings. The selected engineering state is the fixed completed post-update state, not a validation-selected model. It saves only TRAIN post-update logits with model/Adam/RNG state, reconstructs a fresh original model and optimizer, restores all streams, verifies exact parameter/buffer state and finite Adam state, and checks TRAIN logp roundtrip within `1e-6`.

Every preset has a result, including failure stage and partial costs. Records expose source/role/config/seed identity, actual parameter/topology storage, support and label opportunity, forward counts, timings, cumulative process RSS and CUDA allocated/reserved peaks. CPU RSS is a process high-water mark. A skipped/failed static comparison cannot produce a placement pass. No VALID metrics are computed, and the summary establishes no baseline competence or performance. Ordinary Tolokers remains an original benchmark exploratory task.
