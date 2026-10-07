# Setup and complete WikiCS inputs

Use Python3.11 and the separately published
`portable_internal_be_public_interface_20261007_v2` directory (manifest SHA256
`190940ca9f8141ac45f739d064cb1ef76aaf1965ba8c91adb544ad8e38fef724`).
Keep it beside this folder or pass its path with `--public-interface`. The public
dependency supplies the native Session, complete data checks/batching and original
model/factor/selection primitives. No private research directory is read at runtime.

Recorded provider family: NumPy1.26.4, Torch2.1.2+cu118, PyG2.7.0,
torch-scatter2.1.2+pt21cu118, torch-sparse0.6.18+pt21cu118 and OGB1.3.6.
Choose Torch/scatter/sparse wheels matching your OS and CPU/CUDA ABI. Actual
versions enter PUBLIC_COST; this does not assert numeric equivalence across builds.
Torch uses2 threads/1 interop thread, TF32false and cuDNN benchmarkfalse.
OGB and torch_sparse are transitive imports of the public library even for WikiCS.

Example Linux CUDA11.8 setup commands for the caller (not executed here):

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install torch-scatter==2.1.2 torch-sparse==0.6.18 -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
python -m pip install numpy==1.26.4 torch-geometric==2.7.0 ogb==1.3.6
```

For CPU choose corresponding CPU wheels. Refer to the
[PyG installation instructions](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).
This package does not install dependencies or claim arbitrary platform support.

Obtain [Polynormer upstream](https://github.com/cornell-zhang/Polynormer) at commit
`fc8c276c9c5dfbd616d83f65338a3392188a5e08`. Pass its `model.py` with SHA256
`9b4e533f46ae7f91a23a552f88bbb47a01359e224b53996cf1a68c10a865f6a8`.
The public Session checks that exact source. No third-party native code, weights
or NCN source is bundled or needed; preserve upstream notices when obtaining it.

Supply numeric-only complete TRAIN/development NPZs:

| Role | Fields | Required population |
|---|---|---|
| TRAIN | x, edge_index, ids, y | full11701×300 features, prepared442907 directed edges,580 unique TRAIN IDs/labels |
| development | ids, y |5274 unique IDs/labels, disjoint from TRAIN |

Features are finite float32; edges/IDs/labels are integer int64 with the public
WikiCS domain checks. Ordered TRAIN IDs define mask rows and stable cosine ties.
The development role is the official split0 validation+stopping union. The
trainer consumes no TEST role or TEST metric.

You can use the public converter on official raw WikiCS JSON obtained independently:

```sh
python portable_internal_be_public_interface_20261007_v2/export.py wikics \
  --raw-json data.json --output roles/wikics
```

See public v2 `DATA_FORMAT.txt` and converter/source documentation for exact
preprocessing and pinned official ordered-array checks. Caller NPZ serialization
hashes need not equal the author’s NPZ files. Supplying arbitrary compatible arrays
does not become official data certification; the loader records that limitation.
No benchmark download or data-provider hydrator runs inside this training CLI.
