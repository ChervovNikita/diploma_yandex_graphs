# Exact isolated environment recreation recipe

This recipe is prepared from allocation metadata. Only syntax, capture and
dry-run checks are performed here. No package is downloaded or installed and
the currently running allocation `.venv` is untouched. No sudo, server77
access, CUDA pilot, scientific fit or GO occurs.

## Base interpreter prerequisites

The source environment is CPython3.11.14 on Linux x86_64, glibc2.35, SQLite3.50.4
and OpenSSL3.5.5 (27 January2026). A qualified base interpreter with these exact
recorded properties must be supplied before creating a new venv. `pip` does
not recreate CPython, glibc, SQLite or OpenSSL. This is a concrete portability
prerequisite, not a claim that the future server77 environment is replicated.
If its base platform differs, root must resolve and declare that difference
before environment qualification; the script rejects it instead of hiding it.

## Wheel and dependency contract

`prepare_sync_environment.py capture` records the exact installed version of
the transitive dependency closure of Torch/NumPy/SciPy/scikit-learn/PyG/scatter/
sparse/OGB, evaluating source Python/platform markers and checking declared
metadata constraints. Unused environment packages and optional extras are not
copied or installed. A direct-URL dependency needs its own frozen wheel receipt.

The mandatory numerical pins are Torch2.1.2+cu118, NumPy1.26.4, SciPy1.14.1,
scikit-learn1.5.2, PyG2.7.0, torch-scatter2.1.2+pt21cu118,
torch-sparse0.6.18+pt21cu118 and OGB1.3.6. The source CUDA11.8/Torch2.1 ABI uses:

- Torch wheel index `https://download.pytorch.org/whl/cu118`.
- Compiled PyG wheels `https://data.pyg.org/whl/torch-2.1.0+cu118.html` (Torch2.1.x ABI).
- Remaining explicitly pinned distributions from `https://pypi.org/simple`.

All downloads are binary wheels with `--no-deps`; no source build or silently
new transitive version is admitted. Installation resolves the complete exact
lock offline from the new wheelhouse, then runs `pip check`. Every acquired
wheel receives a size/SHA256 receipt. Wheel bytes are not claimed to have been
obtained or hash-qualified during this dry run. Source author CLI requirements,
private overlay and absolute library paths remain allocation-only and unchanged.

## Later authorized setup

The script defaults to a dry run. It prints exact argv and quoted shell commands
from a pinned recipe without touching the filesystem or network. Example:

```sh
python -B prepare_sync_environment.py dry-run --project-root PHASE --recipe TOOLING/RECIPE.json --recipe-hash PINNED_RECIPE_HASH --base-python /qualified/CPython3.11.14/bin/python
```

After later server77 reauthorization, `recreate` additionally requires explicit
expected hostname/GPU UUIDs matching the frozen final manifest. It creates a
fresh named venv and wheelhouse inside the project, rejects any existing target
or default/running `.venv`, and checks the supplied base before any installation.
It starts no worker and writes no scientific approval. The future environment
still needs the exact metadata/CPU binary-import check and separate GPU/resource
qualification. Existing `start_massive.sh` remains the only scientific startup
command after synchronization and explicit human GO.

Current one-second fit-cost placeholders are not an ETA. Environment capture
and dry-run checks do not establish training throughput or an execution window.
