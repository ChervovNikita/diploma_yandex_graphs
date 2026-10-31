# Compatible isolated environment recreation recipe

This recipe is prepared from allocation metadata. Only syntax, capture and
dry-run checks are performed here. No package is downloaded or installed and
the currently running allocation `.venv` is untouched. No sudo, server77
access, CUDA pilot, scientific fit or GO occurs.

## Base interpreter prerequisites

The source observation is CPython3.11.14 on Linux x86_64, glibc2.35,
SQLite3.50.4 and OpenSSL3.5.5 (27 January2026). These values remain immutable
provenance. Required base compatibility is CPython3.11/Linux/x86_64 plus usable
stdlib modules, SQLite transactions and an SSL context. A different CPython
micro release, glibc, SQLite or OpenSSL version is accepted and reported as an
observed difference. Pip does not recreate those interpreter/system libraries.

CPython major/minor and numerical wheel build pins govern the compiled ABI.
SQLite supplies registry storage; SSL supplies transport. Their version strings
are not scientific method parameters. Actual stdlib capabilities and native
wheel import/operation probes establish usability; a provenance match alone
does not establish binary or GPU execution compatibility.

| Field | Admission requirement | Saved observation |
| --- | --- | --- |
| Interpreter/platform |CPython3.11/Linux/x86_64|Full interpreter version and platform|
| Numerical packages |Exact eight pins below|Full package versions/build suffixes|
| Stdlib |Required modules, SQLite transaction and SSL context work|SQLite/OpenSSL/glibc versions and differences|
| Native extensions |CPU imports and actual scatter/sparse operations pass without CUDA init|Probe receipt|
| GPU environment |Separate root driver/resource qualification|Host/resource evidence|

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
python -B prepare_sync_environment.py dry-run --project-root PHASE --recipe TOOLING/RECIPE.json --recipe-hash PINNED_RECIPE_HASH --base-python /qualified/CPython3.11/bin/python
```

After later server77 reauthorization, `recreate` additionally requires explicit
expected hostname/GPU UUIDs matching the frozen final manifest. It creates a
fresh named venv and wheelhouse inside the project, rejects any existing target
or default/running `.venv`, and checks the supplied base before any installation.
It starts no worker and writes no scientific approval. The future environment
still needs mandatory numerical metadata locks, the CPU native binary probe
and separate GPU/resource
qualification. Existing `start_massive.sh` remains the only scientific startup
command after synchronization and explicit human GO.

Earlier exact-micro/system-library recipes and manifests remain preserved as
historical preparation. The v2 compatibility recipe supersedes that admission
rule while retaining their source observations and numerical package locks.

Current one-second fit-cost placeholders are not an ETA. Environment capture
and dry-run checks do not establish training throughput or an execution window.
