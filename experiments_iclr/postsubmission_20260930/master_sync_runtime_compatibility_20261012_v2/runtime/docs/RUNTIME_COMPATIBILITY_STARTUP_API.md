# Runtime compatibility startup API

Required execution policy: CPython3.11/Linux/x86_64, exact mandatory numerical
package/build locks, usable stdlib capabilities, and a successful CPU native
extension probe. Captured CPython micro, SQLite, OpenSSL and glibc versions
remain provenance; differences are returned explicitly. No bitwise runtime or
score-equivalence claim follows from compatibility.

## Root integration

Freeze both `prepare_sync_environment.py` and `prepare_sync_bundle.py` beside
the common controller. Their startup API imports only stdlib at module load.
`packaging` or pip's vendored parser is lazy and used only for recipe capture.
The API itself needs the eight numerical distributions it is checking; it
imports them in a separate CPU child using the caller's `sys.executable`.

```python
from prepare_sync_environment import verify_runtime_compatibility

receipt = verify_runtime_compatibility(plan["runtime_requirements"])
```

Root should bind the environment JSON to the final plan/source receipts, copy
its observed map into `plan.runtime_requirements`, and invoke this function
after physical host/source/approval checks and before detaching either startup
process. Preserve the returned receipt under that plan/host identity. It is a
plain JSON-serializable dictionary; errors fail admission before worker start.
It neither writes a file, alters an environment, registers a job nor launches
a worker. A later environment mismatch cannot be hidden by an old receipt.

The receipt includes source and observed environment hashes, the actual Python
executable, observed package/platform/library versions, mandatory pins/capability
policy, all provenance differences and actual CPU native-operation results.
The child validates imported module versions, then executes tiny scatter-add
and sparse-matrix multiplication probes and asserts CUDA is uninitialized.
Sparse module source may expose the public version without its wheel build
suffix; distribution metadata retains the exact mandatory pt21cu118 suffix.
The probe uses fixed synthetic values and no model, optimizer, target file or
fit. It does not consume the parent process's model-construction RNG stream.

## Launcher audit

The inspected launcher/start script checks source hashes, assignments, physical
hosts and scientific approval but has no standardized numerical environment
gate. There is no exact CPython micro/SQLite/OpenSSL/glibc equality rule there
to remove. Root must add the call above to make one-command startup enforce
the same compatibility policy as synchronization preflight.

The unchanged author worker's exact `cli.runtime_versions` and absolute
Python/PYTHONPATH/library bindings remain allocation-only. This compatibility
API does not broaden author placement or modify its recipe. Root GPU driver,
resource and whole-master gates remain separate. No server77 access, package
installation, final archive or scientific GO occurred during this preparation.
