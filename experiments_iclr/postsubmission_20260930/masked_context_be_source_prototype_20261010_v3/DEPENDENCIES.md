# Dependencies

The numerical surface needs an existing ordinary Python3.9+ runtime with Torch,
NumPy, SciPy and PyG, plus the retained hash-bound PolyFormer class/utility source
and existing internal factor helper. Author environment is Torch2.0/PyG2.3;
the future caller's exact provider versions must be bound and numerically
qualified. No particular newer runtime's native parity is claimed here.

The new packet does not import the author dataloader or training script. It
avoids their unused DGL/OGB/data/download/plotting dependencies and module-level
seeding by compiling only the exact retained class/preprocessing definitions.
The source hashes are checked before extraction. No native author source is
copied into this new packet or cleared for redistribution by this work.

`plan.py`, `native.py` at import time, `static_check.py`, and CLI describe/help
use stdlib only. The numerical imports in `method.py` occur only if a caller
explicitly imports that surface; the provided CLI imports it after its separate
root engineering admission. There is no install, hydration, server or queue tool.
