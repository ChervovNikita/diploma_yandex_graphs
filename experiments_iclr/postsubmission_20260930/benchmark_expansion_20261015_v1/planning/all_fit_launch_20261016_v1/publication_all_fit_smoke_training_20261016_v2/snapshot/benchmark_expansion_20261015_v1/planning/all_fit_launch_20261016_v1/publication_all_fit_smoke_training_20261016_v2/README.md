# Repaired-path source publication, 16 October 2026

Source-only successor to `b7faa8f5061fa5acfba09e2c1bb5bab9edc3bc9f` on
`codex/all-fit-smoke-training-20261016`. The live pilot checkout remains at
`48329b19941ea869c439e5a220a3df0e30aa1237`.

This captures frozen port2, BUDDY path-portability, smoke2, execution3 and
projection-cache2 packages and their finite dependency closure. It preserves
the scientific design: 6,460 fits, 270 independent ensemble exports, unchanged
settings, and the user's conditional GO requiring every fit's actual smoke PASS
before full training. Publication does not launch or qualify training.

## Review and scope

All nine package manifests and their members match the selected source bytes.
The parent Git blobs were compared for 377 finite candidates: 212 files are
inherited byte-identically, 163 are new and two operational helpers changed.
The large frozen roster remains inherited. The exact user authorization is
selected even though it is already present in the parent. The closure audit
records dependency pins and the preserved original snapshot digest.

Six mocked stdlib publication-helper tests passed. Selected Python/JSON parse,
UTF-8, credential-marker and exact snapshot-byte checks are in CHECKS.json.
The two modified helpers add execution3 observation and compressed staging.
The coordinator source, README and test were frozen after concurrent edits;
its receipt records five passing offline tests and actual pending-state probes
on both hosts, with zero global markers and zero native jobs. The live
CONTINUATION.md is excluded; its parent version remains. An incomplete local
snapshot attempt was preserved after detecting the concurrent test edit.

## Known execution limitation

Root reported four execution3 BUDDY seed1/seed2 constructor failures at
provider.py:99: observed static_tensor_bindings differed from STATIC_PROVIDER
before any optimizer update. The path repair was reached; tensor identity
remains under separate diagnosis. All four logs remain with the execution
owner. No live numerical outputs are selected. A later provider successor may
be required. This snapshot makes no claim of all-fit PASS or training readiness.

## Git isolation

Snapshots are staged under this fresh publication directory. The helper uses
a temporary Git index and advances only the exact expected branch parent.
The live checkout/index and original publication snapshot are preserved.
The push verifies the exact remote parent and final commit. Credentials,
datasets, checkpoints and generated campaign payloads are excluded.
