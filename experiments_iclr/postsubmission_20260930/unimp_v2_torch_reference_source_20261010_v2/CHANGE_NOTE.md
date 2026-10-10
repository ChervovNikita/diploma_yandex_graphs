# Narrow v2 readiness repair

Root's source review identified two issues in the unexecuted sealed v1 packet:
Torch 2.1.2 does not guarantee `get_device_properties(0).uuid`, and the README
misidentified the hidden edge-mask RNG as NumPy. V1 is retained unchanged.

V2 changes:

- Verify literal `anogena-2-0` and the sole authorized `nvidia-smi` UUID with
  stdlib before project metadata/payload access or numerical imports. CUDA
  visibility, when configured, must use the UUID. The exact authorized SSH
  destination/port remain frozen. CUDA count/availability and provider version
  checks still follow numerical imports; there is no UUID index/name fallback.
- Resolve and confine freeze/release, adoption receipts, TRAIN/VALID archives,
  own source hashes and the frozen fresh output before opening them. Reject
  outside/symlink-escaped study paths, reused output directories and outputs
  under sealed v1/v2 source packets. Ordinary runtime caches remain allowed;
  no namespace, mount, HOME or package location is changed.
- Document the actual RNG ownership: isolated NumPy RandomState shuffles TRAIN
  label episodes; Torch RNG produces hidden edge masks and feature dropout.
  Existing checkpoint fields already save both streams.

`unimp_v2.py` is byte-identical to v1. Architecture, graph transformation, float
0.1 pre-softmax adaptation, optimizer, loss, 377/203 label partition, seeds,
horizon and selection/replay rules are unchanged. Legacy int64-feed uncertainty
and the absence of Paddle runtime equivalence remain explicit.

Verification is stdlib source AST/compile, byte comparisons and manifest checks
only. No `nvidia-smi`, Torch/NumPy import, training, dataset, outcome or remote
operation was executed to prepare this repair. Numerical/full-input
qualification and execution still require root's separately bound release.
