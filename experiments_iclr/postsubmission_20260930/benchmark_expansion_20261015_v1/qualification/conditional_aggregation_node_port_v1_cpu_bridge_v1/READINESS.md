# C6 native node bridge CPU qualification

**All three frozen C6 controls passed** the exact existing CiteSeer-Cheb native source case.

- One guarded allocation attempt; CUDA hidden; two CPU threads; Torch2.1.2+cu118/PyG2.7.0.
- Six real native constructions and six optimizer transitions: two fullTRAIN1929 updates per mode across fresh-session restore.
- Fifteen complete VALID665 passes: six initial native/bridge parity passes and nine post-update/restore passes.
- Original source mean initialization and fresh-restored actual/same-bank mean outputs reproduced bit for bit.
- All native/boundary/gate gradients present and finite. First F/L/J identical across controls.
- Exact native `_routes(True)` activation checkpointing, source per-parameter Adam groups, and native Parameter identity retained. Gates inherit existing lin3.weight group50 LR.0005/WD1e-5.
- Full transductive inputs/basis and TRAIN/VALID target/role state bindings preserved; no TEST targets hashed or scored.

Node bridge manifest: `aa90be4c1f989e6b4f81250eb3f30bb6a56ad75d9b242e445d539821be956a0c`.
Original C6 manifest remains `edef4b1f552a495e09e64855afbd96c7eb14ce5ef02d26cc13ec530d80e9805f`; the frozen core/package and molecular ports remain byte-identical.

Measured case-loop wall time after imports: 46.9772s. Full launch-to-final-log wall interval: 52.9688–53.9688s (launch timestamp has one-second precision, includes bootstrap/imports). Process CPU user69.9066s + system22.5063s; peak RSS4365624KiB. Resource preflight and every attempt/source/result/checkpoint/cost receipt are preserved in this directory.

Only the exact CiteSeer-Cheb CPU engineering cell is qualified. No Physics/link task, full scientific fit, TEST scoring, GPU model/worker operation, original score change, convergence or predictive-benefit claim. Other node cells, CUDA/resource admission, link C6 and final selected-state TEST reconstruction remain pending.

The native session exposes actual and same-bank mean VALID diagnostics. The unchanged generic runtime_v2 NodeAdapter propagates actual served outputs/accuracy for selection and drops extra diagnostic fields; campaign archival of mean diagnostics requires a later explicit runtime/analysis integration. No frozen runtime was modified here.
