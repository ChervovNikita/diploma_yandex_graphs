# C6 molecular CPU bridge qualification

**Passed all three controls** on the actual frozen MolBACE/GIN source cell.

- One guarded allocation attempt; CUDA hidden; two CPU threads; Torch2.1.2+cu118.
- Six native constructions and six optimizer transitions total: two per method, across fresh-session restore.
- Same32 real TRAIN graphs reused for bounded engineering. Native5-layer300-wide GIN, dropout.5, full categorical features, Adam.001 retained.
- Nine complete official VALID151 passes; both actual served and same-bank mean outputs restored bit for bit.
- Exact initial native mean, unchanged gate-attachment global RNG, equal first F/L/J across controls, original Parameter identity and gate state verified.
- No TEST access, scientific fit, selection/quality claim, original-paper rerun, or GPU-worker change.

Frozen method manifest remains `edef4b1f552a495e09e64855afbd96c7eb14ce5ef02d26cc13ec530d80e9805f`.
The exact script/plan freeze, launch, transfer, observation, full result and cost receipts are in this directory.

Timing: 12.2919s measured case-loop wall time after imports; 18.9955–19.9955s launch-to-final-log wall time including bootstrap/imports, bounded by the launch timestamp's one-second precision. Process CPU user23.6166s + system1.8580s; peak RSS1282984KiB.

Only this exact CPU MolBACE/GIN cell is qualified. Other molecular settings, node/link native ports, CUDA/resource admission, and final selected-state TEST reconstruction remain pending. The frozen method package's earlier pending-native status is preserved as source-freeze history; this separate result supplies the narrower later evidence. No predictive benefit is established.
