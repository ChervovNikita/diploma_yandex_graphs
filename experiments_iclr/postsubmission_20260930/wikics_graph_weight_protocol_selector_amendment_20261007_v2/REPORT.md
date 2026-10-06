# WikiCS selector amendment v2

Root selected author-native selector semantics before outcomes. The single and shared banks retain one best VALID selector spanning all 100 local and 1000 global epochs. At the transition, restore the selected local model and optimizer, retain the live RNG and enable global mode. The best value is not reset.

Independent4 uses four separately optimized native models, each with its own author-native selector and transition. Freeze selected members, then pool their fixed probabilities. Charge all four 1100-epoch update/evaluation budgets. The shared bank uses pooled selection; independent models use individual selection. This difference is disclosed.

Persist and restore the selected forward-stage `_global` flag, which the native state dictionary omits. This repair permits a selected local state to be scored correctly after global training.

All weights, initialization, source bindings, splits, seeds, backbone, optimizer and update horizon remain as v1. No source, data, score or v1 packet was changed.
