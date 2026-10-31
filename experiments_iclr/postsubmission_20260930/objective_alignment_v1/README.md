# Prepared objective-alignment diagnostic

The local standard-library freeze has completed. `INPUT_LOCK.json` already exists; upload it with the source package and start at `verify_cpu.py` in the protocol commands. Do not recreate or overwrite it. AST parsing and local dependency-byte checks are preparation checks only. **No torch/PyG CPU gate or real six-cell collection was executed by the author.**

Inputs remain the exact P25 source, existing selected/random/raw bases, and previously audited training graphs9/18 with their original masks. The experiment takes no optimizer step and opens no validation trajectory. Every source/criterion outside this new directory remains unchanged.

Runtime sequence for the coordinator, through the existing repository-local cache wrapper and supervisor:

1. Upload this exact folder into the corresponding server phase.
2. Run `verify_cpu.py` with a new gate output filename. Continue only on `OBJECTIVE_SYNTHETIC_CPU_GATE_PASS`.
3. Run `collect_objectives.py run` with the existing input lock and passed gate. It checks inherited checkpoint bytes on the server before loading.
4. Run `collect_objectives.py replay` in a separate process with a new output directory.
5. Retain the supervisor records, raw objective-specific gradients, null evidence, numerical discrepancies and any failures. Numerical repair requires a new sibling version and new freeze.

The main quantities are local gradients of the native mean-member and pooled-logit losses with respect to centered private coefficients. They can explain objective mismatch; they cannot certify novelty, useful ensemble diversity, predictive gains, or AdamW benefit. Full purpose, numerical conventions, equations and limitations are in `PROTOCOL.md`.
