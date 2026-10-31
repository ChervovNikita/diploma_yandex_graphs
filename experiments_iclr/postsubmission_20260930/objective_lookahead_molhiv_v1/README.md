# MolHIV actual-optimizer lookahead — source-only v1

This packet prepares a training-only mechanism screen for the exact unchanged
GINE model. It compares tied, first-private, middle-private and last-private
models after eight paired fresh-AdamW updates. Both member-wise BCE training
and direct pooled-logit BCE training are essential controls. The comparison
always uses the **actual tied-after-eight** model from the same objective family.

**Real-input admission is closed. No dataset, checkpoint, CPU execution result,
accuracy improvement or novelty claim is delivered here.** Read
`PROTOCOL.md` before integration. CPUv3 and failed grouping v5 are not valid
parents. Qualification of a grouping v6 parent does not by itself qualify the
matching CPU sibling or six P-only epoch-25 source cells.

The correct authorized one-GPU allocation must first be verified. Nothing from
the unauthorized seven-GPU allocation can qualify this packet's real inputs.

## Files

| File | Purpose |
| --- | --- |
| `lookahead_contract.py` | Fixed recipe/rule, source hashes, exclusive project outputs, real admission refusal |
| `lookahead_core.py` | Paired actual optimizer trajectories, all-screen pooled BCE, complete-matrix abstention |
| `verify_cpu.py` | Analytic autograd, actual GINE paired-null and scope/abstention synthetic gates |
| `runner.py` | Standard-library source preparation; explicit real-cell refusal |
| `PROTOCOL.md` | Prospective design, controls, future requirements and interpretation limits |

## Coordinator commands

Run from the project phase directory, using the approved repository wrapper and
supervisor. The following are child commands to wrap; they do not launch jobs
by being documented here. `PYTHONDONTWRITEBYTECODE=1` prevents writes outside
this sibling during model imports. Use a separately named output for retries.

```sh
PYTHONDONTWRITEBYTECODE=1 python objective_lookahead_molhiv_v1/runner.py prepare \
  --output objective_lookahead_molhiv_v1/SOURCE_PREPARATION_run01.json

PYTHONDONTWRITEBYTECODE=1 python objective_lookahead_molhiv_v1/verify_cpu.py \
  --output objective_lookahead_molhiv_v1/CPU_GATE_run01.json
```

The synthetic gate requires the later qualified CPU runtime's existing torch,
torch-geometric and OGB libraries. No dependency installation is performed by
the packet. A fresh process is required for the one-thread inter-op contract.

The real command below always exits with a source-only refusal **before opening
the binding file**:

```sh
PYTHONDONTWRITEBYTECODE=1 python objective_lookahead_molhiv_v1/runner.py real-cell \
  --binding objective_lookahead_molhiv_v1/FUTURE_APPROVED_BINDING.json
```

Real execution needs a separately versioned admission implementation after the
coordinator approves the successful exporter, frozen CPU sibling, gate and all
six complete P25 checkpoint/contracts/replays. No packet version is implicitly
promoted by editing the `SOURCE_ONLY` flag. A passing short mechanism screen
would still require a separately planned predictive study.
