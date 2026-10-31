# Current-member mean propagation: source-only prototype

This packet implements the one unadopted hypothesis in the sibling `propagation_cost_scout_v1`. It supplies sparse execution and a bounded synthetic gate. It contains **no dataset, checkpoint, predictive result, measured speed advantage, or scientific launch authorization**. The original molecular packets and earlier scores are unchanged.

## Computation and claim

Four full-width source TABM/SAGE members keep their original shared blocks and boundary factors. At each layer, compute `Z_m = LayerNorm(H_m)`, its differentiable member mean `Zbar`, and the centered deviations `Delta_m`. The candidate replaces full propagation of every member by:

```
G_m = P Zbar + 4 C_c Delta_m
```

`P` is the complete incoming-neighbor mean. The four fixed color supports partition canonical coalesced CSR entries and keep their original weights. The same color is used for every member. Only the **linear aggregation mean** is exact: `mean_m G_m = P Zbar`, in exact arithmetic. Uniform training colors give conditional unbiased aggregation relative to the current input states. Nonlinear states, logits, losses, gradients and AdamW updates have no such guarantee. Existing control-variate and graph-sampling methods remain closest-prior constraints; this packet establishes no novelty.

## Implemented controls

| Arm | Linear propagation | Parameters | Execution paths |
|---|---|---|---|
| Candidate | full mean plus quarter-edge deviations | source TABM/SAGE tensors | coupled packed states |
| Exact member | complete `P Z_m` | same tensors | packed; serial; serial CE backwards with one optimizer step |
| Naive shared sample | `2(C_c+C_(c+1)) Z_m` | same tensors | packed/serial; cached half union or two quarter calls |
| Single BASE | complete `P Z` | source ordinary SAGE tensors | single trajectory |

The candidate and naive control both use `2 E D` sparse edge channels **in expectation**; realized color counts are recorded. The exact ensemble uses `4 E D`. These are arithmetic counts, not latency claims. Candidate serial full trajectories would change its coupled computation and are refused. Exact serial training is an important low-memory baseline: it accumulates four own-member CE gradients and performs one AdamW update.

Whole residual blocks can be checkpointed for all arms. Colors and per-member dropout keys are fixed on recomputation. Every sparse recomputation and backward appears in the operation ledger. A score-blind future gate must choose the fastest source-equivalent exact training and inference paths before scientific fits; it cannot compare only against a slow packed implementation.

## Source and graph fidelity

- `_pinned/frozen_models.py` and `_pinned/frozen_centered_paths.py` are byte-identical copies of the scout's specified source, checked by SHA256 before model import.
- Transfer copies every boundary factor, shared SAGE linear, FFN linear, LayerNorm weight and bias. The ordinary BASE transfer is implemented too.
- Raw incoming degrees are computed before duplicate coalescing. A coalesced weight is multiplicity divided by that original degree. Existing self edges are retained; isolates aggregate zero; no loops are inserted.
- Coloring happens **after** canonical coalescing. This resolves the raw-duplicate ambiguity while preserving disjoint color supports and workload accounting.
- Fixed CSR and explicitly prebuilt CSR adjoints use sparse matrix multiplication. No `E × M × D` edge-feature gather exists. Adjacency values are fixed; higher-order derivatives are deliberately unsupported.
- Training dropout uses member/epoch/layer/site keys independent of the NumPy PCG64 color stream. It has the source Bernoulli distribution and pairs packed/serial arms, but its RNG order is changed. Source numerical equivalence is tested with dropout disabled; paired training-noise equivalence is a separate synthetic check.

## Verification status

`STDLIB_GATE_local_run04.json` records 30 passed rational/source checks. They cover duplicates, self edges, isolates, color partition, conditional linear unbiasedness, shared-mask mean identity, the differentiable-centering adjoint, and failure examples for independent masks, sampled-degree normalization and nonlinear expectation. They do not import Torch or NumPy.

`runtime_fixture.py` is runnable source, **not run by this packet's author**. It checks source logits and all parameter gradients, CSR adjoints, four-color means, packed/serial dropout pairing, one-step serial AdamW equivalence, lazy/cached naive equivalence, and checkpoint gradients/recomputation accounting. Its inputs are twelve hand-constructed edges on six nodes. Passing it would establish synthetic mechanics only.

With working directory set to the project `postsubmission_research_20260930` folder, an independently authorized operator can run:

```
PYTHONDONTWRITEBYTECODE=1 python -m propagation_cost_impl_v1.runtime_fixture \
  --device cuda:0 \
  --output propagation_cost_impl_v1/runtime_gpu_fixture_run01
```

Use `--device cpu` and a distinct output name for CPU. It writes only inside this implementation directory, refuses to overwrite evidence, and supplies no remote connection or GPU-account discovery. Torch, NumPy and PyG must already be available; this packet installs nothing.

## Remaining admission blockers

See `FEASIBILITY.md` and `SYMBOLIC_RESOURCES_local_run01.json`. Actual complete Reddit2/products CSR edge counts, graph preparation cost, sparse-kernel correctness and performance on the authorized A100, full autograd peak memory, and the entire 24-fit forecast remain unmeasured. Full products feasibility is the largest uncertainty. The source-only fixture bounds and real-pilot refusal are permanent in this packet; a separately reviewed/frozen admitted implementation must add the complete-data interface and resource guard. No automatic adoption follows from the synthetic gate.

Every gate in the scout remains required: complete graphs and official splits; all four arms and three seeds; 100 full epochs with full validation; quality, useful diversity, naive mechanism control, and measured execution criteria. OOM or an infeasible family cannot be repaired by a smaller graph, mini-batch fallback, shortened runs or selected successful arms within this proposal.
