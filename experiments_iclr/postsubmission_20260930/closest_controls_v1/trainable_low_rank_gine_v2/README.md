# Trainable low-rank GINE ensemble control

Prepared for source review. No remote launch, dataset/model-checkpoint access or predictive fit has occurred. Local Torch/PyG/OGB are unavailable; runtime gates are supplied and unexecuted.

This v2 sibling preserves v1 and its supervised initializer failure. The parent identified that failure as acquired on the unauthorized seven-GPU allocation using the wrong SSH account. It is retained to diagnose an API mismatch and is **not an authorized engineering result**. Torch2.1's `nn.init.kaiming_uniform_` rejected the `generator` keyword before any data access. V2 uses tensor `uniform_` with the same private generator and the exact legacy gain/std/bound calculation for Kaiming-uniform at `a=sqrt5` (algebraically, bound `1/sqrt(fan_in)`). No model, loss, rank, seed, scaling or environment was changed. `REPAIR_PROVENANCE.json` binds the failure and its scope correction. The added gate requires exact factor draws against legacy Kaiming inside a restored RNG fork for float32 and float64, plus global-RNG preservation and distinct member draws.

V2 is source-only and unlaunched. Run its future gate only after the parent verifies the original **one-GPU** allocation through SSH account `anogena-2.ai0001053-01174`, with all paths inside its authorized project repository. The inherited runtime fields are compatibility metadata; this packet does not certify them as the authorized allocation's runtime. If that runtime differs, a prospectively recorded compatibility amendment is required before execution.

This is a **TabLoRA-inspired GINE adaptation**, retaining the existing molecular projector ensemble's boundary factors. It is not an exact TabLoRA author implementation and claims no novelty or performance benefit.

## Method and attribution

TabLoRA §3.3 jointly trains shared dense weights and predictor-specific low-rank factors; §§3.4–3.5 describe separate predictors and averaging their training losses/logits. The retained primary paper is [Luo and Xu, 2026](https://arxiv.org/html/2607.10077v1). `SOURCE_BINDINGS.json` records its exact hash and previously inspected passage. `METHOD_SPEC.json` declares every adaptation.

For each hidden GINE MLP affine, member `m` uses `W x + b + B_m A_m x`. All six hidden dense maps are covered: `blocks.i.conv.nn.0` (128→256) and `.3` (256→128), for `i=0,1,2`. Shared weights and biases remain trainable, alongside every member's `A` and `B`. Scaling is one (`alpha=rank`), fixed before outcomes.

The existing AtomEncoder, per-layer BondEncoder, scalar GINE epsilon, LayerNorm, GELU/dropout, residual GINE message passing, input/output BatchEnsemble projectors, and mean-node graph readout are preserved. Shared biases and normalization/bond parameters remain shared. Each member retains its own complete node activations; the only pooling across members is the final mean raw logits. Loss is the exact inherited mean-member, equal-graph BCE over finite target entries, copied without semantic changes.

The tabular paper's group feature adapter, independent prediction heads, ReLU MLP and dataset/training recipe are not ported. Existing graph boundaries and GINE operations are retained to isolate additional hidden private capacity. This distinction precludes an exact author-reproduction claim.

## Initialization and fixed rank

`TrainableLowRankGINE(original, rank, init_seed)` copies a fully tied, width128/L3/M4 molecular model. Every inherited tensor is unchanged. Each `A_m` uses declared Kaiming-uniform random initialization and each `B_m` is zero. This is an explicit adaptation choice, not recovered author initialization. A private CPU generator consumes no global/dropout RNG. Before learning, every member exactly preserves its corresponding source function. The first `B` gradient is live; the first `A` gradient is zero as expected, then becomes live after `B` updates. Setting both factors to zero would create a dead branch and is rejected by the gate.

The primary rank is chosen **only by actual registered parameter counts**. `rank_plan()` constructs/counts legal ranks 0–128 and picks the count nearest to making the middle GINE block completely private for all four members; ties choose the lower rank. No graph, label, forward pass, optimizer or quality score enters this choice. If the mismatch exceeds 1%, the result must be described as a near-count control rather than a match.

The retained count receipt records 244,619 parameters for the tied model and 448,142 for one copied-private block; its allocation authorization is not certified by this packet. New shape accounting predicts rank22 and 447,371 parameters, **771 fewer (0.1720%)** than the target. This is not an exact count match. These new-model values remain predictions until the supplied gate constructs and records them on the verified authorized allocation. All twelve trainable `A/B` tensors are included: 202,752 additional scalars (811,008 bytes in float32). There are no added frozen basis buffers; actual total buffer counts/bytes are emitted by the gate. Parameter storage is not a propagation, activation-memory or latency saving claim: all 12 member graph propagations remain.

## Engineering gates

`verify_synthetic_cpu.py` runs only three constructed molecular graphs, including directed and repeated edges, a self-edge, an isolated node and an empty-edge graph. It checks:

- Actual production-width count-based rank choice, every private factor and all buffer storage.
- Effective dense-map outputs and gradients against an independent `W+B_m A_m` oracle.
- Complete inherited state, zero-delta evaluation logits, matched dropout outputs/RNG streams and shared initial gradients.
- Live first-step `B`, expected zero first-step `A`, then nonzero `A` gradients and actual factor updates during three fresh AdamW steps.
- Private gradient isolation and an intervention that changes only one member's complete graph path.
- Nonzero full-graph bond/scatter/residual/norm/readout outputs and autograd against an independent oracle.
- Rank-zero exact storage/function/dropout and equality after a real optimizer update.
- Rejection of zero-A/zero-B dead initialization; exact inherited BCE and final raw-logit pooling.

These are engineering tests, not a representative predictive experiment or evidence of superiority. Their losses and gradients must not enter research result tables.

## Parent integration

Upload this exact folder and run `COMMANDS_CPU.sh` only after source review and verification of the authorized one-GPU allocation/account. The script does not establish SSH route identity. It performs a separately supervised synthetic gate and no dataset preflight or predictive training. All outputs stay in this folder. Immutable source inventory explicitly excludes logs and generated outputs; failed attempts are preserved.

When a prospective molecular training protocol later includes this control, bind its model source and the **actual gate-produced rank plan** before any outcome. Construct a common tied source with that protocol's initialization/warmup, then call `make_model(original=common_source, init_seed=29001+1000*optimizer_seed)`. The control copies source weights and uses fresh optimizer state; the parent must declare any warmup/optimizer policy consistently for every arm. Use the same observed labels, graph order, dropout seed streams, original BCE, complete-validation schedule, raw-logit pooling and checkpoint criterion as its paired comparator. Do not tune rank after seeing results.

This packet does not amend a frozen protocol, launch training or release evidence. The failed earlier spectral development gate is not promoted by this new comparator preparation.
