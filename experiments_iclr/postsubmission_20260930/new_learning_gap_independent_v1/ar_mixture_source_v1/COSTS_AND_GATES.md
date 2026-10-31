# Costs and outstanding gates

Status: source formulas/counters only, with no executed measurements or forecast.

## Complete charged workload

Let K be total generated schedules, with S schedules per branch in the balanced case `K=M S`; let `p=d+2`, and let N,E describe the supplied full graph. One objective batch uses **one** source graph encoding, with propagation/dense counts supplied by the audited source recipe. H and its common encoder autograd graph are reused throughout the batch.

Each complete generation or density trajectory performs N GRU-style steps. Per step, dense matrix multiply terms are `3h p+3h^2+2h`. The factored arm adds `p+8h+2` diagonal multiply elements and `6h+2` private gate/output bias additions. The symbolic ledger also counts `3h` sigmoid/tanh elements per step. These are selected operation counts, not a complete FLOP count. Prefix-mask neighbor traversal has a `2E` upper bound per trajectory; early termination means this is not an exact measured visit count.

The minimal implementation performs:

- K generation trajectories under no-grad, consuming N caller-supplied uniform slots per trajectory.
- `K M` complete teacher-forced density trajectories, including K generating self-density recomputations and `K(M-1)` extra cross-density forwards.
- A full `K by M` density matrix and K log-mixture reductions for **both** mixture-KL and mean-branch-KL objective calls.

For `K=M S`, the density matrix has `M^2 S` contributions. Generating self-density reuse from sampling is not implemented: sampling is discrete/no-grad and the K generating scores are replayed to obtain score derivatives. With a separately qualified value-reuse implementation, there are `M(M-1)S` extra nonself cross-density forward calls beyond self scores. No sparse message work repeats for those decoders.

Only the K generating-score replays retain autograd; nonself cross-density trajectories and generation are forward only. Backward work/activation storage therefore need **not** be quadratic in M. The current source retains the common encoder graph plus K generating recurrent score graphs, while nonself density replay needs transient state storage and its matrix scalar values. Backward and exact peak-memory costs remain unmeasured. Processing/reduction/chunking changes need qualification before a memory benefit can be claimed.

The mean-branch-KL objective intrinsically needs only own densities. Its natural cheaper version would omit the `K(M-1)` cross forward calls and mixture reductions. This source intentionally computes the full matrix for the objective-only ablation; any efficiency report must also disclose the cheaper intrinsic-cost mean-KL control. One conventional single decoder likewise has M=1 and a cheaper natural density workload; parameter/resource matching remains a prospective comparison choice.

## Parameter accounting

Let `W=3h p+3h^2+2h` be decoder matrix parameters. The factored arm has W shared matrix entries and `M(p+15h+4)` private factor/bias/initial-hidden entries. A conventional single decoder has `W+7h+2` entries. An untied M-decoder arm has `M(W+7h+2)` entries. These exclude the one common source encoder and assume distinct injected parameter storage with the documented shapes. No width or parameter cap is fixed here.

## Unmeasured and incomplete counters

The ledger does not measure wall time, GPU utilization, bytes or full backward arithmetic. Source recipe audits, encoder activation memory, recurrent state/gate/score activations, parameter/optimizer memory, density matrices, assignment/uniform Python objects, adjacency/order storage, log-softmax, scalar reward reductions, labels/log-fugacity tensors, host/device transfers and allocator overlap require measured accounting.

The serial Python implementation performs finite-value checks and sampling-threshold `.item()` calls that synchronize host/device execution. This may dominate GPU latency. Reference/task construction, uniform/branch RNG generation, exact reference normalization, all repeated graphs/batches, independently obtained baselines and adverse/failed work must also be charged. Feasibility/provenance validation and short-circuit mask work are not fully measured by the current symbolic counters. There is no end-to-end latency or scalable-training claim.

## Required prospective qualification

No routine below was run. Future admitted checks should cover:

1. **Source/architecture:** authorized encoder provenance and dropout-zero/node-wise-normalization audit; full source/conflict-graph binding; one actual encoder call per batch; fresh paired initialization, tensor storage/aliasing and parameter counts.
2. **Normalized law:** exhaustive tiny-instance branch normalization; all and only independent-set support; teacher-forced complete densities; identical sampling/density mask/state equations; source node-order remapping and symmetry/order sensitivity. Forced-zero decisions must have zero direct density derivative while their states still influence later steps.
3. **Sampling:** external uniforms/branch draws obey the declared design; continuous ideal versus actual finite RNG precision; frequency/density agreement; log-softmax saturation/underflow and explicit boundary failures. No exact floating-point sampler claim is made.
4. **Estimator:** on tiny fully enumerated laws compare exact objective gradients with expected score gradients through shared/private parameters; verify the explicit normalized-density derivative cancellation; inspect detach boundaries and baseline provenance; include nonuniform pi and stratified counts. No own-sample-including baseline or leave-one-out mixture reward is admitted.
5. **Controls:** full-matrix same-work objective ablation, naturally cheaper mean-KL and single-decoder cost disclosures, matched untied decoder and appropriate structured/variational baselines. Feasible outputs alone do not qualify target-law accuracy.
6. **Resources and scientific freeze:** representative training-only CPU/GPU/memory/timing qualification under unchanged phase/diagnostic caps and one-A100 limit; full graph-level task/outcome/control/configuration/seed/reuse freeze before fits. No claims about arbitrary future-lock conditional sampling.

The exact hard-core reference's brute-force CPU/fugacity arithmetic envelope remains independently unqualified, as disclosed in the sealed earlier packet. This AR generator does not require an oracle normalizer in its training reward, but reference evaluation and any exact conditional comparison still require their own qualified resource plan.
