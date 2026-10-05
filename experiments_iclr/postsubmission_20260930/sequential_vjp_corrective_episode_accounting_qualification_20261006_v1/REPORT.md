# Sequential corrective episode accounting and qualification plan

6 October 2026. **Unadopted complementary blueprint; prospective accounting only; no implementation, execution or fit admission.** The staged VJP schedule preserves the original finite private-SGD objective and derivative ownership. Its expected costs require measured callback and derivative counters before use. The old protocol and its 5100-call bill remain unchanged.

The preserved full-native run failed with CUDA OOM after 7.865677 seconds during the complete public episode, with peak allocated 80,001,296,896 bytes and reserved 81,570,824,192 bytes. Sequential scheduling is a proposed lifetime change; it has not demonstrated memory feasibility. The immutable receipt is bound in SOURCE_BINDINGS.json.

## Counted operations

M is the member count, P the number of supported unordered class pairs, and T=8 the solver steps. A native forward is one complete `family.forward_member` callback on the full graph, including its native message passing and global reductions. A private-gradient construction is one own-CE or independent-Q main partial with respect to a member's complete private row. A native member VJP is one reverse construction returning theta/private/Q adjoints as applicable; its nested private partial is counted separately. Small leaf-query and pair-map VJPs have separate counters.

These units count actual constructions, not equal FLOPs, equivalent backward passes or equal wall time. Backward work, derivative output dimensions, sparse solver kernels, cloning, snapshots and failed attempts must also be billed. In particular, uniform/stop-Q retain direct mixed shared credit through the main private update.

## Fixed stage ledger

| Stage | Native forwards | Private partials | Native member VJPs | Small work |
|---|---:|---:|---:|---|
| Detached response values | 2M | M | 0 | P primal maps |
| Detached main-adapt/query values | 2M | M | 0 | Query value |
| Joint leaf query cotangent | 0 | 0 | 0 | One small VJP |
| Differentiable main partial and query VJP | 2M | M | M | Accumulate independent-Q adjoints |
| Normalization plus finite Q-map VJP | 0 | 0 | 0 | P additional primal maps and P small VJPs when Q credit is retained |
| Separate original-before margin VJP | M | 0 | M | Negative cotangent for response costs; positive for current margins |
| Own-probe through after-margin VJP | 2M | M | M | Response-cost arms only |
| Complete theta+ value recomputation and private commit | 3M | 2M | 0 | P primal maps and complete diagnostics |

The first stage reuses the own-CE forward for before-margin values. The commit similarly reuses its own-CE forward, then evaluates the probe-adapted member and constructs the main private partial. This is permitted only under the unchanged pure deterministic eval callback and must pass parity; it is not an assumed saving for stochastic training callbacks.

At every stage, original private states are fixed. Query-logit cotangents come from all member logits jointly and are constants during subsequent VJPs. The map VJP starts from raw costs and includes member centering, the shared per-pair RMS scale and all eight finite steps. It is not an implicit derivative of an exact optimizer. Q remains independent inside the main private partial. At theta+, every probe/cost/Q/main step is recomputed from original phi; the old virtual private state is discarded.

## Control totals and prospective bill

| Arms | Native forwards per episode | Private partials | Native member VJPs | Pair primal maps | Pair-map VJPs |
|---|---:|---:|---:|---:|---:|
| live, graph_free, permuted | 12M | 6M | 3M | 3P | P |
| margins | 10M | 5M | 2M | 3P | P |
| uniform, stop_q | 9M | 5M | M | 2P | 0 |

Every episode also constructs one small query-logit VJP. All controls pay value probes and eight-step primal maps, including uniform's map before its output is replaced by 1/M. Current margins retain response diagnostics but have before-only Q credit. Gamma=0 does not imply a separable map. Permutation changes responsibility affinity only; native forwards and serving use the unchanged graph.

For M=4 and six H16 arms:

| Arm | Continuation forwards | Private partials | Native member VJPs |
|---|---:|---:|---:|
| live | 768 | 384 | 192 |
| uniform | 576 | 320 | 64 |
| margins | 640 | 320 | 128 |
| graph_free | 768 | 384 | 192 |
| permuted | 768 | 384 | 192 |
| stop_q | 576 | 320 | 64 |
| Total | **4096** | **2112** | **832** |

The continuation has 96 small query VJPs and 640 pair-map VJPs at P=10. Its primal work is 256 assignment banks, 2560 pair maps and 20,480 solver iterations. Retaining the original warm diagnostic adds one bank, 10 maps and 80 iterations:257 banks/2570 maps/20,560 iterations in the prospective complete bill.

The member-forward bill is **1600 common acquisition +4096 continuations +16 original warm diagnostic +28 fixed served evaluations =5740** (5712 fitting-phase callbacks and 28 served-evaluation callbacks), replacing the old 5100 estimate only if a future amendment is reviewed and admitted. The increase is640 calls. Private constructions rise from 1536 to 2112 in continuations; adding the unchanged warm diagnostic gives 2120. Acquisition's 400 ordinary optimizer/backward updates are separate operations, not these nested private partials.

A separately reviewed streaming warm diagnostic would cost3M=12 forwards and2M=8 private partials, yielding5736 rather than 5740. It is not assumed here. Keep H16, all six arms, warm 400 and all fixed evaluations intact; no shortened study is substituted to fit the old bill. Qualification costs and failures are additional, not hidden inside 5740.

## Memory and inference boundaries

The intended largest retained native-forward groups are main-support plus adapted-query, or probe-own plus probe-adapted-after. The original-before VJP is separate. Each member helper must finish and return only detached values/cotangents before the next member. Small per-pair maps are differentiated after native graphs are released.

The actual peak includes differentiated private-gradient intermediates in addition to those two forward graphs, full parameters/gradient accumulators, public features/edges, private states, cached logits/costs/Q, snapshots, workspaces and CUDA allocator/context memory. `retain_graph=False` is not a lifetime proof if a closure or returned tensor retains a graph. GPU reserved memory need not fall after release. MEMORY_PLAN.json specifies what must be measured.

For the declared roles, stored raw pair costs or Q have S(C−1)M=39,184 scalar entries; query logits/cotangents have MRC=49,000 each. These small detached banks do not replace native full-graph activation costs. No scientific tensors were read or allocated here.

Serving remains four complete native forwards and the arithmetic mean of four softmax probability vectors. It requires no probe, assignment solve, private update, meta-VJP or inference-time adaptation. The additional work belongs to training and engineering qualification; model-storage sharing supplies no inference compute-saving claim.

## Fixed numerical qualification specification

Use the original synthetic state, features, graph, role indices, dtype and complete native architecture. Compare all six controls with the unchanged monolithic operator at that same state. Each control starts from fresh identical theta/phi values; check raw responses/current costs, normalized costs, Q, probes, virtual private states, complete query logits/objective, every theta gradient coordinate, any reported outer-phi gradient, theta+, committed private states and all existing diagnostics. Outer-phi inspection gradients never replace the private SGD commit.

Use the existing CPU-FP64 comparison conventions: values/costs/Q/logits at atol 1e-10/rtol 1e-8; full derivative vectors at atol 1e-8/rtol 1e-6; returned commit states at atol 1e-10/rtol 2e-5. Use the fixed mixed direction and local FD displacements 1e-6 and 1e-7 with tolerance 5e-6+1e-3|analytic|. Preserve the earlier coarse failure. Bind prior same-state local FD evidence separately and charge every repeated signed value call. Require exact zero unused local-head tangents and unchanged unused private rows, live/stop-Q identical pre-core forward values, and stopped/direct credit matching independently anchored Q. Do not demand equal live/stop committed states after their different core gradients.

The per-oracle matrix in QUALIFICATION_PLAN.json records the physical charges: original response4M, original outer gradient/value5M, complete original public episode9M, original probe reconstruction3M and anchored-Q gradient2M. It separates these from staged calls and small cotangent/map checks. Raw/probe capture that adds no callback must not be charged as an extra model call, while a separate reconstruction must be charged. The root qualification worker must freeze its actual invocation matrix before execution; observed counters, including failed calls, are authoritative. Inherited callback/alias/sparse checks require matching pins; any rerun adds its actual bill. A streamed scalar FD value costs 4M forwards and 2M private partials per signed point; a full episode must not be mistaken for that cheaper value call.

The original complementary proposal retained here specified one common probe oracle, per-control response/gradient/commit/fixed-Q oracles, small cotangent/map checks, four signed local FD values and two unused-head values. Its unadopted planned aggregate was **301M=1204 forwards and148M=592 private partials, plus13M=52 staged native member VJPs and12 monolithic outer reverse constructions**. This historical proposal is preserved without using it as root's actual oracle bill. Root has explicitly declined to adopt or require reconciliation of that aggregate; its frozen worker invocation scope, measured candidate counters and separately scoped oracle work govern actual execution.

After synthetic parity and independent source review, full qualification uses the reviewed safe public+B accessor, exact preprocessing/context, FP32 and the authorized device. Complete all six read-only public episodes and fresh streamed theta+ commitment checks; discard constructed states. This plans82M=328 forwards,45M=180 private partials and13M=52 native member VJPs. Do not rerun the known-OOM full monolithic oracle. Log actual allocated/reserved/RSS peaks, phase/member wall times, shapes, all counters and restoration. Full-graph directional FD remains a separate explicit optional scope; its additional signed values are charged and an omitted test remains unqualified.

Retain the existing fixed limits of 180 seconds for the complete synthetic block and 900 seconds for the complete full block. A prospective root-reviewed limit change requires a distinct specification before execution. No automatic retry, allocator tuning, precision reduction, smaller graph, width change, frozen features, stopped gradients or state substitution follows failure. A timeout/OOM retains its last phase, partial counters and external failure receipt.

State/mode/attribute/buffer/input/RNG/alias and process-local gate restoration are mandatory. Source counters are currently **unmeasured**. This document grants no numerical pass, resource feasibility, fit budget, W/S/R fitting, A scoring, predictive benefit or novelty. The source-only independent algebra review is separately bound; implementation review and numerical execution remain root-owned.

## Binding to root’s adopted engineering scope

ROOT_SCOPE_BINDING.json separately binds the sealed root decision. Root adopted a 600-second all-six complete-native CPU float64 synthetic comparison and a first 900-second oneLIVE full-context float32 memory gate. The latter is not all-control qualification or fit authority. Successful all-six full-context float32 control and commitment/resource qualification under a separately fixed successor remains mandatory before acquisition or fitting. This report’s original 180-second/full-six/301M blueprint is explicitly unadopted; no old limit or failure is retroactively rewritten. Scientific fitting remains disabled.
