# Representative CPU workload qualifier, source v2

## Status and admission

This packet contains prospective source only. It was authored and inspected as text, with file hashing, without importing or compiling it, constructing a model, generating a fixture or uniforms, running numerical work, installing software, accessing a server/GPU or launching a child. Earlier sealed packets remain unchanged. Root must review, pin and admit the packet before execution. An admission ID is a receipt, not an authorization mechanism.

The root's exact numerical diagnostic is separate prior work. Its pass does not qualify this actual-encoder resource workload or the copied evaluator. Existing phase/diagnostic caps and the one-A100 limit remain unchanged. No automatic run is authorized by this packet.

This version repairs the independent v1 review's two admission blockers: the optimizer is AdamW and an actual-full15 scalar/batched parity gate is added. The sealed v1 bytes are preserved and recorded in `PREDECESSOR_HASHES.json`; v1 remains unadmitted. This version adds no scientific evidence or control arm.

## Complete fixed workload

`FREEZE.json` declares one complete 15-node fixture, `resource_fixture_15_seed1729`, seed 1729, from the pinned spatial-family v2 generator. Keep every node and every generated edge; make one fixture after the parity gate. Evaluate homogeneous fugacities in order 4 then 1. The integer and sealed rational oracles both examine all 32,768 binary states, with an exact equality check on the full feasible `(mask, weight)` tables and partition. Rational caps are 32,768 states and 64 bits. Coordinates/order are recorded as provenance; coordinates, identities and oracle values are not encoder features.

The encoder is the actual pinned source `Model`: GCN, two layers, input 1, hidden 16, output embedding 16, hidden multiplier 2, one head, normalization `None`, dropout 0. Supply the raw simple graph as both directed edge orientations and one known `log(lambda)` value per node. CPU float64 and one intra-op/inter-op thread are fixed. The source encoder has 2,992 parameters by manual shape accounting, checked against the instantiated source before probing. Each encoding audit counts two propagation calls, `4E` directed aggregation visits and `15 * 2832` dense multiply terms; these counts do not attempt to count PyG normalization, allocations or kernel instructions.

| Arm, in execution order | Members | Hidden width | Expected raw decoder parameters |
|---|---:|---:|---:|
| Private recurrent factored | 4 | 16 | 2,712 |
| Shared transition, ordinary heads | 4 | 16 | 1,880 |
| Shared transition, factored heads | 4 | 16 | 1,856 |
| Single | 1 | 16 | 1,778 |
| Approximate parameter matched single | 1 | 22 | 2,840 |
| Compact untied, **SERIAL** | 4 | 8 | 2,792 |

These are source-shape predictions checked against runtime parameter counts. Add the common 2,992 encoder parameters to each row. The compact untied arm uses the existing serial private-decoder API; it is not a packed implementation.

Each arm/fugacity has one full15 implementation parity gate, one full initialization-state law evaluation and two objective probes, mix then mean. Thus a successful child completes 12 full15 parity gates, 12 full-law evaluations, 24 probes and 36 separately charged source encoder forwards. Every probe generates eight complete schedules: two/member for M4, eight for M1. It computes every complete cross-member density (32 entries for M4, eight for M1), invokes the unchanged scalar source score surrogate with uniform mixture weights and a predeclared zero baseline, completes one backward and one AdamW step. AdamW is learning rate .001, betas (.9, .999), epsilon 1e-8, weight decay zero, `amsgrad=False`, `maximize=False`, `foreach=False`, `capturable=False`, `differentiable=False`, `fused=False`. Fresh moment/state tensors are included in measured step work and raw memory; actual optimizer state entries/elements are recorded. Nothing evaluates the post-step law. Resource clocks include imports, construction/initialization, exact references, every tiny/full15 parity call, sampling, scoring, gradients, optimizer work and accounting/serialization.

## Initialization and objective pairing

Role seeds are the first eight SHA-256 bytes, big-endian, of `ar_cpu_resource_v1|1729|` plus the role. Every encoder matrix is Xavier uniform and every encoder bias is zero. The GRU has independently initialized Xavier input-gate blocks, orthogonal recurrent-gate blocks, a Xavier output base, and zero biases/initial states. Private recurrent input factors are Rademacher; output factors are Rademacher and paired with shared factored heads. Other diagonal factors are one. Ordinary shared heads begin at the same effective output maps as their factored counterpart. Untied width-8 branches use separate role streams. Model constructor RNG work occurs in a fixed scoped seed before the pinned encoder state is copied; it is included in resource clocks.

Generate one external `8 x 15` uniform table and copy those values to every probe. Forced-zero nodes consume ignored slots through the unchanged source sampler. Objective pairs begin with bit-identical parameter values and distinct storage. Construct both probes while the reference parameters are live, compare SHA-256 parameter digests, and check all three live storage address sets are disjoint before either step. Optimizers are freshly constructed for each probe; no optimizer state is reused. Compare each pre-step encoder output exactly against the initialized reference output. Reference and both probe storage are briefly live together, so the recorded raw memory includes this scaffold.

## Evaluation-only vectorization

`batched_evaluation.py` explicitly copies/vectorizes the pinned reset-after-affine GRU equations, affine output heads, fixed-order prefix inputs and hard-core masks. Assignment rows are vectorized in fixed chunks of 512. Nodes remain sequential. Private recurrent arms retain `A*M` logical trajectories; shared-transition arms use `A` trajectories and `A*M` head paths for A supplied assignments. Every head remains the actual two-logit source head. Private branch state/factors remain private. Masks depend on supplied complete y and are computed once per assignment for all members; their neighbor-inspection bound is `A * 2E`, independent of recurrent reuse. State and logits must be finite at every step. Forced-zero steps skip softmax and contribute log density zero; free-step log probabilities must be finite. All evaluator operations use no-grad.

Before any full15 evaluation or fixture draw, every arm must match the unchanged scalar source density on **every feasible tiny state** of empty0, edgeless3, path3 and triangle+isolate4, at fixed float64 `atol=1e-10, rtol=0`. Tiny H values are a fixed arithmetic proxy solely for this implementation parity gate. This gate does not prove parity on every full15 state. The full15 pass nevertheless retains every feasible assignment, accounts for all infeasible states as structural zeros, checks strictly positive feasible masses and normalization at `atol=1e-10, rtol=0`, and preserves a streaming digest of all full member log densities. There is no target-law fidelity statistic.

After the one fixture and exact-oracle check, every arm/fugacity compares the unchanged scalar density API with the copied batched evaluator using the actual initialized source H. The pre-frozen ordinal rule is **sorted unique `floor(j*(F-1)/15)` for `j=0,...,15`**, where F is the complete feasible-state count and the feasible table is in ascending binary-mask order. Retain all unique selections even if F is smaller than 16. Selection uses neither model scores nor parity errors. Compare all members at the same fixed float64 `atol=1e-10, rtol=0`, then require a pass before that arm's complete-law evaluation. Record selected ordinals/masks, maximum parity error, scalar source counter deltas, batched recurrence/head/mask counters, and separately completed scalar/batched row counts. The source/vector ledgers charge started paths; completion counts distinguish an interrupted call. All parity work occurs within measured stage/child clocks. This check samples implementation parity only and supplies no quality result or proof for unselected rows. The complete-law pass still evaluates **every feasible full15 assignment**.

One complete initialization-law pass is reused as a resource/normalization reference for the two identically initialized objective clones. Its actual source encoder forward and complete evaluator use no-grad. Both clones still perform their own actual source encoder forwards, scalar generation and scalar cross-density replay. Vectorization does not replace the sampling, gradient estimator or optimizer APIs. No post-update evaluation, support deduplication, support subset, numerical repair, free-step probability clipping or fixture replacement is offered.

## Binary heads and parameter interpretation

The pinned root `BINARY_HEAD_PARAMETER_NOTE.md` applies. An ordinary affine binary head stores `2h+2` two-logit parameters but has an ideal-arithmetic Bernoulli representation with `h+1` affine parameters. For the executed ordinary sources, replacing only those heads algebraically gives the following equivalent parameter totals:

| Ordinary-head source | Raw decoder total | Bernoulli-equivalent decoder total |
|---|---:|---:|
| Shared transition M4/h16 | 1,880 | 1,812 |
| Single h16 | 1,778 | 1,761 |
| Single h22 | 2,840 | 2,817 |
| Untied M4/h8 | 2,792 | 2,756 |

The factored source has its own shared-factor parameterization and is not assigned an ordinary-head subtraction. Any fixed set of its four effective binary heads can be represented by four affine Bernoulli heads, but this does not preserve its training parameterization. No additional compact-head architecture is constructed, trained or timed here. Actual parameters/latency describe the executed two-row sources. These raw counts do not justify a factor-sharing storage advantage over compact ordinary Bernoulli heads.

## Guard, clocks and incomplete results

`launch_guard.py` verifies the authoring manifest/seal and all external source pins, reserves a private new `runs/<run-name>` directory and launches exactly one `python -B qualifier.py` process group. It sets an empty CUDA visibility string and one-thread environment variables, preserves stdout/stderr and the incremental child journal, and prohibits a second run reservation. The internal alarm is 170 seconds; the supervisor hard timeout is 180 seconds from child-launch measurement, followed by SIGKILL of the process group. Failure/timeout is retained without retry or post-result protocol change. Runtime sufficiency is unqualified until this admitted complete workload finishes.

The guard's 180-second bound covers the child-launch envelope only: prelaunch hashing/reservation and post-timeout kill/reap/final serialization are outside it. The guard records the entire real launcher elapsed time but imposes no absolute whole-coordinator cap or cumulative-budget check. Root must supply the external whole-coordinator bound and charge all surrounding work; root has stated it will wrap the total coordinator envelope. The guard's timeout is not a claim that all coordinator work finishes within 180 seconds.

Child stages report actual wall/process CPU time and raw `RUSAGE_SELF`; the launcher reports real child/launcher wall time and `RUSAGE_CHILDREN` user/system CPU deltas. Both retain **raw `ru_maxrss`** with platform units; memory maxima are never subtracted, attributed per model or converted into stage memory increments. Raw child maxima include imports, templates, oracle/support tables, encoder/autograd objects, simultaneous paired storage and serialization. Incremental completed support rows and actual completed backward/step stages survive an internal failure; a hard kill preserves the latest successfully written journal. The old source `generating_score_backward_trajectories` ledger counts retained score graphs, so actual completed backward and optimizer calls are reported separately.

This is a representative CPU workload/source qualifier only. It produces no accuracy, development or confirmation evidence, original-score recalculation, scientific fit matrix, methodology promotion, scientific novelty or measured GPU throughput claim.
