# Common native features, private recurrent graph memory

11 October 2026. **One prospective nomination; no execution admission.** Preserve the complete negative scanner study and the sealed reset-before GRUBE comparator memo. Nominate a new staged package: a selected ordinary P1 first-layer feature basis, a common native second-layer decoder, and member-private factors and biases in a previous-state-conditioned **reset-after** GRU memory. This is a practical graph adaptation of known recurrent BatchEnsemble ideas, not a faithful reset-before GRUBE reproduction or a novelty certificate.

## What the closed study establishes

The scanner's frozen criteria were read before its complete scalar report. SW4 is closed: raw accuracy 93.013229%, below P1 93.212683%, S4 93.110023% and WI4 93.285034%; SW4−WI4 mean members is −.318247 pp. Every raw-reference, mean-member and attribution conjunction fails; calibrated confidence protection also fails. The tied ordinary PV1 loses to P1 in all three blocks, while V1−F1 is small and mixed. These are not a basis for a rescue grid.

Correct alternatives and useful served predictions diverge. SW4−WI4 has +73 net correct-member coverage but +351 pooling losses, with 475 repairs and 753 harms. In the S4 common-wrong-rival cohort, SW4 supplies 608 correct alternatives, loses 520 in the probability mean and serves 88 repairs; outside I4 coverage, the corresponding counts are 355/339/16. The proposed mechanism must retain member quality and make acquired correct scores survive serving without offsetting population harms. Neither hidden misalignment nor native signs have been identified as the cause. The competent no-memory S4 history contradicts a universal large native-factor deficit.

The complete report records 12 inherited banks/21 trajectories and 18 new banks/27 trajectories. Its paid acquisition inventory contains 9,306 inherited and 14,014 new Adam steps, respectively 16,209 and 41,872 TRAIN routes plus the same SELECT counts. The recorded new paid acquisition is 1,152.137299 s, peak CUDA allocation 1,647,672,832 bytes, reservation 1,889,533,952 bytes and RSS 2,696,798,208 bytes. Those costs remain attached to their closed study and owners; they are not costs or forecasts for this nomination.

## Two memory mechanisms, one nomination

| Mechanism on common native H | Private adaptation and information | Execution implication | Assessment |
| --- | --- | --- | --- |
| Input-conditioned affine memory | Private fast factors can change input/carry maps; diagonal carry depends on current H, with `s=a(H)*s_prev+q(H)` | Existing affine prefix machinery remains applicable; private maps can be projected once per node | Plausible cheaper package, but remains close to the failed input-only scanner. Its gates cannot respond directly to retained private history. No utility follows from adding factors. |
| Previous-state-conditioned GRU | Reset/update decisions depend on H and the private prior state; dense recurrent maps mix channels; private gate factors/biases adapt those decisions | Nonlinear sequential critical path; practical reset-after fused RNN dispatch is available in pinned source, subject to runtime qualification | **Nominate this one package.** It tests a materially different memory operator and allocation, while preserving a common demonstrated local feature basis. |

GRUBE, arXiv:2601.21581v1, already supplies private recurrent gate factors/states, mean own losses and probability averaging. Local graph propagation, graph serialization and node occurrence readouts also have known ancestry. MV-GMN and GraphScan were boundedly inspected in the sealed ancestry addendum; complete absence or primitive novelty was not established. A useful result here would be a graph-specific staged allocation package, supported by predictions, controls and costs.

The proposed GRU changes state-dependent gates, dense recurrent mixing, cell convention, native training stages and private parameter placement together relative to SW4. This design does not isolate one of those changes. The same-GRU placement control below is necessary; it still does not equalize private capacity or optimization geometry.

## Competence anchor and fixed native computation

Use the exact three encountered Physics blocks 10103/10427/10949 and original source graph/features, h=32, K=5 and TRAIN/SELECT/CAL/DEVELOPMENT roles. No TEST is opened. Body 0 is each block's previously selected ordinary P1 checkpoint, reused only after root verifies source/data/roles/selector/restore and custody. A strong independent bank additionally needs three fresh ordinary P1 acquisitions per block, each with its own source-law optimizer, SELECT selector, stop and RNG. No old DEVELOPMENT outcome selects among these bodies or checkpoints.

At stage 2, freeze each acquired P1 first-layer maps and bias; compute its native clip/ReLU H **in evaluation mode without first-layer dropout**, once. This caches a competent native local feature basis, not the complete final P1 predictor. Train the native second-layer aggregate/skip matrices and common bias, initialized from the selected P1. Apply its original p=.4 dropout independently in each body/view tail, feeding the same dropped H′ to aggregate and skip; keep original adjacency, clipping and output law. GRU dropout is zero. The changed stage-2 dropout law is explicit and matched across the stage controls.

Fixed H caching is outside autograd only in the common-H arms. Memory, residual, tail dropout/aggregation and the native decoder remain in the gradient graph; GN4 must also retain gradients through its first-layer computation to live factors and offsets. Do not apply `no_grad` around native decoding of H′.

The zero memory readout gives the selected P1 serving function at stage-2 checkpoint 0 in exact arithmetic; repeated clipping can introduce floating point roundoff, so bitwise identity is not asserted. Include checkpoint 0 in the same SELECT minimum rule. SELECT protection is not a DEVELOPMENT accuracy guarantee: decoder/memory adaptation may still weaken the predictor. Even a frozen decoder would not guarantee that a selected nonzero residual retains accuracy.

Use a fresh source TF1-Adam optimizer at stage 2: LR .001, β1=.9, β2=.999, ε=1e−8 before bias correction and float32 beta powers. Mean own-view TRAIN CE, no loss on pooled probabilities, plus one ordinary source first-layer .05 penalty, which is constant in common-stem arms. GN4 instead retains the mean live effective first-layer penalty. Memory is unpenalized. Latest tied minimum total SELECT loss includes step 0; stop after 50 strictly worse consecutive totals or 100,000 updates. Restore complete selected parameters, optimizer, modes and global/private RNG. Train decoder and memory jointly; no additional staged warming, schedule, grid or auxiliary loss.

New independent stage-1 body b∈{1,2,3} uses private source initialization seed `block+459000101+1000003*b` and persistent source dropout seed `block+459000111+1000003*b`, preserving the source draw law and leaving role/tour RNG untouched. Stage-2 ordinary/control shared GRU slow matrices use seed `block+459000131`; GI16 body b uses `block+459000131+1000003*b`, so body 0 is paired with the tied controls. All fast factors and bias/readout starts are deterministic. Tail view j of body b uses its own persistent generator with seed `block+459000151+1000003*b+1009*j`; the body-0 streams have paired starts across all stage controls, with separate generator objects and states. Native/readout/memory sharing is within a bank only; every comparison arm has its own trainable parameters, optimizer and selected state. No memory draw consumes a native/dropout stream. Root records exact generator states and draw-device binding before fits; this packet supplies no runtime admission.

## Concrete reset-after gate memory

For each gate g∈{r,z,n}, shared slow input and recurrent matrices A_g,B_g are each 32×32. Member m has private vectors r^x_g,r^h_g,s_g, each length 32, and private input/recurrent biases b^i_g,b^h_g. In PyTorch's column-vector convention:

`A_eff[g,m] = diag(s[g,m]) A[g] diag(r_x[g,m])`

`B_eff[g,m] = diag(s[g,m]) B[g] diag(r_h[g,m])`.

The effective matrices enter the standard reset-after cell:

`r_t = sigmoid(A_eff[r] x_t + b_i[r] + B_eff[r] h_prev + b_h[r])`

`z_t = sigmoid(A_eff[z] x_t + b_i[z] + B_eff[z] h_prev + b_h[z])`

`n_t = tanh(A_eff[n] x_t + b_i[n] + r_t*(B_eff[n] h_prev + b_h[n]))`

`h_t = z_t*h_prev + (1-z_t)*n_t`.

This differs from primary GRUBE's reset-before candidate map and its single concatenated bias. The two bias vectors and PyTorch's retention-gate convention are declared adaptations. All newly introduced fast factors start **ones**; all gate biases start zero except input z bias **+log(99)**. Here z retains the old state, whereas the older GRUBE memo's Z weights the candidate and used −log(99). Nominal .99 retention applies at zero learned affine contribution, not uniformly over learned inputs/states. Slow matrices use the documented GRU uniform ±1/√32 law. The shared residual readout W_o starts zero and has no private factors or bias:

`z_v = mean over all occurrences of v of (h_forward+h_reverse)/2`

`H′_v = clip_rows(H_v + .25*z_v W_o)`.

Use exactly the sealed four persistent label-free randomized-DFS spanning-forest Euler tours, with the same sampler seeds/draw order, occurrence means and component resets. Root reuses exact tour payloads after custody audit, rather than silently drawing new orders. Forward and reverse use the same effective gate weights and biases, h0=0 at every component and call; no learned initial state, carry between optimizer steps, cross-member hidden mixing, class input or truncation. Finite seeded tours remain ID-dependent; no per-realization equivariance or uniform spanning-tree claim is made.

Private factors and biases occur **only inside GRU gates** in GM4. The unfactorized W_o is common. Thus the old memo's private-readout confound is removed. Zero W_o delays gate learning until the readout acquires a nonzero path; this optimization effect is disclosed and shared by all equipped controls.

The falsifiable hypothesis is that this common local evidence plus privately adapted state decisions learns complementary, competent score functions whose correct alternatives survive the fixed mean. Memory can distinguish a local alias using different retained neighborhood histories, but cannot reconstruct raw attributes absent from H and the supplied structure/history. Frozen compression may discard exactly the useful distinctions; common slow memory maps may homogenize members; 100 labels may overfit the new memory; own losses may still create weak correct alternatives and pooling losses. Componentwise tanh bounds and convex retention do not bound long-BPTT Jacobians or guarantee trainability. These remain risks, not diagnoses of the closed scanner.

## Strong controls, fixed before outcomes

| Arm | Training/serving ownership | Resident parameters | Stage-2 trainable |
| --- | --- | ---: | ---: |
| B4 | One selected P1 stem; common native decoder; no memory; four own tail dropout streams, tied parameters and mean CE; four-view probability mean | 538,917 | 325 |
| G1 | Ordinary same-cell GRU, all native/memory/readout/bias parameters tied across all four tours | 546,277 | 7,685 |
| GF1 | One factorized same-cell memory parameter set, tied across all four tours | 546,565 | 7,973 |
| **GM4** | Common P1 H and decoder/readout; four private GRU factor/bias sets; member m uses tour m | **548,005** | **9,413** |
| GN4 | Same reset-after cell and four tours; ordinary memory/readout tied; four private native factor/bias-offset routes | 614,297 | 75,705 |
| GI16 | Four genuinely separate ordinary G1 learners; each has its own acquired P1 stem, native decoder, GRU/readout, optimizer, selector/stop and all four tours; mean of 16 body-view probabilities | 2,185,108 | 30,740 |

GN4 freezes the same selected-P1 native first-layer slow maps/bias, but recomputes H_m with live native aggregate/skip factors at both layers and additive private bias offsets. Factors start ones and offsets zero so it shares the initial P1 serving function. Common second-layer slow maps/bias remain live, as in GM4. First-layer dropout is disabled, as in the other stage arms. Private native capacity, penalty gradients, initial gate perturbations after updates and computational cost differ; **GM4−GN4 supports the specified allocation package, not placement independent of capacity**. No claim that ones initialization explains any closed failure is made.

Each tied single has one complete parameter set, including biases. Four logical views must not create four private parameter rows. There is no hidden averaging between views. Serving averages final class probabilities. GI16 bodies remain wholly unshared despite paired initial draws; body 0 may share the source checkpoint used to initialize the other comparison arms, but its stage-2 optimization and selected state are independent. Evaluate the acquired four-P1 probability mean, **GA4**, as an additional no-memory independent reference requiring no new fit. Its four selected source bodies are the same ancestors as GI16.

B4 tests whether the staged decoder/dropout law alone helps. G1 and GF1 test general single-GNN memory augmentation and factorized parameterization separately from private-member superiority. GI16 removes the old missing stronger independent-view reference. No superiority is inferred from parameter count, expressivity or a weak single reference.

## Utility, attribution limits and failure

Proposed prospective screen, requiring root admission before any acquisition: GM4 raw served accuracy must exceed each of P1, B4, G1, GF1, GN4, GI16, GA4 and the inherited WI4 by ≥.20 pp in mean, with strictly positive deltas in all three blocks. Report all other inherited fixed arms without changing their old gates. Compare mean GM4 member accuracy with mean **GI16 body accuracy**, where each independent body itself serves its tied four-tour mean: ≥−.25 pp overall and no block below −.50 pp. Also report all individual body/member and GI16 view accuracies; do not substitute view union for competent body prediction.

Separate confidence protection: CAL NLL may not worsen by >.005 nat in any block against P1/B4/G1/GF1/GN4/GI16/GA4/WI4, and its mean must be no worse than the best reference mean. Keep raw and confidence verdicts separate. Retain the source scalar-temperature law on cached selected outputs: float64 CPU Adam LR .05, 500 steps, logT start 0, clamp [−5,5], latest tied minimum CAL NLL over steps 0…500. Serving is the mean of `softmax(member_logprob/T)` across four GM4 members or tied views, and across all 16 GI16 body-view outputs. ECE remains unavailable/null; no extra binned metric rule or historical ECE is invented.

Close all new fits, export every fixed arm/ancestor selected state, complete all new CAL endpoints, then release one complete DEVELOPMENT readout. Stage-1 ancestors are selected with TRAIN/SELECT alone; no new stage-1 DEVELOPMENT release conditions the memory study. Against all strong references report repairs, harms, coverage C, covered-but-pooled-wrong loss L and aggregation-only correctness A, with Δcorrect=ΔC−ΔL+ΔA; report existing fixed S4/common-rival cohorts without using them to select the method. Mechanism success means served corrections with limited population harm, not merely larger C. No new inferential p-value or universal transfer claim follows from three protocols on one encountered graph.

Any primary raw/member failure closes this exact whole package. A successful G1/GF1 augmentation can remain useful even if private ensemble superiority fails. A positive GM4 result supports this staged allocation package; it cannot prove that previous-state gating alone, common features alone or long-range information alone caused the gain. No gate-zeroing, horizon, ordering, factor-init, loss, width, residual or decoder grid is included. Unused coherent confirmation is a later separate root decision.

## Practical fused execution and paid work

Materialize each effective gate tensor through live tensor products, stack in PyTorch gate order r/z/n and pass the four tensors `[W_ih,W_hh,b_ih,b_hh]` directly to a version-pinned GRU tensor-list operation. The nominated implementation is the `_VF.gru` entry used by official `torch.nn.GRU.forward`, rather than copying into module Parameters or in-place flattening. It is a private API and must be bound to the actual runtime. Public `torch.func.functional_call` is a possible wrapper, but `nn.GRU` refreshes flat weights and performs an in-place `no_grad` flattening path; its treatment of derived nonleaf tensors must not be assumed safe without qualification.

Pinned PyTorch v2.1.2 source supplies a concrete prospective gradient path: RNN.cpp dispatches GRU to cuDNN when input is acceptable; cuDNN's composite passes the original tensor-list weights to `_cudnn_rnn`; if they lack a common flat buffer, the forward packs/copies them; derivatives.yaml explicitly returns gradients for the **weight list**, and cuDNN backward remaps gradient tensors to its original shapes. Therefore a backend-owned packing copy does not imply factor detachment. Ordinary chain rule can propagate those weight gradients into slow matrices and factors. This is source-level evidence, **not a tested gradient or cuDNN dispatch receipt for the allocation runtime**. Never detach effective weights, use `.data`/copy into leaf Parameters, wrap products in new Parameters or cache them across an optimizer update. Reuse them only within that update's forward/reverse calls. Only first-order derivatives are required.

For multiple graph components, precompute an immutable packed-sequence gather/inverse permutation for the component tours and separately reversed components. Packed recurrent batch entries start at zero, so components never carry into one another; no per-token Python CUDA loop is needed. One route uses two fused RNN forward calls; GM4/G1/GF1/GN4 each use eight per complete four-view forward, GI16 uses 32 across its four independently selected bodies. These are API calls, **not one kernel each**. Packed descriptor cost, long sequence length, batch often near 1, dtype/layout, cuDNN acceptance and reserve/workspace size are genuine risks.

At h=32,N=34,493 and the upper bound L=68,985, standard fused input projections repeat per tour occurrence/direction. Per four-view forward, dense gate/readout projection work is `4*(12*L+N)*h² = 3,532,034,048 MACs`, excluding nonlinear gates, gather/means, native second layer, packing, backward and workspace. GI16's full four-body pass is four times that. The older memo's cheaper node-projection-reuse count belongs to a different implementation, not the nominated fused calls. Freezing H saves repeated native first-layer acquisition work after the cache, but stage-1 acquisition and H construction still cost work. A one-layer fixed native tail adds 44,151,040 dense projection MACs per four contexts, plus original sparse aggregation and other operations.

A projection-only full-gradient budgeting heuristic for one TRAIN update plus SELECT is about four forward projection totals, 14,128,136,192 MACs per four-view learner; this is not a bound or runtime estimate. Frozen GRU inputs and actual backend backward may change the count. Explicitly storing states/reset/update/candidate values for both directions and four contexts would occupy 269.47 MiB, before opaque reserve, adjoints, workspace, native states, indices and optimizer. Stream view backward contributions into shared gradient buffers before **one** Adam update to reduce concurrent tapes; preserve the exact mean objective, accumulation order and RNG receipt. Do not truncate or detach within-tour state. No empirical speed, peak-memory or readiness claim is made.

Acquisition inventory: stage 1 has 12 logical ordinary P1 fits (three exact prior references plus **nine new**), each owned independently. Stage 2 has **18 new fit banks /27 new optimizer trajectories**, with 96 recurrent view contexts and 12 no-memory tail contexts over the three blocks. Total fit inventory is 30 logical banks/39 trajectories; new acquisition is 27 fit banks/36 trajectories. GA4 adds three pooled-output/CAL endpoints and no fit trajectory. If each new stage-1 body receives the source single-output CAL procedure, new scalar-temperature work is 4,500 stage-1 +9,000 stage-2 +1,500 GA4 =15,000 steps; the three reused P1 CAL endpoints carry their original 1,500 steps separately. These counts do not predict training horizons. Charge TRAIN/BPTT, SELECT, cache construction, restores/exports, calibration, serving, packing and every stage-1 body; measure wall time and peak memory. Reused historical study costs remain inherited, never zero-cost replacements or newly acquired work.

## Custody and disposition

The new packet contains scalar/document arithmetic, a compact closed-outcome extract and public official-source evidence. No scientific library import, data/checkpoint/prediction/tour-array read, model/code edit, GPU/SSH operation, qualification, job, install or scheduling occurred. No sealed packet was edited. This is one distinct architecture for root assessment, with explicit capable controls, costs, causal limits and a known unqualified fused implementation path.

Primary and local authorities are listed in [READ_SCOPES.json](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/common_native_private_GRU_memory_nomination_20261011_v1/READ_SCOPES.json>), [REUSED_SOURCES.json](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/common_native_private_GRU_memory_nomination_20261011_v1/REUSED_SOURCES.json>) and the download receipts. The key official passages are [GRU cell and reset convention](https://github.com/pytorch/pytorch/blob/v2.1.2/torch/nn/modules/rnn.py#L900), [GRU cuDNN dispatch](https://github.com/pytorch/pytorch/blob/v2.1.2/aten/src/ATen/native/RNN.cpp#L1198), [original-weight-list forwarding](https://github.com/pytorch/pytorch/blob/v2.1.2/aten/src/ATen/native/cudnn/RNN.cpp#L1670), [packing](https://github.com/pytorch/pytorch/blob/v2.1.2/aten/src/ATen/native/cudnn/RNN.cpp#L1034) and [weight-list autograd](https://github.com/pytorch/pytorch/blob/v2.1.2/tools/autograd/derivatives.yaml#L2673).
