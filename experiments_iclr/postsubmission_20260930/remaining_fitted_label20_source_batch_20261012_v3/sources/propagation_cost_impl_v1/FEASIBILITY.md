# Unmeasured feasibility and comparator requirements

## Current status

Only rational/source fixtures ran locally. No provider graph was opened. Torch, GPU timing, peak allocator memory and complete-graph engineering have not been executed by this author. The packet is an executable mechanics prototype, not an admitted study or feasibility pass.

## Node-state memory is the main risk

At products metadata size (2,449,029 nodes), one float32 state `[N,4,64]` is about **2.34 GiB**. One source FFN concatenation `[N,4,128]` is about **4.67 GiB**. The complete 100-dimensional feature matrix is about **0.91 GiB**. These are individual payloads, not peak-memory estimates.

Training retains several normalized states, deviations, propagated features, feed-forward activations and dropout masks. Temporary packing, centering and output tensors add bandwidth and memory. Some objects share storage, while autograd retains others, so summing a short activation list does not establish the true peak. The 60 GiB reserved-memory ceiling could fail. Optional whole-block checkpointing reduces saved activations but recomputes sparse and dense work; those costs must enter the same forecast and speed comparison.

The exact serial CE control keeps only one member trajectory's autograd graph alive at a time. Its single `[N,64]` state is about 0.58 GiB on products. It shares parameters and accumulates own-member loss gradients before one AdamW update. Therefore candidate versus packed exact peak memory is insufficient evidence of an advantage; the fastest feasible exact control must be measured.

## CSR and preparation storage

Graph weights are float32; column indices and row pointers are int64. Each sparse entry costs 12 payload bytes. Full graph, four color graphs and their adjoints contain `4 E` stored entries and ten row-pointer arrays. Keeping all four consecutive half unions and their adjoints raises this to `8 E` entries and eighteen row-pointer arrays. Original degrees, multiplicities, offsets and color IDs are counted separately.

`SYMBOLIC_RESOURCES_local_run01.json` records payload calculations for Reddit2's raw-entry upper-bound scenario and two explicit products edge-count scenarios. **Actual products directed/coalesced E is unknown.** The twice-metadata scenario is not a claim about the provider. Do not use metadata edges as measured sparse work.

Preparation lexicographically sorts raw edges and separately builds CSR adjoints. NumPy sorting, coalescing, masks, selected positions, transpose sorting and file writes can require substantial temporary CPU RAM and time. Current kernels are bounded fixture kernels; full-scale external sorting/memmap engineering is not qualified. The future limit is 16 CPU hours and 200 GiB additional disk, but no measurement shows that this source meets it. The prepared array payload is not a CPU/GPU peak-memory estimate.

Load only structures needed by each execution arm in an adopted version and charge all H2D copies. Candidate requires full and four quarter operators plus adjoints. Cached-half naive requires unions plus adjoints; lazy-two-color naive avoids unions but uses two sparse calls and an additional dense sum. Holding every variant on GPU at once is unnecessary and must not be imposed on the fastest exact comparator.

## Sparse arithmetic is not end-to-end time

Candidate requires an exact width-64 common pass and a sampled width-256 deviation pass. Exact packed requires one width-256 complete pass. Candidate adds dense member means, subtraction, addition, and a second sparse kernel. Half-union naive can use one width-256 sparse call, whereas its lazy form uses two. Degree imbalance, CSR kernel throughput and memory bandwidth may eliminate the nominal edge-channel benefit.

The future score-blind gate must compare packed/serial exact training and inference, lazy/cached naive paths, and checkpointed/noncheckpointed forms where feasible. Keep initialization, own-member objective, dropout draws, precision and graph weights unchanged. Cache preparation and recomputation are charged. Select execution policies using complete graphs without validation/test predictive scores, and freeze them before fits.

The 6.5 GPU-hour family forecast includes 24 fits, 100 full validations per fit, final complete evaluation and profiling. Dividing that ceiling by 2,400 training-plus-validation epochs gives an illustrative mean allotment below 9.75 seconds per epoch; final evaluation and profiling consume part of it. This is not a runtime estimate. Engineering attempts and failures count toward the overall eight-GPU-hour cap.

## Mathematical and scientific limits

The exact property stops at linear aggregation. Member-specific nonlinear Jacobians can turn cancelling aggregation errors into a noncancelling pooled change. Uniform color weighting does not make nonlinear states or AdamW updates unbiased. Current-member disagreements need not vanish when parameters are fixed. Neither useful diversity nor preserved accuracy follows from a centered residual.

Passing a small synthetic fixture cannot support a complete-graph performance claim. Passing the proposed two-graph pilot would support its scoped quality/cost gates and independent confirmation planning. It would still require nearest-prior access gaps and broader baselines to be resolved before claiming methodological novelty or general superiority.

## Concrete next gate

1. Independently inspect and run the bounded CPU/CUDA mechanics fixture on the authorized account.
2. Freeze a separate admitted full-data preparation/interface and comprehensive resource guard; this packet's real-data gate remains closed.
3. Freeze actual full operators, raw degrees, coalesced counts, color assignments, node/features order and official splits, while protecting predictive outcomes.
4. Measure complete-graph memory and score-blind unit times for all required source-equivalent execution paths. Forecast all work within 6.5 GPU hours, ≤60 GiB reserved peak, and preparation/storage caps.
5. If infeasible, defer this specific two-graph family. A substitute requires a new prospective hypothesis/protocol and does not retroactively rescue this one.

No resource or scientific gate is marked passed here.
