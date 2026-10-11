# Source cost plan, pending full-graph qualification

All quantities are analytic proposals, not runtime measurements. N=41,554; exact feature width D, canonical directed edge entries E and eligible/role counts remain unknown until custodial payload admission. H=128, depth L=2, C=2 for common native SAGE; original C=1. The conservative metadata-derived symmetrized-support planning bound is E≤2,724,458 (twice the source note's 1,362,229 edge count), not a measured canonical edge count.

## Parameter and operation formulas

| SAGE arm | Trainable float32 parameter count |
| --- | ---: |
| Ordinary CE2 M1 | 128 D + 165,762 |
| Internal-factor CE2 M1 | 129 D + 168,324 |
| Internal-factor CE2 shared4 | 132 D + 176,010 |
| Matched-boundary CE2 shared4 | 132 D + 166,794 |
| Genuine independent internal-factor CE2 I4 | 516 D + 673,296 |
| Original boundary BEBlock BCE1 shared4 | 132 D + 167,048 |

For native SAGE each residual layer has SAGEConv lin_l/lin_r (2 H²) and FFN `2H→H→H` (3 H²), totaling **5 H²** dense weights; it is not the ordinary 4 H² FFN/graph approximation. Native CE2 count follows `H D + H + L(5 H² + 5 H) + 2H + H C + C`. Internal factors add M(D + (2+9L)H + C); boundary factors add M(D+2H+C). Original private B and missing boundary shared biases are included separately.

One full native forward's dense multiply-accumulates (MACs) are approximately `N*(D*H + L*5*H² + H*C)`, excluding factors, biases, norm/GELU/dropout, sparse aggregation and backward. Each SAGE layer aggregates about `E*H` coordinates. Dense FLOPs are roughly twice MACs; do not turn these source counts into seconds or an isolated-speed claim. First-pass allocation, graph scatter work, sparse providers, device/host transfer, tapes and calibration add work.

The fixed LINKX point has trainable count `H*(N+D)+3H²+H*C+6H+C` = `128D + 5,369,090`, including final BatchNorm's trainable affine terms. Running mean/variance/counter buffers are additional state. Its sparse adjacency projection is approximately `E*H` MACs; remaining dense MACs are `N*(D*H+3H²+H*C)`. The adjacency weight is N×H, but no N² dense adjacency is required by the saved forward. Native SparseTensor→COO construction per pass and backend sparse-linear compatibility need representative qualification. The source's row.min() behavior must preserve node identities on the complete graph.

## Proposed ten-seed upper ceilings

The original native-recipe ceiling of 5,000 is shown separately from common controls at 1,000; root may choose a different finite proposal before freeze. Each update includes full-graph TRAIN routes plus the same number of full-graph SELECT routes. I4 has four own 1,000-update fits. Early stopping remains authoritative, and failures/actual calls are recorded.

| Arm | TRAIN routes per seed | SELECT routes per seed | Optimizer steps per seed | Backward calls per seed |
| --- | ---: | ---: | ---: | ---: |
| Internal shared4 | 4,000 | 4,000 | 1,000 | 1,000 |
| Original boundary BCE1 | 20,000 | 20,000 | 5,000 | 20,000 |
| Matched boundary | 4,000 | 4,000 | 1,000 | 1,000 |
| Ordinary M1 | 1,000 | 1,000 | 1,000 | 1,000 |
| Internal-factor M1 | 1,000 | 1,000 | 1,000 | 1,000 |
| Genuine I4 | 4,000 | 4,000 | 4,000 | 4,000 |
| SAGE total | 34,000 | 34,000 | 13,000 | 28,000 |
| Fixed LINKX | 500 | 500 | 500 | 500 |

For ten proposed seeds: 70 banks/100 optimizer-body fits; SAGE TRAIN routes 340,000 and SELECT routes 340,000; one final full-graph extraction per selected SAGE route adds 180 routes. At depth two these are **1,360,360 SAGE graph blocks**. SAGE optimizer steps 130,000/backward calls 280,000; LINKX adds 5,000 of each, yielding total 135,000 optimizer steps and 285,000 backward calls. LINKX's 5,000 TRAIN, 5,000 SELECT and ten final full-graph extractions total 10,010 LINKX forwards. Stored final full-graph logits can be sliced for all roles without another native pass; any future implementation's repeated calls must be charged in its actual ledger.

These ceilings exclude full-graph source qualification, warmups, checkpoint writes/restores, transfers, public preprocessing and temperature fitting. CAL's proposal is 90 scalar temperature fits × 500 CPU steps = 45,000 scalar updates, using cached predictions. Alternate current/boundary fusion adds no native fit or forward under the cached-logit contract.

## Memory planning

- Public X uses `4*N*D = 166,216 D` bytes; canonical COO uses `16E` bytes before additional sparse/provider structures.
- One four-route N×H float32 node state uses 85,102,592 bytes (81.16 MiB). This is one buffer, not peak memory.
- If the backend materializes a four-route E×H float32 edge buffer at the conservative E bound, it uses 5,579,689,984 bytes (5.20 GiB). Native execution is four sequential forwards; whether edge buffers coexist depends on retained autograd tapes. It is an exposure estimate, not a required tensor or measured peak.
- Float32 AdamW parameters, gradients and two moments are about 16 bytes per trainable parameter, excluding temporary foreach allocations, buffers, checkpoints, device/provider memory and tapes. Four independent models can be acquired sequentially; final pooling still uses all four selected predictions. Do not count sequential acquisition as concurrent wall time.
- Original sequential backward releases each route's tape after its backward, while the current combined-backward wrapper can retain four complete tapes. Preserve the actual interfaces and measure their different peaks; do not assume factor sharing yields fourfold speed or memory savings.

## Qualification before resource scheduling

Require a representative complete Penn94 graph on the actual reviewed native providers for every distinct integration. Record synchronized wall time and peak host/GPU memory for preprocessing/transfer, complete TRAIN forward+backward+step, complete SELECT, selected checkpoint save/restore and raw serving. Exercise original real checkpoint cadence and LINKX sparse linear/BatchNorm state. Ownership/seed/role contracts for singles and I4 must be checked on complete native bodies, with no shared I4 storage or optimizer ownership.

No ETA, GPU admission, concurrency plan, launch or practical-feasibility conclusion follows from this packet. Do not schedule from a tiny subgraph, reduced width/member count, toy parity, favorable microbenchmark or invented quadratic dictionary. Root binds the finite qualifier budget and source/data/config hashes; preserve failures, then use measured full-graph cost/peak bounds to choose normal-host scheduling within the authorized route. Numerical feasibility and predictive competence remain distinct.
