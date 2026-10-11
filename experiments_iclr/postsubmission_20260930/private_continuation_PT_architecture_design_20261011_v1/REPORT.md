# Private continuing P/T routes: one prospective architecture memo

2026-10-11. **Architecture definition; no fit admission.** Current GRU and PubMed LM4 studies retain priority and their contracts remain unchanged. Saved conclusions and native source text suffice: zero fresh literature passages, numerical imports, remote operations, current partial quality or older-packet edits. `ARCHITECTURE_BINDING.json` records the complete forward and ownership definitions; fit roles, selectors and resource limits remain unbound.

## Recorded failure and conditional question

Completed WikiCS private sparse-kernel SAGE acquired 458 covered readouts but removed 1,062; 700 BASE-error repairs were outweighed by 1,086 harms, and mean members weakened. These sum dependent seed-node readouts on one encountered graph. They are not independent graph observations.

**Hypothesis:** separate original-support P/T continuations might retain competent local or neighborhood cues that an early mixed continuation suppresses. Own class supervision might make the resulting alternatives useful to the original probability mean. Neither that suppression nor a causal role for raw-X conditioning/support union is established by the failure. A capable ordinary routed predictor may suffice.

## Exact native anchor

Use one competently selected ordinary meanSAGE source. Bind source semantics to the saved native core and runtime, not GNNMoE's undefined helpers. For inherited prepared edge `src→dst`, `A[dst,src]=1`; `degree_i=sum_j A[i,j]`; `P(Z)_i=(AZ)_i/degree_i`. Degree is fixed, positive and includes inherited self entries. The saved Physics authority has 530,417 entries, exactly 34,493 diagonal entries; its provider preparation added one loop per node before source acquisition. The continuation inserts no further loops and applies no degree clamp or feature-neighbor union. These source dimensions do not nominate a new Physics fit.

Define `C(Z)_i=Z_i/max(1,||Z_i||₂)`. With acquired first-layer aggregate/skip maps `U0,V0` and bias `b0`, freeze all three and cache

`H0=ReLU(C(P(XU0)+XV0+b0))`.

Raw X remains unchanged; stem dropout is off. **Native order is clip then ReLU.** Width `d` equals the stem width (32 in the saved binding). H0 has no gradient; every continuation and decoder path does.

## Operators, controllers and continued states

Four routes start at `H_m^0=H0`; two blocks expose a retained-history distinction. In each block ℓ, create four distinct roles `j∈{PT,TP,TT1,TT2}`:

`T_ℓ,j(Z)=C(ReLU(ZW_ℓ,j+b_ℓ,j))`.

Each W is d×d and each b is d. **New T order is ReLU then clip**; retain both return values. Roles and blocks own distinct maps, but each role's same physical W/b objects serve every route. Define exactly:

`F_PP(H)=C(P(PH)); F_PT(H)=T_PT(PH)`

`F_TP(H)=C(P(T_TP(H))); F_TT(H)=T_TT2(T_TT1(H))`.

There is no intermediate dropout, additional ReLU, normalization, residual or learned propagation. Mixture outputs receive no extra activation/clip. All four operations are evaluated; no hard/top-k dispatch or pretrained expert bank.

Each block owns one common gate:

`g_ℓ(c)=ReLU(cG_ℓ,1+β_ℓ,1)G_ℓ,2+β_ℓ,2`, widths `3d→d→4`.

From incoming private state `H=H_m^(ℓ−1)`, recompute

`c_m=[H,|PH−H|,|P²H−H|]`

`a_m=softmax_rows(g_ℓ(c_m)+e_m^ℓ)`

`H_m^ℓ=sum_o a_m,o F_o(H)`.

Softmax temperature is one, order PP/PT/TP/TT, and weights multiply node rows. Each `e_m^ℓ` is a private four-vector: **32 private parameters total**. Gate maps/biases are common, including its final bias. States are private activations reset every forward, never optimizer-step memory. Gradients remain live through P, gate contexts and both blocks.

Initialize T maps and the first gate maps independently by Glorot uniform; their biases are zero. Set both final gate affines to zero. Initialize `e_m^ℓ=onehot(m)−1/4` in both blocks, fixing differentiated soft preferences without labels. Shared initialization draws are isolated from native dropout streams; exact seeds are future bindings.

## Common head, objective and starting function

Use shared bias-free d×d `Wo`, initially zero:

`H′_m=C(H0+.25 H_m²Wo)`.

The native live decoder has one common `U1,V1,b1`, initialized from the selected ordinary source. For TRAIN, `D_m=dropout(H′_m,p=.4)` once, with an own persistent stream per route; the same D feeds aggregate and skip. For eval, `D_m=H′_m`. Return logits

`L_m=C(P(D_m U1)+D_m V1+b1)`.

There is **no decoder ReLU**. Train `mean_m CE(L_m[TRAIN],y)` plus the one constant frozen native first-layer penalty `.05(||U0||²+||V0||²)`; add no memory penalty. Serve `mean_m softmax(L_m)` with dropout off, never mean logits. No private class head/bias, pooled CE, auxiliary/diversity loss, learned edges or cross-route exchange is introduced. Zero Wo contains the initial evaluation function at the selected source in exact arithmetic; it delays all upstream memory gradients on the first update. Neither statement certifies bitwise identity or later competence.

## Prior difference and closest collision

The failed kernel used common raw-X projection Φ and private 16-coordinate metrics: supported-edge softmax was computed **once from raw X per factual forward**, reused in both native layers, on a top-20 raw-cosine support union. It already had shared maps, private nonlinear states and own CE. The narrower difference here is repeated current-private-state operation choice and nonlinear placement on original P. It is not the first adaptive private graph filter.

**GNNMoE**, WWW 2025 Companion, DOI [10.1145/3701716.3715462](https://doi.org/10.1145/3701716.3715462), is the nearest collision: current-state PP/PT/TP/TT gating into one continuing mixture, with separate expert T maps in official code. **Node-MoE** supplies graph-context inputs and differentiated filter specialists. GNNMoE's source-incomplete release is attributed ancestry, not an executable reference; any later benchmark needs a complete authority or transparently declared reimplementation. This construction is also a wider grouped network. No expressivity theorem, new ensemble principle or benefit is claimed.

## Minimal controls and failure conditions

| Control | Exact comparison |
| --- | --- |
| Head-only staging | Same common source/cache/live decoder, four own-dropout CEs; `H′_m=H0`. |
| Capable mixed single | One continuing state; same full F bank, two context gates, zero residual readout and native decoder. Ordinary gate biases remain trainable; remove route e. Use four TRAIN tail-dropout draws of that one state and mean their CEs; eval is one prediction. Preserve ordinary/factorized/original-GNNM boundaries where applicable. |
| Genuine equipped I4 | Four complete capable mixed-single bodies, independently acquired/fitted/selected, with own source, transformations, gates, head, optimizer, RNG and stop. No coupled selector; all extra source acquisition is charged. |
| **Middle-state mean** | Candidate unchanged except set every block-two input to `mean_m H_m^1`. Keep block-two private e, shared gates/maps, four final predictions and own CEs. Never average final states before classification. |

The middle-state control preserves parameters, source and initial draw law while removing first-block private history. It also changes backward coupling and permits more computation reuse; attribution concerns that full interface choice. Candidate must improve declared paired net correctness over this control to support private-continuation attribution. Utility requires net served improvement versus capable single/equipped I4 with acceptable competence and proper-loss protection. Staging/single matches explain generic refitting/routed augmentation. Coverage, gate diversity and repaired subsets alone cannot pass. Numeric margins, protection populations, roles, calibration, selectors, horizons and finite costs must be fixed before any fit; current memory-study values are not copied.

## Paid work and disposition

Reuse PH/P²H for gate/PP/PT; TP needs one further propagation. Each route/block costs 3 P and 4 T calls: 24/32 for four routes/two blocks without sharing. Common H0/maps permit the first bank once: **15 P/20 T**, five gate calls, four native decoders. The middle-state mean permits both banks once: 6 P/8 T, two gate calls. I4 shares no banks across bodies. Candidate memory storage is `15d²+18d+40` scalars (15,976 at d=32), including 32 private e scalars; source/native-head storage is additional. These are algebraic inventories, excluding backward, sparse preprocessing, clipping, optimizer, cache, SELECT, restore, calibration and source acquisition. They are not runtime savings receipts.

**Disposition:** seal one prospective architecture question and its middle-state falsifier. No implementation, job, fit grid, manuscript claim or current-study change is admitted.
