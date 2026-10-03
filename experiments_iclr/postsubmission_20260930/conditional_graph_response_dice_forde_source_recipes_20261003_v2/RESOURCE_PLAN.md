# Expected implementation/resource blockers

Source estimates only. No graph/model was loaded and no resource/time measurement was made. Bind the actual immutable P, complete native token bank and exact `(N,F,K,H,layers,M)` before runtime profiling. Let L=K+1, B=actual target minibatch, M=4 and float32 element width4 bytes.

| Item | Leading storage / work |
| --- | --- |
| Complete native tokens | `4*N*L*F` bytes, plus native sparse P and warm predictor state |
| Direct full-X gradients | `4*M*B*N*F` bytes before mixed-autograd activation overhead; this is the avoided allocation |
| Selected row-power matrix A | `4*B*L*N` bytes; prepared routine transiently retains row list plus stacked A, approximately twice that storage |
| Row Grams G | `4*B*L*L` bytes; cache per target/operator identity, target order and dtype |
| FoRDE live token coefficients q | `4*M*B*L*F` bytes, plus normalization/pair/mixed-autograd graphs |
| Row propagation | K native sparse transpose products with `[N,B]` dense columns: work proportional to `K*nnz(P)*B` |
| Gram formation | Work proportional to `B*L*L*N`; done once per selected target cache, not once per member |
| Native selected-row predictor | M sequential row-local forwards and mixed derivatives; activations depend on B,L,H,layers and order-attention width |
| DICE partner means | At most `B + 4*choose(M,2)*2*B` unique planned node IDs before duplicates; gather/forward base means in ordinary row minibatches |
| DICE discriminator | Per auxiliary step: `4*choose(M,2)*B` positives and twice that many negatives; four RMSProp steps per learner cycle |

For a **purely illustrative** shape `N=25000,F=300,L=11,B=128,M=4`, direct gradient values alone take 15.36 GB, one A takes 140.8 MB, G takes about 62 KB and q about 6.76 MB. These are formula evaluations for a fictional shape, not inspected Amazon dimensions or a feasibility certificate. Operator construction, complete token caches, model activations and allocator/autograd overhead are additional.

The proposed exact Gram path removes the full feature dimension from graph propagation and avoids sparse second derivatives through P: P is fixed, G is detached graph state, and mixed private derivatives occur in the dense node-local predictor/q contraction. This improves implementation prospects under a normal runtime. It does not certify sparse COO support, float32 roundoff or peak memory on the intended server.

## Minimal qualification sequence, after authorization

1. Verify root's ordinary runtime/dependencies and source composition. The v4 adapter's existing continuation helper is intermediate-only/guarded; the proposed all-private source-style partition needs its own exact ownership receipt. Do not claim an existing qualified entrypoint.
2. Run tiny synthetic source-force, row-locality, direct-pullback and Gram oracles with full mixed derivatives. Stop on derivative/runtime errors before any graph acquisition or fit.
3. Profile one immutable real-graph target batch: token-cache residency, native COO transpose propagation and G preparation, then one all-member forward/mixed-backward without optimizer mutation. This requires separate graph/resource authorization, not this packet.
4. If needed, stream row-power/Gram preparation in target chunks while keeping a declared B=128 logical force. Cache only G, not A. Recompute/checkpoint dense predictor activations only after force equivalence is qualified. Charge preparation, recomputation and all DICE detached mean gathering.
5. If runtime cannot hold a logical batch's live q graphs, an exact two-pass force VJP may use detached full-batch normalized coefficients/h, then per-target live q recomputation with the frozen kernel cotangent. It needs its own direct-oracle gate and cost accounting; it is not implemented or qualified here. Do not silently shrink B, switch to token metric, approximate powers or truncate neighborhoods.

No candidate graph-removal caches, backtracking guards or per-step optimizer-state copies are needed for the proposed source-style main comparison. They remain additional costs only for a separately adopted guarded protocol. DICE's auxiliary sample volume and base partner means can still dominate a small native minibatch; this is a concrete cost to profile, not a reason to weaken its source estimator without labeling it.
