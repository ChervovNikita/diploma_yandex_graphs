# Native neighborhood quartile source feasibility

The proposed descriptor fits the existing native hidden boundary without changing a graph convolution or adding a native forward. Source preparation is authorized; no scientific execution is admitted. The complete recurrent class-decoder recipe is closed independently. Its outcomes do not choose a projection, degree rule, loss, control or selector here.

## Specified operation and source boundaries

The proposal fixes current preclassifier H and native Z, incoming factual nonself records with duplicate multiplicity, one normalized learned direction per route, three interpolated centered quartiles plus log degree, a zero-start private 4×C score map, the half-native/half-corrected full-TRAIN loss and mean corrected-probability serving. Degree zero gives all-zero descriptors; degree one gives zero centered quartiles and log2. Projection normalization uses the declared 1e-8 floor. No label or cohort is a message value, fit mask or selector.

`NativeModelAdapter.native` already captures the input of `body.output_linear`, after the native output normalization. `SharedRoutes` returns that live H together with logits. An ordinary/factorized M1 or an independently fitted member can use that same adapter capture, under its existing RNG scope. No frozen hidden cache, extra layer capture, detached field or teacher is necessary.

The existing full native `Family` supplies model/factor construction, Rademacher stem initialization, route seed streams, safe TRAIN/VALID loading, strict probability-pool VALID accuracy selection, patience, full state/optimizer/stream restoration, metrics, archives and finite completion. A thin derived extension can register directions and score maps before its existing AdamW constructor, graft the mixed loss and corrected logits into that fit loop, and archive native logits separately at the corrected-selected state. No separate model wrapper, owner, retry loop or fit lifecycle is required.

## Complete neighborhoods and live gradients

Prepare label-free degree buckets once from the factual nonself support. For each positive incoming record degree d, store the recipient IDs and every source record as an `[N_d,d]` table. Keep source order stable within recipients and preserve duplicate records. Every arm, including genuine I4, reuses the same immutable tables. Native graph support is untouched.

Each current projection gathers `[routes,N_d,d]` scalars and sorts along the complete d axis. At q=.25/.50/.75, interpolate the two adjacent order values at `(d−1)q`, then subtract the neighborhood mean. This implements the exact proposed multiset descriptor, with O(routes Σ_v d_v log d_v) comparison work. There is no degree cap, subsampling, degree scaler, fallback to a mean, large dense N×max_degree padding or changed projector. Buckets are only a storage/computation arrangement; they do not change the mathematical evidence.

Ordinary Torch sort/interpolation/index-copy operations retain gradients through the gathered H and learned directions. Stable sorting specifies which factual record receives an order-statistic gradient at ties; it does not make the piecewise operation smooth. The full degree-one/empty-neighborhood behavior follows the descriptor definition. Actual ties, full high-degree sorting, backward cost and device compatibility require later bounded runtime qualification.

## Five-arm ownership and controls

The candidate registers four directions and four private 4×C maps under one shared native body. The common-direction control registers one direction and the same four private maps; different route H can still produce different descriptors. The moment arm retains four private directions and uses a private `[mean,std,min,max,log_degree]` score map. It has more score coordinates, so it is a capable ingredient comparator rather than an exact parameter match.

Root's subsequent control instruction strengthens genuine factorized I4 to four separately initialized/fitted copies of the capable all-signature M1 corrector, with own selected checkpoints. Each M1 has four directions and a `Linear(D+13,128,bias) → ReLU → Linear(128,C,bias)` residual readout of its H, twelve centered quartiles and one degree field. There is no added H/descriptor normalization or dropout, and its output layer starts at zero. Thus I4 has sixteen learned directions across four native bodies, rather than four narrow one-direction correctors. Its richer capacity and work remain explicit. Plain native archives remain contextual references. Five arms×three paired seeds give 15 banks, 24 optimizer/body fits and 51 native route trajectories per family-wide forward. These are prospective arithmetic, not measured work.

## Method specification and root freeze

The method researcher has now supplied all source-level choices. Moments use the FP32 population mean and `sqrt(mean((a−mean)^2)+1e-8)`; nonempty constant/singleton standard deviation is 1e-4 and empty fields remain zero. Direction draws use fixed local CPU FP32 normal seeds `base+5,000,011+1,000,003*(4*b+k)`. Candidate/common/M1 use b=0, with k equal to route/0/0..3; genuine I4 uses body b=0..3 and k=0..3. All-signature heads use standard `nn.Linear` first-layer initialization inside an isolated CPU RNG scope seeded `base+6,000,017+1,000,003*b`, followed by a zero final weight/bias. No direction search, orthogonalization, added normalization or dropout is introduced. These choices are prospective method definitions, not adjustments to decoder outcomes.

The proposal's base seeds are 7301/7403/7507, with native depth2/width128 and 1000-update/300-patience opportunity. Root freezes the final source, streams, AdamW grouping, gain/NLL gates and cost cap before qualification. Identity at a zero output map establishes only the initial algebra; projection and upstream residual gradients along the added correction path are initially zero and activate after the output learns. The native-gradient coefficient remains one initially. No competence, novel distribution principle, runtime or portability result follows.

PNA and Fourier Sliced-Wasserstein embeddings supply established distribution-aware aggregation ancestry. Three sampled quartiles inherit none of the cited full-embedding injectivity or metric guarantees. The source decision remains inactive feasibility pending root's operating-point freeze and bounded qualification.
