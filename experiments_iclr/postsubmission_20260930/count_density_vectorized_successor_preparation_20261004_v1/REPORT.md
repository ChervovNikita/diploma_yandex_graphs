# Exact batched C_mu implementation successor

4 October 2026. **Source preparation only: no numerical implementation import
or test was possible locally because Torch is unavailable.** The sibling
supplement retains all runtime probe failures. This successor leaves v1 and
the frozen P0/J_P/F_P/C_mu arms, seed, architecture, lambda, optimizer, target
BCE, exact law and expected-completion interface unchanged.

The scalar source loops over65,536 queries separately and recomputes all count
states. `vector_density.py` batches reachable log-ESP/count-state updates,
normalizers, NLL and fixed-context unary reverse passes. Padding is an internal
workspace: impossible states stay minus infinity, invalid slots have zero
influence, and each genuine count state remains. Safe dummy operands prevent
an autograd derivative through `logaddexp(-inf,-inf)` on masked branches.
The correlated full two-dimensional g is preserved; neither independent bits
nor separate side potentials replace it. FP64 overflow fails rather than
introducing clipping, precision fallback or support truncation.

`ragged_density.py` packs all actual ordered side slots and scatters outputs
back to their exact original query/slot positions. Length grouping and bounded
query chunks are engineering work partitions, not predictor features or learned
sorting. Default budgets are256 queries and262,144 nominal DP/count workspace
cells per chunk. A single query exceeding the workspace budget retains all its
slots/counts in its own chunk; it may still exhaust resources. No actual slot or
count is removed. Auxiliary checkpoint recomputation now occurs per query chunk,
instead of per query. Every empty query retains its zero in all-query means.

The unchanged count head is530→64 ReLU→1, symmetrized across both full contexts.
`vector_head.py` uses its original parameter views and the exact affine identity
`W[C,f]+b = W_context C+b + W_count f`. The expensive520-dimensional context
projection is computed twice per query; the ten-dimensional candidate-count
projection and nonlinear output are evaluated twice for **every** feasible
count pair in4096-state workspace blocks. No parameter, feature, activation or
sharing changes. This is algebraically equivalent; FP32 GEMM regrouping can
round differently and requires the prepared numerical/gradient parity tests.
It is not claimed bitwise equal to the scalar neural head.

Without this decomposition the two head evaluations pay roughly
`2*(530*64+64)=67,968` matrix MACs per count pair. The decomposition pays
`2*520*64=66,560` once per query plus
`2*(10*64+64)=1,408` per pair. For the **fabricated sensitivity** a=b=8,81 pairs,
that is5,505,408 versus180,608 MACs/query, about30.5 times less head matrix work.
At a=b=32,1,089 pairs the ratio is about46.3. These are source arithmetic,
not measured degrees, wall time, GPU speedup or full-native-batch feasibility.
Every recomputed head/DP forward and backward remains paid work.

Straight DP still costs quadratic side work, and the full potential grid still
costs(a+1)(b+1) states. Padded chunks can add wasted workspace and DP operations;
grouping reduces but does not eliminate this. Prefix stacks can temporarily
overlap their input row lists; the nominal cell budget is not a byte-perfect
peak-memory bound. Neural feature/scorer/context autograd, graph construction,
raw features and the native65,536-query batch are additional global memory.
Actual support maxima, padded-work overhead, CPU/CUDA peaks and wall time remain
unmeasured. Exactness does not imply that the complete native batch is feasible.

`vector_count_model.py` is the runnable native integration adapter for a future
root assembly: inherited native encoder/decoder/context/head constructors,
same two complete depth0 calls and empty-call schedule, same detached1.05*mu
and one nonlinear native target decode. Its source uses no extra parameters.
P0/J_P/F_P continue to use v1. The native adapter was only statically parsed;
neither Torch Sparse nor private native code was imported locally.

`parity_qa.py` prepares all45 padded density geometries with finite nonzero
padding sentinels, scalar normalizer/marginal/NLL parity, first-order gradients
of every unary/potential cell and all-query mean, empty/one-sided/extreme cases,
the original dense scalar neural-head reference, full neural/context/parameter
gradient parity, chunk invariance and checkpoint parity. It explicitly probes
padding zeros and exact native slot coverage. FP64 tolerance is2e-10; FP32 head
values use2e-6 absolute/1e-5 relative, gradients2e-6/2e-5. The tests remain
unexecuted. They cannot qualify the native full model by implication.

Root can run the sibling CPU runner on its already installed admitted runtime,
then qualify the complete native adapter, paired replay and an uncapped ordinary
65,536-record positive/negative batch before scientific fitting. The new source
requires its own numeric and native-batch gate; no old receipt promotes it.
The frozen screen remains4 fits,6,800 updates,400 selectors and408 complete
VALID evaluations. This implementation work does not choose a hyperparameter,
read outcomes or establish graph posterior/global active-dropout coherence,
higher-order necessity, ranking benefit, novelty or manuscript evidence.
