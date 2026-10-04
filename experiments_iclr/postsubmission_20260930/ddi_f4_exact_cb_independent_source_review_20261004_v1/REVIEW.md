# Independent static source review

Reviewed manifest `b94b173bd280231b82ec655c7e949ca7d15fcc50ab84c6263ce50fd203ba62be` and pinned F4/native/exact-CB sources. Source hashes match; the exact core is byte-identical. This is a method/test-design review, not manuscript acceptance or numerical qualification. **execution_authorized=false**.

No confirmed mathematical, masking or scoring source blocker was found for the intended **one representative native/F4/auxiliary TRAIN-only runtime check**. The supplied main is a gated full-family driver and cannot serve as that limited invocation. No numerical/native imports, graph/score/checkpoint payload reads, source changes, server actions or PDF compilation occurred.

## Concrete findings

1. **F01 — full-family adoption blocker:** `paired_comparison.py:47-54` uses `row.get`. Missing epoch negative hashes across all arms, or missing auxiliary fields across joint/separate, compare equal as `None` and can permit `paired_eligible=true`. A prospective successor must require mandatory fields, types and valid nonempty hashes before eligibility.
2. **F02 — execution configuration:** `train_f4_cb_ddi.py:49-53` resolves the relative default output inside the sealed packet (observed mode 0555). Any execution needs an absolute writable output outside that directory. An external config location alone does not move the default; no implementation change is needed to supply a proper path.
3. **C01 — unmeasured cost:** head checkpointing retains its two gathered 512-wide feature inputs across every support chunk until backward. Four float32 heads require `2*4*L*512*4` input bytes, at most 4,472,176,640 bytes (about 4.17 GiB) at the 64-query/4267-node support bound. That excludes target activations, graph states, exact ESP history and allocator overhead. Actual support/counts, time and peaks remain unmeasured. Graph rebuilding, GPU-to-CPU receipt hashing and synchronization also enter practical update cost.

F01 and actual budget measurement must be resolved before full-family adoption; F02 is satisfied with an external writable path. These findings do not require another qualification sequence before the intended practical TRAIN-only check. This review authorizes no execution.

## Source and method assessment

F4 representations `[4,N,512]` and logits `[4,slots]` are consistent. Left slots score `(v,w)` and right slots `(u,w)` through inherited raw Hadamard MLPs. Checkpoint closures bind each head, and propagated constant channels preserve affine bias. There are no added auxiliary parameters or serving overrides.

The whole positive minibatch and selected queries are masked as undirected edge identities with deduplication. Both endpoints and visible common neighbors are excluded; supports remain complete and sorted. Detached teacher bits label opposite incidences in complete TRAIN. No direct hidden incidence, count or bit enters masked representation/scoring. Learned TRAIN embeddings and complete-TRAIN targets remain intentional transductive reconstruction context.

Selection is uniform without replacement within 32-positive and 32-native-negative strata. Selected positives are already in the whole-batch mask and native negatives are absent from TRAIN, so selection does not change that mask. Half each stratum mean estimates the declared equal-stratum objective conditional on minibatch/negative pool, with scorer dropout understood in expectation. It is not a full-graph likelihood or equal weighting of all 4B origins. If a future sampler admits TRAIN edges, the fixed-mask argument needs revision.

Missing weights dispatch literally to AUC sum over 3B squared margins, averaged over members. Adding `3B*lambda*A` gives the stated mean-margin units, including the tail. Lambda 1 is a fixed adaptation, not measured gradient balance. Exact reachable ESP, complementation, centering, padding and deterministic zero sides are coherent in static inspection; float32 gradients and dense-DDI backward remain unverified.

Fresh parameters, initialization and seeds match across arms. Native target computation precedes the auxiliary, z is reused, scorer RNG is owned/forked, and checkpoints preserve forward RNG. Source checks CPU/current-device RNG after forward and backward. Joint/separate use the same schedule and select different loss keys. Runtime preservation and CUDA numerical reproducibility remain unverified.

Serving remains uniform raw-score averaging on complete TRAIN. Fixed native VALID pools, strict Hits@20 improvement with first ties, 500 epochs, every-5-epoch validation, own selected checkpoint and complete selected replay are retained; no TEST path exists. Each cell has 100 scheduled validations plus one replay.

Joint/separate same-logit laws and the log-space responsibility-overlap identity are correct. Deterministic sides erase coupling; collapse remains possible. Independent head initialization does not guarantee diversity. Common-offset cancellation does not eliminate candidate-degree effects. Uniform/visible-degree probes match supports, labels and denominator. Wedge/triangle mask structure, shared TRAIN embeddings and overlapping contexts remain confounds. End-fit joint/separate differences include optimization and changed marginals; target-only contrasts also include augmentation and compute. The protocol states these limits without claiming diversity, degree invariance or predictive gain.

Metadata supports 17 updates/epoch with 19,335 tail records, 76,500 family updates and 51,000 added masked 15-hop passes. ESP autograd history scales with `M*n*min(k,n-k)`. Prior synthetic float64 equivalence, Collab exact-CB results and the NCNC Boolean census cannot establish actual DDI update cost or a full-family budget. The numerical result SUMMARY pin is retained without reading that payload. Exact bindings and full findings are preserved in `SOURCE_INPUT_BINDINGS.json` and `REVIEW.json`.
