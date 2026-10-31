# Training-only objective alignment at the fixed PPI source

Status: a separate diagnostic specification. The earlier recurrence/spectral rules, checkpoints, masks, bases, and predictive studies remain unchanged. This is not a predictor, promotion rule, novelty claim, or an independent confirmation experiment. F2 was already used in the preceding local denoising audit; no unused-data claim is made here.

## Purpose

The actual training objective averages each member's BCE. Deployment averages raw logits before predicting. These are different losses. This diagnostic asks whether the same centered private coefficients would receive similar gradient directions from the two losses, and whether an own-member update on one training graph points toward lower pooled loss on another training graph. Positive own-member gradient recurrence alone does not establish that alignment.

Use exactly the six original seed0/1/2 × observed-node label10/100 cells, their reproduced P-only epoch25 checkpoints, and the existing fixed selected/random/raw bases. Graph units are exactly `ppi:train:09` and `ppi:train:18`, in that order. Only the existing training cache is loaded. The observed-node masks are checked against the existing bytes and seed rule; hidden target rows are physically NaN. Validation/test trajectories, loaders and metrics are absent.

The exact unmodified `spectral_privacy/v2_abi_fix_01/adapters.py` supplies `SourceSAGEEnsemble`. Its three neighbor corrections are `Delta_m = U sum_q C_mq A_q`, with original source dtype float32, frozen U/C buffers, and A=0. The source snapshot, shared weights, normalization, root maps, FFNs, biases and boundary projectors are unchanged. Selected/random/raw match every per-target rank, including the all-zero seed1/10% cell.

## Objectives and numerical conventions

For each graph, forward all four members in eval mode, with autograd enabled and no dropout. Compute:

- **Mean member BCE:** mean across members of graph-mean BCE over exactly the finite observed targets.
- **Pooled BCE:** graph-mean BCE of the native-dtype arithmetic mean of the raw member logits on those same targets. This is the deployed pooling rule; probabilities are not averaged.

Original source forwards, pooling, elementwise BCE and parameter autograd retain float32. Elementwise BCE values are promoted to float64 before observed-entry/member scalar reductions. Recorded gradients are promoted to float64 before norms, dot products, cosines, projections and cross-graph aggregation. This changes scalar reduction precision relative to earlier native means; the pointwise function and mathematical losses are unchanged. The archived own-member source-matrix gradients are checked within the explicitly fixed float32 tolerances. The synthetic gate uses double source tensors solely for a precise chain-rule verification; real checkpoints are never converted to double.

Each whole graph has equal weight. Keep the two graph gradients separate; the equal-graph mean is their arithmetic average. Nodes, labels and members are not treated as independent graph observations.

## Collected quantities

Collect the direct gradient of each objective with respect to A at zero, for selected/random/raw separately. An empty rank produces an empty coefficient gradient, zero inner product, and undefined cosine represented by JSON null. Empty coefficients are not passed to autograd.

For a separate rank-zero copy of the common source, collect unscaled own-member source-weight gradients `G_m`. For the pooled objective, differentiate the pooled loss with respect to each member's logits, then backpropagate through that member's branch to its source neighbor matrix. Define `H_m` as **four times** this branch contribution. The direct coefficient gradients must match:

    a_own,q = (1/4) sum_m C_mq U.T G_m
    a_pool,q = (1/4) sum_m C_mq U.T H_m .

The second identity uses the pooled residual and member Jacobian; substituting own-member G for H would compute a different objective. U/C in these checks are the actual native adapter buffers, promoted to double, including their float32 rounding. Save raw source-response matrices, coefficient gradients, basis bytes and hashes, losses, observed counts and logit fingerprints.

For each target and the concatenated disjoint coefficient vector, report:

- own-versus-pooled norms, dot product and cosine within each graph;
- own/own and pooled/pooled dot products and cosines between graphs9/18;
- own9/pooled18 and pooled9/own18 alignment, their symmetric dot-product average, and alignment of the equal-graph mean gradients.

Cross-basis coefficient coordinates cannot be compared directly. Lift gradients into the common centered neighbor-weight space with `V_m = U sum_q C_mq a_q`, then report selected/random/raw pairwise direction comparisons there. The lift describes an infinitesimal coefficient-SGD tangent. It does not describe AdamW's coordinatewise moment-normalized update.

At A=0, a coefficient-SGD step using own graph9 gradients changes graph18 pooled loss to first order by `−eta <a_own,9,a_pool,18>`. Pooled/pool recurrence uses a different direction. These are local derivatives, without a noise correction, final-loss guarantee, metric guarantee or inference about convergence. No optimization step is taken by this collector.

## Difference from the earlier Gram proxies

The earlier block statistic centers unscaled own-member gradients for complete residual blocks, including parameters outside these three neighbor maps. Its `C` and `N` concern member disagreement across graph-fold averages. This diagnostic differentiates the actual centered coefficient parameterization, includes native mean-member 1/4 scaling, conditions on each fixed output basis, and explicitly changes the loss residual for the pooled objective. Existing own-member block Gram summaries contain neither the pooled residual/Jacobian contributions nor output-feature projection information needed to reconstruct these gradients. The present cross-graph coefficient dot products are not the previous `trace(U.T T_F U)` denoising score.

## Identical-members null and engineering gates

Create a new null clone by copying member0's R/S/B boundary rows into all members, keeping shared source tensors and the same bases. With eval-mode dropout disabled, logits must be identical. Both centered coefficient gradients must then vanish within the fixed dtype-specific numerical tolerance. Run this null for every cell, basis group and F2 graph; preserve its raw gradients and measured residuals. The existing sources and checkpoint files are never mutated.

The synthetic CPU gate checks both coefficient chain identities, unequal-size graphs with equal graph weighting against joint autograd, zero-rank handling, original-source forward identity, an identical-members null, rejection of non-training units, and preservation of existing `.grad` references and module modes. It loads no dataset and uses no optimizer. Local syntax inspection alone does not constitute a passed CPU gate.

Before real autograd, verify all inherited input hashes, exact source loading, existing F2 graph/mask fingerprints, frozen selected/random/raw bytes and matching capacities. A new `PREGRADIENT_GATE.json` records these identities. Only then collect the six diagnostic records. Collection preserves model values, existing gradients, module modes and RNG. Independent arithmetic replay reconstructs projections/directions from saved gradient bytes without labels or a model forward; it is **not a fresh autograd reproduction**.

Two previously inspected training graphs provide no useful confidence interval or p-value. The six cells reuse graph identities and nested masks; they are not six independent tasks. No diagnostic result can modify the current frozen predictor or its promotion gates. Report unfavorable, zero and numerical-failure results equally.

## Coordinator provenance and execution

Only the coordinator launches runtime jobs. Upload this exact new directory to the existing repository phase. Existing dependencies, checkpoints and cached data are reused from their original phase-local paths. The standard-library freeze can run locally; inherited checkpoint hashes are verified on the server before loading. Preserve all failures; repair source only in a new version with a new freeze/gate.

From the phase root, source `protocols/repo_env.sh` and execute CPU commands through `protocols/run_logged.py`. Do not replace its repository-local cache/temp paths or create a CUDA context. Use the existing repository interpreter. Each output path must be new.

```bash
python objective_alignment_v1/freeze.py --output objective_alignment_v1/INPUT_LOCK.json
python objective_alignment_v1/verify_cpu.py --output objective_alignment_v1/CPU_GATE_run01.json
python objective_alignment_v1/collect_objectives.py run \
  --lock objective_alignment_v1/INPUT_LOCK.json \
  --gate objective_alignment_v1/CPU_GATE_run01.json \
  --output objective_alignment_v1/collection_run01
python objective_alignment_v1/collect_objectives.py replay \
  --lock objective_alignment_v1/INPUT_LOCK.json \
  --run objective_alignment_v1/collection_run01 \
  --output objective_alignment_v1/replay_run01
```

Runtime fixes intra-op and inter-op CPU threads to1, matching the inherited spectral replay contract. Supervisor logs, exit status, code/wrapper/environment hashes and elapsed time accompany every runtime command. No GPU-resource assumption is made.

## Primary context

[TreeNets](https://arxiv.org/html/1511.06314v1) studies ensemble-aware losses alongside shared same-task ensembles. [TabLoRA §3.5](https://arxiv.org/html/2607.10077v1) uses mean member training loss and raw-logit aggregation. [TAG](https://arxiv.org/html/2109.04617v1) motivates treating update effects as objective-specific quantities. Exact inspected passages, bibliography and retrieval hashes remain in `adaptive_sharing_novelty_v1`; no further literature search is performed. The present diagnostic is not claimed as a new method.
