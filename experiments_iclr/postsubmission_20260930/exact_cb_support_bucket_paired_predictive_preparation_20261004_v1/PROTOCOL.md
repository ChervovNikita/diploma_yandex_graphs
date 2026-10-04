# Prospective three-block development protocol

## Question and evidence entering this comparison

The prospective decision memo asks whether joint conditioning of member responsibility across an edge's two residual neighbour supports improves its count-free serving score. Conditional Bernoulli likelihoods, cardinality inference, mixtures, topology supervision and shared graph mixture components have precedents. Predictive transfer beyond the target-only and separated controls must support a graph-specific contribution. Generic mixture novelty and batching efficiency are not predictive contributions.

The complete TRAIN census reports 5.8242% of positive queries with both supports variable and 1.3599% with both genuinely informative. This sparsity threatens useful transfer. The first native batch has a nonzero joint-minus-separated model derivative L2 norm of 0.000570229453061945 versus target L2 norm 1.016701034901922, or 0.05608624693855585% of the target norm. These norms are obtained by combining the disjoint parameter-block summaries already recorded in the completed diagnostic; no derivative is rerun for this preparation.

Mean-reduced slot deltas agree under the original absolute/relative tolerance, `atol=rtol=128*2^-23=1.52587890625e-05`. The absolute tolerance exceeds these small mean-normalized slot derivatives. This agreement is preserved. No tolerance or coefficient is relaxed, and no useful mechanism or large numerical distinction is inferred. This family is a bounded exploration of a weak opportunity at one initialization and mask. Gradient cosine does not forecast generalization. The completed bucket/direct same-state comparison passes the unchanged rule; that establishes an implementation prerequisite, not predictive success.

## Frozen nine fits

The three blocks are seeds 0, 1, 2, chosen as the first three of the existing native frozen five-block sequence before the new predictive outcomes. Each block contains three independently fresh F4 constructions with width 64 and 43,790 total parameters. The constructor, dedicated lexical factor-sign seed, native Adam learning rates 0.0082/0.0037, float32 profile and per-seed initial model/empty optimizer/RNG/flags are identical across arms.

Each cell runs all 100 epochs. Each epoch uses the existing default native negative sampler once, the unchanged native permutation iterator, 17 full 65,536-record masks/batches and the same dropped 64,940-record positive tail policy. Every fit makes 1,700 updates; all nine make 15,300. No resources or checkpoints from qualifications, the completed baseline family, another arm or another seed become a donor.

The source records the initial typed state hash and RNG hash; all 100 epoch negative-draw/permutation/tail hashes; each batch's record-index hash and before/after RNG hashes. Checkpoint recomputation keeps the existing complete depth-zero call with `preserve_rng_state=True`. The auxiliary loss introduces no new draws. Equal streams are checked after the complete family. Any mismatch is reported and makes paired interpretation ineligible; no replacement cell is selected to hide it.

## Fixed objectives

All arms retain the exact native target loss:

`L_target = -mean(logsigmoid(positive_native_logits)) - mean(logsigmoid(-negative_native_logits))`.

The means retain all native query/member entries. Target completion scorer outputs remain detached on the native target path. Only the two auxiliary arms enable the qualified scorer autograd branch. One native encoder forward is followed by positive and negative queries in the existing member/left/right order. One combined backward and one native Adam step occur per batch.

For each query and member, let `ell_left` and `ell_right` be exact conditional-Bernoulli negative log likelihoods of the ordered residual observation bits given each side's observed count. Let `R=max(n_left+n_right,1)`. The optimized loss source preserves the original algebra, TRAIN authority and slot order:

`J_K = -log(mean_members(exp(-(ell_left+ell_right)))) / R`.

`J_K_sep = (-log(mean_members(exp(-ell_left))) - log(mean_members(exp(-ell_right)))) / R`.

`target_only` uses `L_target`. `joint` adds `mean_positive(J_K)+mean_negative(J_K)`. `separate` adds `mean_positive(J_K_sep)+mean_negative(J_K_sep)`. The auxiliary coefficient is exactly 1. No responsibility weighting changes the target. Every query, including empty or deterministic support, remains in the population means. Uniform member prior 1/4, denominator, batch size, support, precision and normalization remain fixed.

The complete TRAIN teacher supplies detached observation-membership labels only. An unobserved entry is not a verified latent nonlink. Neither teacher labels nor counts are predictor or serving inputs. Each selected model serves through its private own native completion route on the complete TRAIN graph, averaging four raw logits.

## Selection, replay and retained outputs

After every epoch, score the complete official VALID positive and shared negative pools on the complete TRAIN graph. Use the pinned official strict shared-pool Hits50 rule, with no VALID loss gradient and no TRAIN RNG consumption. Compare all 100 epoch candidates by strict improvement; retain the first exact tie.

After epoch 100, restore the own selected state and repeat one complete VALID traversal. Serialize and restore the full model/Adam/all-RNG/flags state into a fresh own model, verify the typed state roundtrip, and perform the second complete replay under the original float32 absolute/relative rule. Selection quality and RNG must reproduce. There are 102 complete VALID traversals per fit and 918 across the family.

Keep current/best own states in two rotating durable journal slots. Retain the selected checkpoint, private selection, complete selected VALID score vectors with canonical query hashes, compact epoch/batch stream receipts and physical terminal/cost/artifact custody. Count/time progress may be observed during execution. The prepared driver does not expose predictive values in its live progress or completion summary. No per-step activation or per-step large-logit dump is required.

## Fixed comparisons and uncertainty

After all nine cells complete and their physical/artifact custody verifies, report the following contrasts using every seed0,1,2:

1. `joint - target_only`: primary predictive-transfer question.
2. `joint - separate`: primary cross-side member-association question.
3. `separate - target_only`: secondary auxiliary-supervision question.

Report all three differences, selected epochs, their mean, sample SD, range and signs. Scores and differences use Hits50 fractions; percentage points multiply differences by 100. The fixed descriptive Student-t 95% interval has 2 degrees of freedom. Enumerate the eight exact sign flips; with three blocks the smallest nonzero two-sided p is 0.25. These describe seed variation on one development graph. No query bootstrap or independent-graph uncertainty is claimed.

Any failed or missing cell leaves the family incomplete, preserving all completed cells, costs and failures with no contrasts or success-only subset mean. If all cells complete but pairing differs, preserve all differences and expose the mismatch; they are not eligible for a paired interpretation. Do not summarize the best seed or compare only convenient epochs.

If joint improves over target-only without a supported advantage over separated supervision, the evidence supports auxiliary supervision and the claim is simplified. If neither auxiliary improves, retain both and close the candidate instead of expanding a coefficient or hyperparameter grid. Three-block descriptive evidence is insufficient to establish superiority over the field.

## Development history and next scientific requirement

The native single and independent ensemble in the pinned completed 35-fit family are context. Their results are not newly matched runs for this family and their states cannot be reused. Prior Collab TEST history is consumed. This source has no TEST path, accessor, selector or release. VALID findings remain development-associated.

A promising outcome would require a separately frozen successor, an independent graph/link-prediction benchmark with official splits, a competitive native baseline/independent ensemble and an identifiable recent strong public recipe before a methodological or generalization claim. This preparation does not authorize those studies, TEST evaluation or any execution.

## Resource and terminal policy

Use the existing one-fit supervision on GPU1 with a 24h wall cap, 32GiB host RSS, 70GiB allocated and 75GiB reserved CUDA caps. The measured batch-derived TRAIN proxy is 20.1451 GPU-hours for nine fits; the planning total is 24–30 GPU-hours serially. Full-fit time and fit memory are unmeasured. The prior 28.81GB allocated/44.93GB reserved diagnostic held both loss graphs and four reverses, so those peaks are not fit memory. See `RESOURCE_EXPECTATIONS.json` for exact arithmetic and omitted costs.

Reuse existing fit supervision and source/runtime admission; no exhaustive environment closure or new gradient/CPU ladder is added. Preserve every cap, nonfinite, custody, replay or process failure. No automatic retry/resume, donor state, favourable-batch substitution, precision/batch/budget/normalization fallback, tuning grid or outcome-dependent freeze is allowed by this packet.
