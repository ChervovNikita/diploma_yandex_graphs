# Conditional auxiliary label semantics audit

## Finding and scope

The current Collab NCNC and DDI HL-GNN F4 auxiliaries reconstruct **complete-TRAIN counterpart-incidence patterns on visible residual supports**. Each current arm applies one law to both positive and sampled-negative queries: joint/joint or separate/separate. A positive-joint / negative-separate successor would be a new supervised, query-origin-conditioned regularizer. Its semantics are coherent, but the current source does not establish that positive queries have a common latent cause or that TRAIN-unobserved queries have independent latent causes.

This audit reads local source and custody metadata only. It contains no predictive results, empirical support frequencies, dataset or model payload reads, numerical imports or runs, literature conclusions, server actions, or edits to existing packets. Selected source hashes match their source manifests; the two conditional-loss cores are byte-identical. [SOURCE_BINDINGS.json](/Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/conditional_auxiliary_label_semantics_audit_20261004_v1/SOURCE_BINDINGS.json) records the exact files, hashes and audited line ranges.

## What the source implements

Let `G` be complete TRAIN and `V` the visible graph. For query `(u,v)`, left candidates are `N_V(u) \ N_V(v)` and right candidates are symmetric. The left teacher bit for candidate `w` is `1[(v,w) in G]`, not the query-class label; the right bit is `1[(u,w) in G]`. Thus a teacher one reconstructs a counterpart incidence hidden from `V`. Visible common neighbors are excluded from these supports. A sampled-negative query can have teacher ones on other edge coordinates without a label contradiction. Teacher zero means TRAIN-unobserved, not a verified latent nonlink. [C2, C3, D2]

| Feature | Collab NCNC | DDI HL-GNN F4 |
|---|---|---|
| Positive/negative auxiliary law | Same arm key for both strata | Same arm key for both strata |
| Visible graph | Remove positive minibatch **record positions**, then symmetrize/coalesce | Remove every positive minibatch edge identity plus selected query identities, both directions |
| Duplicate identities | An unmasked duplicate record can retain an edge | Cannot restore a removed auxiliary edge |
| Auxiliary support | Complete residual neighbors; no explicit endpoint exclusion | Complete residual neighbors; both query endpoints explicitly excluded |
| Native target graph | The same minibatch record-masked graph | Complete TRAIN; masking is auxiliary-only |
| Auxiliary exposure/reduction | All native positive/negative minibatch queries; sum of stratum means | Uniform owned selection from each native stratum; half the sum of stratum means, multiplied by native margin-term count |
| Serving | Mean raw member logits on complete TRAIN | Mean raw member scores on complete TRAIN |

These are source contracts, not an assertion of identical scientific exposure across models. [C1–C4, C7, D1–D4]

### Native-negative collisions

**No admitted sampled negative can coincide with a visible TRAIN positive under either declared path.** Collab uses the bound PyG negative sampler on the symmetric raw TRAIN graph and explicitly rejects self-links and any returned TRAIN-edge collision before training. The source also checks that raw and official TRAIN record multisets agree. [C4]

DDI requires the global sampler. It adds self-loops to the exclusion graph and calls PyG `negative_sampling`; any shortfall copies already sampled negatives. The native `edge_index` is reconstructed from `adj_t`, and the auxiliary teacher verifies that `adj_t` equals the symmetric complete-TRAIN membership graph. Since the auxiliary visible graph is a subset, global negatives cannot become visible positives. This conclusion relies on the declared PyG sampler contract; no runtime/library execution was performed here. [D1–D3]

The absence of TRAIN collisions does not make these queries verified latent or future nonedges. A different local or unconstrained sampler would need its own collision policy; those alternatives are not the sealed recipes audited here.

## Cardinality and member responsibilities

For one side with logits `eta_m`, detached bits `b`, support size `n` and observed count `k=sum(b)`, the member loss is

`a_m = log e_k(exp eta_m) - sum(b * eta_m)`.

It is the exact subset law conditional on cardinality `k`, with complete finite support. A common side-logit shift cancels. If `k=0` or `k=n`, there is one feasible pattern and the loss and gradients are exactly zero; those queries remain in the stratum mean. Conditioning removes sensitivity to a common intercept, not candidate-specific preferences, degree effects, exposure differences, or the need to define support. It fits no count distribution or query-link probability. [C6]

For side losses `a_m,b_m`, `M` members and `d=max(n_left+n_right,1)`, the implemented laws are

`J = -log[(1/M) sum_m exp(-a_m-b_m)] / d`

`S = {-log[(1/M) sum_m exp(-a_m)] - log[(1/M) sum_m exp(-b_m)]} / d`.

Joint uses one common member with posterior responsibility `softmax(-a-b)`. Separate has independent side responsibilities `softmax(-a)` and `softmax(-b)`. Independent draws **may select the same member**; separate does not enforce different members, repel assignments, or establish different biological causes. Both priors are fixed uniform. Neither law changes the native target's member weighting or the serving pool. [C1, C6, C7, D1, D4]

Algebraically, `J-S = -log[M * dot(rho_left,rho_right)] / d`. There is no universal ordering. One deterministic side makes `J=S`; identical member laws also make them equal. The benefit or prevalence of informative two-sided patterns cannot be inferred from this source audit.

The source mixes member laws **after each member has been conditioned on the counts**, retaining a uniform prior. This is a valid conditional model. It is generally different from conditioning a uniform mixture of unconditioned Bernoulli models, which would update component weights using their count probabilities. A successor should declare the former interpretation explicitly; the current source provides no count model for the latter.

## Concrete incompatibilities with stronger claims

1. **Class-specific latent assignment is absent today.** Neither current arm implements positive-joint / negative-separate. That proposed assignment must use supervised query origin during training. It must not be described as an existing label-blind mechanism or a generative query classifier. Keeping the choice out of serving avoids inference-label dependence, but does not erase training supervision. [C1, D1]
2. **Collab record masking does not guarantee every positive query edge is hidden.** Duplicate records can retain the query identity. Supports also do not explicitly remove endpoints: if `(u,v)` remains visible and the corresponding self-incidence is absent, endpoint `v` can be a left residual candidate whose teacher coordinate is `(v,v)` and bit is zero. This is a conditional source consequence, not a measured occurrence. Calling Collab's mask an identity mask or its support endpoint-free would be incorrect. DDI has both guarantees for its auxiliary, while its native target remains on complete TRAIN. [C3, D1, D2]
3. **Negative origin is not an all-zero completion label.** Query class and teacher coordinates differ. Conversely, an all-zero side supplies no conditional learning signal; changing its law cannot create a negative penalty. A claimed negative-discrimination mechanism needs informative pattern evidence beyond the query's origin label. [C2, C6, D2]
4. **Conditional calibration is not absolute calibration.** There is no Collab conversion error: with `p0=sigmoid(scale*(raw-offset))`, the native completion weight is `alpha*pt*p0/(pt*p0+1-p0)`, so weight/alpha equals `sigmoid(t)` for the auxiliary `t=scale*(raw-offset)+log(pt)`. Fixed-count conditioning nevertheless cancels the common `offset` and `log(pt)` terms. DDI directly treats raw MLP scores as auxiliary logits; its missing-weight native dispatch is an AUC squared-margin sum, not a probability likelihood. Neither path establishes calibrated query probabilities. Logit scale remains consequential. [C5, C6, D1, D5]

No internal sampler/teacher inconsistency or Collab native-to-auxiliary logit mismatch was established. The incompatibilities above concern the stronger interpretations a contingent successor might claim.

## Serving boundary

On complete TRAIN, every residual candidate is outside the counterpart's TRAIN neighborhood. Hence the **teacher-positive subsets are empty**, even when candidate supports are nonempty: both counts are zero. Removing only the query edge also leaves all teacher bits zero once both query endpoints are excluded. The exact conditional member likelihoods are then one, `J=S=0`, the member responsibilities are uniform, and the joint/separate Bayes factor is one. [C2, C3, C6, D2]

Native predicted NCNC completion weights can still be nonzero. They do not make this deterministic conditional teacher pattern informative. A nontrivial posterior or Bayes-factor serving proposal therefore requires a separately fixed mask/view/support protocol that can hide counterpart incidences, with its additional inference work charged. Simply reusing the current training conditional law on unmasked TRAIN is insufficient. Both existing uniform-serving studies remain unchanged. [C5, C7, D4]

## Fair controls for a contingent successor

Use a fresh prospective packet and preserve both frozen studies. Within each model, compare target-only and the following complete factorial assignment with identical stratum coefficients:

| Positive law | Negative law | Role |
|---|---|---|
| Joint | Joint | Current joint control |
| Separate | Separate | Current separate control |
| Joint | Separate | Proposed successor |
| Separate | Joint | Reverse assignment control |

Hold the native target, sampler, epoch/batch stream, initialization, parameterization, dropout/RNG schedule, selected queries, mask, ordered complete supports, teacher bits/counts, denominator, logit transform, optimizer and stopping/selection budget fixed. Keep deterministic queries in every mean. Preserve each model's existing reduction and scale; silently switching Collab's sum to DDI's half-sum changes auxiliary strength.

For a mechanism contrast, keep query exposure identical across all four assignments. Declare whether Collab retains its native record mask or adopts an identity mask with endpoint exclusions. Any repair of masking, support, calibration or sampler policy must be shared by every arm of a separately declared comparison; otherwise it is confounded with the proposed label-dependent law. Freeze the current raw-score map/temperature for the primary contrast. A calibration or temperature change is a separate common factor.

Record source/mask/support/teacher/query-stream identities prospectively. If later authorized diagnostics characterize where the laws differ, use the same fixed query set and report deterministic and informative cases without selecting training exposure from observed outcomes. Preserve count-free serving and mean native scores. Predictive gains, mechanistic support, novelty and practical cost remain questions for separately authorized evidence.

## Source references

All references below are local, source-only. The binding file includes complete SHA-256 pins and line ranges.

- C1: [Collab training](/Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/exact_cb_support_bucket_paired_predictive_preparation_20261004_v1/paired_train.py:26).
- C2: [Collab teacher](/Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/exact_cb_support_bucket_paired_predictive_preparation_20261004_v1/pattern_teacher.py:15).
- C3: [Collab graph/mask/support](/Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/graph_ncNC_member_completion_qualification_preparation_20261003_v2/graph_ops.py:34).
- C4: [Collab raw-TRAIN and sampler admission](/Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/exact_cb_support_bucket_paired_predictive_preparation_20261004_v1/pilot_data.py:76), [sampler binding](/Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/exact_cb_support_bucket_paired_predictive_preparation_20261004_v1/pilot_model.py:55).
- C5: [Native completion clamp](/Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/graph_ncNC_member_completion_qualification_preparation_20261003_v2/prototype.py:130), [auxiliary capture/logits](/Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/exact_cb_support_bucket_paired_predictive_preparation_20261004_v1/pattern_model.py:43).
- C6: [Count-conditioned core](/Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/exact_cb_support_bucket_paired_predictive_preparation_20261004_v1/conditional_loss.py:46), [mixtures](/Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/exact_cb_support_bucket_paired_predictive_preparation_20261004_v1/conditional_loss.py:222).
- C7: [Collab serving](/Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/exact_cb_support_bucket_paired_predictive_preparation_20261004_v1/pilot_evaluate.py:43).
- D1: [DDI conditional model/target/reduction](/Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/hlgnn_ddi_f4_exact_cb_integration_preparation_20261004_v2/f4_cb_model.py:73).
- D2: [DDI query selection](/Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/hlgnn_ddi_f4_exact_cb_integration_preparation_20261004_v2/ddi_pattern_support.py:25), [supports/mask/teacher](/Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/hlgnn_ddi_f4_exact_cb_integration_preparation_20261004_v2/ddi_pattern_support.py:71).
- D3: [DDI sampler dispatch](/Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/hlgnn_ddi_f4_private_hop_target_only_preparation_20261004_v1/native/utils.py:15), [global sampler](/Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/hlgnn_ddi_f4_private_hop_target_only_preparation_20261004_v1/native/negative_sample.py:6), [graph-to-edge-index mapping](/Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/hlgnn_ddi_f4_private_hop_target_only_preparation_20261004_v1/baseline_train_ddi.py:115).
- D4: [DDI serving](/Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/hlgnn_ddi_f4_private_hop_target_only_preparation_20261004_v1/f4_model.py:84).
- D5: [DDI raw MLP](/Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/hlgnn_ddi_f4_private_hop_target_only_preparation_20261004_v1/native/layer.py:146), [native loss dispatch](/Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/hlgnn_ddi_f4_private_hop_target_only_preparation_20261004_v1/native/model.py:108), [AUC loss](/Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/hlgnn_ddi_f4_private_hop_target_only_preparation_20261004_v1/native/loss.py:5).
