# Forward and loss ordering

Pseudocode is a source specification, not an executed adapter. The ordering below makes the teacher boundary and gradient paths reviewable.

```text
positive_queries = train_records[selected_record_ids]
negative_queries = native_epoch_negative_rows[selected_record_ids]
G = symmetric_coalesce(train_records excluding selected_record_ids)
h = native_width64_encoder(features, G with native encoder-only edge dropout)

for queries in [positive_queries, negative_queries]:
    support = native_neighbors(G, queries)  # before encoder dropout, no teacher
    U = deterministic_order(left-only and right-only counterpart slots)
    endpoint_product = h[i] * h[j]
    h_outer = h + native_xlin(h)             # outer feature graph stays live

    # Native left and right depth-zero paths include their own second xlin.
    # Auxiliary needs autograd through these entire scorer calls.
    score_left = depth_zero_score(h_outer, G, left_counterpart_queries)
    score_right = depth_zero_score(h_outer, G, right_counterpart_queries)
    t = native_clamp_logit(concatenate_and_order(score_left, score_right))
    q = sigmoid(t)
    context = masked_only_context(h, h_outer, support, q)
    count_logits = shared_count_MLP(context, all k in 0..R)
    log_pi = fp64_logsoftmax(count_logits)    # R=0 is special: pi_0=1

    # Use detached parameters; teacher oracle is not passed to this function.
    B[0..3] = fp64_count_conditioned_sample(stopgrad(t), stopgrad(log_pi),
                                         independently_keyed_uniforms)
    for draw in 0..3:
        common = sum(h_outer[v] for retained common neighbors v)
        common += sum(1.05 * stopgrad(B[draw,r]) * h_outer[node(r)] for r in U)
        raw_logit[draw] = native_single_decode(endpoint_product, common)

    # Label lookup comes after graph/context/predicted law/target forward.
    Z = train_membership_teacher_only(counterpart_queries_from_U)
    K = sum(Z)
    loss_aux_query = stable_pattern_NLL(t, log_pi, Z, K) / R if R>0 else 0

loss_main = mean(-logsigmoid(positive_raw_logits))
          + mean(-logsigmoid(-negative_raw_logits))
loss_aux  = mean_over_all_positive_queries(loss_aux_query)
          + mean_over_all_negative_queries(loss_aux_query)
loss_total = loss_main + 1 * loss_aux
backward(loss_total); native_Adam_step()
serve(queries) = mean(raw_logit[0..3])         # no Z/K at evaluation
```

After freezing the primary C64-D4-selected checkpoint, the prebound C64-M4 calculation replaces only B with four independent Bernoulli(mu) vectors from that same predicted law's actual marginals. The model, features, alpha placement, decoder, keyed slot uniforms, draw count and raw-logit averaging remain fixed. There is no C64-M4 training loss or selection pass. Its added marginal DP and scoring are charged. It tests retaining dependence during prediction; it does not isolate auxiliary-training effects.

The source implementation may perform the teacher lookup earlier in wall-clock time only if a reviewed interface enforces identical isolation. The conservative ordered interface above is the preparation contract. The teacher graph must not be stored as an accessible model input or forwarded into the sampler/head. Membership lookup returns labels keyed to the already sealed support; it cannot add or remove slots.

## Allowed paths

| Path | Main target loss | Pattern auxiliary |
| --- | --- | --- |
| Encoder and outer xlin to common/residual features | Differentiable | Differentiable through scorer and context |
| Depth-zero scorer to t | No sampler gradient | Differentiable |
| t/q to count context and pi | No sampler gradient | Differentiable |
| Count head | No sampler gradient | Differentiable |
| Shared native decoder maps | Differentiable on four downstream decodes | Differentiable inside scorer |
| TRAIN teacher Z/K | Labels only | Labels only |
| VALID/TEST labels | Absent | Absent |

For a main-only gradient audit, test the t/count-logit **output nodes** and count-head parameters, not a shared decoder parameter that also lies on the permitted downstream path. Shared parameters can have main gradients while the sampler route is detached. For an auxiliary-only audit, every active scorer path and the head must receive finite gradients on a nondegenerate fabricated fixture; unused ptlin stays unused. No blanket detachment of h_outer is allowed.

## Prospective fixed diagnostic mask

Before any future comparator science, freeze a TRAIN-only diagnostic set by assigning each ascending complete TRAIN record ID the key SHA256 of the ASCII string `ncnc-cardinality-single-v1|seed=0|domain=diagnostic-records|record=<decimal_id>`. Choose the lowest65536 keys in byte order, break any exact key tie by ascending record ID, and remove exactly those records before coalescing. Pair them in ascending selected-record-ID order with the first65536 rows of a separately seeded pinned native negative draw for that complete TRAIN snapshot. Its seed is the first8 bytes, little endian modulo2^31-1, of SHA256 of `ncnc-cardinality-single-v1|seed=0|domain=diagnostic-negatives`. Capture and restore native random states around this diagnostic draw so it does not advance the scientific stream. Preserve chosen ID/negative-row digests before reading any model outcome. This is an auxiliary/target competence audit, not additional training.

Enumerate slots once from the masked diagnostic graph and seal counterpart IDs/labels. The membership teacher remains label-only. Run all admitted methods on the same usable graph and query/slot set with dropout disabled, and score reconstructed observation patterns with their declared law. The complete TRAIN teacher graph is not their encoder input. The diagnostic can expose observation-proxy losses, synthetic-positive versus source-zero fit, observed count strata, and the degree of source-zero domination. It cannot certify hidden true links. It adds no checkpoint selection rule and no training masks to the frozen J/F study.

The diagnostic record selection, lookup, source-zero counts, prediction and likelihood work are paid analysis. This preparation writes only its deterministic recipe; no record IDs, sampled negatives, labels or model values are generated here.
