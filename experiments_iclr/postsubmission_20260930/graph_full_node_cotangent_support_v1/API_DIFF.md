# Exact API and output delta

## Function interfaces

The original final arguments were `tangent_mode='graph', control_seed=None`. The candidate appends:

```python
cotangent_support='train_remasked', homogeneous_full_node_outputs=False
```

The default `False` deliberately requires an explicit full-output contract at every candidate call, including common-only and the remasked comparator. This is a source/interface qualification requirement; it is not a study outcome rule. The caller supplies true homogeneous full-node class logits. Both support modes require exact `0..N-1` coverage in `target_nodes`, each once, in any permutation. A partial or typed target subset requires a different explicit operation.

New exported helper:

```python
signed_graph_contrast(cotangents, candidate_logits,
                      common_logits_same_alpha, alpha,
                      first_order_slope=None)
```

It accepts detached-or-detachable `[4,N,C]` cotangents/candidates and `[N,C]` common logits and performs no model forward. The helper evaluates the supplied functional; its raw return does not infer the caller's support provenance. The initializer annotates every attempted signed record with that provenance.

All other public signatures, the returned `(slices, stats)` tuple, and existing installation/binding helpers are unchanged. Existing stats fields keep their previous TRAIN meaning. The complete line-level change is in `initializer.diff`.

## Changed support operation

For each graph band, remasked support retains the original operation:

```python
cotangent = torch.zeros_like(base_logits)
cotangent[train_rows] = band[target_nodes[train_rows]]
```

Full support uses:

```python
cotangent = band[target_nodes]
```

The residual still contains only TRAIN mean-CE derivatives. The common `g` and the band-gradient sum remain unchanged. Both modes preserve the admitted private slice and all original cap/projection/acceptance expressions.

## New top-level stats

| Field | Meaning |
|---|---|
| `operation` | Changed to `graph_band_cotangent_support_v1`. |
| `cotangent_support` | `train_remasked` or `full_node`, used for the four band pullbacks. |
| `homogeneous_full_node_outputs` | Explicit caller contract, recorded as true after validation. |
| `output_diagnostic_scope` | `all_full_node_target_outputs`. |
| `acceptance_scope` | `unchanged_TRAIN_CE_and_TRAIN_functional_separation`. |
| `full_output_tangent_finite` | Whether all centered full-output JVPs are finite. Off-TRAIN failure adds no acceptance rejection. |
| `full_output_tangent_gram` | Four-by-four Gram of centered `J*tangent`, divided by `N*C`, or null on nonfinite full JVPs. |
| `full_output_tangent_pair_rms` | Six full-output pair RMS values, or null on nonfinite full JVPs. |
| `signed_graph_contrast_functional` | `shared_full_node_q` for both arms. |
| `signed_graph_contrast_cotangent_support` | `full_node`; describes the diagnostic q, not the initializer's pullback support. |
| `signed_graph_contrast_support_matched` | True for `full_node`, false for `train_remasked`; a support-provenance flag, not a success flag. |
| `support_matched_remask_contrast_computed` | False. |
| `signed_graph_contrast_first_order_slope` | `sum_m <q_full,m, J*tangent_m>`; null if unavailable. It need not be a remasked tangent norm. |
| `candidate_trial_forward_calls` | Four forwards per completed existing trial bundle, including fallback. |
| `same_alpha_common_forward_calls` | Extra common diagnostic forwards, counted even if that extra closure raises. |

`line_search_forward_calls` now includes both kinds of line-search forward. Existing `source_tangent_pair_rms` and `actual_centered_pair_rms` continue to describe TRAIN-only separation.

## New per-attempt fields

`full_output_actual_centered_pair_rms` contains the six finite centered full-output pair RMS values. `signed_graph_contrast` contains its status, signed sum, four signed terms, exact alpha, supplied first-order prediction/ratio, and full finite-change Gram divided by `alpha**2` and `N*C` when available. It is annotated with:

```text
cotangent_functional = shared_full_node_q
cotangent_support = full_node
initializer_cotangent_support = train_remasked | full_node
support_matched = initializer_cotangent_support == full_node
```

The common fallback has zero signed contrast because its path is identical to the same-alpha common path; no extra forward is needed. Unavailable/exception/nonfinite signed diagnostics are recorded. All signed records are diagnostic-only. Nonfinite helper returns may omit detailed scalar/Gram fields; consumers must inspect `status`/`finite`. Early zero-gradient returns may omit diagnostics that were never constructed.

The trial is still accepted solely by the original `if quality and useful` condition. Full-output finite trial checks already present in the pinned source are retained; the amendment adds no signed or Gram gate.
