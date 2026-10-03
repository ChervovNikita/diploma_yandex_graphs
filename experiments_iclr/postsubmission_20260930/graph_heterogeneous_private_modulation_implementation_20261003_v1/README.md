# HGB HGT private message factors: source implementation

This packet implements the concrete candidate for a later complete HGB-DBLP comparison. It independently expresses the pinned HGB HGT operations in Torch2.1.2, with no production DGL dependency. It supplies no graph acquisition, dataset loader, optimizer, fit loop, validation selection or heldout launch. Numerical CPU correspondence is pending root execution; source/syntax verification alone does not establish native equivalence or utility.

## Model and fixed sites

`hgt_private.py` supplies native factor-off `NativeHGT`, a shared native core with four full trajectories (`PrivateHGT`), same-site global BE, the rank-one CP output residual, and an unrestricted relation-output table materialized from the CP initialization. Every layer has its own factors. The exact affine value is

`reshape(V_type(h_member * a_member), heads, head_width) @ relation_msg[relation]`, then multiply by `reshape(b_member + c_member*q_relation*u, heads, head_width)`.

The biased V map leaves its shared bias unscaled by `a`; the output factor scales the transformed bias too. Query/key maps receive each member's unmodulated private hidden state. Attention uses softmax across incoming neighbors separately for each canonical relation, then a weighted sum and an equal cross-relation mean. Zero-degree rows participate as zeros in that mean. Native biased typed output, sigmoid skip, optional affine LayerNorm followed by dropout0.2, biased input adapters plus tanh, and the shared biased classifier are retained. No pyHGT GELU or global cross-relation attention normalization is substituted.

Topology contains unique `raw<raw_ID>__<source>__<target>` relation names. Same-type citation/reference relations remain distinct. Edge direction and multiplicity are retained; no self or reverse relations are added. `raw_relation_to_row` explicitly records parameter-row mapping. A production importer must freeze node/relation ordering with its complete raw graph; the CPU fixture derives actual modern DGL etype order and checks that mapping instead of assuming a naming sort. Native author behavior is undefined for an entirely empty relation (`edge id[0]`) or a type with no incoming relation; this implementation refuses those inputs.

All four members re-enter adapters from immutable original features, keep their own hidden/Q/K/V/attention/message frames, and traverse every layer without early pooling or detached caches. Training requires persistent `MemberStreams`: four prospectively bound distinct seeds, one stream per complete trajectory, checkpointed with the model/optimizer. Matched arms use identical initial stream states. The caller's global RNG is restored. The serving function is mean raw member logits, followed by softmax externally. `mean_member_ce` takes compact TRAIN labels aligned with TRAIN IDs and computes ordinary uniform mean own-member CE.

## Initialization and comparison hooks

`initialized_factors` requires a supplied generic seed; this packet chooses no study seed. Each layer uses its own CPU generator with `generic_seed + 1009*layer_index`. Base BE input/output factors are independent signs. CP has exactly `c=0.01*(-3,-1,1,3)/sqrt(5)`, balanced relation signs shuffled once, and independent output-channel signs. No graph, label, cotangent or outcome seeds these factors. Global BE retains the same base signs/core; unrestricted output factors are materialized from the complete initial CP table in the initialization dtype. Construct in the intended dtype before matching; subsequent dtype conversions can otherwise round a free table and a factored expression differently. Matching initial functions does not match CP/free-table optimizer or weight-decay coordinates.

`matched_arms` copies one initialized core into BE/CP/unrestricted models. `untied_core_hook` copies the same native constructor into separate parameter objects, and `wider_be_hook` finds the smallest larger head-divisible width meeting the actual CP parameter count. These are constructor/count hooks for root's remaining comparison recipes; no independent stronger baseline has been qualified or trained.

For DBLP source schema T4/R6/H8/L3/classes4 with norm, let F be the sum of the four native input dimensions after feature preparation. Source counts are:

| Model | Trainable parameters |
|---|---:|
| Native width64 | `64*F + 220320` |
| Global BE width64 | `64*F + 221856` |
| CP width64 | `64*F + 222078` |
| Unrestricted output width64 | `64*F + 225696` |
| Wider global BE width72 | `72*F + 279808` |

CP adds222 parameters to global BE across three layers. Width72 exceeds CP by `8*F + 57730` parameters, a substantial gap; it is not exact parameter matching. Actual F and final counts require the complete verified release. All factor arms retain four full graph trajectories; no arithmetic or speed reduction is claimed.

## Meaningful synthetic CPU fixture

Later root command, using the pinned Torch2.1.2 and isolated modern CPU DGL1.1.3:

```sh
python -B graph_heterogeneous_private_modulation_implementation_20261003_v1/cpu_fixtures.py
```

The fixture uses only fabricated small inputs, not a predictive study. It imports the preserved actual HGB author `model.py` after checking source custody; it never imports `train_hgt.py`. For norm off/on it compares factor-off logits, every named core gradient, feature gradients, and sparse identity adapters. The CP author reference reuses actual author forward/reduce/recurrence and inserts only a V input pre-hook and a post-relation value hook. It compares all four member logits, mean CE, and every core/factor gradient with dropout off. Separate fixtures check a degree-imbalanced relation mean, nonzero affine bias/head order, CP derivative contractions, CP/free-table initial function, full private recurrence/gradient isolation, immutable features and replayable independent dropout streams. Double-precision correspondence uses absolute/relative2e-10 for outputs and3e-9 for gradients; these are fixture tolerances, not study-quality thresholds.

`--torch-only` explicitly omits actual-author correspondence and cannot establish it. `verify_source.py` is read-only stdlib syntax/custody verification and imports neither Torch nor DGL. The author environment has neither installed, so no numerical fixture was executed by the author. Root owns later numerical failure receipts and any revised packet; preserve this packet once sealed.

## Provenance and claims

`PROVENANCE.json` binds HGB commit `ca6fd5bb0c1ca32e63b132c8bfe8f11a4a6629fe`, exact source sites/recipes, the preserved candidate, source qualification manifest/seal, factor/RNG coordinates and input fingerprints. The source qualification's license limitations remain: no HGB NC license or author-reference redistribution grant is inferred. Author reference code remains outside this independently written packet.

BatchEnsemble fast factors, CP/tensor factorization, conditional tensor adaptation, diagonal activation adapters and HGT typed/relation sharing are attributed prior. Same-site diagonal adapters implement the same affine operator. The rank≤2 versus≤1 statement concerns fixed effective member×relation coefficients, not complete HGT function-class superiority. Nonzero CP product Jacobians guarantee neither useful loss gradients nor predictive benefit; antithetic effects may cancel. This packet changes no saved arm/seed/gate policy and reports no real data, fitted outcomes, quality results, remote/GPU execution or final-label access.
