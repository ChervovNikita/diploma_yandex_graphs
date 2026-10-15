# Exact method coverage for benchmark review

This is a readable map of the existing 6,460-fit proposal. It adds no methods,
settings or runs. The campaign remains pending preparation and user GO.

## Comparisons present in every one of the 27 settings

Each setting fixes a dataset, native architecture and published configuration.
Ten trials each train one source-setting reproduction and four independent
native bodies. Body 0 is reused as the matched single; the four bodies also
form the independent ensemble. Every setting also trains original GNNM.
The other rows below are additional shared-backbone methods and their controls.

The source reproduction keeps the paper/release training schedule. The matched
single, independent ensemble and shared methods use the longer recorded
controller. These are separate comparisons; the longer schedule is not called
an exact reproduction of the paper.

## Where each shared recipe is included

Every Yes means all ten trials of every setting in that column. A dash means
that exact recipe is absent; task-specific counterparts are listed separately.
All these methods retain one shared native backbone.

| # | Shared recipe | Node | Molecule | NCN / NCNC | BUDDY |
| ---: | --- | :---: | :---: | :---: | :---: |
| 1 | Original GNNM: factors and uniform averaging | Yes | Yes | Yes | Yes |
| 2 | Private factors also learn from the averaged prediction | Yes | Yes | Yes | Yes |
| 3 | Fixed random positive training weights for each route | Yes | Yes | — | — |
| 4 | Earlier within-route supervised contrastive objective | Yes | — | — | — |
| 5 | Supervised class alignment across route embeddings | Yes | Yes | — | — |
| 6 | Class alignment plus a within-class variance floor | Yes | Yes | — | — |
| 7 | Align the same link query across routes | — | — | Yes | Yes |
| 8 | Contrast the same query against sampled alternatives | — | — | Yes | Yes |
| 9 | Contrast against alternatives sharing an endpoint | — | — | Yes | Yes |
| 10 | Route weights tied to degree or molecule size | Yes | Yes | — | — |
| 11 | Matched randomized control for structural weights | Yes | Yes | — | — |
| 12 | Persistent link weights correlated through endpoints | — | — | Yes | Yes |
| 13 | Link-query weighting control with matched raw marginals | — | — | Yes | Yes |
| 14 | Every route learns the same mixture of context targets | Yes | Yes | Yes | Yes |
| 15 | Different routes learn different context targets | Yes | Yes | Yes | Yes |
| 16 | Factors plus a private nonlinear input correction | Yes | Yes | Yes | Yes |
| 17 | Private nonlinear corrections without diagonal factors | Yes | Yes | Yes | Yes |
| 18 | Train all parameters partly on the averaged prediction | Yes | Yes | Yes | Yes |
| 19 | Learn one global set of route weights | Yes | Yes | Yes | Yes |
| 20 | Learn route weights from prediction information | Yes | Yes | Yes | Yes |
| 21 | Combine geometry with private input corrections | Yes | Yes | Yes | Yes |
| 22 | Combine structural weights with pooled private training | Yes | Yes | Yes | Yes |
| 23 | Combine input corrections with pooled mean training | Yes | Yes | Yes | Yes |
| 24 | Combine input corrections with conditional aggregation | Yes | Yes | Yes | Yes |
| 25 | Keep separate BN running statistics for each route | — | Yes | — | Yes |
| | **Recipes per setting** | 19 | 19 | 18 | 19 |

Molecular supervised alignment uses molecular class labels and graph embeddings.
Link contrast uses query identity and sampled alternatives rather than treating
the absence of an observed edge as certified nonlink truth. The earlier node
within-route SupCon definition is preserved as a separate control. BN state
ownership applies only to architectures that already contain native BN.

## Fixed rules and controls

Here, an own-route loss trains each route to solve the native task. A pooled
loss evaluates the prediction actually served by the ensemble. “Shared” means
the common native weights; “private” means route-specific wrapper parameters.
These are fixed recipe choices, not a search over loss coefficients.

The exact source rules below use F for the mean own-route loss, W for a
positively reweighted own-route loss, and J for the stated mixture of own-route
and served-prediction losses. Raw source rules are included to prevent prose
from silently changing the implemented method.

### 1. Original GNNM: factors and uniform averaging

Recipe: `gnnm_boundary_native`.

Shared native body, original signed input factors and neutral remaining factors; mean own native risk F and mean raw logits.

Controls: `native_M1`, `native_I4`.

### 2. Private factors also learn from the averaged prediction

Recipe: `boundary_private_half_pool`.

Shared gradient F; private gradient .5F+.5L_mean. One native optimizer transition.

Controls: `gnnm_boundary_native`, `aggregation_mean_globalJ_v1`.

### 3. Fixed random positive training weights for each route

Recipe: `boundary_private_fixed_responsibility`.

Preserve predecessor fixed Exp weights and native class/group normalization. Shared F; private weighted W.

Controls: `gnnm_boundary_native`, `structural_class_quartile_private_v1`.

### 4. Earlier within-route supervised contrastive objective

Recipe: `boundary_own_within_route_supcon`.

Exact predecessor node within-route SupCon recipe; do not alias it to the new cross-route geometry.

Controls: `gnnm_boundary_native`, `task_geometry_alignment`.

### 5. Supervised class alignment across route embeddings

Recipe: `task_geometry_alignment`.

All-block F+.05A, temperature .2. Class-balanced anchors/denominator; same-class existing route preheads. Full native task population retained.

Controls: `gnnm_boundary_native`, `task_geometry_alignment_variance`.

### 6. Class alignment plus a within-class variance floor

Recipe: `task_geometry_alignment_variance`.

Same alignment plus conditional within-route, within-class variance floor .5/sqrt(width); coefficient .05 on the declared combined auxiliary.

Controls: `gnnm_boundary_native`, `task_geometry_alignment`.

### 7. Align the same link query across routes

Recipe: `link_same_query_cosine`.

Shared F; private F+.05A. Existing authenticated query pairs, no task-negative invention, same endpoint-eligible anchors as contrast controls.

Controls: `gnnm_boundary_native`, `link_same_query_global`, `link_same_query_endpoint`.

### 8. Contrast the same query against sampled alternatives

Recipe: `link_same_query_global`.

Shared F; private F+.05InfoNCE, temperature .2. Same-query positive; same-relation opposite source-label alternatives; endpoint-matched counts/support.

Controls: `link_same_query_cosine`, `link_same_query_endpoint`.

### 9. Contrast against alternatives sharing an endpoint

Recipe: `link_same_query_endpoint`.

Same rule and source query reductions as global, but alternatives share an endpoint. Max1024 anchors and16 alternatives; disclose zero support.

Controls: `gnnm_boundary_native`, `link_same_query_global`.

### 10. Route weights tied to degree or molecule size

Recipe: `structural_class_quartile_private_v1`.

TRAIN classwise incoming-degree/atom-count fractional quartiles, w=.5+2q. Shared F; private W. Preserve native reductions, no minibatch renormalization.

Controls: `gnnm_boundary_native`, `structural_class_quartile_permuted_private_v1`.

### 11. Matched randomized control for structural weights

Recipe: `structural_class_quartile_permuted_private_v1`.

One whole-vector within-class permutation of the exact quartile weight table, fit seed+7030031, no redraw; same block updates.

Controls: `structural_class_quartile_private_v1`.

### 12. Persistent link weights correlated through endpoints

Recipe: `link_endpoint_responsibility_v2`.

Corrected v2 symmetric positive endpoint field; shared F/private W; exact native group normalization. Normalized marginals need not match query control.

Controls: `gnnm_boundary_native`, `link_query_responsibility_v2`.

### 13. Link-query weighting control with matched raw marginals

Recipe: `link_query_responsibility_v2`.

Corrected v2 query field: two Exp draws for nonloops, one for loops, matching endpoint raw marginal; group normalization retained.

Controls: `link_endpoint_responsibility_v2`.

### 14. Every route learns the same mixture of context targets

Recipe: `context_affinity_common`.

All-block F+.05A; existing prehead donor distribution at temperature .2 learns .5q_A+.5q_B for every route. TRAIN-only target labels; same donor support/work as split.

Controls: `gnnm_boundary_native`, `context_affinity_split`.

### 15. Different routes learn different context targets

Recipe: `context_affinity_split`.

Same loss/donors/budget as common. Routes0/1 use q_A, routes2/3 q_B. No extra inference input, head, teacher or VALID target.

Controls: `context_affinity_common`.

### 16. Factors plus a private nonlinear input correction

Recipe: `portable_input_residual_be_v1`.

BE plus h+U_m GELU(V_m h) at the declared encoded input; rank=min(8,width), zero U and fixed-law orthonormal V frames; one shared native body.

Controls: `gnnm_boundary_native`, `portable_input_residual_only_v1`.

### 17. Private nonlinear corrections without diagonal factors

Recipe: `portable_input_residual_only_v1`.

Four private residual routes around one shared native body, BE hooks/banks removed before training; same native constructor and optimizer semantics.

Controls: `portable_input_residual_be_v1`, `native_M1`, `native_I4`.

### 18. Train all parameters partly on the averaged prediction

Recipe: `aggregation_mean_globalJ_v1`.

All-block J=.5F+.5L_actual. Uniform mean, zero added parameters; alias of predecessor global half-pool, one fit only.

Controls: `gnnm_boundary_native`, `aggregation_global_globalJ_v1`, `aggregation_conditional_globalJ_v1`.

### 19. Learn one global set of route weights

Recipe: `aggregation_global_globalJ_v1`.

Same J, four TRAIN-learned biases, weights .05+.8softmax(score), exact uniform start.

Controls: `aggregation_mean_globalJ_v1`, `aggregation_conditional_globalJ_v1`.

### 20. Learn route weights from prediction information

Recipe: `aggregation_conditional_globalJ_v1`.

Same J,20 parameters, detached entropy confidence/JS descriptors, hidden4 GELU scorer and route biases; same positive weight floor/uniform start.

Controls: `aggregation_mean_globalJ_v1`, `aggregation_global_globalJ_v1`, `same_selected_bank_fixed_mean`.

### 21. Combine geometry with private input corrections

Recipe: `interaction_geometry_input`.

Combine BE input residual with labelled alignment+variance or link endpoint contrast, using each task's exact preceding gradient rule; no coefficient search.

Controls: `gnnm_boundary_native`, `portable_input_residual_be_v1`, `task_geometry_alignment_variance_or_link_endpoint`.

### 22. Combine structural weights with pooled private training

Recipe: `interaction_responsibility_private_pool`.

Shared F; private .5W_structured+.5L_mean. Compare G0, structured W and private half-pool to complete the2x2.

Controls: `gnnm_boundary_native`, `structural_class_quartile_private_or_link_endpoint_v2`, `boundary_private_half_pool`.

### 23. Combine input corrections with pooled mean training

Recipe: `interaction_input_meanJ`.

BE input residual with all-block .5F+.5L_mean; same adapter initialization/site and J as its ingredient controls.

Controls: `aggregation_mean_globalJ_v1`, `portable_input_residual_be_v1`, `interaction_input_conditionalJ`.

### 24. Combine input corrections with conditional aggregation

Recipe: `interaction_input_conditionalJ`.

BE input residual plus conditional20-parameter aggregator, all-block .5F+.5L_actual; compare same selected bank under actual gate and mean.

Controls: `aggregation_conditional_globalJ_v1`, `aggregation_mean_globalJ_v1`, `interaction_input_meanJ`.

### 25. Keep separate BN running statistics for each route

Recipe: `gnnm_private_bn_state_v3`.

Prospective route-private BN running buffers/counters; affine and native weights shared, original BN operators/batches. No BN introduced into BN-free models.

Controls: `gnnm_boundary_native`.

## Counts and next review

| Scope | Settings | Source fits | Independent native bodies | Shared fits | Total fits |
| --- | ---: | ---: | ---: | ---: | ---: |
| Node | 8 | 80 | 320 | 1,520 | 1,920 |
| Molecule | 16 | 160 | 640 | 3,040 | 3,840 |
| NCN / NCNC | 2 | 20 | 80 | 360 | 460 |
| BUDDY | 1 | 10 | 40 | 190 | 240 |

The total is **6,460 training fits plus 270 averaging/export tasks**. Short
execution checks and current development pilots are recorded separately. There
is no hyperparameter search and no rerun of the original five-dataset paper.

The methods share the baseline comparison, but their auxiliary targets differ
where the task requires it. The trial matrix tests hypotheses; it does not
assume all ingredients will help or that combining them will improve accuracy.

- [Overview and native training settings](../BENCHMARK_OVERVIEW.md)
- [Exact frozen roster](../expanded_study_design_v1/PROPOSED_ROSTER.json)
- [Measured costs and unknown costs](../resource_budget_draft_v1/README.md)
- [Completed evidence and conditional next hypotheses](../research_next_decision_v1/MEMO.md)
