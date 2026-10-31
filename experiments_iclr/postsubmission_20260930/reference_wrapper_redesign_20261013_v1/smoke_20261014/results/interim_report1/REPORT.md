# Actual-source short smoke receipts

Authentic-source short engineering tests only. Source-horizon and main tracks reuse this initial code path only where initial operator, optimizer and schedule semantics are identical. These receipts do not qualify full epochs, stopping, later phase transitions, convergence, distributed worker lifecycle, mature-training memory or all 2,480 planned fits. No full campaign fit or scientific accuracy result is implied.

## Matrix coverage

Expected 167; selected seeded terminal cases 107; passed 81; failed 26; missing 60.

Every declared case has a row in `CASE_CHECKS.csv`. All unique attempts, exclusions and failure records remain in `RESULTS.json` and `ATTEMPTS.csv`. Per-step source operations and optimizer clocks remain in `STEP_WORK.csv`.

Selection uses the latest eligible terminal attempt, including failures. An initial unseeded pass or pair-memory child cannot satisfy a matrix case. Identical evidence copies are counted once.

A separate functional-completion diagnostic reports actual four-callback optimizer work, finite losses/gradients, complete VALID, identity and save/reload evidence. Strict floating replay failures retain failed status. Legacy replay evidence is not exact-restoration-qualified; absent old VALID/source-operation receipts remain unknown.

## Actual charged work

| Category | Terminal attempts | Observed optimizer attempts | Completed calls | Unknown costs | Recorded replay callbacks |
|---|---:|---:|---:|---:|---:|
| all_unique_case_work | 119 | 692 | 692 | 0 | 119 |
| selected_matrix | 107 | 620 | 620 | 0 | 107 |
| excluded_attempts | 3 | 24 | 24 | 0 | 3 |
| warmup_or_unseeded | 1 | 4 | 4 | 0 | 1 |
| pair_children | 2 | 20 | 20 | 0 | 2 |

Unknown work is not zero; step-only totals are lower bounds. Failed callbacks and four independent owners are charged by observed optimizer calls. Subcategories overlap and must not be added. Wall sums count worker time, not elapsed queue time.

## Queue duration observations

Completed seeded passing duration observations: 88; median seconds: 13.042040508240461. Pending cases with same-source/procedure duration proxies: 0; pending cases without such coverage: 60.

The report preserves actual completed durations and exact queue membership where available. It does not invent a completion ETA or infer a full-training ETA from four callbacks. Family-level proxies may span different dataset sizes and are explicitly labeled.

## Pair-memory observations

Terminal pair records: 1. Separate sampled joint memory and child outcomes are in `PAIR_MEMORY.csv`. Samples are physical observations, not complete mature-training upper bounds.

## Remaining evidence

Parse errors: 0; terminal dispatcher attempts without a case receipt: 0; unexpected RESULT schemas: 0.

Missing eligible terminal case IDs:

- `ogb-ogbg-molhiv-gin-drop0.5/independent4`
- `ogb-ogbg-molhiv-gin-virtual-drop0.0/native_single`
- `ogb-ogbg-molhiv-gin-virtual-drop0.0/gnnm_boundary_native`
- `ogb-ogbg-molhiv-gin-virtual-drop0.0/boundary_global_half_pool`
- `ogb-ogbg-molhiv-gin-virtual-drop0.0/boundary_private_half_pool`
- `ogb-ogbg-molhiv-gin-virtual-drop0.0/boundary_private_fixed_responsibility`
- `ogb-ogbg-molhiv-gin-virtual-drop0.0/independent4`
- `ogb-ogbg-molhiv-gin-virtual-drop0.5/native_single`
- `ogb-ogbg-molhiv-gin-virtual-drop0.5/gnnm_boundary_native`
- `ogb-ogbg-molhiv-gin-virtual-drop0.5/boundary_global_half_pool`
- `ogb-ogbg-molhiv-gin-virtual-drop0.5/boundary_private_half_pool`
- `ogb-ogbg-molhiv-gin-virtual-drop0.5/boundary_private_fixed_responsibility`
- `ogb-ogbg-molhiv-gin-virtual-drop0.5/independent4`
- `ogb-ogbg-molbace-gcn-drop0.0/gnnm_boundary_native`
- `ogb-ogbg-molbace-gcn-drop0.0/boundary_global_half_pool`
- `ogb-ogbg-molbace-gcn-drop0.0/boundary_private_half_pool`
- `ogb-ogbg-molbace-gcn-drop0.0/boundary_private_fixed_responsibility`
- `ogb-ogbg-molbace-gcn-drop0.0/independent4`
- `ogb-ogbg-molbace-gcn-drop0.5/native_single`
- `ogb-ogbg-molbace-gcn-drop0.5/gnnm_boundary_native`
- `ogb-ogbg-molbace-gcn-drop0.5/boundary_global_half_pool`
- `ogb-ogbg-molbace-gcn-drop0.5/boundary_private_half_pool`
- `ogb-ogbg-molbace-gcn-drop0.5/boundary_private_fixed_responsibility`
- `ogb-ogbg-molbace-gcn-drop0.5/independent4`
- `ogb-ogbg-molbace-gcn-virtual-drop0.0/native_single`
- `ogb-ogbg-molbace-gcn-virtual-drop0.0/gnnm_boundary_native`
- `ogb-ogbg-molbace-gcn-virtual-drop0.0/boundary_global_half_pool`
- `ogb-ogbg-molbace-gcn-virtual-drop0.0/boundary_private_half_pool`
- `ogb-ogbg-molbace-gcn-virtual-drop0.0/boundary_private_fixed_responsibility`
- `ogb-ogbg-molbace-gcn-virtual-drop0.0/independent4`
- `ogb-ogbg-molbace-gcn-virtual-drop0.5/native_single`
- `ogb-ogbg-molbace-gcn-virtual-drop0.5/gnnm_boundary_native`
- `ogb-ogbg-molbace-gcn-virtual-drop0.5/boundary_global_half_pool`
- `ogb-ogbg-molbace-gcn-virtual-drop0.5/boundary_private_half_pool`
- `ogb-ogbg-molbace-gcn-virtual-drop0.5/boundary_private_fixed_responsibility`
- `ogb-ogbg-molbace-gcn-virtual-drop0.5/independent4`
- `ogb-ogbg-molbace-gin-drop0.0/native_single`
- `ogb-ogbg-molbace-gin-drop0.0/gnnm_boundary_native`
- `ogb-ogbg-molbace-gin-drop0.0/boundary_global_half_pool`
- `ogb-ogbg-molbace-gin-drop0.0/boundary_private_half_pool`
- `ogb-ogbg-molbace-gin-drop0.0/boundary_private_fixed_responsibility`
- `ogb-ogbg-molbace-gin-drop0.0/independent4`
- `ogb-ogbg-molbace-gin-drop0.5/native_single`
- `ogb-ogbg-molbace-gin-drop0.5/gnnm_boundary_native`
- `ogb-ogbg-molbace-gin-drop0.5/boundary_global_half_pool`
- `ogb-ogbg-molbace-gin-drop0.5/boundary_private_half_pool`
- `ogb-ogbg-molbace-gin-drop0.5/boundary_private_fixed_responsibility`
- `ogb-ogbg-molbace-gin-drop0.5/independent4`
- `ogb-ogbg-molbace-gin-virtual-drop0.0/native_single`
- `ogb-ogbg-molbace-gin-virtual-drop0.0/gnnm_boundary_native`
- `ogb-ogbg-molbace-gin-virtual-drop0.0/boundary_global_half_pool`
- `ogb-ogbg-molbace-gin-virtual-drop0.0/boundary_private_half_pool`
- `ogb-ogbg-molbace-gin-virtual-drop0.0/boundary_private_fixed_responsibility`
- `ogb-ogbg-molbace-gin-virtual-drop0.0/independent4`
- `ogb-ogbg-molbace-gin-virtual-drop0.5/native_single`
- `ogb-ogbg-molbace-gin-virtual-drop0.5/gnnm_boundary_native`
- `ogb-ogbg-molbace-gin-virtual-drop0.5/boundary_global_half_pool`
- `ogb-ogbg-molbace-gin-virtual-drop0.5/boundary_private_half_pool`
- `ogb-ogbg-molbace-gin-virtual-drop0.5/boundary_private_fixed_responsibility`
- `ogb-ogbg-molbace-gin-virtual-drop0.5/independent4`

Selected failed case IDs:

- `ogb-ogbg-molhiv-gcn-drop0.0/independent4`: Checkpoint third-update replay outside declared engineering tolerance
- `ogb-ogbg-molhiv-gcn-drop0.5/native_single`: Checkpoint third-update replay outside declared engineering tolerance
- `ogb-ogbg-molhiv-gcn-drop0.5/independent4`: Checkpoint third-update replay outside declared engineering tolerance
- `ogb-ogbg-molhiv-gcn-virtual-drop0.0/native_single`: Checkpoint third-update replay outside declared engineering tolerance
- `ogb-ogbg-molhiv-gcn-virtual-drop0.0/gnnm_boundary_native`: Checkpoint third-update replay outside declared engineering tolerance
- `ogb-ogbg-molhiv-gcn-virtual-drop0.0/boundary_global_half_pool`: Checkpoint third-update replay outside declared engineering tolerance
- `ogb-ogbg-molhiv-gcn-virtual-drop0.0/boundary_private_half_pool`: Checkpoint third-update replay outside declared engineering tolerance
- `ogb-ogbg-molhiv-gcn-virtual-drop0.0/boundary_private_fixed_responsibility`: Checkpoint third-update replay outside declared engineering tolerance
- `ogb-ogbg-molhiv-gcn-virtual-drop0.0/independent4`: Checkpoint third-update replay outside declared engineering tolerance
- `ogb-ogbg-molhiv-gcn-virtual-drop0.5/native_single`: Checkpoint third-update replay outside declared engineering tolerance
- `ogb-ogbg-molhiv-gcn-virtual-drop0.5/gnnm_boundary_native`: Checkpoint third-update replay outside declared engineering tolerance
- `ogb-ogbg-molhiv-gcn-virtual-drop0.5/boundary_global_half_pool`: Checkpoint third-update replay outside declared engineering tolerance
- `ogb-ogbg-molhiv-gcn-virtual-drop0.5/boundary_private_half_pool`: Checkpoint third-update replay outside declared engineering tolerance
- `ogb-ogbg-molhiv-gcn-virtual-drop0.5/boundary_private_fixed_responsibility`: Checkpoint third-update replay outside declared engineering tolerance
- `ogb-ogbg-molhiv-gcn-virtual-drop0.5/independent4`: Checkpoint third-update replay outside declared engineering tolerance
- `ogb-ogbg-molhiv-gin-drop0.0/native_single`: Checkpoint third-update replay outside declared engineering tolerance
- `ogb-ogbg-molhiv-gin-drop0.0/gnnm_boundary_native`: Checkpoint third-update replay outside declared engineering tolerance
- `ogb-ogbg-molhiv-gin-drop0.0/boundary_global_half_pool`: Checkpoint third-update replay outside declared engineering tolerance
- `ogb-ogbg-molhiv-gin-drop0.0/boundary_private_half_pool`: Checkpoint third-update replay outside declared engineering tolerance
- `ogb-ogbg-molhiv-gin-drop0.0/boundary_private_fixed_responsibility`: Checkpoint third-update replay outside declared engineering tolerance
- `ogb-ogbg-molhiv-gin-drop0.0/independent4`: Checkpoint third-update replay outside declared engineering tolerance
- `ogb-ogbg-molhiv-gin-drop0.5/native_single`: Checkpoint third-update replay outside declared engineering tolerance
- `ogb-ogbg-molhiv-gin-drop0.5/gnnm_boundary_native`: Checkpoint third-update replay outside declared engineering tolerance
- `ogb-ogbg-molhiv-gin-drop0.5/boundary_global_half_pool`: Checkpoint third-update replay outside declared engineering tolerance
- `ogb-ogbg-molhiv-gin-drop0.5/boundary_private_half_pool`: Checkpoint third-update replay outside declared engineering tolerance
- `ogb-ogbg-molhiv-gin-drop0.5/boundary_private_fixed_responsibility`: Checkpoint third-update replay outside declared engineering tolerance
