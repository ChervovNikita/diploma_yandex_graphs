# Completed metric audit — 30 October 2026

Snapshot: 19:02 Moscow. The original queue assignments and scientific settings were unchanged. All evaluations here are saved validation outputs; no test split was evaluated.

## Execution and stopping

- 407 completed fits and 32 ensemble exports audited across both servers.
- 0 missing/nonfinite/out-of-range metric or invalid selected-epoch issues.
- All 379 completed longer-protocol fits stopped by validation patience. None hit its epoch cap or ended below the declared minimum.
- Native source-reproduction schedules are reported separately; reaching their fixed horizon is expected.
- This verifies saved result metadata and receipts, not checkpoint replay or optimization convergence.

## Observed validation ranges

These ranges combine different methods and seeds and are not comparative means.

| Model/dataset setting | Metric | Completed outputs | Range (%) |
|---|---|---:|---:|
| collab-buddy | hits@50 | 123 | 65.91–67.88 |
| ogb-ogbg-molbace-gcn-drop0.0 | rocauc | 177 | 54.14–83.52 |
| ogb-ogbg-molbace-gcn-drop0.5 | rocauc | 34 | 61.10–76.56 |
| ogb-ogbg-molhiv-gcn-virtual-drop0.0 | rocauc | 1 | 83.38–83.38 |
| ogb-ogbg-molhiv-gin-drop0.5 | rocauc | 1 | 83.03–83.03 |
| ogb-ogbg-molhiv-gin-virtual-drop0.0 | rocauc | 1 | 86.36–86.36 |
| polyformer-citeseer-cheb | accuracy | 43 | 78.35–84.81 |
| polyformer-citeseer-mono | accuracy | 18 | 80.30–83.61 |
| polyformer-pubmed-cheb | accuracy | 23 | 89.88–92.44 |
| polyformer-pubmed-mono | accuracy | 18 | 88.92–92.37 |

## Matched native controls

The single is the preassigned body 0, not the best member selected after evaluation. Both columns use the longer protocol and the same completed cohorts.

| Setting | Matched cohorts | Single (%) | Independent ensemble (%) |
|---|---:|---:|---:|
| collab-buddy | 7 | 66.350 | 66.514 |
| ogb-ogbg-molbace-gcn-drop0.0 | 7 | 76.253 | 80.487 |
| ogb-ogbg-molbace-gcn-drop0.5 | 2 | 72.088 | 73.114 |
| polyformer-citeseer-cheb | 7 | 81.783 | 82.707 |
| polyformer-citeseer-mono | 3 | 81.404 | 81.855 |
| polyformer-pubmed-cheb | 3 | 90.380 | 91.318 |
| polyformer-pubmed-mono | 3 | 91.521 | 91.741 |

## All available method–I4 comparisons

Each row uses exactly the same completed seed/split cohorts for the method and its independent ensemble. Coverage is out of ten planned trials. Different rows can cover different cohorts. The complete JSON also preserves comparisons with the matched single and original GNNM. No candidate was selected, removed or retuned from this interim audit.

| Setting | Method | Pairs / 10 | Method (%) | I4 (%) | Difference (pp) | Wins / ties / losses |
|---|---|---:|---:|---:|---:|---:|
| collab-buddy | aggregation_conditional_globalJ_v1 | 6/10 | 66.503 | 66.479 | +0.024 | 2/1/3 |
| collab-buddy | aggregation_global_globalJ_v1 | 6/10 | 66.599 | 66.479 | +0.119 | 4/0/2 |
| collab-buddy | aggregation_mean_globalJ_v1 | 5/10 | 66.463 | 66.551 | -0.088 | 2/0/3 |
| collab-buddy | boundary_private_half_pool | 5/10 | 66.555 | 66.551 | +0.004 | 3/0/2 |
| collab-buddy | context_affinity_common | 5/10 | 66.741 | 66.551 | +0.190 | 3/0/2 |
| collab-buddy | context_affinity_split | 5/10 | 66.848 | 66.551 | +0.298 | 2/0/3 |
| collab-buddy | gnnm_boundary_native | 4/10 | 66.495 | 66.572 | -0.077 | 1/0/3 |
| collab-buddy | gnnm_private_bn_state_v3 | 4/10 | 66.552 | 66.572 | -0.020 | 1/0/3 |
| collab-buddy | interaction_geometry_input | 4/10 | 66.639 | 66.572 | +0.067 | 2/0/2 |
| collab-buddy | interaction_input_conditionalJ | 4/10 | 67.002 | 66.572 | +0.430 | 3/0/1 |
| collab-buddy | interaction_input_meanJ | 4/10 | 66.776 | 66.572 | +0.204 | 2/0/2 |
| collab-buddy | interaction_responsibility_private_pool | 4/10 | 66.614 | 66.572 | +0.042 | 2/0/2 |
| collab-buddy | link_endpoint_responsibility_v2 | 4/10 | 66.467 | 66.572 | -0.105 | 1/0/3 |
| collab-buddy | link_query_responsibility_v2 | 4/10 | 66.425 | 66.572 | -0.148 | 2/0/2 |
| collab-buddy | link_same_query_cosine | 4/10 | 66.929 | 66.572 | +0.357 | 3/0/1 |
| collab-buddy | link_same_query_endpoint | 4/10 | 67.129 | 66.572 | +0.556 | 4/0/0 |
| collab-buddy | link_same_query_global | 4/10 | 67.353 | 66.572 | +0.781 | 4/0/0 |
| collab-buddy | portable_input_residual_be_v1 | 4/10 | 66.875 | 66.572 | +0.303 | 3/0/1 |
| collab-buddy | portable_input_residual_only_v1 | 4/10 | 66.788 | 66.572 | +0.216 | 3/0/1 |
| ogb-ogbg-molbace-gcn-drop0.0 | aggregation_conditional_globalJ_v1 | 7/10 | 77.441 | 80.487 | -3.046 | 1/0/6 |
| ogb-ogbg-molbace-gcn-drop0.0 | aggregation_global_globalJ_v1 | 7/10 | 76.787 | 80.487 | -3.700 | 1/0/6 |
| ogb-ogbg-molbace-gcn-drop0.0 | aggregation_mean_globalJ_v1 | 7/10 | 77.080 | 80.487 | -3.407 | 0/0/7 |
| ogb-ogbg-molbace-gcn-drop0.0 | boundary_private_fixed_responsibility | 7/10 | 76.970 | 80.487 | -3.516 | 1/0/6 |
| ogb-ogbg-molbace-gcn-drop0.0 | boundary_private_half_pool | 7/10 | 77.514 | 80.487 | -2.972 | 1/0/6 |
| ogb-ogbg-molbace-gcn-drop0.0 | context_affinity_common | 7/10 | 76.102 | 80.487 | -4.385 | 0/0/7 |
| ogb-ogbg-molbace-gcn-drop0.0 | context_affinity_split | 7/10 | 77.598 | 80.487 | -2.889 | 1/0/6 |
| ogb-ogbg-molbace-gcn-drop0.0 | gnnm_boundary_native | 7/10 | 64.908 | 80.487 | -15.578 | 0/0/7 |
| ogb-ogbg-molbace-gcn-drop0.0 | gnnm_private_bn_state_v3 | 7/10 | 77.577 | 80.487 | -2.909 | 0/0/7 |
| ogb-ogbg-molbace-gcn-drop0.0 | interaction_geometry_input | 7/10 | 76.944 | 80.487 | -3.543 | 0/0/7 |
| ogb-ogbg-molbace-gcn-drop0.0 | interaction_input_conditionalJ | 7/10 | 78.247 | 80.487 | -2.240 | 2/0/5 |
| ogb-ogbg-molbace-gcn-drop0.0 | interaction_input_meanJ | 7/10 | 78.226 | 80.487 | -2.261 | 2/0/5 |
| ogb-ogbg-molbace-gcn-drop0.0 | interaction_responsibility_private_pool | 7/10 | 77.132 | 80.487 | -3.354 | 0/0/7 |
| ogb-ogbg-molbace-gcn-drop0.0 | portable_input_residual_be_v1 | 7/10 | 77.012 | 80.487 | -3.475 | 1/0/6 |
| ogb-ogbg-molbace-gcn-drop0.0 | portable_input_residual_only_v1 | 7/10 | 77.881 | 80.487 | -2.606 | 1/0/6 |
| ogb-ogbg-molbace-gcn-drop0.0 | structural_class_quartile_permuted_private_v1 | 7/10 | 77.802 | 80.487 | -2.684 | 1/0/6 |
| ogb-ogbg-molbace-gcn-drop0.0 | structural_class_quartile_private_v1 | 7/10 | 77.321 | 80.487 | -3.166 | 0/0/7 |
| ogb-ogbg-molbace-gcn-drop0.0 | task_geometry_alignment | 7/10 | 76.248 | 80.487 | -4.239 | 0/0/7 |
| ogb-ogbg-molbace-gcn-drop0.0 | task_geometry_alignment_variance | 7/10 | 77.258 | 80.487 | -3.229 | 0/0/7 |
| ogb-ogbg-molbace-gcn-drop0.5 | aggregation_conditional_globalJ_v1 | 2/10 | 74.634 | 73.114 | +1.520 | 2/0/0 |
| ogb-ogbg-molbace-gcn-drop0.5 | aggregation_global_globalJ_v1 | 2/10 | 74.908 | 73.114 | +1.795 | 2/0/0 |
| ogb-ogbg-molbace-gcn-drop0.5 | aggregation_mean_globalJ_v1 | 2/10 | 75.495 | 73.114 | +2.381 | 2/0/0 |
| ogb-ogbg-molbace-gcn-drop0.5 | boundary_private_fixed_responsibility | 2/10 | 74.799 | 73.114 | +1.685 | 2/0/0 |
| ogb-ogbg-molbace-gcn-drop0.5 | boundary_private_half_pool | 2/10 | 74.634 | 73.114 | +1.520 | 2/0/0 |
| ogb-ogbg-molbace-gcn-drop0.5 | context_affinity_common | 2/10 | 74.505 | 73.114 | +1.392 | 2/0/0 |
| ogb-ogbg-molbace-gcn-drop0.5 | context_affinity_split | 2/10 | 74.542 | 73.114 | +1.429 | 2/0/0 |
| ogb-ogbg-molbace-gcn-drop0.5 | gnnm_boundary_native | 1/10 | 61.099 | 74.176 | -13.077 | 0/0/1 |
| ogb-ogbg-molbace-gcn-drop0.5 | gnnm_private_bn_state_v3 | 1/10 | 75.421 | 74.176 | +1.245 | 1/0/0 |
| ogb-ogbg-molbace-gcn-drop0.5 | interaction_geometry_input | 1/10 | 73.553 | 74.176 | -0.623 | 0/0/1 |
| ogb-ogbg-molbace-gcn-drop0.5 | interaction_input_conditionalJ | 1/10 | 76.557 | 74.176 | +2.381 | 1/0/0 |
| ogb-ogbg-molbace-gcn-drop0.5 | interaction_input_meanJ | 1/10 | 76.447 | 74.176 | +2.271 | 1/0/0 |
| ogb-ogbg-molbace-gcn-drop0.5 | interaction_responsibility_private_pool | 1/10 | 74.103 | 74.176 | -0.073 | 0/0/1 |
| ogb-ogbg-molbace-gcn-drop0.5 | portable_input_residual_be_v1 | 1/10 | 76.374 | 74.176 | +2.198 | 1/0/0 |
| ogb-ogbg-molbace-gcn-drop0.5 | portable_input_residual_only_v1 | 1/10 | 75.788 | 74.176 | +1.612 | 1/0/0 |
| polyformer-citeseer-cheb | aggregation_conditional_globalJ_v1 | 1/10 | 80.752 | 84.211 | -3.459 | 0/0/1 |

## Interpretation

- BUDDY same-query contrast is promising in the available four cohorts: the global variant wins all four versus I4. This is an exploratory result within a large campaign, with six planned trials still missing.
- MolBACE/GCN without dropout shows a substantial weakness in original GNNM. Separate running statistics for each route recover much of the deficit while keeping learned backbone parameters shared, but the available newer variants still trail I4 on mean ROC-AUC.
- The smaller MolBACE/GCN dropout-0.5 sample has gains for several extensions. It is too incomplete to counter the no-dropout result or establish robustness.
- Large BUDDY validation losses and a MolHIV loss near 195 require interpretation of the loss computation and confidence of predictions. Finite ranking scores do not establish calibrated probabilities. See the separate loss audit.
- Source-paper test scores are not comparable to these selected validation scores. The frozen full reporting and heldout evaluation remain necessary.
