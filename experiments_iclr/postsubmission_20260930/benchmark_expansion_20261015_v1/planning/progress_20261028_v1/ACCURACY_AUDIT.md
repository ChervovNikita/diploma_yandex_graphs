# Completed-score audit — 28 October 2026

Server51 access is restored. The supplied password was correct. A temporary private-terminal reader lost one byte; whitespace framing drains the competing read, and a private integrity handshake now verifies exact delivery before SSH. No password was saved to project files or written to the other Mac. The authentication terminals were closed.

## Scope and numerical checks

Inspected all 168 completed fits and 21 independent-ensemble exports found in the three frozen worker plans. No selected metric/loss was missing, nonfinite, or outside its expected metric range. Every fit reports complete source VALID evaluation. This is a metadata sanity audit, not an independent checkpoint replay or proof of paper reproduction.

| Model/dataset cell | Metric | Completed result rows | Observed validation range |
| --- | --- | ---: | ---: |
| collab-buddy | hits@50 | 89 | 65.91–67.59% |
| ogb-ogbg-molbace-gcn-drop0.5 | rocauc | 2 | 75.49–76.52% |
| polyformer-citeseer-cheb | accuracy | 42 | 78.35–84.81% |
| polyformer-citeseer-mono | accuracy | 18 | 80.30–83.61% |
| polyformer-pubmed-cheb | accuracy | 20 | 89.88–92.37% |
| polyformer-pubmed-mono | accuracy | 18 | 88.92–92.37% |

Ranges include different completed methods, cohorts, and protocol tracks. They are not mean performance estimates and are not interchangeable across cells.

## Training duration

All 150 longer-protocol fits stopped by validation patience; none hit its maximum epoch cap or ended below its declared minimum. Two BUDDY native-replication runs ended at the prescribed 100-epoch source horizon. These observations support execution of the intended stopping policy, not a proof of globally converged optimization.

## Incomplete BUDDY comparisons

All available shared-method comparisons are retained below. Each mean uses only the stated matched cohorts out of ten planned. These are incomplete descriptive observations. Full reporter binding/convergence resolution is not assessed by this scalar audit; no interval or superiority conclusion is reported. No method was promoted, removed, or retuned in response.

| Method | Pairs / 10 | Shared mean | I4 on same pairs | Difference, pp | Wins / ties / losses |
| --- | ---: | ---: | ---: | ---: | ---: |
| aggregation_conditional_globalJ_v1 | 4/10 | 66.478 | 66.572 | -0.094 | 1/1/2 |
| aggregation_global_globalJ_v1 | 4/10 | 66.500 | 66.572 | -0.072 | 2/0/2 |
| aggregation_mean_globalJ_v1 | 4/10 | 66.555 | 66.572 | -0.017 | 2/0/2 |
| boundary_private_half_pool | 4/10 | 66.660 | 66.572 | +0.088 | 3/0/1 |
| context_affinity_common | 4/10 | 66.733 | 66.572 | +0.161 | 2/0/2 |
| context_affinity_split | 4/10 | 66.960 | 66.572 | +0.387 | 2/0/2 |
| gnnm_boundary_native | 4/10 | 66.495 | 66.572 | -0.077 | 1/0/3 |
| gnnm_private_bn_state_v3 | 4/10 | 66.552 | 66.572 | -0.020 | 1/0/3 |
| interaction_geometry_input | 4/10 | 66.639 | 66.572 | +0.067 | 2/0/2 |
| interaction_input_conditionalJ | 4/10 | 67.002 | 66.572 | +0.430 | 3/0/1 |
| interaction_input_meanJ | 4/10 | 66.776 | 66.572 | +0.204 | 2/0/2 |
| interaction_responsibility_private_pool | 4/10 | 66.614 | 66.572 | +0.042 | 2/0/2 |
| link_endpoint_responsibility_v2 | 2/10 | 66.350 | 66.600 | -0.250 | 0/0/2 |
| link_query_responsibility_v2 | 2/10 | 66.236 | 66.600 | -0.364 | 1/0/1 |
| link_same_query_cosine | 2/10 | 67.193 | 66.600 | +0.593 | 2/0/0 |
| link_same_query_endpoint | 2/10 | 67.108 | 66.600 | +0.508 | 2/0/0 |
| link_same_query_global | 2/10 | 67.227 | 66.600 | +0.627 | 2/0/0 |
| portable_input_residual_be_v1 | 2/10 | 66.829 | 66.600 | +0.229 | 2/0/0 |
| portable_input_residual_only_v1 | 2/10 | 66.782 | 66.600 | +0.182 | 2/0/0 |

## Issues requiring interpretation

- BUDDY ranking metrics are close to the matched native/ensemble baseline range. Several shared variants improve available paired scores; others lose or are essentially tied. The base GNNM wrapper is slightly behind I4 on its four currently available pairs.
- Recorded BUDDY validation loss ranges up to 19.35, including high values for some native runs. Ranking performance alone cannot establish good probability quality. This deserves a separate loss/logit analysis; finite values alone do not resolve it.
- Only two MolBACE/GCN shared-method fits have completed, without their matched native controls in this snapshot. The PubMed shared-method observations also lack the matching current-campaign control cohort. Their scores cannot yet establish a method advantage.
- Current outcomes are VALID results. A source-paper TEST score cannot be treated as a like-for-like reproduction comparator here.

## Evidence

`allocation-completed-scores.json` and `server51-completed-scores.json` preserve source result receipts and all observed scores. `COMPLETED_SCALARS.csv` is the compact complete table. `SCORE_AUDIT_SUMMARY.json` retains exact matched IDs, all partial differences and limitations. Queue owners remain alive with six active fits and no current failed tasks in the host snapshots. Training and scientific settings were unchanged.
