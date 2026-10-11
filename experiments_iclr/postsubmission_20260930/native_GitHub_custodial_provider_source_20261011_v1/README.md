# GitHub custodial provider source, inactive

This source prepares the canonical PyG GitHub NPZ release for the existing safe sequential SAGE consumers. It is not executed here. All template admissions are false. Root owns acquisition, access eligibility, preparation admission, host/runtime bindings and all later qualification or fits.

## Provider and evidence

Pinned PyG commit: `76ff9c2ce18c8cebf52122b57e2aeadce9793d10`. The provider source is preserved as `pinned_sources/PyG_GitHub.py` (SHA256 `421aa192f8ba799f1517f1ea3f5acbda2613a196af14db1087b27559d78cf5d0`). Canonical released URL: https://graphmining.ai/datasets/ptg/github.npz . It is unversioned; root must bind actual acquired bytes by SHA256 before the exporter reads it. Released NPZ has numeric `features`, `target`, `edges`; the provider casts to float/long and transposes edge rows. No official roles, symmetrization or loop insertion are supplied by that source.

Source prose/stats are inconsistent about nodes/features. Fixed intended shape is N37700/D128/C2/stored directed E578006, with later exact custodial checks. The exporter checks graph counts before accessing targets and fails on mismatch. E289003 is the undirected-pair count, not the documented stored directed count. No payload has been acquired or inspected by this agent.

The authored-note audit found no earlier GitHub dataset evaluation in 3587 scoped files; all initial GitHub hits were service/literature false positives. This is not a global unused certificate. Root must bind its steward/access decision receipt before export.

## Prospective roles and graph

RandomState MT19937 seed11709 permutes every node ID once. First60% TRAIN22620, next20% VALID7540, last20% CONFIRM7540. First half of that VALID slice is CAL3770; second half SELECT3770. Output vectors are sorted. This is a new prospective split; it is not an official provider split or a reproduction of the paper's10080/20 splits. Roles are assigned before target access and use no truth or prediction.

The full provider graph is symmetrized/coalesced using `ToSparseTensor()` then `SparseTensor.to_symmetric()`, preserving original diagonals and adding no loops. This support preparation is explicitly chosen for the prospective mean SAGE protocol and is distinct from PyG's raw load. Release128 features are only castfloat32; their derivation from the paper's4005 sparse features is not established and no features are reconstructed.

## Safe interface

`train.npz` has exactly `{x,edge_index,ids,y}` with TRAIN truth only. `select.npz` has exactly `{ids,y}` with SELECT truth only. `graph.pt` has `{x,edge_index,y,train_mask}` with literal zero outside TRAIN. Four `*_ids.npy` files are label free. `cal_targets.npz` and `scorer_only/confirm_targets.npz` are separate, mode0600. Custodial reading of all targets, including CONFIRM, is disclosed; predictions and scores remain zero.

The output provider identity and export receipt bind the full input, source and runtime receipts, feature/support hashes, roles and safe file hashes. Their consumer profile is `{dataset,nodes,input_dim,classes,train_rows,select_rows,edges,self_edges,provider_identity_sha256,graph_support_sha256}`. Root must bind the actual generated identities later.

## Published context

MUSAE paper arXiv1909.13021v3 Table4 reports GitHub GraphSAGE micro-F1 .854±.001 and best supervised APPNP .868±.001. Micro-F1 equals integer accuracy on exhaustive single-label classification. Appendix C uses sampled40 graph-convolution aggregation, depth2/hidden32, Adam.01, dropout.5, L2.001, 200epochs and original4005 sparse features. The public author tree exposes embedding code but no SAGE baseline. The current full-graph meanSAGE core is an explicit adapted prospective protocol; the published number is context, not expected replication on the128-feature release or a tuning target.

## Checks performed

Standard-library AST and compile checks only; source was not imported or run. No dataset, target, model, qualification, fit or score was accessed. Normal future host execution requires hostname `anogena-2-0` and the sole UUID `GPU-44039938-fd82-41d2-fefd-de71514e2fac` before any project metadata/config reads. No downloader, remote transport, namespace operation or model constructor is included.
