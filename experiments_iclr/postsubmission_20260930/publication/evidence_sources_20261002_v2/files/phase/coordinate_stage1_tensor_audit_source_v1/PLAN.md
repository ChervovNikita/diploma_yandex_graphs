# Narrow selected-state tensor audit and replay plan

2 October 2026. Source preparation only. No comparative report/curve/logit/checkpoint outcome or remote state has been inspected. Source remains unexecuted and uncompiled. All earlier assessment files are preserved. This plan makes no empirical-effect or paper verdict.

## Exact scope and prerequisites

One separately admitted invocation processes all54 frozen Stage1 core0 cases, in protocol order: both graphs, all nine arms and seeds17/29/43. Each case performs one selected-state eval/no-grad full-graph forward. There is no fitting, optimizer application, checkpoint selection, alternate pooling selection, hyperparameter search or additional timing benchmark.

The adopted stdlib v2 assessment checks all54 completion/curve/horizon/selection/hash/provenance and supervision evidence. Its binary-semantic limit is explicit. The replay is a complementary check of selected tensor contents and computational correspondence, not a second implementation of the entire training/assessment framework. An inconclusive training cell remains inconclusive even when its saved selected tensors reproduce.

Root must prospectively freeze the numerical integrity tolerances, source/request/evidence-index hashes, allocation and exact bounded route before this replay. The new request template is unadmitted. It binds the adopted v2 specification/adoption, pinned protocol/source manifest, native source, existing whole/authorization/logging supervisors, wrapper/route, and new replay allocation. It enters the same corrected single-A100 route and repository interpreter through bounded_run→run_authorized→this source, using `--request` and `--supervisor`. No direct launch or new GPU is implied by source preparation.

## Native input isolation

The complete tensor read list is:

- For each of54 canonical cells: `selected_state.pt`, `selected_deployment_state.pt`, `selected_validation_member_logits.pt`.
- For each of the two datasets: public `graph.pt` and public `core0/validation.pt`.

Public graphs contain the already adopted transductive features/edges. Only core0 validation targets are used for recomputation. Train labels, test indices, test labels, sealed directories, author raw snapshots, processed PyG objects, alternate datasets and acquisition providers are not opened. Data-manifest JSON can declare sealed references; those paths never enter the tensor read list. No dataset loader is constructed. No tensors are written or recopied.

Every tensor input must have matching immutable evidence-index and original manifest SHA256/length. The loader reads its bytes once, verifies that exact buffer and then calls the original `torch.load(BytesIO, map_location='cpu', weights_only=True)` on it. This avoids a path mutation between hash check and deserialization. No unrestricted pickle fallback or safe-global additions are allowed. Direct torch.load/save is replaced with a rejection function; the reviewed model source has no serialization at import/construction. Exact reviewed Python model/dependency bytes are verified before import. This is a trust-restricted input list, not a security sandbox for arbitrary files in the old framework runtime.

## Per-case tensor and computation checks

1. Bind report and the three selected artifacts to the trusted index/original cell manifest. Reconstruct report provenance from frozen protocol/cell/recipe/source manifest and acquisition metadata. Require complete fit status, competent horizon bounds, test flags false, and selected epoch inside the completed horizon.
2. Require exact checkpoint keys; selected state and saved validation logits share the exact epoch/internal provenance. Model-only deployment shares that epoch, protocol/source hashes and exact factory kwargs derived from the frozen recipe/data dimensions. Optimizer snapshot count is checked, but optimizer states are neither applied nor moved to GPU.
3. Build the exact CPU architecture from the frozen model entry/dependency. Require exact state key sets, shapes and dtypes; finite floating state tensors; every tensor exactly equal in selected/deployment states. Fixed registered buffers must equal fresh construction from the frozen permutation seed. Apply selected state with `strict=True`; independently measure model/index storage against the frozen gate. No reshaping, class/member reordering, casting rescue or alternate source is allowed.
4. Saved member logits must be FP32/finite with `[actual_members, validation_nodes, classes]`; actual members are1 for S/T and4 otherwise. Int64 saved node IDs must equal the public validation bundle in order, and its canonical mask digest must match acquisition metadata.
5. Recompute saved-logit primary arithmetic-softmax-pool and same-checkpoint mean-logit sensitivity NLL/accuracy/macro-F1, all member metrics and argmax disagreement on CPU64. The implementation uses explicit log-probability indexing/classwise F1 counts and does not call the runner's metric helpers. Compare with the reported selected metrics under the frozen integrity tolerances. Retain recomputation/differences; keep adopted reported FP32 gate quantities unchanged.
6. Release optimizer/deployment bundles, transfer only selected model, public graph/features and validation IDs to CUDA0, use the same frozen framework/TF32/determinism/thread settings, disable dropout with eval and run exactly one no-grad full-graph forward. Gather validation rows and compare with saved member logits, with no member/class permutation. Require exact pooled/member argmax equality and recomputed replay metrics within the same report tolerances.
7. Record hashes/provenance, differences, max absolute logit error, forward/gather/host-copy duration and peak memory. That replay duration is not a replacement warm benchmark or a new speedup claim. Release each model/GPU graph/tensor before the next case. The shared file lock is acquired before importing/initializing Torch; only public CPU graph/validation bundles are reused.

## Prospective integrity tolerances and failures

Proposals: elementwise logits `abs(error)≤1e-6+1e-5×abs(saved)`; NLL `abs(error)≤2e-6+1e-5×abs(reported)`; accuracy, macro-F1 and disagreement absolute error≤1e-7. Exact pooled/member argmax equality is also required. These tolerate FP32/GPU versus CPU64 reduction differences under the original nondeterministic runtime. They are not effect-size thresholds, and no post-outcome widening is allowed.

Each expected case receives a retained row. Key/shape/dtype/order/provenance/finite/storage/state/logit/metric/memory failures are `INCONCLUSIVE_TENSOR_REPLAY`. A whole-process timeout retains partial receipts and makes overall integrity inconclusive. It must not become a favorable subset, a failed scientific mechanism gate, an automatic retry or a reason to promote a sensitivity metric. Original training competence and assessment gates retain their own statuses.

## Resource proposal

`RESOURCE_PLAN_v1.json` uses the previously retained eighteen qualification rows; no current comparative costs are used. Three repeats of all eighteen arm/graph warm medians give a1.5735484734-second forward proxy; the analogous p95 sum is1.5861726478seconds. Fresh setup proxies total4.9812120348seconds. These are planning proxies including all-member forward/gather/pooling, not guarantees or pure GPU busy time. Qualification gathering used train nodes; replay gathers validation nodes.

The estimated native input service is3,479,638,128bytes at assumed50MiB/s, or66.3688302612seconds. This includes all54 optimizer-containing selected checkpoint references, model-only deployment/logit references with1MiB per-case allowances, two graph bundles, two validation bundles and metadata allowance. Selected tensors are deserialized once per case, on CPU. Resource planning uses the fresh proxy plus this I/O service,27seconds CPU metric allowance,45seconds fixed overhead and15seconds initial idle allowance, then20% contingency plus one60-second reserve:250.0200507553seconds. No warm/fresh component is counted twice.

Propose one900-second whole cap (normal895 after existing5-second grace),1000seconds new phase reservation,900seconds diagnostic reservation and16MiB new disk reservation. No previous reservation is reclaimed based on faster actual execution. The current old phase bound leaves insufficient room for this new reservation, so root must amend its allocation/cap before launch. Disk writes are only JSON/case/terminal/log receipts; existing fit artifacts are reused.

Against the retained Stage1 allocation alone, these additions yield conservative phase/diagnostic/disk bounds132001.60191523682seconds/99595.729656751seconds/133737100960bytes. Example updated phase/diagnostic caps132140/99600 preserve the existing small reservation slack; the existing214748364800-byte disk cap remains sufficient. Root must reconcile any intervening allocations when freezing the actual amendment.

Largest qualification inference allocated/reserved peaks are2,198,408,192/2,554,331,136bytes. The largest reference checkpoint is168,200,898bytes and largest model56,029,244bytes. Propose8GiB GPU and8GiB CPU RSS working limits; neither is a measured replay peak. Source sets the GPU allocator fraction to0.1 of the pinned80GiB device and rejects measured peaks above8GiB; CPU max-RSS is checked under the Linux remote runtime. Retained input buffers and CPU graph copies are included in planning, without GPU optimizer state or target tensors.

## Deliverables and limits

The one new separate receipt root is `coordinate_ensemble_execution_root_v1/tensor_replay_selected_v1_run01`. Source writes START,54 case JSON rows and TERMINAL; no forward logits/checkpoints are written. Root still audits outer/inner completion and source/manifest hashes. Source uses the v2 stdlib guards by exact digest; it does not duplicate the full assessment code.

A clean replay strengthens selected checkpoint/logit/metric correspondence for these54 cases. It does not prove optimizer trajectories, original timing attribution, historical kernel configuration, mathematical convergence, graph-population significance or novelty. Tests remain sealed, and no continuation or public claim is admitted by this plan.
