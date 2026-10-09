# Inactive full-task NSD competence runner

This fresh packet provides complete **native-single and feature-MLP TRAIN/VALIDATION loops** and a local official Tolokers role exporter. Nothing was launched: numerical packages, graph data, labels and checkpoints were not imported/read/executed during source preparation. The immutable original NSD packet and source seals are unchanged.

## Exposure correction

Ordinary Tolokers is one of the original five paper benchmarks. These new NSD/MLP runs are a representative **exploratory comparison**. Split 0 and remaining splits are not asserted unused; this packet cannot supply fresh dataset/unused confirmation. `EXPOSURE_CORRECTION.json` separately corrects the older packet's wording. Original paper scores and the canonical ledger are unchanged. GraphLand `tolokers-2` is a distinct, unexamined future option; no compatibility or confirmation claim is made.

## Entry points and inactive default

`baseline_runner.py` schedules four fixed native configurations × three paired seeds, then three ordinary Torch MLP fits. `role_exporter.py` accepts only a local official raw Tolokers archive and never downloads anything. Both public CLIs default to an inactive plan without numerical imports or graph/label reads.

Future root activation requires `--execute --release ROOT_RELEASE.json`, the exact sealed source fingerprint, exact input hash, a fresh bound output directory and completed exposure audit. A screen release also binds the role metadata hash, qualified runtime versions, final device, explicit deterministic policy and native numerical/placement qualification. `RELEASE_TEMPLATE_DISABLED.json` is disabled and is not an authorization.

Once root has audited and authorized acquisition/roles/qualification, the future commands are:

```sh
python role_exporter.py --execute --release ROOT_EXPORT_RELEASE.json --raw-npz /authorized/tolokers.npz --output /fresh/export
python baseline_runner.py --execute --release ROOT_SCREEN_RELEASE.json --roles /fresh/export/roles.npz --output /fresh/screen --device cuda:0
```

Transfer this packet **beside** `private_sheaf_native_source_design_20261009_v1`; relative source-seal references require both packets. No interpreter installation, dependency changes, server contacts or source patches are supplied here.

## Numeric role contract and canonical support

`roles.npz` accepts exactly `x`, `edge_index`, `train_index`, `train_y`, `valid_index`, `valid_y`. Arrays are numeric with pickle disabled; features are finite official float32 `[11758,10]`, indices/support are int64, and truth exists only for TRAIN and VALIDATION. Unknown keys, including full `y` or TEST truth, are rejected before any archive array is loaded. TRAIN/VALIDATION must be disjoint, cover all supplied official role rows and contain both binary classes.

The exporter verifies official raw keys and split masks, reads the raw label vector inside its separately authorized process, and selects/emits only TRAIN/VALIDATION labels. It does not read TEST masks, select TEST labels, emit the raw label vector or write TEST truth. Full-graph features may include TEST nodes, as required for the complete transductive graph.

Static inspection of MIT-licensed PyG 2.6.1 source confirms `HeterophilousGraphDataset.process()` applies `to_undirected` to raw edges, and that utility expands reversed entries then row-sorts/coalesces. The exporter reproduces the unweighted support convention using sorted unique integer pair keys. It rejects native-incompatible loops rather than silently removing them. `ROLE.json` records actual raw entries, raw unique directed entries, expanded entries, canonical directed entries, canonical undirected pairs, array hashes and archive/source provenance.

**No directed edge count is guessed or enforced.** No graph has been read here, so no actual graph count is yet established. The runner recomputes reverse-paired unique support checks and verifies the actual counts/hashes from the supplied numeric archive. Canonicalization occurs once at role export to match the inspected official loader; the immutable adapter never reorders or changes caller edges.

## Complete baseline experiment

`RUNNER_PROTOCOL.json` preserves the sealed native protocol's four presets: `d2/f32/L2`, `d2/f32/L4`, `d4/f16/L2`, `d4/f16/L4`, all at node width 64. It preserves seeds `1103,2207,3301`, original `.01` Adam and `.0005` regular/sheaf decay, `.4/.2` dropouts, ELU/tanh, optional second input Linear, 500 epochs and patience 200. These are our prospective full-task choices, not winning author configurations. No subgraph or subgroup is selected.

Every epoch uses all TRAIN labels in the original unscaled NLL. A full graph evaluation reports TRAIN/VALIDATION metrics only. Checkpoint selection uses full VALIDATION AUROC, then lower validation NLL and earlier epoch for exact ties. There is no TEST evaluator or TEST truth input.

The feature anchor is an ordinary **same-Torch `10->64->64->2` MLP**: ELU after both hidden Linear maps, input dropout `.4`, hidden dropout `.2`, 4,994 active parameters. It receives the same features, TRAIN labels, VALIDATION rows, optimization budget and selector. It has no topology input; the native model additionally receives full graph support. This is a feature-signal competence reference, not the later same-information joint transport control. Its device edge tensor is released before MLP fits; host archive arrays remain resident.

`native_placement.py` runs the original constructor and reverse-pair/index preprocessing on CPU, preserving the exact caller order. It then moves registered parameters, the plain model `edge_index/time_range`, and every native builder index/degree tensor; updates the model/builder device fields; and verifies that no plain tensor is left on another device before any forward. Native learned weights initialize on CPU in the original direct-device construction as well. The helper avoids per-incidence CUDA scalar reads without replacing source arithmetic or propagation. Model/builder graph identity remains shared. Exact static-index and numerical parity against direct final-device construction is a required root qualification; this placement is not claimed numerically qualified here.

Selected state is saved with model/optimizer/RNG state and selected TRAIN/VALIDATION logits. After fitting, the trained model is discarded and a **fresh native/MLP constructor** reloads the owned selected checkpoint strictly. Reporting uses this reconstructed state. Exact parameter/buffer equality, maximum role-logp difference and AUROC difference are checked/reported. Only self-produced checkpoints are accepted, with `weights_only=True`; there is no checkpoint input CLI.

All 15 scheduled fits retain records, failures, partial histories and any selected checkpoint. Each record exposes family/config/seed/source/role/selector identity, actual parameter count and cost. The summary selects among native configurations only after the complete panel succeeds; no failed seed/config is silently replaced. Provisional numerical competence gates still require root curve/stability review before architecture freeze. A weak baseline requires a prospective amendment with at most four sensible full-task configurations, keeping TEST truth unopened to the runner.

## RNG and later family ownership

Each baseline fit owns Python, NumPy, Torch CPU and visible CUDA RNG state, seeded once for native/MLP initialization and continued through training. Validation uses one `base+2000003+epoch` stream inside a snapshot/restore context, preserving training RNG. Native CUDA dropout uses device RNG while the original SVD jitter is drawn from a CPU FloatTensor; both states are retained.

The checkpoint helpers also encode the ownership needed by later controls. Independent encoders require separate complete constructors, optimizers, unscaled own NLL and member streams (`base+1000003*member`). Shared/joint banks own a frozen ordered traversal stream with all private factors in state. Fixed assignment buffers must reload; resampled assignment consumes the bank-owned train/eval stream once per member/layer/forward. A joint optimizer must include its readout. Resuming training requires restoring optimizer/RNG after fresh construction/state load. Those later family loops are not implemented or launched here; their exact immutable adapters remain in the previous packet.

## Compatibility and cost

`COMPATIBILITY.json` records the static import chain. The general-map path eagerly imports **torch_householder** through `models.orthogonal` despite not using orthogonal maps. A real compatible import is mandatory. No stub, eager-import removal, scalar normalization substitute or native math rewrite is included. Torch sparse/scatter extensions must match the root-qualified Torch/CUDA ABI; original block normalization, SVD, jitter, clamping and sparse propagation remain untouched.

Saved modern runtime receipts show possible Torch 2.4.0/CUDA11.8/NumPy2.2.4 and separate Torch2.1.2/PyG2.7 source requirements, but neither establishes NSD compatibility or householder availability. The runner uses modern Torch checkpoint/storage APIs; qualify the chosen existing modern runtime. Nothing installs or downgrades a host/runtime. Strict deterministic policy may be rejected by native extensions; failures are retained for root qualification.

Cost records include construction and topology storage, actual persistent native map-diagnostic payload, train/validation/reconstruction forward counts, checkpoint time, complete attempt time and CUDA peaks. CPU RSS is a cumulative process high-water mark including common role arrays; it is not an isolated per-family peak. Numerical stability, finite gradients, source parity, data/role provenance, qualified dependencies and checkpoint/RNG behavior remain execution gates. Static source/AST checks establish no competence or performance claim.
