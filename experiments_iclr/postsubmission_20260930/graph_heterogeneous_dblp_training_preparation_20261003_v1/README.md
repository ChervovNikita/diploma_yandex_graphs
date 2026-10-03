# Runnable complete-DBLP HGT family preparation

This separate packet supplies a ZIP loader, frozen-split preparation and runnable native/global-BE/CP/unrestricted/shared-relation/untied/wider HGT training families. It preserves the sealed implementation and original unadopted nine-arm candidate. The author executed only fabricated stdlib loader/selection checks; no original archive, label payload, fitted outcome, Torch, remote operation, GPU or training driver was executed. Root's earlier actual-author CPU correspondence receipt is bound. The new family/checkpoint CPU fixture remains required before a native release.

## Representative stage and interpretation

Recommend the **complete released DBLP graph, all five seeds131/137/139/149/151, and the original nine arms plus one prospectively declared shared-relation control**:50 configurations. This packet can execute the seven HGT families (35 configurations). Native author GAT, Simple-HGN and SeHGNN supply the remaining15 competent-challenger configurations and need separate complete wrappers. They are required before superiority/manuscript claims; an HGT-family development summary cannot stand in for that benchmark or the saved both-graph gate. No study is adopted by this packet.

The shared control has `b_m + rho*q_r*u`, with shared trainable `rho0=.01` and CP-matched q/u/base signs. Its common relation residual tests whether independently learned member c adds value beyond nearly equal relation capacity. CP exceeds this control by only9 parameters across3layers. The control can also have local rank2 through existing private a/b; no whole-function-class argument follows. Shared rho matches CP's RMS c, not its initial pooled function. Different initialization functions, parameter/decay coordinates and optimization remain disclosed. CP/free-table initialization matches exactly in FP32; base core/signs and member streams are paired across matched arms.

Untied HGT now executes four independently parameterized complete cores, with member0 matching the paired core and members1–3 initialized at `seed+200003*member`. It uses one joint mean-member objective/common mean-logit checkpoint criterion. This is a declared ensemble adaptation, not four independently selected native author jobs. Wider BE now trains a full width72 core using the same implementation; its count and initialization policy are explicit. Neither is an independently new stronger backbone.

## Actual released inputs and native recipe

The acquired archive SHA is `0d3ea4a74399f9cd3e83af206e8e0b67e1844fe2c8463b424189884dd58ad7c8` (2,567,741bytes). Root's byte-bound node/link schema has26,128 nodes,6 directed raw relations and239,566 edges, with no duplicates/self edges and unit weights. Public meta.dat doubles edges and info.dat misstates target features; the actual payload governs. No edges are duplicated to satisfy metadata. The loader streams only `DBLP/node.dat`/`DBLP/link.dat`, preserves every raw relation ID in unique names, and reproduces native per-relation COO→CSR support coalescing with multiplicity accounting. No synthetic reverse/self edges are added. Whole-archive fingerprint hashing reads compressed bytes, while both label ZIP members remain unopened by this loader.

Feature-type2 retains target334 attributes and uses sparse identity features for paper14328/term7723/venue20. The actual input-dimension sum is22405. Native64 has1,654,240 parameters; global BE1,655,776; CP1,655,998; shared control1,655,989; unrestricted1,659,616; wider BE72 has1,892,968 (236,970 above CP); untied4-native has6,616,960. Counts follow source schema, not measured performance.

The development pool has1217 target nodes; every private NumPy RandomState(seed) split uses first floor(.2N)=243 validation and974 TRAIN IDs. Root provides labels through an explicit `TRAIN_VAL_ONLY` JSON input, separate from the archive. TRAIN class coverage is verified for each split. The loader never opens `label.dat.test`, obtains TEST membership or uses TEST to size the classifier. Public info/meta global class counts were already exposed and remain disclosed.

Native HGT64/8heads/3layers/norm/dropout.2, biased adapters/V/output/classifier and attention/skip order are unchanged. AdamW uses native constructor defaults and weight_decay1e-4. OneCycleLR uses300steps/max_lr1e-3/pct_start.05 and native `step(epoch+1)`. Up to300 post-update epochs are eligible; native patience30 and delta0 apply. A validation tie replaces the checkpoint and resets patience. Epoch0 is not eligible. Ensemble training is ordinary mean member CE; validation/checkpoint CE is the served mean-logit predictor, a declared ensemble extension of single-native CE.

## Root preparation and exact commands

Run from the shared postsubmission research directory with root's pinned Torch2.1.2 Python. First verify the additional family/serialized-state behavior on CPU:

```sh
CUDA_VISIBLE_DEVICES='' python -B graph_heterogeneous_dblp_training_preparation_20261003_v1/cpu_training_fixtures.py
```

It runs three synthetic CPU AdamW/OneCycle steps without launching driver main or reading a dataset. It verifies all7 family shapes, FP32 CP/free-table function matching, the common shared residual, and serialized model/optimizer/scheduler/private-RNG replay. Record remote invocation separately in root's immutable receipt.

Root must provide a byte-bound JSON development input with this schema (values come only from the explicitly admitted original label.dat; do not read TEST):

```json
{
  "scope": "TRAIN_VAL_ONLY",
  "archive_sha256": "0d3ea4a74399f9cd3e83af206e8e0b67e1844fe2c8463b424189884dd58ad7c8",
  "source_label_member_sha256": "5f5fd3841ef1ad99de706ac3399e1439b0cb81818bd3d951653c11ac0e94d589",
  "target_type": "0",
  "node_ids": ["sorted original target IDs, supplied as integers"],
  "labels": ["aligned single class integers"],
  "train_class_schema": [0, 1, 2, 3]
}
```

Create a descriptor JSON containing only absolute `path`, SHA256 and `bytes` for that input. Then prepare the five paired ID files in a fresh root-owned directory:

```sh
python -B graph_heterogeneous_dblp_training_preparation_20261003_v1/freeze_splits.py --labels-descriptor /absolute/root/development_descriptor.json --output /absolute/root/split_freeze_v1
```

Copy `FREEZE_TEMPLATE.json` into a separate root-owned freeze, fill the development/split descriptors, independently review configuration/use history, and explicitly set `study_adopted_by_root:true`. Minimum driver arms are native_HGT/global_BE/shared_relation/CP/unrestricted on **all five seeds**; recommended complete HGT family additionally includes untied_HGT/wider_BE. Native challenger completion remains a separate benchmark requirement. The template is unadopted; no CLI can silently change seeds/recipe/sites or enable TEST.

Create a separate execution release with `execution_authorized:true`, exact `study_freeze_sha256`, this packet's `prepared_manifest_sha256`, `implementation_manifest_sha256`=`2c932f9f5e223156c207bb748d34aa47d02e03458bb1d95e088e67d746deed94`, `CPU_equivalence_passed:true`, `training_packet_CPU_fixture_passed:true`, `run_name:"root01"` and `device:"cuda:0"`. Root owns resource/device admission. Then the concrete launch is:

```sh
CUDA_VISIBLE_DEVICES=GPU-44039938-fd82-41d2-fefd-de71514e2fac python -B graph_heterogeneous_dblp_training_preparation_20261003_v1/train_dblp.py --freeze /absolute/root/FROZEN_STUDY.json --admission /absolute/root/EXECUTION_RELEASE.json --run-name root01 --device cuda:0
```

## Saved states, failures and final boundary

Before data/native imports, the driver verifies its packet/dependencies/root release and writes a fresh admission marker. It refuses automatic replacement studies after an existing marker. Every selected checkpoint serializes full model, named optimizer state, scheduler, early-stopping state, global CPU/device RNG, all private member streams and exact input/split/source bindings. Selected member logits contain predictions only. Selected-state validation CE is replayed after restoration. No label payload is saved. Per-case traces/selected scores/costs and all seed/arm terminals are retained. Resource errors are deferrals; any failure or preservation error closes subset comparison eligibility.

`STUDY.json` reports per-seed validation scores and paired CP-minus-control deltas only when every frozen fit selects. Descriptive five-seed SD/SE and an illustrative t95 interval preserve pairing; their assumptions do not establish power or dataset generalization. Reuse this calculation for future independently released heldout scores only after separate final source/selection/calibration closure. Final evaluation, validation-only calibration, competent-native closure and ACM/both-graph practical thresholds remain separate work. This training driver cannot open TEST.

`stdlib_fixtures.py` checks11 concrete loader/selection/closure risks on fabricated ZIP data. `verify_source.py` checks only syntax/source custody/seal, not numerical training correctness. CP/BE, conditional tensor adaptation, same-site diagonal adapters and HGT typed sharing remain attributed prior; no predictive, speed, whole-function-class or independent-baseline qualification claim is made. Original source/CPU/acquisition packets and license limitations remain preserved.
