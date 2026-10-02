# Concrete native qualification path for root

This is a prospective source/AD recipe, not an execution request or driver. No native model, graph data, checkpoint, label array or GPU was opened by this packet's author.

## 1. Qualify the paired CPU source

Run `fixtures/torch_paired_checks.py` under root's authorized repo CPU runtime with CUDA hidden and one CPU thread, retaining the subprocess command, source/fixture hashes, stdout/stderr, exit code, elapsed time and Torch version. The independent reference must match all four returned arms at the exact shared alpha. Preserve failed outcomes instead of adjusting geometry or replacing arms. Verify the source manifest, sealed-v1 packet and active Round17 hashes afterward.

## 2. Bind the existing native common warm closure

`NATIVE_SOURCE_BINDINGS.json` fixes a concrete representative source path: Squirrel, PolyFormer-Mono, seed17/split0, all 2,223 nodes, 2,089 features, five classes, and the 512-coordinate stem/head slice. It binds the existing Round17 v3 precision driver/adapter/qualifier and six modern native source files by exact hashes, plus the prior cold and actual-warm qualification receipts. Prior qualification passed; the new support/shared-step path still needs its own actual-closure check.

The existing remote warm checkpoint is:

```text
/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/graph_init_precision_execution_root_v2/study_v2/warm/Squirrel_seed17/warm_checkpoint.pt
SHA-256 e67a0ab44c966f4980c709d0256a10a4a6916daa7dc564a07c98fc1a867a57b2
```

This descriptor was taken from the existing freeze/receipt; the checkpoint itself was not opened or rehashed by the author. The metadata extraction excludes quality fields. This exposed checkpoint is a representative compatibility qualification, not independent scientific confirmation or a prospective warm fit for a new full training study.

The `execution_root_v2/study_v2` directory is an execution namespace: its freeze context binds **Round17 v3_precision** `SOURCE_BINDINGS.json` SHA-256 `f6d8ff959204ccc793212b0ff0560213f08de7d92b74e162441888058262bc48` and `PROTOCOL.json` SHA-256 `91f20afb455c234ce113a5e3d3035955bc42b412ff7f389e5f511c7de47de446`. It is not the earlier Round17 v2 source protocol. Model parameters, logits and training gradients remain native FP32 for both admitted backbones; the FP64 change is only the per-example CE finite-difference mean/subtraction measurement.

In a separately reviewed qualification process, import the bound source prototypes without altering them or invoking the old driver main/phase orchestration. Use `graph_init_driver.source_inputs(rt, context, ledger, validation=False)` with the exact bound context and input descriptors. Its canonical loader opens the verified whole feature `.npy`, sorted undirected edge `.npy`, role node-ID arrays, and **only** the compact TRAIN `nodes/labels` pack. Role IDs support order/disjointness checks and carry no label values. The `validation=False` branch does not open the validation label pack; final labels are not inputs. The native graph preprocessing must match the bound certificate. Graph input, feature/edge, role freeze, modern certificate and TRAIN-label descriptors are recorded exactly in the native binding JSON.

Load the hash-verified frozen checkpoint with the existing restore primitive, preserving its declared specification and preprocessing:

```python
checkpoint = torch.load(bound_warm_path, map_location=device, weights_only=True)
native, restored_optimizer = integration.restore_native(adapter, checkpoint, device)
del restored_optimizer  # Qualification implements no optimizer or continuation.
k1 = integration.clone_boundary(native, boundary_api, 1)
k4 = integration.clone_boundary(native, boundary_api, 4)
identity_receipt = integration.identity_logits_audit(native, k1, k4, graph)
common_boundary = k1.eval()
```

Do not change the running driver or consume its live model. The restored common warm state and exact raw wrapper must be deterministic eval predictors, with every R/S at identity and all warm weights/biases copied. Existing primitive interfaces in the bound base file are:

```python
theta0, logits_fn, slice_receipt = base.bind_common_model(
    common_boundary, backbone, model_args,
    select_full_node_class_logits, model_kwargs,
)
```

`common_boundary` is the raw wrapper, whose names match `stem.S/head.R` for PolyFormer-Mono (512 coordinates) or `stem.S/global_head.R` for the admitted final Polynormer-r global phase (1024 coordinates). Use only the actual final predictive logits, with a proved homogeneous full-node target universe and recorded output/graph row permutation. Save forward equality between the common warm predictor, the bound closure and all four pre-install route clones. The current full-node semantic assertion excludes a silently broadened typed/partial target universe.

Acquire the already declared canonical normalized topology through the existing reviewed source path. Construct the null with the existing `base.permute_topology_nodes`, retaining seed, permutation and a sparse equality check for `S_permuted=Pi S Pi^T`. The feature/label row order stays fixed. TRAIN rows and compact TRAIN labels are the only label inputs. Do not substitute a sampled/minibatch or surrogate closure for the declared common full-output derivative.

## 3. Qualify actual VJP/JVP, then the shared path

Before initialization, use the bound Round17 v3 **precision** qualifier on that exact native closure, keeping the existing per-example FP32 CE / FP64 mean finite-difference measurement and original FP32 diagnostic side-by-side:

```python
ad_receipt = precision.qualify_gradient_interface(
    logits_fn, theta0, train_rows, train_labels, 90017,
)
```

It checks three deterministic JVP/VJP dualities and centered-logit/CE central differences at the existing two fixed epsilons; charge its one VJP primal, four pullbacks, three JVP primals and twelve finite-difference forwards (sixteen total closure forwards). Retain unsupported AD and both epsilon outcomes. Passing a synthetic CPU fixture does not substitute for this native qualification. An unsupported full-model derivative is a source/AD limitation; do not replace it with coordinate repulsion or suppress cross-node backward paths.

Only after source/AD qualification, invoke the actual new helper on the same frozen state:

```python
S = base.symmetric_normalized_adjacency(
    len(graph.teacher_input), canonical_edges, theta0.dtype, theta0.device)
S_permuted, permutation = base.permute_topology_nodes(S, 80017)
target_nodes = torch.arange(len(graph.teacher_input), dtype=torch.int64, device=theta0.device)
train_rows, train_labels = train.nodes, train.labels
arm_slices, paired_receipt = paired.initialize_paired_four_arms(
    logits_fn, theta0, S, S_permuted, target_nodes,
    train_rows, train_labels, homogeneous_full_node_outputs=True)
```

`train.nodes` is the exact verified TRAIN output-row order in the canonical full-node closure. Preserve that order throughout AD and all four arms. Save all four geometry receipts, six-grid attempt records, accepted shared alpha or joint failure, every exception and all charged calls. Verify shared `g`, row universe, band sums, cap/projection identities, parameter-radius bound, full-output Grams and exact same-alpha signed checks. Common-only bypasses source/finite TRAIN separation, because its routes are identical; it still must pass member/pooled finite TRAIN CE. The three graph arms retain both separation checks. A signed/Gram result cannot add an acceptance gate. Native timing/peak memory must include all rejected work and actual closure/AD overhead; a future GPU preflight requires its separate authorization and declared resources.

## 4. Freeze a separate continuation integration

Jointly accepted factor tensors are ready for root review; no model is installed by this helper. A later integration must verify the original common-route warm equality and use the existing factor installation primitive only on the admitted slices. Warm shared weights, all other factors/biases, optimizer-state convention, random/dropout streams, continuation budget, pooling and selector need one predeclared identical treatment across arms. Preserve joint failure and source/resource deferral in the record. This packet implements none of those training choices.

The later quality comparison must assess full support against remasking, common-only and the topology-null arm at the exact shared step, under paid common continuation. Actual utility and scientific merit are assessed independently of today's available compute. No source/AD pass alone establishes generalization or novelty.
