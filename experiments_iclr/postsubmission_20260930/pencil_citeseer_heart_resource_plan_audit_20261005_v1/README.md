# PENCIL Citeseer: audited qualification successor

## Recommendation

Keep the frozen 36-fit NCN/frame screen untouched. Do not release PENCIL scientific fits yet. The next PENCIL action is an exact one-GPU dependency admission followed by the complete TRAIN-backward / complete VALID-forward **zero-update** probe prepared here. It is not a predictive fit and cannot contribute a scientific result. Retain the existing PENCIL source-plan packet unchanged; this packet supersedes its proposed qualification with four optimizer updates.

The proposal is a worthwhile later competitor: a scratch graph Transformer with author-supported raw-feature early fusion on the same Citeseer-HeaRT input. Its faithful recipe has much larger worst-case tensors than NCN despite the small graph. A resource failure should remain a resource failure; changing precision, batch, sampler, workers or epoch count requires a declared successor.

## Exact runtime and dependencies

The authorized runtime interpreter is `/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/native_ncn_runtime_20261005_v1/.venv/bin/python`. Its proven Python3.11 base packages come from the repository `.venv/lib/python3.11/site-packages`, with native sparse/scatter overrides in `experiments_iclr/postsubmission_20260930/native_ncn_dependency_overlay_20261005_v1`. The installation transport contains **cp311** sparse/scatter wheels and the receipt shows this exact interpreter/PYTHONPATH. A different interpreter or package path is a new runtime qualification.

| Package/group | Pinned author / earlier18.77 | Qualified one-GPU | Audit |
| --- | --- | --- | --- |
| Torch | author2.5.1;77=2.7.1 |2.1.2+cu118 | Exact version mismatch. Do not install the broad author requirements into the active NCN core. PENCIL-specific operators on2.1.2 still need qualification. |
| NumPy | author2.2.6;77=1.26.4 |1.26.4 | Keep1.26.4. The author's NumPy2 pin changes the numerical ABI exposed to older compiled extensions and cannot be adopted as a harmless overlay. |
| PyG | author2.6.1;77=2.4.0 |2.7.0 | Exact mismatch; NCN qualification supports these operators for NCN, not native PENCIL neighbour sampling. Preserve and disclose this adaptation. |
| Sparse/scatter |77=pt27cu126/cp312 |pt21cu118/cp311 | **Binary conflict:** do not copy77 wheel files; they target a different Python/Torch/CUDA combination. |
| Transformers |4.46.2 | Not inventoried in selected one-GPU path | Needs exact4.46.2 import of BertEncoder/BertLayer and private `_prepare_4d_attention_mask_for_sdpa`. The author source forces SDPA. There is no demonstrated incompatibility solely from Torch2.1.2, and no demonstrated PASS. |
| Tokenizers/HF/safetensors |77=.20.3/.36.2/.8.0 | Unqualified | Re-resolve wheels for Python3.11; the77 installed file inventory is host-specific. No model weights are needed. |
| yaml/wandb/rich/rootutils |77=6.0.2/.18.7/14.0.0/1.0.7 | Unqualified | Required at module import even though wandb is never initialized. Rich14.0.0 also differs author's14.1.0. |
| scipy/sklearn/ogb/tqdm/networkx/pandas | Existing77 observations | Unqualified | Needed transitively by native dataset/evaluator imports. scipy1.16 and sklearn1.7 requiring Python>=3.10 would admit Python3.11; Python version alone is **not** a reason to reject those versions. Actual selected metadata/ABI imports remain unknown. |
| psutil | Harness instrumentation | Unqualified | Required by the prepared probe to record its own process and live loader-descendant RSS. |

The retained native closure imports no deepspeed, timm, rdkit, sentencepiece or training datasets package on this execution path. Do not install every optional author requirement to run this BERT baseline. `datasets` must resolve to the pinned native package, never the HuggingFace package with the same name. The probe asserts native module custody.

`metadata_inventory.py` is prepared for a **stdlib-only** selected-path inventory. It was not executed on the server. Import readiness requires a later exact native import check; installed metadata alone is insufficient. Add only missing ancillary dependencies into a fresh repo-owned PENCIL overlay with a saved resolver/install/file inventory and exact versions. Keep the already-qualified NCN core/sparse overlay unchanged and allow ordinary standard runtime caches. No namespace guards, settings, mounts, sudo or process interference are involved.

## Source-derived tensor bounds

Native sampling bounds842 nodes, plus two endpoint/task positions gives844 positions; raw features are3703 per adjacency-row token. All these bounds assume the retained256-query full microbatch and FP32.

| Individual tensor | Shape | GiB |
| --- | --- | ---: |
| Feature input |256×844×3703 |2.981 |
| Structural input |256×844×1686 |1.357 |
| Hidden state |256×844×1024 |0.824 |
| Dense message-passing adjacency |256×844×844 |0.679 |
| Expanded attention mask, if materialized |256×1×844×844 |0.679 |
| Attention logits per layer, if dense |256×8×844×844 |5.435 |
| FFN intermediate per layer |256×844×2048 |1.648 |

These are individual bounds, not a measured peak or a sum of simultaneously live tensors. FP32 SDPA on this older Torch runtime cannot be assumed to avoid dense attention, and autograd may retain additional intermediates. There are two Transformer layers, gradients, DDP buffers, temporary sampling/collation objects and a CUDA allocator. The graph's actual sparse neighbourhoods may shorten sequences substantially; that observation is not available yet.

CPU loading also matters:8 native workers with the Torch default prefetch factor2 allow16 pending batches. Their feature+structural tensor content alone has a69.40GiB joint worst-case bound, plus the current batch, collation temporaries, masks and Python objects. This is not a claim actual Citeseer reaches that bound. The source code pads input/task tensors separately, then concatenates, so additional transient copies are real. Full worker RSS must be monitored; parent RSS alone is insufficient.

`TENSOR_BOUNDS.json` retains exact bytes and the derivation. No numerical library was imported to compute these integer bounds.

## Prepared executable probe

`probe.py` is AST-checked source only; it was never imported or run. It reuses the exact unchanged licensed native closure and thin selective adapter, original source constructor/RNG order, native full loader batch256/workers8/pin_memory=True, native FP32+TF32/matmul-medium precision, and world-size-one NCCL/DDP. The config still declares300 epochs; the resource qualification executes the **complete epoch0 workload only**, not a shortened scientific fit.

Its derived native TRAIN function changes exactly the three `optimizer.step()` call sites to a gradient-finiteness/resource boundary. No Torch optimizer is constructed and its substitute has **no step method**. Native loss division by8, backward, accumulation windows, final partial window and gradient clearing are preserved. Parameter bytes are hashed before/after and must be equal. No scores, loss values, checkpoint, optimizer state or RNG state are saved for scientific reuse.

Complete VALID uses the unchanged native evaluation function with `evaluator=None` and `compute_loss=False`: traverse every unique query, verify sequential native indices, restore all113727 occurrences through the original inverse map, check finite shapes, then discard logits. No ranking metric or heldout TEST data is accessed. The prepared selective API has no TEST argument. Coverage records actual negative underfill and actual loader counts; it does not assume31 TRAIN batches.

The probe records shape maxima, input/query-order hashes, complete traversal counts, setup/TRAIN/VALID time, CUDA allocated/reserved peaks, and sampled parent-plus-descendant RSS. RSS sums may double-count shared pages and sampling at batch boundaries may miss a transient peak; a root-owned external supervisor should observe worker RSS and enforce the admitted physical-device/wall ceilings without interacting with other jobs. Partial failures are preserved. No automatic retry, batch reduction or silent pinning override is supplied.

**Limit:** zero updates deliberately omit Adam's two FP32 moment buffers (8 bytes per trainable parameter) and optimizer-step scratch/time. A PASS establishes native full backward and VALID-forward feasibility with these omissions, not full300-epoch fit memory readiness. The result explicitly reports `full_fit_memory_readiness=False`. Add measured optimizer cost only in a separately declared resource successor if root later wants to release scientific fits. A projected `300*T_train + 150*T_valid` remains incomplete without optimizer, save and setup overhead.

## Launch specification after root admission

The packet has no approved execution release. Root must stage the preserved source closure, adapter and this packet at their same phase-relative paths, qualify the exact PENCIL native imports, admit newly installed package bytes, bind caps to the observed device/queue, and create `ROOT_RESOURCE_RELEASE.json` from `RELEASE_TEMPLATE.json`. Root approval and source/dependency hashes are mandatory. The frozen NCN screen must remain unchanged; queue this probe after it or after an explicit root scheduling decision.

From the authorized repository on anogena-2-0, the exact command shape is:

```sh
RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/pencil_one_gpu_dependency_overlay_20261005_v1:/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/native_ncn_dependency_overlay_20261005_v1:/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/.venv/lib/python3.11/site-packages \
/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/native_ncn_runtime_20261005_v1/.venv/bin/python \
experiments_iclr/postsubmission_20260930/pencil_citeseer_heart_resource_plan_audit_20261005_v1/probe.py \
--release experiments_iclr/postsubmission_20260930/pencil_citeseer_heart_resource_plan_audit_20261005_v1/ROOT_RESOURCE_RELEASE.json \
--release-sha256 ROOT_SUPPLIED_EXACT_RELEASE_SHA256 \
--output experiments_iclr/postsubmission_20260930/pencil_citeseer_zero_update_resource_execution_20261005_v1
```

The proposed first PENCIL overlay path is an uncreated plan path; root may use a declared exact successor and update the dependency admission before release. The command's SHA placeholder is intentionally not a fabricated approval.

## Reusable receipts and exclusions

Reuse only authenticated data and source identities: the exact four-file AVAILABLE manifest SHA1b9c8bb5…, feature qualification v2 SHAcf1f3c2d…, pinned PENCIL native commit2d32e29… and its retained20-file closure/config, exact NCN selective loader source SHA165d51ca… and native NCN runtime v2 receipt. The latter proves NCN Torch/sparse operators, not PENCIL operators. `REUSABLE_RECEIPTS.json` retains exact paths/hashes and allowed scope.

The18.77 overlay/Collab resource receipt qualifies neither this one-GPU interpreter nor this Citeseer model. The old77 pin_memory=False, bf16=True, batch1024 and allocator overrides do not transfer. No wrong-allocation evidence, comparative outcomes, current NCN fit states, TEST inputs or heldout scores were inspected or promoted. This audit made no server connection, dependency installation, numerical/model import, model/data payload fetch, fit or update, and changed no canonical code or frozen decision.
