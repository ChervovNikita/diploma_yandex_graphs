# Representative feature-enabled PENCIL baseline plan

## Decision and scope

There is a concrete route to one **fixed, author-supported feature-enabled ogbl-collab PENCIL recipe** on authorized host 18.77, physical GPU 0. Use commit `2d32e29dbed533288d9d758138e07547a0a7d8a9`, the official Collab YAML and `--use_features --feature_fusion early`. Keep hidden 512, 8 layers, 8 heads, intermediate 2048, 20 epochs, batch 1024 per rank, total accumulation 8, bf16 and 12 data workers. The author default seed is 0. One completed seed 0 reference would not certify published multi-seed performance or estimate seed uncertainty. Native PENCIL uses mean BCE on its mixed query batches. Its 20 epochs of approximately 50% positive exposure are not a compute or supervision-budget match to NCNC's 100 floor-batched epochs. These family-specific training recipes must remain named separately.

This packet is a source-reading and implementation plan. It performs **no server access, staging, environment changes, model imports/execution, data download, fitting or score computation**. It contains no predictive result, tuning grid or PENCIL ETA. `PROPOSED_COMMAND.json` and `RESOURCE_OBSERVATION_PLAN.json` remain disabled proposals.

## Exact launcher and tensor bindings

The newly read author launcher has conflicting comments about four GPUs, but its actual `main()` default is **eight**. It scans all GPUs through `nvidia-smi`, selects by free memory and overwrites `CUDA_VISIBLE_DEVICES`. Even `--num_gpus 1` still selects a GPU automatically. It therefore cannot enforce the user's GPU 0 restriction.

Its underlying training entry point is `torchrun --standalone --nproc_per_node=<count> --nnodes=1 run_lp.py <config>`. The future GPU 0 process can invoke that entry point directly, using the admitted Python interpreter:

```text
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 WANDB_MODE=disabled
<admitted_python> -m torch.distributed.run --standalone --nproc_per_node=1 --nnodes=1
run_lp.py <owned_feature_collab_config.yaml> --use_features --feature_fusion early --seed 0
```

This is a command specification, not a command executed here. `run_lp.py` always initializes an NCCL `env://` process group, so a plain Python invocation without distributed environment variables is insufficient. One node and one process make global rank, local rank and visible device all 0, avoiding the script's single-node rank/device assumption. World size 1 gives accumulation 8 and **8192 query examples per full optimizer update**. A partial last update retains the native loss divisor 8. World sizes above 8 fail; nondivisors of 8 silently reduce the effective accumulation after a warning. No other GPUs are required or selected by this proposal.

### Correction to the retained PENCIL sizing

The actual Collab runner computes `2*(1+75)=152` and passes that value to `STokenizer(num_nodes=max_num_nodes)`. Its adjacency-row width is therefore **306**, and the sampled sequence has at most **154** positions including the two endpoint task tokens. The raw attribute channel is 128 dimensions under `--use_features`.

The previously retained 256-node/514-column values describe the standalone tokenizer default, not this runner-bound Collab recipe. The native wrapper receives model `max_position_embeddings=2048`; the top-level `max_sequence_length=154` assignment is not the wrapper argument. Sampling and the152-node tokenizer assertion bound the actual sequence; the collator pads to the batch maximum. These values must be observed during the full native epoch, rather than replaced by an arbitrary padded proxy.

`CORRECTIONS_AND_DECISION.json` supersedes that PENCIL sizing detail in the earlier source recipe and competitiveness packet. It changes no NCNC recipe or completed-result conclusion.

## Data and dependencies

### Existing data can supply the native Collab path

The author code uses `PygLinkPropPredDataset(name="ogbl-collab", root=DATA_DIR)`, where `DATA_DIR` is `<repo-root>/data` and `.project-root` locates the repository. It calls the combined `get_edge_split()` accessor and then filters TRAIN records by year≥2007. The broad README `data.zip` archive is unnecessary for this official Collab path. HeaRT files are also unnecessary when the dataset name remains `ogbl-collab` and `--heart` is absent.

Saved custody metadata identifies existing official data at:

```text
/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git/experiments_iclr/postsubmission_20260930/buddy_complete_data_cache_preparation_v3/root_run_77_v1/dataset/ogbl_collab
```

Qualified members include `RELEASE_v1.txt`, all six raw CSV gzip files (edges, weights, years, attributes, node count and edge count), and the official TRAIN/VALID splits. The later TEST metadata and actual all25 NCNC root adoption establish prior TEST use. The source plan does not reopen those payloads or verify their current server presence.

Future preparation should copy the bound official files into an **owned** `data/ogbl_collab` directory and let OGB create its processed tensor there. Native processing must not write into the shared BUDDY data/cache root. The BUDDY sketch cache is not PENCIL input. At Torch 2.7, the existing official-data qualification used process-scoped `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`; preserve that explicit legacy-loader setting for the admitted official files.

All 235868 declared nodes and 128 raw feature dimensions remain available. The original TRAIN has 1179052 records; the **native year-filtered record count is unknown in this audit**. It must be measured from admitted data, not guessed. Native TRAIN/VALID use the filtered graph; TEST adds VALID positives to that graph. Dataset edge weights are retained/coalesced, but the selected adjacency-row token path uses unweighted edges by default. Both query-edge directions are removed **after** neighborhood sampling and induced-subgraph extraction. Removing the query earlier would change the author recipe.

The native loader opens TEST splits at startup even with `eval_test=False`. This plan does not claim an unopened TEST barrier. The current official TEST has already been consumed by NCNC.

### Known runtime and unobserved packages

The saved host77 runtime receipt, dated 2026-10-03, confirms Python 3.12.11, Torch 2.7.1 / CUDA 12.6, PyG 2.4.0, torch-sparse 0.6.18+pt27cu126 and torch-scatter 2.1.2+pt27cu126 on an A100 80 GB PCIe. Data/cache metadata additionally confirms OGB 1.3.6, NumPy 1.26.4, SciPy 1.16.0 and pandas 2.2.3. These are recorded existing versions, not a live server recheck.

The author requirements specify Torch 2.5.1, PyG 2.6.1, NumPy 2.2.6 and Transformers 4.46.2. Use of the existing runtime would be a **declared contemporary-runtime adaptation**, not the exact author environment. PENCIL-specific presence of Transformers, rootutils, rich, wandb, PyYAML, tqdm and networkx remains unobserved. Their absence is not asserted. `DEPENDENCY_READINESS.csv` maps each required package to source and saved availability.

A future isolated dependency check should inspect those packages and native imports, then add only missing required packages in an owned environment or overlay. The exact BERT path does not require installing the entire “other models” and development list over the shared environment. Networkx and SciPy are nevertheless required by unconditional dataset import paths. Native sampling needs the compiled CPU `ego_k_hop_sample_adj` and `saint_subgraph` operations; existing torch-sparse import metadata alone does not qualify those operations. BERT forces PyTorch SDPA and uses a private Transformers 4.46.2 mask helper, so its exact import and real-batch attention path need verification. FlashAttention installation is not required by this source path.

### BERT configuration metadata is now pinned

The scratch model calls `AutoConfig.from_pretrained("bert-base-uncased")` for configuration metadata, then overlays YAML dimensions. `from_pretrained=False` avoids language-model weight loading. This packet retains the 570-byte config at Hugging Face revision `86b5e0934494bd15c9632b12f734a8a67f723594`, SHA256 `7160e1553ad2ca51d8c1cb066be533db31826e12d173824c1bb0cb1a4f187d20`.

The native inherited defaults include attention/hidden dropout 0.1, GELU and layer-norm epsilon 1e−12. The YAML overlays hidden 512, layers 8, heads 8, intermediate 2048 and max positions 2048. A future owned HF cache or narrow metadata-only `AutoConfig` binding should use this pinned local asset with offline loading. No pretrained weights were downloaded. This pins a present configuration asset; it does not certify the historical author cache revision.

## One representative resource observation

Before allocating a complete 20-epoch fit, observe **one complete native author epoch on the actual Collab graph**, plus the complete official VALID query set. A native PENCIL epoch exposes approximately 50% of filtered positives and one global negative per selected positive; “complete native epoch” does not mean all original TRAIN positive records. Preserve cyclic positive slicing, epoch RNG reset, all loader rows/tails, batch 1024, 12 workers and accumulation 8. Record actual positive/negative counts, batch count and update count. Do not substitute `--debug`, capped `max_num_samples`, a reduced architecture, small graph or dummy query bank.

The observer should call the native construction/loaders/train loop from a narrow owned harness, not unrestricted `main()`. Preserve its real `get_feature_dim` query sample and resulting RNG advance before model construction. Measure cold setup/data processing separately from TRAIN, full VALID serving and checkpoint serialization. Initialize CUDA before resetting allocator peak statistics. Serve all 60084 VALID positives and 100000 official negatives with the native scorer and sequential-index check, using `evaluator=None` and `compute_loss=False`; publish coverage and resource receipts, not metrics or prediction arrays. Do not invoke TEST scoring during this observation.

Record actual local-node/sequence/input widths, finite parameters and gradients, full query coverage, peak CUDA allocated/reserved memory, sampled owned-process-tree RSS including 12 workers, CPU/data/pinned-memory behavior, checkpoint bytes/hash and output bytes. Include bootstrap, hashing, processing, loading, training, serving, save/finalization and all failed attempts. The native save helper catches write errors and continues; verify checkpoint existence/hash rather than trusting a printed save message or exit 0. Native checkpoints lack RNG-state custody, so automatic restarts are not exact stream continuation. Discard engineering model/optimizer state before any scientific fit.

The JSON proposes **policy ceilings**, not predicted needs: 7200 seconds, 64 GiB owned-process-tree RSS, 70 GiB CUDA allocated, 75 GiB reserved, 16 GiB owned observation output and 128MiB public receipts. Root must bind or adjust them against the actual authorized GPU 0 and host allocation before release. Failure before full epoch/VALID completion remains an incomplete resource observation; it is not a reason to shrink the baseline automatically. Existing NCNC timings are retained reference work scales and cannot supply a PENCIL ETA.

After actual resource evidence is adopted, the concrete scientific reference is one fresh seed 0 run of the fixed 20-epoch recipe. Use a fresh absolute `save_path`; keep receipts/logs outside its generated checkpoint directory to avoid the native automatic-resume heuristic. Preserve strict VALID Hits@50 selection and native graph/query/negative rules. The proposed operational YAML change `eval_test=False` suppresses per-epoch TEST diagnostics; final TEST occurs once after selected-state lock. That visibility adaptation is explicit and gives no fresh-confirmation claim on the already consumed official split. No automatic fit continuation is released by this packet.

## Can GNNM attach meaningfully?

**Yes as an attributed parameter-sharing/optimization experiment; no novel primitive or favorable gain follows from compatibility.** PENCIL already exposes observed pair neighborhoods and their actual attributes through trainable attention and propagation. It therefore already crosses the fixed-cache information boundary that motivated the BUDDY discussion.

A meaningful member ensemble would preserve separate hidden trajectories through feature projection, attention Q/K/V/output maps, FFN maps, propagation maps, normalization/bias states and endpoint readout. A leading member dimension and shared matrices with private feature factors can implement that. Members would still perform their attention/softmax, dense FFN and propagation work. The frozen orthogonal input projection and reconstructed observed adjacency do not become new learned evidence by wrapping them.

Four affine scalar heads on one shared pair representation average to one affine head at serving. Separate member losses may affect optimization, but those heads alone supply no additional pair-context function class. Likewise, independent PENCIL models can reuse the same deterministic query draws; preprocessing reuse is not exclusive to factorization. Freezing the author's stochastic subgraph draws would itself be an adaptation that must be named.

BatchEnsemble already establishes rank-one input/output scaling and complete private input replication (index records 8, 51, 62). The retained modern transformer adapter at record 126 is a nearby prior for private low-rank attention/head paths. PENCIL itself supplies the pair-context/observed-adjacency learner; LPFormer supplies pair-conditioned attention and CN/PPR/feature factor diagnostics; Link-MoE supplies structural expert routing. None proves this proposed GNNM composition superior.

Count/pattern-conditioned initialization or gradient coupling could be formulated on PENCIL queries, but this source contains no NCNC learned missing-edge completion bank. The current NCNC private-completion hypothesis or native-gradient evidence cannot be transferred automatically. A later full-map GNNM comparison would need capable PENCIL single/independent controls and measured whole-model quality/cost under a prospectively fixed objective and sampling contract. This packet adopts no GNNM arm and calls no generic BE wrapper novel.

## Read accounting and retained evidence

Literature memory was reused first. In index_v45, zero-based records 80 and 135 are the same PENCIL identity `arXiv:2602.01553v4` ; 135 is the retained changed-question scope extension of 80. This task adds **zero paper identities, zero index records, zero primary-paper reads and zero full-paper reads**, and changes no index. It newly acquires eight pinned author-code files and two non-weight HF metadata assets, all inside the project. Every author blob matches the previously saved Git tree. Five new tiny/launcher files receive full source-read credit; the three import-only bodies remain scoped reads. Existing source extensions and task-specific revisits are separately recorded. Truncated locators receive no full-read credit.

Key files:

- [Frozen recipe proposal](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/pencil_collab_representative_baseline_source_plan_20261004_v1/FROZEN_RECIPE_PROPOSAL.json>), [exact proposed command](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/pencil_collab_representative_baseline_source_plan_20261004_v1/PROPOSED_COMMAND.json>) and [resource observation plan](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/pencil_collab_representative_baseline_source_plan_20261004_v1/RESOURCE_OBSERVATION_PLAN.json>).
- [Dependency/data readiness](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/pencil_collab_representative_baseline_source_plan_20261004_v1/DEPENDENCY_AND_DATA_READINESS.json>) and [shape correction/decision](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/pencil_collab_representative_baseline_source_plan_20261004_v1/CORRECTIONS_AND_DECISION.json>).
- [GNNM attachment assessment](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/pencil_collab_representative_baseline_source_plan_20261004_v1/GNNM_ATTACHMENT_ASSESSMENT.json>), [retained literature records](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/pencil_collab_representative_baseline_source_plan_20261004_v1/REUSED_LITERATURE.json>), [exact source sites](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/pencil_collab_representative_baseline_source_plan_20261004_v1/SOURCE_SITES.csv>) and [read scopes](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/pencil_collab_representative_baseline_source_plan_20261004_v1/READ_SCOPES.json>).
- [New pinned launcher](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/pencil_collab_representative_baseline_source_plan_20261004_v1/new_source/run_with_best_gpus.py:146>), [native defaults/config lookup](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/graph_link_competing_family_source_20261003_v1/source/pencil__models__transformers__lp_model.py:302>) and [actual tokenizer binding](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/graph_link_competing_family_source_20261003_v1/source/pencil__run_lp.py:1379>).

Author source: [pinned PENCIL repository](https://github.com/quang-truong/pencil/tree/2d32e29dbed533288d9d758138e07547a0a7d8a9). Retained papers: [PENCIL](https://arxiv.org/html/2602.01553v4), [BatchEnsemble](https://arxiv.org/abs/2002.06715v2), [LPFormer](https://arxiv.org/html/2310.11009v4) and [Link-MoE](https://arxiv.org/html/2402.08583v2). Their numerical superiority is not independently established here. This authored source plan is not an independent PASS or an execution release.
