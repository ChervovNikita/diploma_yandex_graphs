# NCN/NCNC and PENCIL: ogbl-collab source readiness

Date: 2026-10-03. Scope: static public author source, saved literature reuse, and prospective protocol boundaries. No adoption, implementation, training, dataset access, checkpoint access, heldout scores, or measured performance/cost claim.

## Decision

**NCN/NCNC has an identifiable public collab recipe and source sites sufficient for implementation planning. PENCIL is the one recent, scientifically relevant competitor qualified here.** Neither family has been run or validated in the project environment. Source availability does not establish reproduced accuracy, superiority, runtime feasibility, or affordable cost. The parent owns adoption and resources.

NCN/NCNC uses the complete official TRAIN input graph without a year cutoff or LCC selection. Its native collab command adds validation positives for TEST and masks the whole positive minibatch during training. PENCIL's native collab loader filters TRAIN records to year ≥ 2007, uses sampled query subgraphs, trains on roughly 50% of its filtered positives per epoch, and unconditionally adds validation positives to the TEST graph. A complete, TRAIN-only PENCIL variant requires explicit source changes. **That variant is an adaptation; it is not an exact native or SOTA reproduction.** Native community settings and a future comparison on matched complete TRAIN topology answer distinct questions.

The NCN recursive tree contains no LICENSE/COPYING path; an explicit reuse grant was not established. PENCIL has a root Apache-2.0 license. Dependency and third-party notices remain separate, including the Meta copyright header in its collator. These observations are availability metadata, not a legal clearance decision.

## Evidence and pins

| Family | Author repository and inspected pin | Recipe anchor | Status |
| --- | --- | --- | --- |
| NCN/NCNC | `GraphPKU/NeuralCommonNeighbor` @ `11d597013750da17ce7468e344bec756a7af39a4` (2023-03-01) | README original lines 50–52; `NeighborOverlap.py`; `ogbdataset.py` | Native collab recipe identifiable; execution and license reuse unresolved |
| PENCIL | `quang-truong/pencil` @ `2d32e29dbed533288d9d758138e07547a0a7d8a9` (2026-09-28) | `args/official/ogbl_collab_bert.yaml`; `run_lp.py`; OGB loader and sampler | Native collab recipe identifiable; complete/TRAIN-only adaptation conditional |

The source snapshots preserve the author path names in flattened filenames under `source/`. Git blob IDs were independently checked against the pinned recursive trees at retrieval. `SOURCE_RETRIEVAL.json` records 24 source/config/license/README retrieval units. README pipe-table lines were excluded before inspection; no result images or result payloads were opened. LPFormer was a saved-literature fallback with public commit/tree metadata only; its source was not retrieved or qualified as another competitor.

Saved NCN/NCNC, BUDDY, PENCIL, LPFormer, and OGB protocol conclusions were reused first. The literature index was `index_v24`, SHA-256 `3618a9d22dbbc6369f850f2fb680ccb3e8727344998e0ec5ea945802c58ba2eb`. No new semantic primary-paper read occurred. Two saved HTML files were mechanically scanned for repository hrefs only. See `REUSED_CONCLUSIONS.json` and `READ_SCOPE.json` for provenance and exclusions.

## NCN/NCNC native collab recipe

The README's `$model` is `cn1` for NCN and `incn1cn1` for NCNC. NCNC2 is a separate depth-2 option; it is not silently included in the qualification. The shared collab command is recorded here as evidence, not as an executed command:

```text
python NeighborOverlap.py --xdp 0.25 --tdp 0.05 --pt 0.1
  --gnnedp 0.25 --preedp 0.0 --predp 0.3 --gnndp 0.1
  --probscale 2.5 --proboffset 6.0 --alpha 1.05
  --gnnlr 0.0082 --prelr 0.0037 --batch_size 65536
  --ln --lnnn --predictor $model --dataset collab
  --epochs 100 --runs 10 --model gcn --hiddim 64
  --mplayers 1 --testbs 131072 --maskinput
  --use_valedges_as_input --res --use_xlin --tailact
```

This is the source recipe, not a new seed freeze. Native runs use integer seeds `0…runs−1`. Collab does not require the README's citation2/DDI pretrained representations; loading is controlled by explicit flags.

| Item | Exact source convention |
| --- | --- |
| Features and graph | `ogbdataset.py:29–45`: official PyG OGB dataset and split; provided collab floating node features; no learned node-ID table (`max_x=-1`). Edge weights are discarded. Symmetric coalesced sparse TRAIN adjacency keeps all declared nodes; no year cutoff/LCC. |
| Native test topology | `ogbdataset.py:63–70` and `NeighborOverlap.py:122–142`: flag concatenates official validation positives for TEST. Both encoder embeddings and predictor neighborhoods switch to TRAIN+VAL; validation remains TRAIN. |
| Encoder | `model.py:40–196`: one PyG `GCNConv` layer, hidden width 64, input dropout .25, edge dropout .25, LayerNorm, dropout .1, ReLU. `gcn` is not the cached-convolution variant. The `--res` path is dimension-conditional; its presence does not imply the first feature-to-hidden residual applies. |
| NCN query structure | `model.py:525–595`: optional residual `xlin` changes pooled node representations; endpoint Hadamard features use encoder outputs captured before that residual; sum common-neighbor node representations with `spmm_add`; endpoint map; combine common-neighbor map with learned beta and endpoint map; biased scalar MLP. `--cndeg=-1` by default. |
| NCNC completion | `model.py:662–805`: uses overlap and each endpoint's residual neighbors; recursively scores missing links under `torch.no_grad`; native probability clamp is `p0=sigmoid(scale*(logit-offset))`, then `alpha*pt*p0/(pt*p0+1-p0)`. Weighted residual embedding sums augment the common-neighbor sum. Completion probability recursion is detached; weighted embeddings retain their downstream path. Default depth 1, fixed `pt` (`--learnpt` absent), no neighbor sampling (`trndeg=tstdeg=cndeg=-1`). |
| Training positives and target removal | `NeighborOverlap.py:42–76`: shuffle official TRAIN positives. `--maskinput` removes the whole positive minibatch from the training supervision edge list, then symmetrizes; both encoder and query predictor use this temporary graph. Full graph encoding is recomputed for each minibatch. |
| Training negatives and loss | PyG `negative_sampling(data.edge_index, num_nodes=...)`, with default requested count from input edge count; indexed negatives paired with shuffled positives. Only TRAIN topology is forbidden; there is no heldout-positive rejection. Loss is mean negative log-sigmoid on positives plus mean negative log-sigmoid on negatives, twice a balanced BCE mean. Adam groups use GNN LR .0082 and predictor LR .0037; source default weight decay is zero. |
| Coverage caveat | `utils.py:8–34`: native `PermIterator` drops the incomplete last training batch, while evaluation keeps the tail. The underlying graph is complete, but per-epoch supervision omits a shuffled remainder. Yielding the tail would be a declared training adaptation. |
| Evaluation and selection | Official validation/test `edge_neg`; shared negative pool OGB Hits@20/50/100, final collab metric Hits@50. Source evaluates TEST each epoch. Strict `>` selection on validation retains the first equal score; 100 epochs, no early stopping in the inspected loop. Native metric tuple tracking is not sufficient to guarantee a saved best checkpoint unless saving flags are bound; a future heldout-once wrapper must persist the validation-selected weights. TRAIN diagnostic Hits uses validation negatives and must not be mistaken for training supervision. |

A future complete/TRAIN-only comparison can omit `--use_valedges_as_input`, preserve all official TRAIN edges/nodes, retain native minibatch target removal, use official validation/test queries and negatives, and restrict TEST to the validation-selected checkpoint once. Removing native target masking would define a different recipe. Per-epoch tail coverage is another separate, declared choice. No such choices were adopted here.

README environment: torch 1.13.0 / PyG 2.2.0 / OGB 1.3.5. `env.yaml` pins Python 3.10.8, PyTorch 1.13.1, CUDA 11.7, and sparse 0.6.16, while the PyG/sparse builds identify torch 1.13.0. This discrepancy is recorded; no environment was installed or tested.

## PENCIL native collab recipe

PENCIL is recent and addresses query-local structural expressivity through bidirectional attention over sampled subgraphs plus adjacency propagation. This makes it an applicable competitor on methodological grounds. Its author's performance statements remain unverified here; no author accuracy or cost number is adopted.

The README original lines 30–31 use `python run_with_best_gpus.py args/official/<config_file>.yaml`, with an optional `--use_features`. The collab config is `args/official/ogbl_collab_bert.yaml`. The launcher's path is present in the pinned tree; its body was not retrieved. `run_lp.py` accepts the YAML as its positional argument and initializes CUDA/NCCL using distributed environment variables. No launcher or training command was executed, and no device allocation is requested.

| Item | Exact source convention |
| --- | --- |
| Native graph | `datasets/ogbl/dataset.py:43–52,95–121`: official OGB dataset/splits, all declared nodes, then unconditional collab TRAIN record filter year ≥ 2007; removes self-loops and converts to undirected with additive edge-weight reduction. Filter applies to native TRAIN supervision and graph, not merely the sampled query computation. No LCC selection found in this route. |
| Native test topology | `datasets/ogbl/dataset.py:438,454–531`: validation uses filtered TRAIN graph; TEST uses filtered TRAIN + official validation positives unconditionally, with validation edge weights 1.0. There is no config flag equivalent to NCN's switch. |
| Features and weights | Official YAML `use_features=False` replaces x with ones in the loader; native default uses structure only. The README also exposes `--use_features`, which retains provided features and defaults fusion to `early`. It is a distinct author-supported configuration, with no superiority claim here. Although dataset edge weights are retained, `DatasetWrapper(...,use_edge_weight=False)` is the default and `run_lp.py:192–257` omits that argument: adjacency tokens therefore use ones, not the stored weights. |
| Sampling and target removal | YAML `depth_neighbors=[[1,75]]`, `replace=False`, no pretraining. `dataset_map.py:999–1053`: torch_sparse ego sampling from both endpoints, induced graph on selected-node union, then removal of both target-edge directions. Target edge is removed **after node sampling**. Node selection can therefore depend on its presence; this is not equivalent to sampling after removal. Full benchmark topology and query-local sampling must be distinguished. |
| Native positive coverage | YAML `percent=50`. `dataset_map.py:502–576`: deterministic permutation seed `base_seed*100 + percent*epoch//100`, cyclic period `round(100/percent)`, slice length `round(N*percent/100)`. Native training uses roughly half the filtered positives per epoch, not every positive each epoch. Two slices aim to cycle coverage; rounding can omit a record for odd N when the half rounds down. No actual N was read. Valid/test use all official query positives and supplied negatives, not the 50% training subsampling. |
| Training negatives | `dataset_map.py:341–401,795–910`: global PyG negative sampling from the full native TRAIN graph with self-loops added, requesting `selected_positive_count*neg_ratio`; ratio 1, no weighting. Does not reject validation/test positives. New epoch sampling/reset seed is `base_seed*100+epoch`. In a complete-graph adaptation, the forbidden set must also become complete TRAIN. |
| Tokens | Local vocabulary capacity 256; query endpoints remap to local indices 0 and 1, other local IDs shuffle in training and use fixed ordering during eval when `node_remap=False`. It is not a learned table of global OGB node IDs. Adjacency-row node vectors have width `2*256+2=514`; two endpoint task tokens copy endpoint structural vectors with task flags. Only connected sampled nodes plus endpoints enter the sequence. Input and task tokens are padded separately and concatenated with masks. |
| Model | Scratch BERT encoder: hidden 512, 8 layers, 8 heads, intermediate 2048, max positions 2048; frozen orthogonal input projection and per-layer trainable Linear+GELU adjacency projection. Each layer applies bidirectional attention followed by `hidden += MPProj(dense_adj@hidden)`. Reconstructed propagation rows combine encoded adjacency and local-ID one-hot vectors, pad task columns, and normalize by row sum. Two endpoint hidden states concatenate into a biased scalar head; BCEWithLogitsLoss. `AutoConfig.from_pretrained("bert-base-uncased")` fetches config metadata; YAML `from_pretrained=False` prevents model weight loading. No pretrained LLM weight requirement. |
| Training | YAML seed 0, 20 epochs, AdamW LR 1e−4, weight decay .01, bf16, DDP, batch 1024/rank, 12 workers, no max sample cap. YAML accumulation 8 is **total across ranks**; `run_lp.py:1430–1449` uses floor(8/world_size) per rank and requires world_size≤8, with effective total altered when not divisible. It is not eight accumulation steps per device. Final partial accumulation takes an optimizer step but loss remains divided by the configured accumulation count. DistributedSampler can pad training records; runtime coverage was not checked. No LR scheduler was found in the inspected runner locator and optimizer/training scopes. |
| Evaluation | Official collab metric route selects Hits@50, not per-positive HeaRT MRR. `evaluator.py:74–110,245–278`: split logits by binary label and rank each positive against the same full official negative pool; a hit requires score strictly greater than the 50th-highest negative (ties fail). Eval loaders use nonshuffling DistributedSampler; gather interleaves ranks then trims to dataset length; sequential index assertion is enabled. This supports intended complete query evaluation but is not runtime validation. |
| Selection and TEST | YAML `eval_every=1,eval_test=True`; native TEST scores/loss every epoch. `run_lp.py:1731–1753` saves only strict validation Hits@50 improvement as best checkpoint; first exact ties remain. Native final evaluation reloads best checkpoint and evaluates TEST again for a fresh, non-debug, non-resumed run. Native resume/skip_val/debug branches differ and are excluded from this fresh recipe. No early stopping in the inspected epoch loop. |

The retrieved `heart_ogbl_collab_bert.yaml` body is identical to the official collab YAML at this pin. Its filename does not make HeaRT queries or metric equivalent to official OGB: the HeaRT flag changes dataset naming, loads external negative files, and evaluates a distinct metric route. That branch is not admitted into an official Hits@50 comparison.

README environment: Python 3.12 and CUDA 12.4 PyG extension wheels; requirements torch 2.5.1 / PyG 2.6.1 / OGB 1.3.6 / transformers 4.46.2. Repository source and config URLs returned 200 at the pin. The README advertises a Google Drive data.zip; no archive or access status was inspected. The ordinary collab path delegates acquisition to OGB; payload availability and dataset identities remain unchecked.

## Prospective boundary, without adoption

| Boundary | Future matched complete/TRAIN-only setting | Native setting preserved as evidence |
| --- | --- | --- |
| Base graph | All official TRAIN records and declared nodes; no year cutoff/LCC; common symmetrization/self-loop/duplicate conventions must be recorded | NCN full TRAIN; PENCIL year ≥ 2007 TRAIN |
| TEST graph | TRAIN only for encoder and every query-structural operation | Native NCN flag and native PENCIL loader add VAL for TEST |
| Queries and negatives | All official validation/test queries and official negative pools; train negative forbidden set complete TRAIN only, excluding self-links; no heldout-positive rejection | PENCIL HeaRT is a separate setting |
| Target removal | Disclose each native order and granularity; any matching target-removal experiment is separately declared | NCN masks whole positive minibatch before encoding; PENCIL masks one query after sampling |
| Features | Prebind structure-only or author-supported feature mode and fusion; record what each comparator receives | NCN uses provided node features; PENCIL default structure-only, optional features supported |
| Training exposure | Retain and disclose native epoch coverage unless a separate adaptation is explicitly selected | NCN drops shuffled last minibatch; PENCIL roughly 50% cyclic positives and DDP padding |
| Selection/heldout | Validation Hits@50 selects persisted weights with explicit tie rule; TEST accessed once after selection | Both native runners inspect TEST each epoch; PENCIL also final reload/test |

For PENCIL, bypass `filter_by_year`, send complete TRAIN to the test dataset instead of `full_graph`, retain official negatives without HeaRT, and disable per-epoch TEST access. These source sites make an adaptation concrete, but it has not been implemented or validated. Changing percent, sampling fanout, target-edge timing, weighted tokens, feature fusion, epochs, or architecture changes additional recipe dimensions. A study seeking community reproduction and a study seeking graph-matched comparison should report their setting names and claims separately.

NCN/NCNC/PENCIL parameter sets must remain independent if used as separate predictors. No structural-cache equivalence with BUDDY is implied: NCN recomputes graph embeddings per minibatch; PENCIL forms and attends to query subgraphs. Saved BUDDY cache-sharing conclusions concern its own fixed inputs. Running BUDDY protocols, artifacts, seeds, and drivers were not edited.

## What remains unknown

Measured epoch time, end-to-end time, peak memory, throughput, accuracy, calibration, ensemble benefit, compatibility with the intended runtime, dataset content hashes, native query/negative counts, official split bytes, checkpoint identity, and exact launch behavior are unknown. Code/config sizes and theoretical computation paths do not substitute for those measurements. Scientific relevance does not depend on current compute availability; this packet makes no compute demand.

The packet is ready for the parent to review source readiness and distinct protocol choices. It is not a run authorization, driver, new arm, seed freeze, baseline adoption, or manuscript result.
