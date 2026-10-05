# PENCIL Citeseer-HeaRT: concrete competitor and resource plan

## Prepared comparison

Use PENCIL's pinned Citeseer-HeaRT recipe from `quang-truong/pencil` at `2d32e29dbed533288d9d758138e07547a0a7d8a9`, with the **author-supported feature/early-fusion mode**. This gives the competitor the same authenticated raw float32[3327,3703] inputs as NCN. It is chosen prospectively, with no validation-based feature-mode search. The official structure-only YAML is preserved beside the explicit proposal; feature mode is a declared adaptation, not a promise that it scores higher. Do not call this SOTA reproduction or reuse an existing Collab fit.

Full proposed comparator: three fresh scratch fits with seeds **[0, 1, 2]**, aligning the existing NCN native-single seed blocks. Keep complete300 epochs and complete VALID every2 epochs. The existing36-fit NCN/frame plan is unchanged. PENCIL results are a separately fixed later comparator, not inserted into its original prospective36-fit hypothesis or chosen from its outcomes. Keep TEST closed and compare only after complete cohorts.

## Native one-GPU interpretation

| Item | Retained Citeseer recipe |
| --- | --- |
| Backbone | Scratch BERT,2 layers,hidden1024,8 heads,intermediate2048; no pretrained weights |
| Query support | Two-hop20-neighbour sampling,no replacement; target link removed **after** sampling |
| Structural tokens | Adjacency-row; tokenizer bound842 nodes,structural width1686,at most844 positions; model position capacity2048 |
| Raw features | Author-supported early fusion,width3703 per adjacency-row token, **not7406** |
| TRAIN | All3870 native nonself positives per epoch; fresh global ratio-one negatives excluding TRAIN/self links; no heldout-positive rejection |
| Microbatch |256 **mixed labelled queries**,8 loader workers |
| Accumulation | Total8/world-size. One NCCL/DDP rank uses8 microbatches/update; full update represents2048 mixed queries |
| Optimizer | AdamW lr1e−4,weight decay.01; native BCE loss; no new scheduler/clip |
| Precision | Native `bf16=False`; retain native TF32/matmul-medium profile and disclose contrast with NCN |
| Schedule |300 complete epochs; VALID every2; no early stopping in this recipe |
| Selector | Strict maximum complete VALID MRR,first exact tie; PENCIL native unrounded precision (NCN's selector rounds4 decimals) |

Intended3870 positives+3870 negatives gives7740 mixed queries,31 microbatches and4 optimizer updates per epoch: three full8-microbatch updates and a final7-microbatch update, the last including60 queries. Preserve native division by8 on that partial update rather than renormalizing it. Native PyG negative sampling can underfill; record actual generated negatives, loader lengths and `ceil(actual_microbatches/8)` per epoch instead of repairing it or asserting31 unconditionally. At world size1 the DistributedSampler does not pad rows.

The842 node bound follows the **native** formula `2*(1+20+20²)`; it is not a measured actual subgraph size. Degree sparsity may make actual sequences smaller, but that does not establish memory feasibility. This differs substantially from the old Collab resource observation (512 hidden/8 layers/bf16/152-node bound). Its32.8GB allocated/50.34GB reserved observation on18.77 cannot qualify this model or this one-GPU runtime.

## Exact TRAIN/VALID adapter

`train_valid_adapter.py` constructs only TRAIN and VALID native datasets from caller-authenticated tensors. It preserves all declared3327 nodes, concatenates the two directions of the3870 TRAIN positives and uses native unit edge weights. There is no year filter or validation-edge addition for this graph. The caller first uses the existing exact available manifest and root feature/pool qualification to load the four allowed files.

Native HeaRT deduplication is retained: concatenate the227 positives **first**, then row-major113500 negative occurrences; normalize each unordered pair `(min,max)`; keep the first insertion of each unique pair; retain an inverse map and original labels. Evaluate every unique query once, then `scatter_preds` restores **all113727 occurrences**, preserving duplicate candidates and any native cross-query collisions. Reshape only the restored negative tail into227×500. Never rank each positive against a shared global negative pool or score only unique negatives. The adapter checks exact inverse-map reconstruction. Actual unique count is unmeasured.

The native loader unnecessarily opens regular valid/test negative text and both TEST pools before replacement. This thin adapter avoids that loader and its pickle cache, directly calling the unchanged native unique-pair helper and dataset constructors. Its API has no TEST argument or cache write. PENCIL's native `node_remap=False` evaluation, global sampler, target-mask timing and loader RNG must remain. Keep native post-constructor `set_seed(seed)`, then the **real** `get_feature_dim` sample call before model initialization; bypassing that sample changes model RNG. At each epoch retain native resets and DistributedSampler.set_epoch.

## Minimum worker changes

Reuse the licensed, byte-bound20 native Python files retained in `pencil_collab_paired_predictive_preparation_20261004_v3/native`, plus its owned source marker and pinned570-byte BertConfig. Keep native `build_loaders/get_model/train_loop/evaluate_loop` functions unchanged. Stage the selective adapter/config and source closure inside the authorized one-GPU phase; do not copy Collab data/runtime authorities or its20-epoch assertions.

The owned worker must: bind the one-GPU route/UUID; use one directly supervised DDP rank and repo-local file rendezvous; call the thin adapter; use empty model-factory scratch directory (no automatic resume); apply the pinned local non-weight AutoConfig instead of network lookup; disable wandb; set TEST loader=None; preserve complete native train/VALID passes; select/save only the strict best-MRR full model/AdamW/RNG state and restored complete VALID arrays. Disable all per-epoch/final TEST code and expose no TEST paths. Report selected state hashes, unsuccessful runs, actual coverage and inclusive work. Save cadence changes are disclosed; no numerical parity claim follows.

Only add shape/finite/provenance instrumentation and owned-process monitoring. Default native pin_memory=True can first be retained on the currently working NCN Torch runtime. The prior18.77 pinning failure/False override is not evidence this host must change it. Any needed loader-transfer or allocator override is a separately recorded successor; no silent repair, batch reduction, bf16 switch or shortened training.

## Actual blockers and one concrete qualification

**Source ownership is available:** exact native snapshots and Apache2.0 license are retained; the thin adapter is newly authored and the license is included. **Dependencies/runtime feasibility remain unqualified on the one-GPU host.** Its NCN receipt uses Torch2.1.2+cu118/PyG2.7.0/NumPy1.26.4, whereas the prior PENCIL receipt is on18.77 with Torch2.7.1 and a host-specific overlay. Do not apply the18.77 dependency admission or install paths here. Verify the one-GPU environment has the declared transformers4.46.2 (including private `_prepare_4d_attention_mask_for_sdpa`), tokenizers,rootutils,wandb,rich,yaml,ogb and their import dependencies without changing the active NCN core environment. Add missing ancillary packages in a repo-owned overlay only after root review; preserve original licensed source.

Then run **one complete resource-only seed0 epoch** with this exact feature-enabled model/batch/accumulation and a complete unique-query VALID forward followed by inverse-map restoration. That is one discarded resource fit with actual optimizer updates, not a scientific300-epoch baseline. Compute no ranking metric and retain no checkpoint/predictions for scientific reuse. Measure inclusive setup/TRAIN/VALID time, all row/order hashes, actual mixed-query/microbatch/update counts, max token/feature shapes, host RSS and CUDA allocated/reserved peaks. Root binds memory/time limits to the verified physical device and current queue; an observation timeout is not permission to restart.

A pass supports a measured300-epoch ETA `300*T_train + 150*T_valid + setup/save overhead`, **not predictive quality**. Only then release the separately frozen three-seed full comparator. If the faithful workload cannot fit or would displace the active NCN cohort materially, preserve that result and discuss resources; do not make a weaker recipe look like the native one. No server action, import, model/data fetch or numerical run occurred in this preparation.
