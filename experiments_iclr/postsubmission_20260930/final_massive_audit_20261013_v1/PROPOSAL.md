# Comprehensive GNNM study: audited run proposal

## Purpose and current status

The study will test whether our changes to a shared-backbone ensemble improve prediction quality against original GNNM, capable single models and independently trained ensembles. It also includes published model recipes so we can check the underlying implementations and interpret any differences from reported results.

The complete source inventory and disabled run plan are frozen. Training awaits your verification and explicit GO. Resource admission remains incomplete: trained-state storage, several source-dependent contexts, concurrent throughput and the physical environment on 18.77 still need qualification. The registered counts below describe planned work. They are not completed experiments or evidence of an accuracy improvement.

The plan's canonical SHA256 is `0a35daf95e7f1ad8e8df1037923a5a38638ed37e8c0a65c9dbcc3bf73f55488b`. Its allocation path is `experiments_iclr/postsubmission_20260930/massive_practical_master_20261013_v29/PLAN.json`.

## Coverage

There are **12 underlying datasets**, with separate task and protocol variants where needed:

| Task group | Datasets |
|---|---|
| Ordinary node classification | Amazon Photo, CiteSeer, Coauthor CS, Coauthor Physics, PubMed, Twitch DE MUSAE raw, WikiCS |
| Heterogeneous node classification | HGB DBLP, HGB IMDB |
| Molecular graph classification | OGBG MolBACE, OGBG MolHIV |
| Link prediction | OGBL Collab; CiteSeer with HeaRT link roles |

The final census distinguishes **21 named architecture recipes**: GAT, GCN, GCN with a virtual node, GIN, GIN with a virtual node, GINE, GT-sep, SAGE, PolyFormer, Polynormer, NCN, NCNC, PatternTwin, BUDDY, DiscreteGeneralSheafDiffusion, HGT, FiLM, SeHGNN, UniMPv2, BayesBundleSheafDiffusion and DiscreteBundleSheafDiffusion. The earlier count of 16 used an incomplete roster and broader grouping. Literal source configurations and aliases remain visible in the census. Frozen features and unspecified parent labels are separate from the architecture count.

The canonical source audit accounts for **263 prepared recipe rows**: 228 procedure variants and 35 reference/control rows. Ten further rows have explicit inactive or unimplemented dispositions. These counts include variants and controls; they do not represent 263 novel algorithms. Concrete runtime identifiers are more numerous because the same procedure can have different source and comparison contexts.

Methods are assigned to compatible tasks and backbones. The plan does not multiply every method by every dataset. Original GNNM remains a reference on compatible new tasks. Roman Empire, Amazon Ratings, Minesweeper, Questions and Tolokers are excluded from the new plan for every method. Original paper scores remain unchanged.

## Comparison design

Each study preserves its declared dataset variant, role partition, source recipe, training policy, seed and comparison controls. A four-member independent ensemble has separately optimized and selected members. A jointly optimized model with four bodies is counted separately from that reference.

Search uses the complete declared training policy for each configuration. Native optimizer, warmup, minimum training and stopping rules remain explicit. Fixed long-cap recipes keep their source patience rules. Search selection uses TRAIN/SELECT information; the subsequent comparison fits use the frozen choice and fresh declared seeds. Studies retain their own split and seed panels, which are listed in the census.

The source inventory covers initialization and contrastive steering, private attention and propagation paths, shared/private corrections, staged label conditioning, structural and cardinality controls, and learned prediction aggregation. Ablations preserve the distinctions between changing the backbone, changing member factors and changing the final prediction rule. Published ancestors and genuine independent ensembles provide the relevant comparisons within each compatible study.

The main experiment log will preserve curves, stopping reasons, selected checkpoints, failures and costs. A training cap reached without adequate convergence will remain an inconclusive comparison. No result will expand the frozen configuration domains. Calibration and heldout evaluation follow the plan's complete-stage gates. Published author recipes retain their separately labelled evaluation protocol.

## Exact registered work

| Work unit | Count |
|---|---:|
| Source packets | 51 |
| Study specifications | 98 |
| Search model-fit groups | 14,436 |
| Comparison model-fit groups | 66,414 |
| Total model-fit groups | **80,850** |
| Paid optimizer starts | 83,637 |
| Author-recipe CLI jobs, counted separately | 88 |
| Search-selection jobs | 892 |
| Served prediction banks | 43,473 |
| Scalar calibration jobs | 43,362 |
| Evaluation jobs | 43,473 |

Fit groups include paid continuation and postfit procedures. Optimizer starts, physical bodies and served banks have different definitions. The count ledger retains those distinctions, native stopping caps and bounded internal optimizer work. Author jobs contain their own declared repetitions. Setup and engineering checks are recorded separately.

The model-fit groups are assigned as follows:

| Worker | Search | Comparison | Total |
|---|---:|---:|---:|
| Allocation GPU 0 | 5,988 | 28,095 | 34,083 |
| 18.77 GPU 0 | 7,240 | 32,619 | 39,859 |
| 18.77 GPU 1 | 1,208 | 5,700 | 6,908 |

These assignments preserve required dependencies and use declared scheduling weights. Measured concurrent throughput is still unavailable, so their balance and completion time are not certified.

## Configurations and training budgets

The final census provides the exact method/control IDs, dataset/backbone cells, seeds, split IDs, budgets, selectors and effective configuration values. Its source-context recipe count includes provenance and dependency fields; it must not be interpreted as a count of independent hyperparameter settings.

The authoritative tables are:

- `final_census_master_v29_20261013_v1/METHOD_STAGE_COUNTS.csv`
- `final_census_master_v29_20261013_v1/DATASET_BACKBONE_STAGE_COUNTS.csv`
- `final_census_master_v29_20261013_v1/WORKER_STAGE_COUNTS.csv`
- `final_census_master_v29_20261013_v1/SOURCE_RECIPE_COUNTS.csv`
- `final_census_master_v29_20261013_v1/HYPERPARAMETER_DOMAINS.json.gz`
- `final_census_master_v29_20261013_v1/SOURCE_RECIPES.json.gz`

The two compressed domain files remain on the allocation to avoid copying large inactive research metadata to the Mac. Configuration lists remain literal lists; marginal parameter ranges are not used to invent a larger Cartesian grid. The frozen study specifications are the execution authority.

## State retention and resource findings

The integrated scheduler permits at most two open releasable search cohorts per GPU and prioritizes completed cohort selection. This limits the registered search-parent frontier to 144, 144 and 36 fit records on the three workers respectively. **76 additional source fits remain protected** because later training consumes their states. Releasing an unused search state's binary files requires exact terminal-owner and retention receipts. Seeds, configurations, curves, ranks, costs and failure history remain preserved.

The final metadata bound for raw prediction banks and IDs is **200,169,515,658 bytes on the allocation** and **279,138,743,322 bytes on 18.77**. Checkpoints, nested optimizer states, recovery, hidden-feature caches, traces and temporary atomic writes require additional space. These bank figures alone cannot establish disk feasibility.

Earlier constructor checks passed for 47 of 55 source-family witnesses. Eight cases require paid parent custody, CUDA replay, runner context or source diagnostic work to qualify their resource footprint. Final additions and aliases are enumerated separately in the resource report. An initialized CPU snapshot does not certify the mature trained state or GPU memory use.

The allocation's disk/RAM observations and these byte bounds are preserved with timestamps. 18.77 has not been accessed under the current allocation-only instruction. A reliable ETA is unavailable. Placeholder scheduling seconds must not be converted into a training-time promise.

## Launch interface and approval

The common wrapper accepts a host, GPU index, exact approval receipt and worker interpreter. Once preparation sets `GNNM_GO_RECEIPT` and `GNNM_WORKER_PYTHON` to the qualified absolute paths on each host, the commands will be:

```bash
bash experiments_iclr/postsubmission_20260930/final_sync_master_v29_20261013_v1/run_gpu.sh allocation 0 "$GNNM_GO_RECEIPT" "$GNNM_WORKER_PYTHON"
bash experiments_iclr/postsubmission_20260930/final_sync_master_v29_20261013_v1/run_gpu.sh server77 0 "$GNNM_GO_RECEIPT" "$GNNM_WORKER_PYTHON"
bash experiments_iclr/postsubmission_20260930/final_sync_master_v29_20261013_v1/run_gpu.sh server77 1 "$GNNM_GO_RECEIPT" "$GNNM_WORKER_PYTHON"
```

The wrapper requires the exact approval receipt; 18.77 also requires an explicitly qualified Python interpreter. Its handoff file records the arguments/environment variables. None of these start commands has been executed.

Two slots per GPU are a concurrency ceiling. The launcher admits two jobs when their summed reservations fit available memory. Two largest jobs exceed the allocation and 18.77 GPU 0 capacities, so those jobs must run individually. Existing unrelated workloads are accounted for through available GPU headroom.

The final audit report binds the census, source coverage, resource findings, manifest/loader checks and transfer receipt to this plan. Your verification and explicit GO are required before any new scientific fit, including a pilot. Physical and resource admission gates remain required after that verification.
