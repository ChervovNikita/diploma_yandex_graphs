# Pubmed expansion of the current private-learning operator

5 October 2026. Source/design preparation under allocation-only access. No server or MacLink contact, data extraction, feature/model/checkpoint/outcome payload read, sampler run or fit occurred. Original studies remain unchanged. The single execution recipe in this folder is disabled.

## Decision and exact scope

Use **Pubmed under the existing HeaRT release**, conditionally after the complete fixed 30+9 Citeseer family passes its existing prospective lead gate. The immediate next step is a model-free TRAIN geometry measurement, not another fitting grid or training admission.

The allocation's saved Citeseer acquisition observation reports successful authentication of the complete `HeaRT.tar.gz`, SHA256 `7b7042476319a353bdb6c50b5f402b89b9006a2fde2d1258b7adcbd1d22629ba`, 880,240,878 bytes. Its launch selected the single-download path; the acquisition source retains the archive at:

`/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/citeseer_heart_official_acquisition_server_20261005_v1/withheld_inputs/HeaRT.tar.gz`

This establishes **archive-resident status at acquisition**, not a fresh file-presence check or a released Pubmed loader on the allocation. The saved Pubmed member manifest from GPU77 binds the same official release's `dataset/pubmed/train_pos.txt` (409,212 bytes; SHA256 `c6de89d86371909f738d620846540168d4b6256fed88dc9d8ab3609cb5357fb4`). It is used only as expected member metadata. No GPU77 action is required or permitted here. Refresh the allocation archive hash and extract this exact TRAIN member into a fresh directory after root source/bounds review. No download, alternative dataset search, VALID/TEST/feature extraction or heldout loader is implemented.

## Known protocol facts and missing evidence

| Item | Saved source/metadata fact | Still needed on the allocation |
|---|---|---|
| Graph population | Pubmed source declares 19,717 nodes; the shared4 bridge declares 500 input features | Node-population metadata adoption; later independent feature identity/provenance and shape admission |
| TRAIN | 37,676 rows, no self-links, no equivalent undirected duplicates, 18,256 distinct endpoints | Fresh exact member authentication/count/range check; graph census |
| VALID | 2,216 positives; fixed `(2216,500,2)` integer pool; first250 source-fixed and last250 target-fixed candidates | Later separate complete pool/order/custody engineering; no payload is opened by this packet |
| Metric | HeaRT tie-aware MRR and Hits10; `rank=1+0.5*(count(neg>=pos)+count(neg>pos))`; native float32 mean rounded4 | Complete evaluator and selected-state replay qualification on the allocation |
| Support | Complete TRAIN support for serving, bidirectional and unweighted; no VALID input edges | New task loader and support construction audit |
| Pool semantics | Native generator may retain duplicates and other-VALID-positive collisions; these are preserved and disclosed | Confirm the existing root-authorized semantic evidence applies to the identical archive bytes; no filtering/regeneration |

Raw feature width or a filename does not prove feature origin. Older saved provenance notes were unresolved; later source bindings refer to an exact raw-feature equivalence receipt. That receipt and its allocation applicability must be bound before fitting. No feature evidence or old baseline score is inferred here.

## The current operator must remain NCN

The current Citeseer operator's donor `heads.py` constructs `CNLinkPredictor`; it has **no recursive NCNC completion**. Its private-learning rule is virtual private Adam → native shared Adam → recomputed private Adam from the old private state, with discarded virtual moments, replayed four streams, frozen previous moments during current differentiation, and adaptation-free mean-raw-logit serving. Preserve `private_adam.py` and `transfer_step.py` exactly, including `.001/(.9,.999)/1e-8`, row roles, loss normalization and the half aggregate/half member BCE objective. Preserve the 256-wide nonrecursive head and its initialization, layer normalization and dropout settings.

The port therefore changes the encoder input dimension from 3703 to 500 and task-specific input/custody/count/shape bindings. It does not adopt the different Pubmed NCNC architecture, dropout values, completion or clamp settings into the candidate. Retain the current encoder's own JK flag from its actual constructor. Recount the actual constructor/partition at qualification; the old hardcoded 1,409,026 outer-coordinate assertion cannot carry over unchanged. Private row geometry is intended to remain 6,402 per row and must be checked, not assumed.

| Future source location | Required change before any fit |
|---|---|
| `custody.py` | New allocation Pubmed input manifest/path/hash authority; full `(19717,500)` safe feature load; 37,676 TRAIN / 2,216 VALID counts and fixed500-pool checks; fresh source/runtime/feature/pool admission |
| `models.py` | Only input width3703→500 and audited partition-count assertions for candidate models; retain current donor head/encoder operations and initialization |
| `qualify_training_step.py` | Pubmed fixture/data/shape/source bindings and fresh exact-source FP32 direct/mixed-gradient and committed-state qualification |
| `run.py` and job/source manifests | Pubmed provenance/shape/gate bindings; retain episode update, all-tail coverage, objective, evaluator and selector; bind new exact bytes |
| Native NCNC baseline wrapper | Allocation paths/runtime authority and independent member state/RNG/optimizer isolation, complete validation pooling and selected-state replay; preserve author numerical bodies |

`private_adam.py`, `transfer_step.py`, donor `heads.py`, `episode_geometry.py` and `native_episode_cycle.py` need no numerical change. These fitting changes are specified, not implemented by this packet.

## Narrow TRAIN preparation in this folder

- `extract_pubmed_train_only.py` reauthenticates the exact resident archive and extracts only its exact regular TRAIN member, checks its bytes, format, full node range, unique nonself facts and count, and emits a TRAIN-only receipt. Archive hashing/header traversal interprets no other member payload. It has no networking or feature/heldout/model loader.
- `episode_geometry.py` and `native_episode_cycle.py` are copied **byte-for-byte** from the current qualified sampler. Negative sampling and the CPU-only restored data RNG, complete outer permutation including tail, four route/class streams, endpoint exclusion, exact full-TRAIN degree/common-neighbor matching and paired union mask are unchanged.
- `inspect_train_episode_geometry.py` reuses the existing inspector's measurement body with Pubmed population/count/seed metadata. Its inherited custody setup is trimmed to a root-reviewed recipe, exact source/input pins and allocation-only bounds/paths. It measures **seed 0, cycle index 0, one full cycle**, outer 64, inner 256 per class per route, four routes. The count implies 589 episodes: 588 full batches plus 44 positive/negative queries in the tail. This is arithmetic, not a measurement result.
- The census retains every draw/rejection; reports all-node and route/class degree/common-neighbor strata before/after masking, eligible positive/negative populations, outer/inner exposure, repeated facts, removed support fractions/counts, endpoint isolation, remaining outer-incident context, matching status and runtime/RSS. It exports geometry metadata and identity hashes, not query payloads. No redraw, padding, skip, fallback update, labels outside TRAIN or optimizer commit is allowed.

Endpoint exclusion restricts **inner queries**, not every support edge incident to an outer endpoint. Support removes the union of outer-positive targets and both arms' sampled inner-positive targets; other outer-incident context remains. A larger graph does not guarantee feasible 256-query classes after hub endpoints are excluded. Matching/removal may change informative common-neighbor structure, and global message passing does not create independent examples. At most 64+8×256 target IDs enter a full-episode union mask before repetitions; actual removal, isolation and strata changes must be measured. Fail the candidate geometry if any episode is infeasible; do not repair it by changing sizes or filtering the graph.

Prospective **review proposals**, with no feasibility claim: extraction 600s soft/900s external hard; geometry 1,800s soft/2,100s external hard, two CPU threads/one interop thread, CUDA hidden, one attempt; external peak RSS 4GiB/output 256MiB. The geometry source records peak RSS but does not provide an ownership supervisor. Root must bind existing reviewed owned CPU supervision, these caps, fresh output, exact runtime/source/recipe and terminal receipt before execution. The allocation sampler versions recorded in the original preparation are preserved; actual Torch/NumPy/PyG versions must match. Bound breach preserves failure; no retry or substitute sampler.

`RECIPE_DISABLED.json` holds both operations' exact member/source pins and concrete fresh output proposals under the allocation phase: `private_transfer_pubmed_train_geometry_execution_root_20261005_v1/train` and `.../census`. Root creates the actual execution release separately after review; the source packet adds no supervisor or launch adapter. Both programs take `--recipe` and `--output` and reject unapproved recipes before numerical imports. The source syntax and copied byte identities are statically checked; no module was imported or executed.

A successful first-cycle census proves only that seed 0/cycle 0 geometry was feasible under those inputs. Other seeds/cycles, live mixed derivatives, native Adam parity, full TRAIN cycle/VALID cost, memory, evaluation and horizon adequacy remain unmeasured. Geometry success never authorizes training.

## Minimal future scientific comparison, only if the current lead survives

Propose five served conditions across fixed base seeds 0,1,2; no geometry, width, learning-rate or schedule grid:

| Condition | Why required |
|---|---|
| Current NCN F4 endpoint live | Exact candidate accuracy transfer to the different graph |
| Same F4 endpoint detached | Isolate whether live private-learning credit helps here |
| Same F4 endpoint ordinary joint | Test the complete learning rule against the paid-exposure ordinary schedule |
| Source-native Pubmed NCNC single | Competent task-native nonlinear/completion baseline |
| True independent-four source-native NCNC | Test accuracy against an ordinary ensemble of strong models |

The native Pubmed NCNC recipe is already saved: one-layer puregcn, width 256, JK, `incn1cn1` depth 1 completion, source-native `.001` encoder/head Adam, source masking/negative sampler, batch 1024 with the native shuffled tail dropped, max 9999 epochs, VALID every 5, eleven misses, first maximum rounded4 MRR. A true independent-four adapter must give each member its own complete native encoder/head, optimizer, initialization and RNG with seeds `s,s+5,s+10,s+15`; each receives its own normalized loss with no 1/4 gradient scaling. Serve the mean of raw logits from four independently selected native checkpoints. Single may reuse member 0's own selected state, disclosed prospectively. Existing Pubmed native single source is a baseline recipe; no qualified allocation independent-four wrapper or baseline performance is established by this task.

This is 15 proposed served comparisons, not a released cohort: nine candidate/control fits plus twelve independent native fits with member 0 reuse, 21 unique fits if adopted. Current-rule controls retain the exact 60-cycle/five-cycle/eleven-miss selector proposal and fixed factor seeds from the original paired blocks. Native baselines retain their competent native convergence/selection recipe; budgets are different and must be charged and disclosed. No claim of equal compute or selection opportunity is available. Full cycle and complete evaluation costs must determine whether this proposal is practical before any fit release; do not shorten a failing baseline to obtain an ensemble claim. Report every seed/failure. Do not add F1/random/untied arms unless the original lead leaves a specific unresolved attribution question requiring them.

## What this confirmation would test

Pubmed supplies a different graph, roughly ten times the TRAIN population, more nodes and different input-feature width. It can test whether an accuracy lead and endpoint-conditioned private learning survive a larger sparse citation graph and changed degree/common-neighbor geometry. It **repeats the citation domain, static transductive link supervision and HeaRT per-query negative-ranking protocol**. It is not a temporal, inductive, new-domain or new-evaluator test. Collab would add temporal supervision and a shared Hits50 pool, but its inspected data custody is only on withdrawn GPU77, its duplicate-record/support policy requires further adaptation, and it is not chosen here.

Next authorized work, after root review, is only archive refresh/TRAIN extraction and the census. A future fresh-task accuracy claim requires the original complete-family lead, task-specific source/runtime/derivative/evaluator admission, measured costs, a prospective scientific release and eventual frozen heldout evaluation. This packet supplies no result, launch authority or accuracy/novelty conclusion.
