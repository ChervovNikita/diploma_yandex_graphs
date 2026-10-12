# Original boundary GNNM adapted to native NCN/NCNC links

## Status

`link_boundary_packet_20261012_v3` is prepared and disabled on the verified allocation, with code and compact metadata mirrored locally. Final frozen CPU QA passed five tests, zero skips, in 9.56 seconds. Allocation and local metadata master ingestion both passed, with all reference cohorts matched and 214,200 model optimizer calls derived from the bound 17-minibatch floor law. No scientific fit, CUDA pilot, CAL, TEST target decoding or GO occurred.

The original `link_plan_packet_20261012_v1` remains unchanged: PLAN file SHA256 `c5cd4075e3b7d743a05e7058c16fd88f2a9484e3003eee47256cea840a22e8b8`. New packet revisions V1/V2 retain their failed CPU receipts. Their test assertions initially inspected gradients after the engine cleared them during selected restore, then compared different native query batching shapes. V3 checks gradients before restore and pools identical complete query batches. Adapter semantics did not change between these three qualification revisions.

| Review artifact | Project-relative path |
| --- | --- |
| Master input | `link_boundary_packet_20261012_v3/MASTER_INPUT_ENTRY.json` |
| Compact summary | `link_boundary_packet_20261012_v3/SUMMARY.json` |
| Disabled plan | `link_boundary_packet_20261012_v3/matrix/PLAN.json` |
| Counts, banks and contract | `link_boundary_packet_20261012_v3/matrix/SUMMARY.json` |
| Exact CPU/source QA | `link_boundary_packet_20261012_v3/receipts/CPU_SOURCE_QA_RESULT.json` |
| Master ingestion proof | `link_boundary_packet_20261012_v3/MASTER_INGEST_VERIFICATION.json` |

Canonical plan hash: `82aea15729e067bc8cc9dadf97271690d5647200a8eac9bd5d27e273916cd059`. PLAN file SHA256: `3097251a64bfd2c5ea848e3ce1ffce9f8d089911ea1456808ee812096c64db1e`. Final CPU QA receipt SHA256: `5568ceea2ec6bf981d7cfa129eb8a45e83f468dbf78b737f4ab139275a3b6028`.

## Tensor and parameter contract

Pinned graph/decoder source is GraphPKU/NeuralCommonNeighbor commit `11d597013750da17ce7468e344bec756a7af39a4`: model SHA256 `5b6cfe26074c83d57900f589b37a4ef57fc7b8eb6dff3179fc52e3f6943aa5e3`; utils SHA256 `29df3b9cf82ae5885ece0fac010b81383f1fb5391068649c88b7d2bc98ad86be`. Original BEBlock is AST-extracted unchanged from pinned native models SHA256 `07a6c1c452486802713a1a040ab24f9e9f8504660d731eb5b6417e2357f0f303`.

Two existing affine boundaries are replaced, with no additional affine transforms:

1. `encoder.convs[0].lin` and its existing post-propagation GCN bias.
2. The existing last `nn.Linear` in `decoder.lin`, from hidden features to one link logit.

With feature width F=128, hidden width H=64 or 128 and M=1 or4 routes, first fast tensors are `R[M,F]`, `S[M,H]`, `B[M,H]`; terminal tensors are `R[M,H]`, `S[M,1]`, `B[M,1]`. Both shared W tensors retain the existing map shapes. Source BEBlock initializes W with Xavier, first R with Rademacher signs, terminal R and all S to one, and B to zero. NCN/NCNC interior encoder/decoder maps, nonlinearities, normalization, beta and completion parameters remain shared source modules.

The first route computes native normalized GCN aggregation of `(X⊙R)W⊙S`, then adds B. The GCN's former shared bias is removed. B stays **after** propagation, as in native GCN's affine law. Original `GCNConv.forward`, SparseTensor normalization, self-loop handling and source edge/feature/dropout calls are retained. The source GCN API is compatible, so no adjacency conversion or new input projection is introduced. S commutes with the featurewise graph aggregation; B does not move through it. Terminal BEBlock uses its original forward verbatim.

If P is the native ordinary parameter count, a boundary body has shared/nonmember parameters `P−H−1` plus fast parameters `M(F+3H+2)`. The factory records its actual total parameter count. Shared4 owns one graph encoder/decoder; independent controls own four separate bodies and optimizer/state allocations. Source constructor initializes the native body before its two replaced maps are reconstructed with original BE initialization; the constructor RNG sequence is explicit, with no paired slow-W identity claim across different route counts.

This is a link-task adaptation of original boundary parameters and supervision. It does not reproduce the published node TABMModel task/API, historical node training horizon or original AdamW optimizer. Native two-group Adam, with source encoder and predictor learning rates and decay 0, is used consistently for all new link controls.

## Training, serving and state

All methods retain the exact prepared TRAIN-only graph, native record masking, original PyG default negative sampler, native floor batching and recursive NCNC source decoder. Once per epoch, the sampler and TRAIN permutation draw from the fit stream. Each minibatch freezes its support, positive records and negative records across the shared routes. For each member, the complete native encoder, positive decoder and negative decoder run sequentially. Its balanced BCE sum is divided by M and backpropagated immediately. One native Adam step follows the accumulated gradients. The same single Adam object created by the native bridge is retained; parameter groups are rebound before any update.

Per-body initialization and sampler/permutation seed is `seed+1000003×body`. Private persistent member dropout streams start at `initialization_seed+3000007+1000003×route`. Each stream spans its encoder and both source decoder calls, including recursive completion, and persists across epochs. Group and private streams, Python/NumPy/Torch CPU/CUDA RNG, model, Adam and SELECT state are restored. These explicit private-stream choices are standardized RNG adaptations; unchanged author recipe replication is unclaimed.

Shared4 SELECT and serving use mean raw binary logits from four routes. Genuine I4 uses the same raw-score mean across four own-selected bodies. M1 reuses the prospectively fixed body0 of its corresponding I4 bank. The original boundary is a single selected shared group with four internal routes; its exported row stores their raw-logit mean. There is no class-softmax pool, calibration or SELECT-as-CAL reuse.

Every fit binds `members`, `source_optimizer_count=1`, `train_minibatches_per_epoch=17`, source graph model receipts and explicit initialization/shuffle/dropout seeds. TRAIN contains 1,179,052 official temporal records; native batch 65,536 pays 17 updates, processes 1,114,112 positive records and drops 64,940 tail records per epoch. No support, tail or TEST-source protocol repair is introduced. Cost receipts distinguish actual member encoder calls, top-level decoder passes, recursive NCNC calls, backwards, shared optimizer steps, sampled/unused negatives and partial failures.

## Finite source-anchored study

NCN and NCNC each search ordinary M1, boundary equipped M1 and original shared4 independently on the same six points: source hidden64/128 × joint learning-rate multiplier 0.5/1/2 on encoder 0.0082 and predictor 0.0037. All other effective source constructor/optimizer flags remain at the native anchor: one GCN layer, encoder dropout 0.1, predictor dropout 0.3, encoder edge dropout 0.25 and feature dropout 0.25, source normalization/residual/decoder/completion settings, decay 0. The unused native predictor depth argument and nonpure GCN tail-dropout behavior are inherited transparently.

Two search seeds 73101/73103 and three comparative seeds 73201/73203/73207 are fixed. Every paid fit completes all 100 source epochs, with strict first SELECT Hits@50 each epoch, no checkpoint0, early stop, continuation or extension.

| Declared work | Count |
| --- | ---: |
| Search / comparative model fits |72 /54|
| Model fits, optimizer objects, slow bodies |126 each|
| Full epochs / model optimizer updates |12,600 /214,200|
| Member backwards / native TRAIN encoder traversals |367,200 each|
| Top-level TRAIN decoder query traversals |734,400 plus native NCNC recursion|
| Search selections |6|
| Served banks / exports / evaluations |30 each|
| Genuine I4 banks / fixed body0 reuses |12 each|
| Original shared4 served banks |6|
| CAL fits |0|
| Complete run records |192|

These compute declarations cover TRAIN only; SELECT/export construction and inference costs are separately charged. Recursive NCNC decoder calls are retained from actual source execution counts, never inferred as two top-level calls.

## Custody, qualification and remaining gates

Compiler metadata reuses actual official GRAPH/TRAIN/SELECT and sealed query/target receipts by hashes. It decodes no role tensors. The original TEST targets stay separate from label-free shuffled opaque query IDs. New endpoint code retains the earlier source query coverage, source fit ancestry, selected restore, full TRAIN-only support, whole-roster gates and target/evaluator checks. No source TEST protocol or old link study is changed.

Five final CPU tests cover finite counts/reference matching and guard-before-read; forbidden TEST fit inputs; original BE initialization; no added maps; exact native GCN normalization/self-loop/flow flags and unit-factor/source-forward equality; nonzero post-propagation B placement; sequential member gradients/costs; persistent private RNG/model/Adam selected replay; complete native query tails and exact raw-score pool; disjoint genuine I4 state and partial failure accounting. They use synthetic roles only. Numerical QA is not a scientific result or CUDA resource qualification.

Resources are explicitly provisional and excluded until engineering qualification: 32 GiB fit/export reservations, 512 MiB endpoint declarations, 78 GiB sole allocation scheduling capacity, and unmeasured timing placeholders. CUDA sparse kernels, optimizer/serialization/endpoint peaks and concurrent headroom remain unqualified. Root combined-master closures/rendezvous and explicit human GO remain required. The packet closes preparation of the mandatory adapted original GNNM link reference; it introduces no novelty or author-replication claim.
