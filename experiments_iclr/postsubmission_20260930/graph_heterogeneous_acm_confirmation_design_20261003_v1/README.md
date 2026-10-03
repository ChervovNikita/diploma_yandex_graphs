# Prospective ACM confirmation design

2026-10-03. **Design and incomplete freeze template; no execution admission.** Prepared before DBLP outcomes were inspected. This packet reuses the existing DBLP→ACM design review and the qualified HGT/native source. It contains no fitted outcomes, new training, or heldout label access. Root has frozen the five split descriptors and owns source adoption, execution admission, and later heldout release.

## Released data and remaining freeze work

The official HGB README at commit `ca6fd5bb0c1ca32e63b132c8bfe8f11a4a6629fe` announces the public benchmark including test data on 2023-03-02 and links the [Node Classification release folder](https://drive.google.com/drive/folders/10-pf2ADCjq_kpJKFHHLHxr_czNNCJ3aX?usp=sharing). Root located `ACM.zip`, file ID `1xbJ4QE9pcDJOcALv7dYhHDCPITX2Iddz`, and acquired it in the authorized repository. The saved common-loader Tsinghua URL is legacy metadata; its HTML response was not an archive receipt.

The measured source receipt is [ORIGINAL_REMOTE_ACQUISITION_SCHEMA.json](../hgb_acm_confirmation_root_v1/ORIGINAL_REMOTE_ACQUISITION_SCHEMA.json). Archive SHA256 is `787766fef7526310321b8ac94eb220209c0876a536f6603640d812220ca62134`, size 4,131,275 bytes. Root opened `node.dat`, `link.dat`, and the authorized source development `label.dat`; the heldout label payload was not opened. This agent read the metadata receipt only.

| Released type | Meaning / target | Nodes | Global offset | Provided feature width | HGT feature type 0 width | GAT / Simple-HGN feature type 2 width |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | P: paper, supervised target | 3,025 | 0 | 1,902 | 1,902 | 1,902 |
| 1 | A: author | 5,959 | 3,025 | 1,902 | 1,902 | 5,959, identity |
| 2 | C: conference | 56 | 8,984 | 1,902 | 1,902 | 56, identity |
| 3 | K: field/term | 1,902 | 9,040 | absent | 1,902, identity | 1,902, identity |

Total: 10,942 nodes. These measurements agree with the saved SeHGNN ACM source comments; the actual archive receipt governs the freeze. The target has three single-label classes, IDs 0–2, established by the source development pool. No semantic class names are inferred.

| Raw relation | Released direction | Records / distinct directed pairs | Released self records |
| --- | --- | ---: | ---: |
| 0 | P → P (`PP`) | 5,343 | 8 |
| 1 | P → P (`PP_r`) | 5,343 | 8 |
| 2 | P → A | 9,949 | 0 |
| 3 | A → P | 9,949 | 0 |
| 4 | P → C | 3,025 | 0 |
| 5 | C → P | 3,025 | 0 |
| 6 | P → K | 255,619 | 0 |
| 7 | K → P | 255,619 | 0 |

All 547,872 records have unit weight and no duplicates within a raw relation. Cross-relation ordered-pair overlap is a separate native Simple-HGN audit; it is not zero merely because within-relation duplicates are zero.

The released development pool has 907 papers: class counts 280 / 293 / 334. For each seed **131, 137, 139, 149, 151**, apply a private `NumPy RandomState(seed)` to the same sorted source development IDs, take the first `floor(0.2 × 907) = 181` for validation, and sort each partition; the remaining 726 are TRAIN. Root writes each descriptor once, binds its path/hash/size, and verifies disjointness, exact pool coverage, the seed algorithm, and TRAIN coverage of all three classes. All HGT and native arms consume those identical descriptors. These are five paired split/seed blocks on one transductive graph.

Root subsequently froze all five splits using the original pure development reader and split verifier. [SPLIT_BINDINGS.json](../hgb_acm_confirmation_root_v1/SPLIT_BINDINGS.json) binds the canonical TRAIN_VAL_ONLY development descriptor (SHA256 `be809e6b861e1aedb57411d133dbd0fe85f6cbf4f0cf2efb2749083706dd0d00`, 15,398 bytes) and all five split descriptors. Every split has 726 TRAIN / 181 validation and all three TRAIN classes. The metadata receipt records no heldout label payload opening, model execution, or DBLP outcome read; label values were not copied locally.

The template binds measured archive, node/link members, canonical development descriptor, and all five verified split descriptors. It deliberately leaves the fresh ACM source release, resource admission, native channel/audit bindings, and final-evaluation implementation unset. It cannot be used as a completed freeze. Do not fill them from source comments, guessed counts, DBLP outcomes, or heldout labels.

## Matched HGT family

Use the literal author command `python train_hgt.py --feats-type 0 --dataset ACM`: width 64, eight heads, **two layers**, **normalization false**, actual HGT-layer dropout 0.2. The saved paper prescription uses normalization true; this design chooses the literal executable source recipe prospectively and declares that discrepancy. Apply the choice identically to all seven HGT arms. The CLI dropout 0.5 and LR 0.005 do not override the actual layer dropout and optimizer/schedule in the author source.

Preserve the qualified AdamW constructor defaults with weight decay 1e-4, OneCycle maximum LR 1e-3 / 300 total steps / `pct_start=0.05` / `step(epoch+1)`, 300 maximum updates, patience 30, post-update raw mean-logit validation CE, latest tied minimum, and complete-state selected replay. Epoch 0 is an initialization diagnostic only, with a served-prediction fingerprint and validation metrics; it is ineligible for checkpoint selection. Preserve the qualified paired core/factor/member RNG rules, four private complete trajectories, shared-control initialization, untied initialization, and wider-BE count-hook policy. Record the ACM wider width and parameter counts from the bound actual schema before score opening; DBLP counts are not ACM counts.

Freeze this arm order: `native_HGT`, `global_BE`, `shared_relation`, `CP`, `unrestricted`, `untied_HGT`, `wider_BE`. All **35** terminals are required. Every HGT arm retains all four node types and all eight separately named raw relations, including released self records. Unique names `raw{raw_id}__{source_type}__{target_type}` repair the author endpoint-only dictionary collision between raw 0 and raw 1. Add no synthetic reverse relations or self edges. Freeze the canonical row order shown in the JSON template.

The served predictor remains FP32 `raw.mean(0)[ids]`. Keep any FP64 ambiguity diagnostic and temperature fitting separate; they do not replace the served aggregation arithmetic. Classifier sizing and macro-F1 use the frozen TRAIN class schema, never `labels.max()` over heldout data.

For **each** primary control, `global_BE` and `shared_relation`, use CP-minus-control paired differences and require:

1. All 35 frozen terminals valid under the input, selection, replay, and cost receipts.
2. Mean selected raw NLL difference ≤ **−0.005 nats**.
3. Strict raw NLL wins in **at least four of five** paired seeds.
4. Mean paired fixed-three-class macro-F1 difference **≥ 0**.

Use the same practical gates separately for DBLP and ACM; no cross-graph pooling or threshold adjustment after outcomes. The ACM development gate is a graph continuation description after validation selection. Independent final evaluation remains a separate frozen release and must repeat the two primary gates on each graph. For a final probabilistic-utility claim, calibrated final NLL must also retain the 0.005-nat mean gain against both controls on each graph. Failures, deferrals, incomplete denominators, or near-ties are retained and do not pass. Report every CP-minus-control vector, mean, SD/SE, signs, illustrative t95 interval, and leave-one-block-out means. Intervals have no significance or pass role.

## Required native challengers: 15 additional terminals

Use the same five paired splits and source labels. Native recipes are competence checks, not published-score substitutions or promises of an independently tuned optimum. Complete native coverage is required for a broad benchmark claim; an HGT-family pass alone supports only the matched HGT-family result.

| Native arm | ACM features / topology | Architecture and training |
| --- | --- | --- |
| GAT | `--dataset ACM --model-type gat --feats-type 2`; paper attributes, other types identity; full undirected support union and one graph self loop per node | Two hidden stages plus class output: three convolutions, heads [8,8,1], 64 hidden channels per head, biased typed input maps, native bias-free message maps, ELU, slope .05, feature/attention dropout .5, no feature residual or final logit L2 normalization. Adam LR 5e-4 / decay 1e-4; 300 epochs / patience 30; post-update validation CE, latest tied minimum. |
| Simple-HGN | `--dataset ACM --feats-type 2` via `baseline/run_new.py`; same feature and graph support policy; literal author edge-type dictionary | Same stages/heads/width/dropout; edge embedding 64; detached previous attention with alpha .05 in hidden stage 2, reset at output; feature residuals in hidden stage 2 and output; retain final class-logit L2 normalization with clamp 1e-12. Same Adam/epoch/patience/tie recipe as GAT. |
| SeHGNN | Native P/A/C graph after PP union/coalescing/diagonal; omit K by `ACM_keep_F=False`; all provided P/A/C attributes; four feature hops and four TRAIN-label hops | Hidden/embed 512, two feature-projection layers, one task layer, **residual false**, dropout/input-drop .5; transformer one head, attention dropout 0, activation `none`; Adam LR 1e-3 / decay 0; 200 epochs; TRAIN batch 10,000; literal `epoch-best_epoch > 50`; strict validation improvement, earliest tied minimum. CUDA TRAIN AMP/GradScaler if admitted; CPU disabled and inference without autocast. |

### GAT / Simple-HGN graph details

Start from all raw directed support. Form `adjM + adjM.T`, remove graph self loops, then add one graph self loop per node, in the existing qualified edge-order convention. GAT discards raw relation labels. For Simple-HGN, reproduce `edge2type`: assign raw types in the verified author relation iteration order; the last raw relation wins on the same ordered pair; assign a synthetic self type only if that ordered self pair has no raw label; assign a synthetic reverse type only if the reverse pair has no raw label. With eight raw types, self type is 8, synthetic reverse type is `raw+9`, and the embedding has **17** types. Released raw self labels survive in `edge2type` even though graph self loops are removed and re-added.

Raw PP/PP_r self records guarantee at least eight collision pairs. The DBLP port's `collisions==0` guard therefore cannot be carried over. Before admission, bind the raw encounter/iteration order, full cross-relation overlap count, overwritten labels, self labels, and native homogeneous support count. Preserve and disclose the literal source overwrite rule; do not silently convert Simple-HGN into the HGT relation-preserving graph. These are preprocessing audit fields, not extra benchmark runs.

### SeHGNN preprocessing, exactly

The saved author loader treats the raw file row as **destination**, column as source, unlike HGT's raw direction convention. Local operators before normalization are raw 0→`PP`, 1→`PP_r`, 2→`PA`, 3→`AP`, 4→`PC`, 5→`CP`, 6→`PK`, 7→`KP`. Verify the source attribute/graph identities before normalization: nonzeros of P match PK, `AP @ PK == A`, `CP @ PK == C`, and PA/AP, PC/CP, PK/KP have the author transpose support correspondence.

Merge `PP` and `PP_r` as binary support, coalesce, and set the diagonal. Then retain adjacency order `PP, PA, AP, PC, CP`, omitting K and its two relations according to the author ACM recipe. Row-normalize each retained support operator **after** this merge/diagonal step in FP32. The P/A/C initial features are the released attributes. CPU native mean feature propagation prepends the receiving node-type letter, retains all paper-destination paths through four hops, and retains the zero-hop `P` feature. This symbolic schema yields **41 feature channels** (1 / 3 / 5 / 11 / 21 at hops 0–4), each width 1,902.

For label propagation, seed a 3-column target one-hot tensor **only at TRAIN IDs** for that seed. Left-extend the same row-normalized operators through four hops, retaining paper-to-paper products. This yields **20 label channels** (1 / 3 / 5 / 11 at hops 1–4). Remove the diagonal from **each complete metapath product**, then multiply by TRAIN one-hot labels. Do not remove each base operator's diagonal or renormalize the completed product after diagonal removal. Rebuild the label channel cache for each split. Never seed validation or heldout labels.

Freeze exact feature and label key/order records and propagated-cache fingerprints in the fresh ACM preparation. Reuse the accepted native port's initialization/stack ordering; declare it and keep it fixed, rather than introducing a new order from results. The symbolic counts above are source expectations, not a completed cache qualification.

The final class BatchNorm has `affine=False, track_running_stats=False`. Freeze one 3,025-paper inference batch in the order sorted TRAIN, sorted validation, sorted remaining topology-derived target IDs. Extract validation metrics only. This preserves the existing label-safe whole-target evaluation policy; verify released membership coverage separately before claiming identity to the literal author's labeled-batch composition. Do not evaluate validation or heldout nodes in a separate batch after selection. Remove test-label copying, author `check_acc` (which computes test metrics even with `show_test=False`), and all per-epoch/final test diagnostics before heldout release.

## Calibration and final evaluation

Freeze a shared scalar-temperature implementation before DBLP score opening. Use selected served FP32 mean logits, converted to FP64 only for fitting; fit on that seed's validation labels after selection. The template proposes bounded `log(T)` minimization of mean CE with T in [0.05,20], a deterministic solver/tolerance/limit, and a T=1 fallback if the fit is invalid or fails to improve its validation objective. Apply the saved temperature to the separately released heldout predictor. Preserve raw and calibrated NLL and multiclass Brier score; report 15 equal-width confidence-bin ECE as auxiliary. Validation temperature fitting does not create an independent calibrated development estimate.

Freeze exact final membership, evaluator source, temperature solver, selected-state replay, no-retraining/unchanged TRAIN-label-cache policy, and one-time label opening before final evaluation. This design does not authorize that opening. A broad claim requires all 35 HGT and 15 native terminals per graph, independently valid final scoring, and disclosure of native preprocessing/selection differences. GPU peaks from a family built together remain process receipts; isolated per-arm memory efficiency is not established.

## Minimal driver generalization

Create fresh ACM preparation files, leaving sealed DBLP packets intact. Preserve the model math, optimizer, scheduler portability, objective, served metric arithmetic, stopping/selection, replay, and RNG functions. Only the following adaptations are needed:

| Existing site | Necessary ACM change |
| --- | --- |
| HGT `dblp_inputs.stream_schema/materialize` | Retain all provided type attributes; parameterize feature type 0 versus 2 and target schema; materialize absent K as sparse identity; keep per-raw relation support and unique names. Existing descriptor verification and split verification can be reused. |
| HGT `families.build` | Accept the bound recipe's layers and normalization instead of hardcoded 3/true, including factor depth, untied cores, and wider count-hook arguments. Width/head/RNG/factor/control policies stay fixed. |
| HGT driver guards / records | Bind ACM archive/schema/dev pool/classes, all seven arms, and 726/181 split descriptors; bind fresh source release. `fit` and `metrics` already accept model-sized classes and target type 0. Add only the prospective ineligible epoch-0 diagnostic around the preserved fit if adopted. |
| Native inputs | Parameterize type letters/endpoint schemas/classes; add source ACM PP union/diagonal and four-hop P/A/C feature/label construction; add declared literal Simple-HGN collision policy with an audit, instead of the DBLP no-collision guard. |
| Native `HGBGAT` call sites | The existing model already accepts a feature list. Pass `[paper_attributes, identity_A, identity_C, identity_K]` for ACM in TRAIN, validation, and replay; bind input widths and 17 edge types. Do not take its DBLP `model(graph)` identity shortcut. |
| Native SeHGNN construction / metrics | Supply three classes, one task layer, residual false, P target, and bound channels; replace hardcoded class-4 metric/reporting guards and DBLP sample counts. The accepted SeHGNN numerical blocks need no rewrite. |
| Native evaluation / records | Freeze 3,025-target batch IDs and cache/source fingerprints; preserve native stopping/RNG/AMP/selection/replay. Keep heldout labels closed. |

Reuse provenance: HGT preparation v2 manifest `c94153106d6df85f1d825bf834f190d2808350d3dea4a16892c0294c3b4dc633`; `hgt_private.py` SHA256 `97a401fd289bbea9a21ae98d3b488ecea91a69547a2ec71d9d7392adb0195042`; native preparation v2 manifest `5956ed69b1c9646aedda0ce67d950081b23ca6ca77796ac7f98c4b900544af43`. Scientific gates/claim boundary come from [the existing review](../heterogeneous_dblp_acm_design_review_20261003_v1/REPORT.md); author recipes and source sites come from [the existing source packet](../graph_heterogeneous_private_modulation_source_20261003_v1/BASELINE_RECIPES.json), especially HGT `run_acm.sh`/`train_hgt.py` and SeHGNN `hgb/utils.py:372–427`, `main.py`, and `model.py`. No retained papers were reread.
