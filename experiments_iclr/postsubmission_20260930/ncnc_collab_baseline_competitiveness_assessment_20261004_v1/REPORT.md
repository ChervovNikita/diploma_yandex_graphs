# NCNC Collab baseline competitiveness assessment

## Decision

The completed **N64 baseline matches the saved official NCNC Collab core recipe**: hidden width 64, one GCN layer, completion depth 1, decoder default 3 and 100 epochs per fit [A1–A3]. The official README command uses ten runs; the completed baseline reports five seed blocks. This reduces the replication denominator, but does not reduce each fit's capacity or training budget. A width 256/depth 3 run would be a new recipe, not restoration of this pinned Collab command.

Here “official” means the released command at GraphPKU commit `11d597013750da17ce7468e344bec756a7af39a4` and its inspected defaults. It does not certify that every published figure used exactly those flags. Recipe and architecture identity are well supported; **reproduction of the published numerical performance is not certified**. The retained evidence has no verified NCNC published result table or historical run states. No author numerical score is imported from memory.

The conditional request for a stronger baseline because NCNC was materially reduced is therefore **not triggered**. The completed I4 independent ensemble and N70 capacity single are already useful within-family controls. Broader competitiveness against recent structural predictors remains unverified, so this packet names one concrete future comparison candidate without adopting an experiment or claiming an empirical winner.

## Official recipe versus frozen execution

| Dimension | Saved author Collab recipe | Completed N64 | Assessment |
| --- | --- | --- | --- |
| Architecture | GCN + `incn1cn1` | Same native NCNC binding | Core architecture matches |
| Width / depths |64 / GCN 1 / completion 1 / decoder default 3 | Same | No capacity reduction; NCNC2 depth 2 is separate |
| Training |100 epochs; batch 65536; floor batching |100 epochs; 1700 updates; 17 full batches/epoch | Same per-fit budget |
| Runs |10, native seeds 0–9 |5 N64 blocks 0–4 | Fewer replications; same fit recipe |
| Features / nodes | Provided OGB features; no global learned node-ID table; all nodes |128 raw features; 235868 nodes | Same input type and coverage |
| TRAIN graph | Complete official TRAIN; unweighted symmetric; no year/LCC filter |1179052 records; same graph policy | Match |
| Training target masking | Remove whole positive record minibatch | Same; surviving duplicate records retained | Match |
| VALID / TEST graph | TRAIN / TRAIN+VALID | Same; TEST records never added; no evaluation query removal | Native VAL-at-TEST parity |
| Negative sampling | Global PyG defaults, TRAIN/self exclusion | Once per epoch; duplicate-preserving reciprocal TRAIN; sparse; `force_undirected=False`; no future-positive rejection | Default semantics match; historical realized draws/count remain unobserved |
| Tail exposure | Drop incomplete shuffled positive minibatch |64,940 records dropped/epoch | Native exposure retained |
| Optimizer | Adam, encoder .0082 / decoder .0037, weight decay 0 | Same; no schedule/clipping | Match |
| Regularization / completion | xdp .25, tdp .05, gnndp .1, gnnedp .25, predp .3, preedp 0; scale 2.5, offset 6, alpha 1.05, pt .1; LN/LNNN/res/xlin/tailact | Same; fixed pt; no degree caps | Match |
| Metric / coverage | Official shared-pool Hits@50; eval batch 131072 | All official positives and 100,000 negatives; strict score greater than 50th negative; ties fail | Match |
| Checkpoint selection | Strict VALID Hits50 improvement, native multi-metric bookkeeping | Strict served Hits50; first tie retained; epoch 1 always persisted; atomic state | Disclosed persistence/bookkeeping adaptation |
| TEST visibility | Per-epoch TEST diagnostics | One locked all 25 TEST evaluation | Disclosed evaluation governance adaptation |
| Runtime / RNG | Torch 1.13 era, PyG 2.2, CUDA 11.7; historical environment has discrepancies | Python 3.12, Torch 2.7.1, PyG 2.4, CUDA 12.6; Python seed added | Contemporary execution; no historical numerical equivalence claim |

Full machine-readable mapping is in `RECIPE_COMPARISON.csv`. Frozen ordinary training is float32, no autocast/TF32, deterministic algorithms False; it does not promise bitwise seed reproducibility. The heldout scorer used its separately qualified strict True2 profile. The saved all 25 adoption, rather than the earlier disabled preparation policy, establishes actual loaded-state/graph/query custody [A5–A7].

N64 is the individually VALID-selected member 0 of I4, not whichever member 0 happened to occur in the selected ensemble bank. I4 trains four separate encoders/decoders with independent optimizer/RNG streams. It admits100 synchronized banks plus candidate 101 assembled from individually selected member states, with the extra VALID work charged. It is a prospectively stronger comparison but has a different selection budget from a single. N70 uses the same100 epoch recipe at width 70 and is a parameter-capacity control; it is not a compute match or the best tuned structural single [A2,A3].

## What the completed results establish

These values are copied from the adopted saved summary; no scores, contrasts or selections were recalculated [A5].

| Model | TEST Hits@50 mean (%) | Seed SD (pp) |
| --- | ---: | ---: |
| N64 |66.4426|0.6233|
| I4 |67.6298|0.2267|
| Private F4 |67.2909|0.7676|
| Pooled F4 |67.0673|0.2595|
| N70 |66.7642|0.7862|

The saved report says the frozen private-versus-pooled primary contrast was not robustly confirmed. It does not establish an architectural advantage, superiority over I4, or field leadership. These are five optimizer-seed blocks on one graph/time split, not uncertainty over independent datasets. The current TEST is consumed; this assessment cannot turn a later candidate into a fresh heldout confirmation.

Calling the current N64 “underpowered NCNC” solely because width 64 looks small would misdescribe the source evidence. Calling its66.4426% a reproduced published result would also exceed the evidence. The appropriate statement is: **a completed contemporary execution of the pinned official core Collab recipe, with disclosed replication, runtime/RNG and checkpoint/visibility adaptations**.

## Recent comparison context

The literature memory was reused before any new acquisition; there was no acquisition [A8,A9]. In index_v45, zero-based records 80 and 135 both refer to arXiv:2602.01553v4: record 135 is a retained changed-question scope extension of the identity already at record 80. The saved PENCIL author-source/config packet [A1] is reused separately. This audit adds zero paper identities, zero index records, zero primary reads and zero raw author-source reads; it edits no index. The following are method/protocol conclusions, not verified rankings or current outcome audits.

| Reference | Why it matters | Comparison boundary |
| --- | --- | --- |
| BUDDY/ELPH [P2] | Sketch-derived pair structure and endpoint features; competent independent predictors can reuse the same deterministic cache | Retained active context is width 1024, complete weighted TRAIN and TRAIN-only TEST. NCNC uses unweighted TRAIN+VALID at TEST. Current completed BUDDY outcomes were not inspected here; a direct score table would need these graph-access/weight differences stated. |
| LPFormer [P3] | Pair-conditioned context attention, learned PPR-relative encoding and structural counts provide a capable structural reference | Retained scoped method evidence only; saved author-code and performance qualification are absent. |
| PENCIL [P4] | Recent2026 sampled-subgraph transformer with adjacency propagation; source-pinned Collab config available | Published superiority, runtime and full launcher/data pipeline are unqualified. Native year ≥2007 filtering, half-positive epoch exposure and optional features are material. |
| Link-MoE [P5] | Pair-specific supervised expert mixture gives a relevant stronger combination reference | Collab gate training uses 80% of official VALID and selects on 20%; this adds supervision and cannot be silently equated with VALID-only selection. |
| HeaRT protocols | Harder query/negative construction can test another deployment question | Different negatives/query/metric setting; scores are not interchangeable with the official shared100,000-negative Hits@50 setting. |

### One concrete future broader comparison

Use **PENCIL's author-supported feature-enabled Collab BERT recipe**: commit `2d32e29dbed533288d9d758138e07547a0a7d8a9`, `args/official/ogbl_collab_bert.yaml` (SHA256 `e488a5273dd9545b0adff0e9b665ee7e672a2da891daf40ac6c1f8856177de08`) plus `--use_features` with default early fusion [A1,P4]. This is a named author-supported variant; the YAML default is structure-only. Keep this one fixed recipe, with no tuning grid.

Its saved configuration is hidden 512, 8 layers, 8 heads, intermediate 2048, 20 epochs, AdamW lr 1e−4 / weight decay .01, bf16/DDP, batch 1024 per rank and total accumulation 8. It samples 75 one-hop neighbors with maximum 256 local nodes and exposes roughly 50% of positives per epoch. World size affects the actual effective batch and must be fixed. Native TRAIN filters year ≥2007, TEST uses filtered TRAIN+VALID, and target edges are removed after sampling. Official negatives/Hits50 must remain official; HeaRT is a separate setting.

This candidate addresses broader 2026 structural competence, **not a weakened-NCNC repair or a verified stronger empirical result**. Report its native protocol separately; any complete-TRAIN adaptation would need to be named and frozen before outcomes. Its launcher body, complete data/query custody and representative full-epoch resource behavior remain to be qualified. No new arm, seeds, allocation, fit or TEST release is authorized by this packet.

## Measured resource anchors

Existing frozen NCNC runs used an NVIDIA A100 80 GB PCIe under the contemporary runtime [A7]. The following are already measured work scales, not PENCIL runtime estimates [A4,A10].

| Saved unit | Measured inclusive100 epoch cost, seconds | Qualification allocator peak: allocated / reserved bytes |
| --- | --- | --- |
| N70 single, one fit | Seed 0–4:1274.8710,1305.8969,1273.9873,1262.1085,966.5810 |8662047744 /19862126592, one complete engineering TRAIN epoch plus full VALID |
| I4 bank, four fits plus candidate 101 work | Seed 0–4:4140.9087,4238.7062,4242.6149,4162.7428,4115.9263 |8471838720 /18406703104, four sequential complete engineering TRAIN epochs plus VALID |

The complete cost rows include setup/reads/hashes/transfers/training/VALID/selection/journals/finalization, with zero failed or interrupted attempts, but explicitly exclude the unmeasured terminal-accounting write tail. N64 donor reuse is explicit; dividing I4 cost would not create a separately measured standalone N64 inclusive cost. Qualification allocator peaks are not full-fit maxima or total session/RSS capacity.

**PENCIL runtime and memory scale are unknown in these saved receipts.** Its transformer, sampled-query preparation and distributed training differ from NCNC, so NCNC seconds cannot supply a credible PENCIL ETA. A representative complete-epoch qualification with full query coverage is the necessary resource evidence before allocation. `RESOURCE_ANCHORS.json` preserves the exact copied measurements and their limits.

## Evidence and limits

This packet uses saved text/JSON metadata, frozen source read scopes and selected literature-memory records. It performs no server execution, model imports, training, data download, tensor/checkpoint/array access, score recalculation, heldout reselection or primary-paper retrieval. Truncated locator displays receive no full semantic-read credit; required claims were repaired by bounded direct projections. Hashing input bytes does not imply full semantic reading. No manuscript, literature index or global status was edited. This assessment is an authored audit, not an independent PASS.

Local evidence:

- [A1: saved native recipes](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/graph_link_competing_family_source_20261003_v1/NATIVE_RECIPES.json>) and [exact retained author scopes](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/graph_link_competing_family_source_20261003_v1/READ_SCOPE.json>).
- [A2: frozen plan](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/graph_ncNC_collab_predictive_pilot_design_20261003_v1/PILOT_PLAN.json>) and [A3: family source analysis](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/ncnc_base_complete_family_analysis_source_report_20261003_v1/REPORT.md>).
- [A4: saved cost metadata](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/ncnc_complete_family_saved_VALID_summary_20261004_v1/RESULTS.json>), [A5: adopted TEST summary](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/ncnc_frozen_all25_heldout_root_adoption_20261004_v1/RESULTS_SUMMARY.md>), [A6: frozen heldout policy](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/ncnc_frozen_all25_heldout_source_preparation_20261004_v2/PROTOCOL.json>) and [A7: runtime authority](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/graph_ncNC_predictive_runtime_authority_root_20261003_v1/RUNTIME_AUTHORITY.json>).
- [A8: retained comparison context](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/link_prediction_quality_gap_20261003_v1/REPORT.md:49>) and [A9: literature index](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/literature_memory/index_v45/LITERATURE_INDEX.json>), selected records 78, 79, 80, 89, 135 only.

Primary identities, reused without a new primary read:

- [P1: NCNC, 2302.00890v4](https://arxiv.org/abs/2302.00890v4); [official pinned Collab command](https://github.com/GraphPKU/NeuralCommonNeighbor/blob/11d597013750da17ce7468e344bec756a7af39a4/README.md#L50-L52).
- [P2: BUDDY/ELPH, 2209.15486v3](https://arxiv.org/abs/2209.15486v3).
- [P3: LPFormer, 2310.11009v4](https://arxiv.org/html/2310.11009v4).
- [P4: PENCIL, 2602.01553v4](https://arxiv.org/html/2602.01553v4), retained version date 2026-09-28; [pinned Collab config](https://github.com/quang-truong/pencil/blob/2d32e29dbed533288d9d758138e07547a0a7d8a9/args/official/ogbl_collab_bert.yaml).
- [P5: Link-MoE, 2402.08583v2](https://arxiv.org/html/2402.08583v2).

`CITATIONS.json`, `READ_SCOPES.json`, `LITERATURE_LOOKUP.json` and `INPUT_BINDINGS.json` preserve claim scopes, pins and byte identities. No new published numerical superiority or novelty claim is made.
