# Shorter appendix for a later complete V4

V3 remains sealed and unchanged. First retain its complete source, bibliography, figures and six evidence copies as `code_evidence/v3_archive/`; the following cuts apply only to a later V4 PDF. Move entire blocks verbatim, including captions, vectors, protocols and adverse outcomes. No score is recomputed or discarded. The supplement destinations below are proposed files, not completed migrations.

`H` means `historical_appendix.tex`; `F` means `sections/F_protocol_diagnostics_appendix.tex`; `E` means `sections/conditional_value_E_PE_appendix.tex`. Ranges are exact V3 source lines.

## Retain in the paper appendix

| Block | Content to retain |
|---|---|
| H 10–19 and 41–47; H 517–546 | Exact boundary factor initialization/bias, member loss/pooling, function inclusion, benchmark reconstruction/ties/seed/selector and historical custody limits. Keep a short native-operation map from H 21–39. |
| H 584–627 | Complete original absolute benchmark table, all methods/datasets/backbones and deviations. Main heatmap/resource/component tables and original scores stay unchanged. |
| H 338–356; H 374–405 | Roman matched protocol, full fixed-mask tied/untied vectors and late-checkpoint limitation; adverse/mixed GAT mask table. Move engineering attempt detail H 357–362. |
| H 427–504 | Compact OGB node and collaboration tables, including all budgets/seeds, late selection, later-budget choice, stronger Adamic–Adar, future-negative and temporal-feature limits. Condense surrounding prose, not results. |
| H 675–684 plus a short H 506–515 summary | Separate original training-step sweep from later A100 inference profiles; preserve units, decay difference, resident-graph/timer scopes and unidentified historical device. |
| F 4–10 and 14–19; all `F_absolute_results_appendix.tex` | Native internal-factor/bias/RNG/selector differences, five complete references, member/coverage figure, absolute scores and the failed I4 mean-NLL rule. |
| E 4–16, 20–51, 53–70 and 115–131 | Exact E/N operation and controls, raw/calibrated absolute outcomes, primary paired contrasts, frozen rules, adverse I4 proper losses and full cost table/overlap limitations. Seed vectors in long rule prose may move only with a visible compact rule/outcome summary. |
| `sharing_scope_appendix.tex` 4, 8, 10; `auxiliary_explorations_appendix.tex` 1–3 | One compact summary preserving sharing/preprocessing failures, higher matched-size latency, unresolved Cora repeats, and affinity/fixed-key negatives. |

## Move intact to code evidence

| Exact V3 block | Proposed destination | Keep visible in the paper |
|---|---|---|
| H 21–39 | `history/native_operations.tex` | Short backbone/concat/root/self-loop map and current-code versus missing launch-source distinction. |
| H 49–328 | `history/secondary_grid_and_optimizer.tex` | Six candidates per arm, twice the partial-family search, shared within-graph runtime, encountered splits, all negative extensions and unresolved Cora outcomes. |
| H 330–337, 357–372, 417–425 | `history/roman_sensitivities.tex` | No unseen depth rule; finite-budget repeat; untuned MC Dropout loses; source/numeric attempts remain archived. |
| H 506–515, 548–573, 629–669 | `history/profile_and_selection_details.tex` | No independent-mask inference; selection sensitivity can transfer labels; profiling scopes remain explicit. |
| H 687–801 | `history/matched_storage_complete.tex` | Both local gains, all latency/memory tradeoffs, post hoc placement, selected/default distinction and unresolved normalized-Cora discrepancy. |
| F 12, 22–28 | `native_F/complete_prediction_accounting.tex` | Figure/table already retain central comparisons; keep one sentence on weaker members, better pool and adverse calibrated NLL. All raw counts/repairs/cost vectors survive here. |
| E 72–112 | `native_E_PE/complete_contrasts_and_accounting.tex` | One packed setting, accuracy–NLL tradeoff, and the main competence/coverage paragraph. All five PE contrasts and all signed error-account rows remain intact. |

## Remove repetition from the PDF only

Replace H 1–8 with a two-sentence appendix guide. Remove the table-introduction repetition H 577–582. Merge F 2/12/22/26/28 with the retained figure, absolute table and new competence paragraph. Remove repeated study descriptions in `sharing_scope_appendix.tex` 2/6 once the protocol summary is retained. Merge E 87/133 with the main packed/scope paragraph. Every removed sentence remains in the V3 archive; material limitations listed above remain visible once in V4.

## Exact editing dependencies

- Replace main.tex 132 table references with the named code-evidence grid/extension locations; do the same in `sharing_scope_appendix.tex` 4. Replace main.tex 134 and `sharing_scope_appendix.tex` 8 matched-size table references. Retain short PDF anchors `app:validation-tuning`, `app:cora-preprocessing`, `app:optimizer-history`, `app:matched-budget` and `app:sharing-scope-summary`. H 728's interval reference moves with its supplement context. All callers are inventoried in `CUT_MAP.json`; do not leave a PDF reference pointing to a removed table.
- Preserve figure bytes `architecture_sharing_compact.png`, `gnnm_vs_ens_heatmap.png` and `F_member_coverage_v2.png`, all historical table/source labels in the archive, and the existing bibliography/citation carriers. Keep raw vector/selected-record custody; compact uploads must still disclose omitted full checkpoints. `DEPENDENCY_MAP.json` and `EVIDENCE_SOURCE_MANIFEST.json` must be updated for the later V4 destinations.
- Use fragments 1 and 2 in `MAIN_TEXT_FRAGMENTS.tex` instead of E main paragraph 24 plus repeated accounting explanation; retain E main paragraph 22 with uncertainty and adverse NLL/Brier. Fragment 3 can replace the E main operation narrative 2–4, while exact equations/controls stay in E appendix 4–16. It states an association question, not a completed intervention result.
- Completed common-rival/strata evidence comes from `WikiCS_E_frozen_strata_diagnostic_root_20261011_v1/{REPORT.md,MARGINAL_DESCRIPTIVE_COUNTS_V1.json}`; the closed head result comes from `WikiCS_E_fixed_feature_head_capacity_summary_root_20261011_v1/{INTERPRETATION.txt,SUMMARY.json}`. Add one scoped sentence on that failed anchored affine probe, with no absent-information conclusion. The source-context dependency is `WikiCS_E_source_context_association_diagnostic_source_20261011_v1/PLAN.json` (SHA256 `8ccdb82b2291884705450e6bca992eba1a34175e1a958019b5c5b68d0d0c1092`). Add no outcome sentence before its future `COMPLETE_ASSOCIATION_DIAGNOSTIC.json` closes all 36 served banks, 108 native trajectories and 18 selected restores. Keep `PERMUTATION.npy`/receipt, all 36 forward archives, factual replay and original archived decision authority. Interpretive dependency: `native_SAGE_source_context_association_interpretive_guide_20261011_v1/GUIDE.md`; source/attribution remain owned by the other agents.
- A later reproducibility sentence may cite `portable_conditional_SAGE_source_20261011_v2/RUNTIME_CHECK_REPORT.md` and the actual root receipt. State successful existing-runtime one-update E4 binding only; clean installation/external/full-CLI verification and historical benchmark replay remain unverified. Physics/Penn outcomes enter V4 only from their eventual complete released evidence.

This is an editorial cut map and three draft fragments, not a manuscript review, publication verdict, V4 edit, new test or scientific execution.
