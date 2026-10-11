# Readable manuscript section draft

This folder prepares specific sections for a later substantive revision. It is not a standalone or delivery-ready paper, a completed confirmation, or a review milestone. No compilation or editor opening was performed.

## New text

- `introduction.tex`: replaces the historical introduction with a prediction-quality purpose before model details; weight savings remain secondary.
- `internal_sage_methods_results.tex`: compact internal-SAGE method, all five capable F arms, correct-alternative analysis and the failed mean NLL protection. Penn94/E slots are LaTeX comments only.
- `sections.tex`: suggested fragment order. It requires an existing manuscript preamble and the later integration of the remaining historical scope, related work, conclusion and appendix. The historical abstract is not copied into this section assembly.

## Preserved source

`original_gnnm_model.tex`, `original_five_dataset_benchmark.tex` and `original_member_paths_analysis.tex` are byte-identical contiguous extracts from the recovered historical `main.tex`. The five-dataset scores, heatmap reference, original selection/provenance statements and Roman analysis are unchanged. Historical `main.tex`, `appendix.tex` and `references.bib` were not edited.

## Exact existing references and dependencies

| Dependency | Source / status |
| --- | --- |
| Original architecture figure | `architecture_sharing_compact.png`, referenced at recovered `main.tex:41`; no local PNG found in the authorized project inventory. |
| Original five-dataset heatmap | `gnnm_vs_ens_heatmap.png`, referenced at recovered `main.tex:67`; the original reference is preserved, with no local PNG found. |
| Existing bibliography | `/Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/fresh_manuscript_review_preparation_20261011_v1/historical_round18_source/references.bib`; all newly used citation keys exist here. |
| Existing appendix | `/Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/fresh_manuscript_review_preparation_20261011_v1/historical_round18_source/appendix.tex`; retain it when integrating the draft. |
| Conference style and bibliography style | `iclr2027_conference.sty` and `iclr2027_conference.bst`; referenced by the historical main source, with no local files found. |

No binary asset was copied or regenerated. `DEPENDENCIES.json` records exact source paths/hashes, extraction spans and these dependency statuses. Pending study comments contain no numerical outcome or successful mechanism claim. No historical manuscript/canonical status, live scientific output or reviewer was changed or invoked.
