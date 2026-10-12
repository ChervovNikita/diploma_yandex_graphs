# Massive study: current roster and comparison design

13 October 2026. Preparation snapshot, not a final approved run list. No new scientific fits are authorized before the user verifies the complete proposal and gives GO.

## Counts

- 12 distinct datasets, used in 13 dataset/task settings. CiteSeer appears in both node classification and the HeaRT link-prediction setting.
- 16 named backbone architectures. Source-specific configurations are not extra architectures.
- Coverage catalogue: 273 canonical entries in 73 families, including 235 scientific variants. Other entries cover references, controls, diagnostics and a proposal. These are coverage targets; final executable coverage is not yet frozen.
- Current packet union: 92 study specifications and 79,485 model-fit groups. The independently read pre-typed-context union had 91 studies and 79,440 groups; the sealed typed-context packet adds one study and 45 groups. This sum remains provisional pending final compilation, dependency reuse, coverage and resource checks.
- Registered method IDs also include controls and alternate readouts. They must not be presented as distinct scientific methods.

## Datasets

| Task | Datasets |
| --- | --- |
| Node classification | CiteSeer, PubMed, WikiCS, Amazon Photo, Coauthor CS, Coauthor Physics, Twitch-DE, HGB-DBLP, HGB-IMDB |
| Link prediction | ogbl-collab, CiteSeer-HeaRT |
| Graph classification | ogbg-molhiv, ogbg-molbace |

HGB-IMDB uses multilabel node classification. Amazon Photo also has a separately declared binary adaptation. Case aliases and aggregate author-roster identifiers do not increase dataset counts.

## Backbones

GCN, GAT, GraphSAGE, Graph Transformer, PolyFormer, Polynormer, UniMP, GINE, NSD, BSNN, HGT, SeHGNN, FiLM, NCN, NCNC and BUDDY.

## Comparison design

1. Test implemented methods only on compatible tasks and backbones. This is not the Cartesian product of every method, dataset and architecture.
2. Include a capable single model, original GNNM on compatible new data, and a genuine four-member ensemble with independent training and checkpoint selection. Add same-information/capacity or mechanism controls where required for attribution.
3. Search finite source-specific configuration lists on TRAIN/SELECT using two search seeds, retaining native defaults. Search receives the declared full training policy rather than a short pilot budget. Fixed source recipes do not acquire artificial hyperparameter axes.
4. Freeze selected configurations before calibration or heldout scoring. Comparative fits use three fresh paired seeds and the prepared split panel, usually ten masks where available; source-constrained studies declare their narrower panels explicitly.
5. Preserve complete training curves, selected and terminal states, stopping/cap diagnostics, failures and all declared outcomes. Distinguish independently trained members from jointly optimized four-body models, and distinguish optimizer calls from functional private updates.
6. Keep author-recipe replication separate from standardized comparisons. Pin source, data and recipe; label adaptations. Do not rerun Roman Empire, Amazon Ratings, Minesweeper, Questions or Tolokers for any method.
7. Prepare distribution across the allocation GPU and the two 18.77 GPUs, with at most two concurrent fits per GPU when measured memory permits. Large cases serialize. The current authorization is allocation-only preparation; 18.77 runtime admission and scientific launch wait for later authorization.

## Remaining before the final proposal

Close the all-method coverage join; compose the final controller and plan; complete resource/storage admission, worker assignments and source audit; produce the exact stage/configuration counts and transfer bundle. The preparation snapshot is not evidence of predictive improvement or novelty.

## Count sources

- `preparation/controller_v19_dispatch_correction_root_20261012_v1/PROVISIONAL_ROSTER_COUNTS_QUESTION.json`
- `docs/METHOD_INVENTORY.csv`
- `../../unresolved_source_ports_20261012_v1/typed_context_ready_v1/matrix/SUMMARY.json`

The typed-context objective-trace clarification may produce an immutable successor; it does not authorize a training change or new fit.
