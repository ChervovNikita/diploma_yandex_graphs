# Saved-evidence dataset and experiment census

Scope: 30 September–11 October 2026. This is a conservative lower bound from the current saved ledger and completed records, not an exhaustive global count. No remote access, new numerical/model execution, arrays, partial predictive outcome reads or PDFs were used.

**At least 17 named real datasets have completed predictive training. At least 30 whole training experiment families completed; one additional molecular family reached terminal status with 17 successful fits and one failed control. The 31 listed families contain at least 1,061 completed training fit/bundle records.** There is also a separate completed graph-serving experiment with 855 CPU fusion fits. These are different units and should not be merged into a body-trajectory total.

Datasets: Reddit2, ogbn-products, AmazonPhoto, CoauthorCS, Squirrel, HGB-DBLP, HGB-ACM, Amazon-ratings, ogbl-collab, ogbl-ddi, CiteSeer, PubMed, WikiCS, ogbg-molhiv, HGB-IMDB, CoauthorPhysics, Penn94.

DDI is counted because two predictive arms physically completed, although its fixed12 family was not adopted complete in the latest saved record. MolHIV has17/18 successful fits. IMDB has a complete predictive readout but is omitted from the fit subtotal. Dataset repetitions, splits and seeds do not create new datasets.

An experiment family here means a distinct fixed comparison/cohort, grouping controls and all seeds together. The DBLP typed cohort and native references are grouped as one family (35+15fits), as are the Amazon native baselines and authored comparison (6+15fits). Backbone companions are grouped into one common-wrapper family (39+78fits). The molecular family is separately labelled terminal with failure. This conservative granularity avoids counting each arm or progress update as an experiment.

## Counted training subset

| Family | Dataset(s) | Completed fit/bundle records |
| --- | --- | ---: |
| Shared-message study | Reddit2; ogbn-products | 30 |
| Coordinate ensemble stage1 | AmazonPhoto; CoauthorCS | 54 |
| Typed HGT study plus native references | HGB-DBLP | 50 |
| Mixed shared/private objectives | HGB-DBLP; HGB-ACM | 40 |
| Full-support paired continuation | Squirrel | 12 |
| Native baseline and authored original comparison | Amazon-ratings | 21 |
| BUDDY paired study | ogbl-collab | 24 |
| NCNC family | ogbl-collab | 35 |
| Endpoint-frame study | CiteSeer | 36 |
| Graph initialization | CiteSeer | 21 |
| Growth followup | CiteSeer | 15 |
| Internal-factor molecular family (17/18; terminal failure) | ogbg-molhiv | 17 |
| Common wrapper across SAGE/GCN/GAT | WikiCS | 117 |
| Private feature rotation | WikiCS | 24 |
| Private class-balanced bootstrap | WikiCS | 18 |
| Live-route Polynormer | WikiCS | 9 |
| Stronger factorized references | PubMed | 18 |
| Private hop | PubMed | 18 |
| Private Adam normalization | PubMed | 3 |
| Private CMCL | PubMed | 18 |
| GNCL/SupCon factorial | WikiCS | 27 |
| Nonlocal label retrieval | WikiCS | 63 |
| Shared recurrent class decoder | WikiCS | 30 |
| Neighborhood distribution | WikiCS | 24 |
| Sparse feature kernel | WikiCS | 27 |
| Fixed private keys | WikiCS | 48 |
| Context-positive exclusion | WikiCS | 30 |
| Native GNN-FiLM | WikiCS | 18 |
| Matched joint-reference F | WikiCS | 24 |
| Penn94 confirmation | Penn94 | 100 |
| Physics endpoint confirmation | CoauthorPhysics | 90 |

The CSV gives the exact current-ledger key for every row. The evidence JSON preserves just those selected entries and relevant exclusions. All entries are deduplicated at cohort level; inherited references in later joins are omitted. For example, rotation adds24fresh acquisitions and reuses66; context adds30native fits and reuses75scalar endpoints; fixed keys add48optimizer fits, not81joinedselected records; the graph-serving family adds zero native base fits.

## Units and limits

Fit/bundle records are the saved training unit: a jointly optimized bank may be one optimizer fit while independently optimized four-body ensembles contribute multiple fits. They are not necessarily independent model bodies or route trajectories. The exact global fresh-body/trajectory count is unresolved. The subtotal is explicitly an audited subset; additional completed WikiCS, HGB-IMDB, PubMed, graph initializer, private-transfer and Physics studies were not added where fresh/inherited count reconciliation was not immediate.

CAL/serving endpoints are separate. Examples: graph reliability855fixed CPU fusion fits on45oldbanks; Penn94 90CAL endpoints on70banks/100body attempts; Physics90fits/60banks; FiLM18fits/9banks/45CAL endpoints; fixed keys150fresh+75reused scalar endpoints. CAL and serving fits, selected banks, body records, model-route calls, gradient diagnostic cases and checkpoint replays are not interchangeable experiment counts.

The completed synthetic AR screen (24fitted screens, 8generated development graphs, 384learned laws) is separate from the17named benchmark datasets. Its generated graph count is not eight additional benchmark datasets.

Excluded: wrong-allocation spectral PPI24; training-only PPI probes; engineering/preflight/qualification and timing fixtures; acquired/planned leads such as Tolokers2, LastFM, Artnetviews, ogbn-arxiv and ogbg-mollipo unless completed predictive provenance was established; original/pre-submission study; repeated ledger/snapshot records and rescoring/replay; inherited/reused training in later joins. A later report mentions historical Tolokers21, but its authorization/date was not established in this bounded audit, so the dataset lower bound deliberately does not count it.

Current Physics GRU and PubMed LM4 are live cohorts. Their completed-prefix metadata does not establish a complete predictive family or authorize partial outcome interpretation; they are excluded from whole-family counts.

## Sources

- `AGENTS.md` — authorized scope and original-score preservation.
- `RESEARCH_STATE.md` — current complete findings and two live memory cohorts.
- `research_ledger.json` — selected exact keys copied to `SELECTED_LEDGER_EVIDENCE.json`.
- `SAGE_matched_joint_reference_77_closed_history_20261011_v1/REPORT.md` — 15banks/24optimizer bundles; 75CAL endpoints.
- `complete_quality_gain_reference_confirmation_evidence_map_20261011_v1/REPORT.md` — complete Penn, Physics, WikiCS and PubMed scopes.
- `graph_mixed40_frozen_scientific_decision_20261003_v1/DECISION.json` — both HGB-ACM and HGB-DBLP completed.
- `paired_squirrel_completed_scientific_review_20261003_v1/REPORT.md` — 12/12selected fits.

This report does not modify research state, ledger, scores, manuscript or predictive evidence.
