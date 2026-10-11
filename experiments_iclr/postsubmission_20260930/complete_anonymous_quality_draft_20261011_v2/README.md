# Complete anonymous source working draft — V2

Entry point: `main.tex`. All TeX inputs, bibliography, conference style and three figures are local to this folder. V2 is a writing-only revision of the sealed `complete_anonymous_quality_draft_20261011_v1`; V1 remains unchanged. This is an uncompiled working draft, not a verified submitted manuscript, delivery-ready PDF or peer-review milestone.

The recovered round-18 `main.tex`, `appendix.tex` and `references.bib` are historical source. Round-18 has **not** been verified as the submitted version. This provenance status belongs in this README, the dependency map and the claim map. Scientific prose retains the actual limitation that historical launch-time source, environment and data fingerprints are incomplete.

## Writing revision

- The abstract starts with the ensemble setting and the problem of improving combined prediction while learning common graph weights. It names the original five graphs and keeps their benchmark central; F is summarized as an accuracy and confidence tradeoff with three-seed uncertainty. F's harm value and policy threshold are removed from the abstract.
- The original model, five-dataset benchmark and Roman member-path analysis are preserved verbatim, as is the Roman matched-budget paragraph. All original numerical evidence, negative tables and three figure bytes are retained.
- The main scope section is reduced to the Roman budget comparison and a graph/recipe summary with exact table and appendix references. The original secondary numerical contrasts and runtime/selection caveats are placed in `sections/sharing_scope_appendix.tex`; complete protocols and tables remain in the unchanged historical appendix.
- F's descriptive confidence intervals, raw accuracy gains, member/coverage counts, development reuse and probability-quality tradeoff stay in the main text. Its frozen accuracy/NLL rules and complete failure outcome appear beside its absolute appendix table. The appendix adds the previously recorded exact mean, per-seed NLL harms and maximum joint harm from the existing closed F report; no new experiment or outcome is introduced.
- The manual title/header override and printed working-draft banner are removed. The unchanged ICLR style controls the anonymous title and header; `\iclrfinalcopy` remains commented out. Anonymous PDF metadata is retained.
- Reproducibility prose describes experimental evidence and historical replay limits. The AI statement describes assistance during earlier research and manuscript preparation. No new reviewer pass is claimed or invoked.
- Auxiliary failures remain appendix-only; unfinished Penn94/E studies remain absent.

## Local package

`main.tex` uses `sections/introduction.tex` and `sections/internal_sage_methods_results.tex`. `appendix.tex` includes `historical_appendix.tex` once, followed by `sections/sharing_scope_appendix.tex`, `sections/F_absolute_results_appendix.tex` and `sections/auxiliary_explorations_appendix.tex`.

`figures/` contains byte-preserved `architecture_sharing_compact.png`, `gnnm_vs_ens_heatmap.png` and `F_member_coverage_v2.png`. `references.bib`, `iclr2027_conference.sty` and `iclr2027_conference.bst` are unchanged. No citations are added or removed from the complete manuscript.

The appendix uses A plus Arabic section numbers, distinct hyperref anchors and Arabic footnotes. `STATIC_CHECKS.json` records local resolution, uniqueness, numeric preservation and source comparisons. Static checks do not establish visual layout or pagination.

## Change and evidence records

`V1_TO_V2_CHANGES.md` records every textual change, relocation and supplemental precision detail. `V1_TO_V2_SOURCE.diff` contains the exact TeX diff, including the new appendix fragment. `CLAIM_MAP.md` identifies evidence locations and scientific limits. `DEPENDENCY_MAP.json` binds upstream sources, the sealed V1 parent and local assets. `MANIFEST.json` and `SEAL.json` bind the V2 package bytes.

No PDF compilation, editor/PDF opening, rendering, Desktop access, reviewer invocation or data/model/remote execution was performed. Historical sources, integrated fragments, canonical status and current runs were not mutated. Dataset arrays, model checkpoints and the separate historical code/evidence supplement are not included in this source package.
