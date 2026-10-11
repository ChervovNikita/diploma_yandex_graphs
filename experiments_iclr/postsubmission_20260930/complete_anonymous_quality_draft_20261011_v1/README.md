# Complete anonymous source working draft

Entry point: `main.tex`, with all TeX inputs, bibliography, conference style and three figures local to this folder. This is an uncompiled working draft. It is not a verified submitted manuscript, a delivery-ready PDF or a peer-review milestone.

The recovered round-18 `main.tex`, `appendix.tex` and `references.bib` are historical source. Round-18 has **not** been verified as the submitted version. This status is visible in the draft title area and provenance prose. The title/header is anonymous and says working draft; it does not claim a review or publication status.

## Assembly

- A quality-led abstract and conclusion surround the integrated v2 introduction and internal-SAGE section.
- The original model, five-dataset benchmark and Roman member-path analysis are preserved verbatim. The original five-dataset heatmap and architecture PNG are copied byte-for-byte.
- The complete F finding is labeled selected development evidence. Its main coverage figure, paired uncertainty and failed mean NLL protection remain visible.
- F's absolute five-arm table appears only in the appendix. Auxiliary failures appear only in one short appendix section. The historical appendix is included once; these are the only two appended sections, with no metric tutorial.
- Unfinished Penn94/E slots are absent from manuscript text and comments. No result or successful mechanism is inferred from current runs.

## Local package

`main.tex` uses `sections/introduction.tex` and `sections/internal_sage_methods_results.tex`. `appendix.tex` includes the preserved `historical_appendix.tex`, then `sections/F_absolute_results_appendix.tex` and `sections/auxiliary_explorations_appendix.tex`.

`figures/` contains the exact recovered `architecture_sharing_compact.png` and `gnnm_vs_ens_heatmap.png`, plus the saved `F_member_coverage_v2.png`. `references.bib`, `iclr2027_conference.sty` and `iclr2027_conference.bst` are local unchanged copies. The small title/status override is in `main.tex`; the recovered style file itself is unchanged.

The appendix uses A plus Arabic section numbers and distinct hyperref anchors, with Arabic footnotes. Twelve appendix sections are resolved statically. All 57 labels are unique, all cross-references resolve, and all 25 used bibliography keys exist.

## Maps and limits

`DEPENDENCY_MAP.json` records exact source/asset hashes, local input resolution and preservation checks. `CLAIM_MAP.md` states the evidence and limits of each material claim. `MANIFEST.json` records the assembled source files.

No PDF compilation, editor/PDF opening, rendering, Desktop access, reviewer invocation or scientific/model/remote execution occurred. Static inspection cannot verify pagination, float placement or visual layout. Historical sources, integrated fragments, canonical status and current runs were not mutated. Dataset arrays, model checkpoints and the separate historical code/evidence supplement are not included in this source package.
