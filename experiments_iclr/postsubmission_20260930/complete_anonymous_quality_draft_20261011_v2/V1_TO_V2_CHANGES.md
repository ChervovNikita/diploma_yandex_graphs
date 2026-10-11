# V1 → V2 writing changes

Parent: sealed `complete_anonymous_quality_draft_20261011_v1`. V2 was copied into a separate directory. V1 and all upstream source/evidence files were left unchanged. `V1_TO_V2_SOURCE.diff` records the exact TeX changes.

## Text and relocation inventory

| Location | Explicit change |
| --- | --- |
| `main.tex` preamble/title | Removed the manual `\@maketitle`/header override, its comments and printed working-draft center banner. Kept anonymous metadata, author and commented `\iclrfinalcopy`; conference-style bytes are unchanged. |
| Abstract | Added a plain ensemble introduction and declarative problem statement about improving combined predictions under common graph weights. Named all original five graphs. Kept 28/40, four members, SAGE/GT variants, intended selection and Roman 2.94-point result. Replaced the secondary F 0.721/0.537-point summary and 0.021448/0.02 policy sentence with broader coverage, weaker members, higher calibrated NLL, three-seed uncertainty and development reuse. Kept the quality/storage conclusion. |
| Introduction | Replaced only the sentence about exceeding F's mean harm allowance with a plain statement of higher NLL than the genuine independent ensemble. All other paragraphs are unchanged. |
| Main F accuracy paragraph | Removed the sentence saying the frozen raw screens pass. The exact gains, all-positive seed statement and both descriptive intervals are unchanged. Passing rules/outcome moved to the F appendix. |
| Main F figure caption | Replaced the 0.021448-versus-0.02 gate sentence with a plain higher-calibrated-NLL sentence. All caption coverage counts, rounded 0.537 gain and rounded [-0.265,1.340] interval remain unchanged. Figure bytes are unchanged. |
| Main F confidence paragraph | Replaced the exact threshold/cap/pass/fail narration with higher calibrated NLL, favorable Brier, a probability-quality tradeoff and a link to the appendix. Full frozen outcome retained there. F methods, acquisition definitions, costs and coverage/repair/harms paragraphs remain unchanged. |
| Main scope section | Retained the entire Roman matched-budget paragraph verbatim. Compressed the remaining six paragraphs into two graph/recipe/budget summary paragraphs, with exact references to the primary four tables, two extension tables, two preprocessing tables and two matched-budget tables. Kept the main limitation that no new-graph choice rule has independent validation. |
| New `sections/sharing_scope_appendix.tex` | Relocated the original primary graph contrasts and matched-budget gain/latency paragraphs without changing their numerical content; replaced general appendix references with exact labels. Condensed the private-first/last definitions and matched-budget setup; directs readers to the existing full six-candidate protocol instead of duplicating it. Retained post hoc choice, three-seed single-split scope, raw-Cora discrepancy, small Chameleon test set, runtime and convergence limitations. The original dataset citation group moved here unchanged. |
| Main conclusion | Replaced “failed mean NLL protection” with higher calibrated NLL. Removed the recovered-manuscript/submitted-source sentence; the actual missing historical fingerprint limitation remains. Other conclusion/limitation text is unchanged. |
| Reproducibility statement | Replaced source assembly and compilation prose with paper-style model/selection/evidence types and actual historical source/environment/data replay limits. Source/package/compilation status moved to records. |
| AI use statement | Made earlier research/manuscript assistance explicit. Retains the historical activities and authors' responsibility, without claiming a new reviewer pass. |
| F appendix | Kept the existing five-arm absolute table byte-for-byte. Added the frozen raw rule (mean ≥0.2 pp, all seeds nonnegative, ≥2 positive), calibrated-NLL mean/per-seed caps (0.02/0.05) and complete pass/fail outcome from the already closed F report. Added a unique subsection label. |
| Appendix wrapper | Includes the new sharing summary between the unchanged historical appendix and F appendix; updated its comment. Historical appendix and auxiliary fragment remain byte-identical. |
| Records | Rewrote README/claim map for V2; updated dependency map, added exact source diff and static verification record, then rebuilt manifest/seal. Round-18-not-verified-submitted status remains in records only. |

## Numerical and citation accounting

No numerical score, experiment, citation identity, bibliography entry or asset was changed. The original model, five-dataset benchmark and Roman analysis block, including all tables, is verbatim; the Roman budget paragraph is verbatim. All historical appendix cells, including negative results, are byte-preserved. Every numerical token removed from the secondary main scope remains in the relocated appendix fragment or unchanged Roman main paragraph.

The abstract no longer displays F's already-rounded gains of 0.721/0.537 or mean NLL harm 0.021448 and threshold 0.02. Exact F gains remain 0.720516/0.537227 in the main and appendix; these were not rounded anew. The original main figure's 0.537 gain and [-0.265,1.340] interval are unchanged. The main descriptive intervals remain [0.2712,1.1698]/[-0.2653,1.3398].

The F appendix preserves the original six-decimal mean harm 0.021448 and explicitly gives the already recorded unrounded value 0.021448383545404. The per-seed I4 harms 0.007869/0.032667/0.023809, maximum joint harm 0.021990 and frozen raw rule are existing closed-report details that V1 did not reproduce. They are newly transcribed supporting details, not newly calculated results. Mean joint harm 0.013310 and the 0.02/0.05 NLL rules moved from the main. Brier's favorable direction remains descriptive; it does not replace the frozen NLL rule.

Existing partial-sharing values retain their V1 precision: 0.17, 0.74, −0.44, −3.09, −1.83, −0.87, +2.43; Chameleon 194/0.52; WikiCS 0.5%, +0.684 and 11.13/8.56 ms; normalized Cora 0.31%, +1.63 and 4.22/3.77 ms. More precise historical table values remain unchanged. All 25 used citation keys remain present; the existing Cora/WikiCS/Actor/filtered-Chameleon citation group was relocated to the appendix.

## Verification boundary

Only static local source and content checks were performed. No compilation, editor/PDF opening, rendering, Desktop access, reviewer invocation, live run reading or data/model/remote execution occurred. Static checks cannot establish pagination or visual layout.
