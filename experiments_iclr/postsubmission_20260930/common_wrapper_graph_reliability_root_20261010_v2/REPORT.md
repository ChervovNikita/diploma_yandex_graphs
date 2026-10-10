# Complete graph-context aggregation comparison

10 October 2026. All45 banks and855 fixed fusion endpoints finished at15:24:28UTC, with no failed endpoint. The full reader ran once after owner closure. Original base scores and members remain unchanged. No training, new model forward or TEST access occurred in the reader. This report covers all original SAGE/GCN/GAT banks and all declared aggregation controls.

## Measured quality

Mean of three paired seeds on encountered WikiCS development. Accuracy is percent. Ordinary and factorized I4 are four genuinely independent models. The same graph scorer receives equal fitting opportunities for every bank.

| Backbone | Bank | Native accuracy | Graph-scored accuracy | Graph-scored NLL |
|---|---|---:|---:|---:|
| SAGE | ordinary_genuine_I4 | 80.0215 | 81.0643 | 0.671709 |
| SAGE | factorized_allmap_genuine_I4 | 80.0847 | 80.8874 | 0.681870 |
| SAGE | shared4_unchanged | 80.5714 | 81.0707 | 0.968328 |
| GCN | ordinary_genuine_I4 | 81.0327 | 81.7722 | 0.640067 |
| GCN | factorized_allmap_genuine_I4 | 80.8874 | 81.7975 | 0.652018 |
| GCN | shared4_unchanged | 80.7357 | 81.5194 | 0.731729 |
| GAT | ordinary_genuine_I4 | 81.9808 | 82.0882 | 0.631041 |
| GAT | factorized_allmap_genuine_I4 | 81.8165 | 81.9429 | 0.642227 |
| GAT | shared4_unchanged | 80.9443 | 80.9695 | 0.697616 |

Graph-context scoring adds0.499305/0.783719/0.025281accuracy points to shared SAGE/GCN/GAT. SAGE andGCN improve at all three seeds. GAT has one negative seed and two positive seeds. Relative to the identical scorer using own-node predictions only, the graph scorer improves shared accuracy at every seed for all three backbones. Mean differences are0.815320/0.796359/0.151688points.

The graph scorer also improves ordinary independent-four by1.042852/0.739477/0.107445points. When both banks receive graph scoring, shared minus ordinary I4 is+0.006320/−0.252813/−1.118695points. Shared versus graph-scored factorized I4 is+0.183289/−0.278094/−0.973328points. SAGE has mixed signs against ordinary I4. GCN andGAT do not establish superiority. All frozen accuracy transfer flags fail, as do all NLL protection flags. Every per-seed comparison, class, repair, harm and control is retained in the complete results.

Shared graph scoring lowers NLL versus original shared averaging, but calibrated temperature controls have lower NLL. For SAGE the graph NLL is0.968328 versus0.682065 with global temperature and0.671709 for graph-scored ordinary I4. These distinct accuracy and confidence outcomes remain visible. Better accuracy versus averaging is not an overall superiority result.

## What this changes scientifically

The full result supports a scoped development clue: neighbor prediction summaries can recover some correct alternatives that averaging discards. It fails the broader hypothesis that this specific aggregation rule establishes a shared-ensemble advantage over equally processed capable independent ensembles. No generic aggregation or universal-wrapper impossibility follows. Combining newly acquired alternatives with a serving method is a new hypothesis only if complete repair evidence and a matched prospective decision support it.

A scalar convex pool cannot reverse a wrong class that outranks truth in every original member. In this study graph scoring corrected zero such strict-rival cases across the nine shared banks. This is the exact mathematical restriction of these weights, not proof that every remaining error is an absent-evidence error or that richer aggregation cannot help.

## Evidence and limits

The full server report is COMPLETE_ANALYSIS.json,11877332bytes, SHA2565cf34c9116016c7ad3c4a35c4333edcfda1619a71d98abaee874709be2016cb6. The compact per-backbone files preserve all fixed rule outcomes and paired intervals, all class contrasts and original fit/failure/cost records. COMPLETE_ANALYSIS_SUMMARY.json preserves source/support/closure/fold identities and cost accounting. COMPLETE_SCALAR_SUMMARY.json is a complete all-bank/all-rule scalar/count index. Models, fit states, OOF arrays and native predictions remain server-side.

All base checkpoints used the full VALID role for selection before aggregator folds were formed. The held aggregator folds therefore measure encountered development, not whole-pipeline cross-fitting or unused confirmation. The descriptive seed intervals describe three optimization repeats on one graph. They do not account for trying prior methods or establish variation across independent graphs.

The owner took1199.665seconds and fixed head fitting1181.516seconds.855smallCPU fits are not855newGNN fits.89new selected forwards/116member trajectories plus the preserved10/10failed-prefix calls represent99/126. No inference speed claim follows. The original descriptor-bound failure and its costs remain preserved. Post-readout download hit a local2MB indented-file limit, then a fetch trailer error. Recovery parsed the complete sealed emitted report, stored compact partitions and made no repeat reader, fit or forward. Those engineering failures do not change the scientific result.

## Disposition

Close this exact fixed aggregation superiority recipe without a head-width, rate, fold or hop rescue grid. Retain its serving gain and full confidence limitations. It can be an attributed ingredient/reference in a distinct mechanism. Any candidate still needs capable matched singles/ensembles, a frozen whole pipeline and unused graph confirmation before manuscript promotion. No new paper acceptance or methodological novelty is established.
