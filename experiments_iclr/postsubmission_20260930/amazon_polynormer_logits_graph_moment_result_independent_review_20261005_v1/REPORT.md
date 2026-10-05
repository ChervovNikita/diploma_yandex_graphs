# Independent audit of the complete Amazon aggregation result

Status: **PASS for aggregate reconciliation; the fixed NO_GO decision is preserved.** One minor wording finding in root V1 was repaired in a separately sealed V2. No unresolved numerical, selection or gate error was found.

This audit read RESULT.json and frozen protocol/source/adoption text. It used only Python standard-library arithmetic. It did not open prediction, label, graph, model, checkpoint or fold payloads, import the inspected source, fit models, run inference, use SSH, score TEST, or edit canonical/manuscript files.

## Closure and selection

All nine exact bank/split groups are complete: six four-member bank groups with 15 configurations each and three single groups with three each. All 99 configurations retain raw and corrected metrics (198 panels), count 6,123, complete class support and weighted accuracy closure. All setting choices and finalists reproduce the corrected OOF Brier selector and prescribed ties.

Every bank finalist is operator 7, score-context residual MLP, settings 1,0,1. The processed single selects operator 10, settings 1,0,0. The strongest cheap reference for every bank/split is that processed single. Operator 6, the analytic local-full moment proposal, is never selected.

| Selected corrected pipeline | Accuracy | Brier | NLL |
| --- | ---: | ---: | ---: |
| Shared four | 52.5450487% | 0.637247101 | 1.833855367 |
| Independent four | 53.2527628% | 0.642338913 | 1.784915555 |
| Single | 53.1275519% | 0.613543703 | 1.492954628 |

Reported fit counts close by group and total: 90 MLP + 36 calibration = 126 fits; the fixed 150-update schedule gives 18,900 reported updates. Per-call propagation records close to 144 batched H calls, 2,880 sparse steps, 684 logical H applications and 13,680 logical sparse steps. The C&S total is 594 logical applications for 297 outer configuration predictions. Frozen call geometry also gives 198 QP calls and 2,351,268 QP solutions, exactly reported. No final refit or base-model fit/replay is reported.

These are aggregate/counter checks. RESULT does not expose the per-fit trajectories/competence diagnostics or whole-fold rows; this review does not certify their numerical contents. Source schedules support the reported arithmetic.

## Every failed practical screen

Differences are candidate minus reference. Brier improvement is minus the mean difference; accuracy gains are percentage points. The prescribed thresholds are Brier improvement >=0.002, at least two improved Brier splits, mean accuracy gain >=0.25 pp, worst split accuracy loss <=0.5 pp, and mean NLL harm <=0.01.

| Candidate / reference | Brier improvement | Accuracy gain pp | NLL harm | Failed predicates |
| --- | ---: | ---: | ---: | --- |
| Shared / native | 0.246635968 | 0.1415428 | -4.359183654 | mean accuracy, worst accuracy split |
| Shared / cheap single | -0.023703398 | -0.5825031 | 0.340900739 | mean Brier, Brier split count, mean accuracy, worst accuracy split, mean NLL |
| Independent / native | 0.130809306 | 0.0762154 | -2.018358727 | mean accuracy |
| Independent / cheap single | -0.028795210 | 0.1252110 | 0.291960927 | mean Brier, Brier split count, mean accuracy, mean NLL |
| Shared / processed independent | 0.005091812 | -0.7077141 | 0.048939813 | mean accuracy, worst accuracy split, mean NLL |
| Shared / processed single | -0.023703398 | -0.5825031 | 0.340900739 | mean Brier, Brier split count, mean accuracy, worst accuracy split, mean NLL |

All six screens fail. Shared versus native also loses 0.9309162 pp on split 2. Shared versus independent has Brier gains in all three splits but loses 1.1105667 and 1.5188633 pp on splits 1 and 2 and worsens mean NLL. Both full bank routes fail against their strongest cheap references. The complete shared confirmation screen therefore remains NO_GO; none of the failed predicates has been dropped.

## Analytic moment attribution

| Bank | Local-full minus global-full Brier | Minus local-diagonal | Minus full-information stacker | Minus posterior projection |
| --- | ---: | ---: | ---: | ---: |
| Shared four | +0.000714106 | -0.000019911 | +0.148801156 | +0.011446870 |
| Independent four | +0.001153589 | +0.001086944 | +0.063047406 | +0.012275332 |

The fixed mean improvement >=0.001 requirement fails against both global-full and local-diagonal controls in both banks. The selected-route maximum harm <=0.002 predicates pass, which does not rescue the conjunction. The stacker/posterior match flags are true because IDs 8 and 9 match or beat ID 6 within 0.001; they are not flags of analytic superiority. Analytic moments are substantially worse in Brier than the full-information stacker and posterior projection.

All eight already-available matched-rho comparisons were also recomputed:

| Bank | rho setting | Local-full minus global-full Brier | Local-full minus diagonal Brier |
| --- | ---: | ---: | ---: |
| Shared four | 0 | +0.000714106 | -0.000025112 |
| Shared four | 1 | +0.000522511 | +0.000198048 |
| Independent four | 0 | +0.001696705 | +0.001624370 |
| Independent four | 1 | +0.001151480 | +0.001086944 |

No matched-rho mean reaches the 0.001 required improvement. On independent setting 0, the maximum split Brier harms exceed 0.002 (0.002755074 versus global and 0.002718910 versus diagonal). These are reported contrasts using the existing settings, not new gates or tuning.

For completeness, shared analytic moments minus the independently processed independent/single finalists are:
- shared_analytic_moment_vs_processed_independent: mean Brier +0.166181860, accuracy -0.8002613 pp, NLL +1.942671871.
- shared_analytic_moment_vs_processed_single: mean Brier +0.194977070, accuracy -0.6750504 pp, NLL +2.234632798.

Gains over badly calibrated native pools do not identify a special local/off-diagonal mechanism. The processed single has lower mean Brier and NLL than either selected bank pipeline. The result supports preserving the negative development outcome; it does not establish causal sharing, representation defects, independent confirmation or paper acceptance.

## Root adoption and repaired wording

Root V1/V2 means, signs, split differences, all practical predicates, retained moment contrasts and NO_GO interpretation reconcile. V1 said "No ... new model inference" although frozen study.py fits and serves the small heads. This minor scope overstatement has no metric/gate effect. V2 changes it to "new base-model inference". The V2 source and report are exact phrase replacements, its source bindings are byte-identical, and its summary differs only in UTC. Both sealed versions remain intact; REPAIR.json binds V1.

The physical child duration and prior failed-attempt history are outside the new aggregate arithmetic check. Historical acquisition remains a separately recorded cost; current RESULT only reports the successful study cost. The existing retrospective VALID and overlapping-split limitations are preserved.

## Artifacts

- REVIEW.json: concise conclusions, raw/corrected all-operator means and supplementary comparisons.
- CHECKS.json: complete reconstructed synthesis, every gate/interval, counter closure and V2 repair checks.
- SOURCE_BINDINGS.json: exact authorized input hashes; descriptor targets were not opened.
- audit.py: standalone stdlib audit, with no inspected-source imports or execution-payload reads.
