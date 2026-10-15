# Completed MolBACE convergence audit

**Current finding:** none of the 36 completed fits exhausted its epoch cap. All stopped under the frozen VALID-AUC patience rule. The retained original 24-fit TRAIN summaries nevertheless show continuing loss reduction, so optimization convergence is not established. These observations do not demonstrate a VALID benefit from adding epochs.

## Exact completed scope

The completed 30-bank analysis supplies exactly 36 unique training IDs and terminal file pins. There are 12 original native bodies, 12 original shared-BN factorial fits, 3 private-BN controls and 9 private-BN factorial additions. Equivalently, the original 24-fit pilot plus 3 plus 9. M1 reuses I4 body 0; it is not counted as another fit. No residual-only partial run or live GRU state was read.

| Source group | Fits | Completed epochs range | Selected epochs range | Cap hits |
| --- | ---: | ---: | ---: | ---: |
| original_native_body | 12 | 149–190 | 7–140 | 0 |
| original_shared_BN_factorial | 12 | 149–210 | 2–160 | 0 |
| private_BN_control | 3 | 149–166 | 54–116 | 0 |
| private_BN_factorial_additions | 9 | 149–178 | 15–128 | 0 |

Total: **5,517 completed epochs**. Thirty fits ended at epoch 149; the other six ended at 151, 152, 166, 178, 190 and 210. Selected epochs span 2–160 (median 39). All 36 terminal records say `patience`; none reached the 1000-epoch cap.

## What the frozen controller means

Every fit uses maximum 1000, minimum 100, patience 50, strict improvement of whole-VALID ROC-AUC, and the first checkpoint on ties. Staleness is held at zero before epoch 100. A non-improvement at epoch 100 starts the counter, so the earliest patience stop is 149. This accounts for the large 149-epoch group without implying an accidental 150-epoch cap. A record after the minimum restarts patience, allowing the observed longer horizons.

The original runtime literally sets `convergence_established = (stop_reason == "patience")`. That flag records validation early stopping. It is not a gradient, training-loss, parameter-stationarity or optimization-convergence test.

## Local late-trajectory evidence

The completed original 24-fit report already audited its full curves and retained TRAIN tail summaries. In **24/24 fits**, mean update loss in the final 20 epochs was lower than in the preceding 20: declines of **4.99–12.83%** within each fit. The final 20 best VALID AUC was below the selected AUC in 23/24 and tied it in 1/24. Thus lower training loss coexists with no better recorded VALID selection; that is compatible with overfitting, noise or metric/loss disagreement.

These TRAIN values average minibatch update objectives, are not graph-weighted, and are measured as weights change with training-mode dropout/BN. Auxiliary and native objectives differ. Full fixed-model TRAIN evaluation was disabled by the frozen policy, so TRAIN AUC and a comparable eval-mode TRAIN-BCE curve are unavailable. Do not compare raw objective magnitudes between methods or call these curves proof of stationarity.

## One remaining metadata extraction

Only four complete seed 101 VALID curves are retained locally; the full 36 TRAIN/VALID curves are not. The four actual retained curves replay exactly under the original selector and contain zero new VALID-AUC records in their last 50 epochs. Full 36 late TRAIN/AUC/BCE trends and last 50 improvement counts remain **unverified locally**, rather than inferred as a completed curve audit.

`extract.py` performs one guarded, read-only extraction of only the 36 exact RESULT.json files and their sibling CURVE.jsonl files. It verifies the completed 30-bank authority and its manifest, the exact terminal pins,24 prior curve pins, the common frozen policy and historical pure selector/loss source. The 12 other completed curves receive current metadata pins. It replays every logged selector decision and emits per-fit last 50 versus preceding 50 means, within-tail 25/25 changes, OLS slopes, cap/selected horizons, strict VALID record counts, ties, and diagnostic BCE minima. It reads no checkpoint, dataset, prediction tensor, live fit or TEST data and starts no native work.

Root command after staging these source files on the guarded allocation (stdout is the compact JSON result; save it locally as REMOTE_EXTRACTION.json):

```text
CUDA_VISIBLE_DEVICES='' /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/.venv/bin/python -B -u /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/analysis/completed_molbace_convergence_v1/extract.py
```

## Does this support more epochs?

It supports a narrow correction to the wording: these fits met their validation stopping rule, while the available TRAIN losses were still decreasing. It provides no evidence of cap-driven truncation, and lower TRAIN loss alone does not establish that additional epochs would improve VALID AUC. A patience stop does not prove that a later rebound is impossible; that counterfactual was not observed. No continuation or new stopping threshold is recommended from the current evidence. The full 36 metadata extraction will show whether any late trend warrants a separately specified follow-up question.
