# Nine terminal selected VALID collections

`JOBS.json` binds the nine missing banks from the unchanged published input authority. `launch.py` invokes the existing, hash-pinned `analysis/selected_molecular.py` exactly. It adds no collector logic, training callback, TEST call or tolerance option. The existing analyzer supplies terminal and factorial gates. Source preparation has not executed either collector or analyzer.

## Fixed jobs

Paths below are relative to the terminal study output. Collection paths are relative to the fresh collection root.

| Index | Seed | Cell n geometry correction | Fit directory | Collection |
| --- | --- | --- | --- | --- |
| 0 | 101 | 1 1 0 | fits/000_c53897a759a16a32 | fit_000 |
| 1 | 101 | 1 0 1 | fits/001_559b911a4356905a | fit_001 |
| 2 | 101 | 1 1 1 | fits/002_630ca81a2ef5dc87 | fit_002 |
| 3 | 202 | 1 1 0 | fits/003_6473ca8a8e9e3823 | fit_003 |
| 4 | 202 | 1 0 1 | fits/004_71baa4e9dc02b445 | fit_004 |
| 5 | 202 | 1 1 1 | fits/005_5f1c6fe266bcc813 | fit_005 |
| 6 | 303 | 1 1 0 | fits/006_ab8841b9576a761f | fit_006 |
| 7 | 303 | 1 0 1 | fits/007_4919b7ca8bd6f743 | fit_007 |
| 8 | 303 | 1 1 1 | fits/008_a9b00808438972cb | fit_008 |

`--project-root` is repository R; relative paths resolve beneath `R/experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1`.

## Root binding and commands

After complete study termination, copy `ADMISSION_TEMPLATE.json` to a separate root-owned receipt and bind the actual study/collection/analysis paths, exact terminal END/OUTCOMES/REPORT hashes, allowed older-GRU compute PID, and one project-contained slot lock path. Root must reserve one collector slot and approve the exact launcher/job hashes. The template has approval disabled; it authorizes no execution.

Use the existing allocation Python environment. These commands have identical arguments except mode:

```text
<python> -B -u <expansion>/pilot_20261015_bn_geometry_correction_v1/collection_launch_v1/launch.py --project-root <R> --admission <root-receipt> --admission-sha256 <receipt-sha256> --mode commands

<python> -B -u <expansion>/pilot_20261015_bn_geometry_correction_v1/collection_launch_v1/launch.py --project-root <R> --admission <root-receipt> --admission-sha256 <receipt-sha256> --mode collect

<python> -B -u <expansion>/pilot_20261015_bn_geometry_correction_v1/collection_launch_v1/launch.py --project-root <R> --admission <root-receipt> --admission-sha256 <receipt-sha256> --mode analyze
```

`commands` emits the nine concrete argv arrays and the single saved-tensor analysis argv after terminal/input checks; it spawns nothing. Each collector argv is exactly:

```text
<python> -B -u <expansion>/analysis/selected_molecular.py --project-root <R> --fit <study-output>/<fixed-fit-directory> --output <collection-root>/<fixed-collection-directory>
```

`analyze` requires the nine-collection END receipt to be complete and invokes the pinned complete-factorial analyzer once with the three reviewed terminal hashes, the same collection root, a fresh analysis output, and `CUDA_VISIBLE_DEVICES=''`. Its output path cannot overwrite an earlier analysis.

## Gates and bounded work

All nine ordered successful study END outcomes must pass before any score-bearing report or fit record is read. Then all 24 factorial report identities are checked. Every new JOB must exactly match its frozen descriptor; its RESULT/job hash, selected-file paths, controller and runtime hashes must match the collector's original expected inputs. The collector itself retains its existing checkpoint/export content hashes, source/dependency checks and selected-output sanity comparison. No input or source is repaired in place.

GPU collection uses a nonblocking exclusive slot lock held across the sequence, one child at a time, and a GPU PID check before every child. The only permitted external compute process is the root-bound older GRU PID, if still present. An unexpected GPU PID blocks further collection. The lock serializes launcher copies using that slot path; it does not control unrelated launchers. Root's resource reservation must prevent an unrelated worker from racing the PID check. This is the remaining operational binding, together with final terminal hashes and paths; no predictive or method-definition gap is introduced.

Each collector has the existing supervisor's 180-second bound, process-group termination on timeout/interruption, and zero automatic retries. Stop on the first failed collection; retain all nine planned outcomes, including not-started jobs. Preserve logs, exit/timeout state, actual attempted/completed forwards, time, memory, failures and reconstruction warnings. If a terminal result is absent, mark forward cost unknown rather than zero. A sanity warning is retained without recollection. There is no retraining or TEST access. Completed controls are never recollected.

Before root admission, the true integration limit is that the existing collector has not yet restored these nine composed selected states. Runtime/source/input prechecks are prepared, but successful collection is established only by its actual result receipts. No synthetic or source check is presented as that qualification.
