# WikiCS shared unit-factor component attribution (disabled)

Question: does the exploratory +0.506 percentage-point unit-factor gain seen in
Wiki24 reflect supervised alignment, residual route contrast, or their joint
effect? This is a post-Wiki24 attribution study, using its three development
seeds again. It is not a novelty screen, tuning grid, independent confirmation,
or a comparison that pools allocation and GPU77 scores.

## Fixed experiment

| Condition | Alignment weight | Residual weight |
|---|---:|---:|
| plain (P) | 0 | 0 |
| alignment_only (A) | .05 | 0 |
| residual_only (R) | 0 | .05 |
| combined (C) | .05 | .05 |

Seeds are 6101, 6203, 6307; all four conditions are fresh fits per seed. Seeds
6101 then 6307 use physical GPU0 (a998); seed6203 uses GPU1 (8ced). Run
P,A,R,C sequentially within each seed, at most one full fit per GPU. The roster
contains 12 cells and 13,200 full TRAIN updates. Plain and combined references
on GPU77 are mandatory; historical Wiki24 scores cannot replace them.

All cells use the sealed portable v2 core, identical to V6, the pinned native
Polynormer, shared four-member unit factors, original architecture, Adam, member
dropout streams, and two full own-loss views. The unchanged portable train.main
runs 100 local plus 1000 global epochs, restores the strict-first best complete
development local joint state at transition, and selects the strict-first best
complete development joint checkpoint across the full horizon. No early stop,
factor adaptation, extra views, objective tuning, or automatic retry is allowed.

The adapter shallow-copies the Session core dictionary and replaces only its
objectives entry. Active original loss functions run unchanged; inactive terms
return a differentiable zero. Session.train_step retains its original .05
multipliers, TRAIN-only targets, deterministic maximum512-object auxiliary
selection, and two CE views. Plain retains its original noncontrastive flag.
Actual condition, underlying Session arm, coefficients and adapter SHA enter
session.config, run/progress/completion metadata and selected snapshots.
Complete requires exactly1100 updates and exactly1100 calls per active term.

## Data and reporting

The pinned safe NPZ roles contain x[11701,300], edge_index[2,442907], all580
TRAIN ids/labels and all5274 development ids/labels. Development is the union
of official split0 validation and stopping masks. Root's completed conversion
verified all six ordered arrays/raw fingerprints against public pins and wrote
no TEST labels. Report development accuracy, not published TEST accuracy.

First report fresh paired C-P on77, then A-P, R-P, C-A, C-R and interaction
C-A-R+P, in percentage points for each seed and as mean/SD. Treat df2 exploratory
intervals as descriptive; three optimizer seeds on one graph do not constitute
independent graphs or confirmatory evidence. If C-P fails to reproduce the
reference direction, report that limit before attributing a mechanism. Do not
rescue the result by selecting seeds, checkpoints, coefficients or extra fits.
At the fixed selected checkpoints, report member accuracy, pooled/member NLL
and Brier, and fixed paired error flows against the same-seed plain checkpoint
(plain wrong -> condition right, plain right -> condition wrong, unchanged).
Interpret accuracy with competence/calibration and error flows; route repulsion
alone does not establish useful diversity or preserved class information.

## Engineering and root admission

No new numerical work ran during source preparation. All templates are disabled.
Root must bind a separate exact enabled release, source review, runtime/data
equality evidence, actual resource evidence, and the existing owned supervisor.
Reuse the reviewed GPU77 owner.run_fit and ownership helper; this packet contains
no scheduler or ownership framework. Full scientific cells retain external
32390s active +10s cleanup =32400s hard bounds, soft28800s. The aggregate108h
hard-bound sum is a safety envelope, not an ETA.

Existing full8view BE evidence observed80,939,581,440 peak GPU bytes. Before
qualifier or fit, require at least83,087,065,088 fresh free GPU bytes, exclusive
one-process-per-GPU admission, a proposed79GiB owned GPU cap, and root-bound RSS
limits. The old40GiB native-prefix cap cannot cover this model. The source checks
fresh free memory and the declared cap; the existing external owner must enforce
the wall/GPU/RSS bounds and own all cleanup.

qualify.py uses the same facade and public data helpers for all four conditions:
one fullgraph original update in local mode, original CPU snapshot/reload,
complete finite5274-development serving, then the same in global mode. It checks
four-member logits, all580 own-label objects, finite backward/Adam, active loss
call counts, CPU byte RNG states, CUDA parameters/Adam moments, and restored
local/global state. Its8 updates are discarded engineering work. It asserts no
numerical equivalence and admits no scientific fit. Run it under a separately
released finite owned cap (template1800+10s) before scientific admission.

Preserve the venv executable path, empty PYTHONPATH and existing providers on
peptide. Portable author-execution equivalence remains unqualified. Source seal
and AST/JSON/CLI checks establish preparation integrity; actual GPU qualification
and root scientific/resource admission remain separate.
