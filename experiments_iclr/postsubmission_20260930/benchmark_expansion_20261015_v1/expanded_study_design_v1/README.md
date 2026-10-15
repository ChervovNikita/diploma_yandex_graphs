# Expanded benchmark proposal

The purpose is to determine which shared-backbone learning mechanisms improve
prediction quality and whether they compete with four ordinary independent
models. The native model is held fixed within every comparison. New methods
change only declared private wrapper coordinates, training credit, auxiliary
objectives, aggregation or running-state ownership.

**This is a proposed scientific roster, not an approved or executable campaign.**
It names **6,460 physical training fits**. The earlier 2,480-fit proposal remains
intact and its baseline/training rules are retained, with an exact alias counted
once. New native integrations, serving and resource checks remain necessary.
No fit in this expanded campaign has been launched.

## Coverage

| Task | Datasets | Native architecture families | Fixed source presets |
| --- | --- | --- | ---: |
| Node classification | CiteSeer, PubMed, Coauthor-CS, Coauthor-Physics | PolyFormer, each with source Mono and Cheb settings | 8 |
| Molecular graph classification | ogbg-molbace, ogbg-molhiv | GCN, GCN-VirtualNode, GIN, GIN-VirtualNode; both declared source dropout presets | 16 |
| Link prediction | ogbl-collab | NCN, NCNC, BUDDY | 3 |
| Total | 7 datasets | 8 architecture families | 27 |

These are source settings rather than a new hyperparameter search. Mono/Cheb
and two source dropout settings are separately reported. A winning preset is
not selected using TEST. The original GNNM paper's experiments are not rerun.

## Required comparisons in every one of the 27 presets

1. **Reference-setting native single:** ten separate source-horizon runs,
   retaining the recorded reference selector and disclosing seed, runtime or
   release ambiguities. These check the reference result; they are not extra
   paired observations of the new method effect.
2. **Matched native single:** use native independent-body0 below, under the
   same longer training controller as the shared methods. No duplicate fit.
3. **Ordinary independent4:** four native models with independent weights,
   optimizers, RNG streams and own selected checkpoints. Average their native
   raw logits. Each body is a separately charged physical fit.
4. **Original boundary GNNM:** one native body with four private factor routes,
   original own-risk training and mean native raw-logit serving.

Ten paired cohorts per preset give 270 reference fits, 1,080 matched independent
body fits and 270 base GNNM fits: **1,620 required training fits**. All other rows
are additional shared-backbone hypotheses or necessary ingredient controls.

## The scientific questions

| Family | What changes | What establishes an interpretable comparison |
| --- | --- | --- |
| Pooled training credit | Give shared and/or private parameters a half-own, half-ensemble loss | Own-only GNNM; global-J mean; shared-F/private-J. Mean global-J is counted once as the aggregation control. |
| Task geometry | Add supervised alignment to existing native preheads; add a separate conditional-variance control for labelled tasks | No auxiliary versus alignment versus alignment+variance. Links use authenticated same-query cosine/global/endpoint contrasts, without inventing nonlink truth. |
| Persistent responsibility | Emphasize different TRAIN structure ranges in each route while keeping every supervised weight positive | Degree/atom-count quartiles versus an exact within-class permutation. Links compare corrected endpoint-correlated and raw-marginal-matched query fields. |
| Context information | Teach existing preheads two factual similarity targets | Every route fits the common target mixture versus two routes per target; same donor opportunities and work. Exact targets are in TASK_CONTEXT_SPEC.md. |
| Private nonlinear input | Learn zero-start low-rank nonlinear corrections around one shared body | BE+correction versus base GNNM and four-route correction-only. The latter still has one shared body and four routes. |
| TRAIN-learned aggregation | Learn global or prediction-conditioned route weights | Uniform/global/conditional under the same half-own, half-served loss; also evaluate the selected gated bank under a fixed mean. |
| BN state ownership | Separate route running buffers while sharing all learned native affine weights | Applies only where the native model has tracked BN: OGB models and BUDDY. No BN is inserted into a BN-free architecture. |

These are 25 exact recipe definitions across tasks, including base rules and
controls. They are not 25 claimed inventions. Task-specific versions have
different targets and are not silently treated as the same implemented loss.
The predecessor node within-route SupCon and fixed positive bootstrap rules are
also retained under their original identities.

Four narrowly defined interaction recipes complete useful ingredient designs:
geometry+input correction; structured responsibility+private pooled credit;
input correction+mean-J; input correction+conditional-J. The last two, together
with no-adapter mean/conditional J, test whether aggregation uses the adapter's
alternatives. These are selected combinations, not a Cartesian product of all
methods, losses, placements and coefficients.

## Exact physical fit count

| Task | Source references | Matched native bodies | Shared method fits | Total |
| --- | ---: | ---: | ---: | ---: |
| Node | 80 | 320 | 1,520 | 1,920 |
| Molecule | 160 | 640 | 3,040 | 3,840 |
| Link | 30 | 120 | 550 | 700 |
| Total | 270 | 1,080 | 5,110 | **6,460** |

M1/body0 reuse, uniform-responsibility/base reuse and global-J/aggregation-mean
aliasing prevent duplicate training. Same-bank readouts, uncertainty analysis,
prediction exports and engineering checks are not counted as scientific fits.
The currently running24-fit development pilot and three-fit BN follow-up are
separate studies; they are not silently substituted for prospective fresh
benchmark cohorts.

The count is generated by `build_design.py`, which checks that every retained
predecessor rule is included in every applicable cell. PROPOSED_ROSTER.json has
the exact method list per preset and explicit pending gates. It deliberately
contains no runnable dispatcher or launch command.

## Native hyperparameters and convergence

NATIVE_SETTINGS.json records each of the 27 presets' exact source LR, WD,
dimensions, dropout, batches, native evaluator, reference score location and
unresolved replication issues. It is exported from the existing immutable
contracts, with exact native-config/optimizer equality assertions. A new wrapper
coefficient is a fixed recipe choice, not evidence of paper-equivalent training.

Reference-setting runs retain their own source horizon. Matched runs retain
the agreed longer controller: PolyFormer minimum 2,000/cap 10,000/patience 100;
OGB and NCN/NCNC minimum 100/cap 1,000/patience 50; BUDDY uses its separately
recorded source-compatible controller in NATIVE_SETTINGS.json. Patience begins
after the minimum and required source phases, and resets only on strict whole-
VALID improvement. Native source schedules and regularizers remain explicit.
The cap is a bound, not convergence evidence; a cap without a plateau is
reported as unresolved rather than called adequate training.

## Uncertainty and promotion

Report every paired cohort, mean difference, variation and paired uncertainty
on the native metric, alongside native proper loss where meaningful, member
quality, actual pooled repairs/harms and costs. Three development seeds and
separated standard deviations do not establish significance. Several presets
of one dataset do not become several independent datasets.

This broad matrix is exploratory. It must not inherit the old impossible
ten-trial/226-contrast exact-test policy or turn many selected wins into a
confirmatory claim. Before launch, freeze which data are for development, the
confirmation candidate rule, heldout access, primary contrasts and multiplicity
policy. A single frozen candidate's confirmation is a different scientific
question from testing every exploratory cell. No acceptance or significance
outcome is promised by the design.

## What remains before a giant launch

Complete native ports and ingredient composition; verify whole epochs,
selection, stopped/recovery resume and actual serving; resolve all source
split/provider bindings; measure CPU/GPU/disk/work profiles; then assign jobs
to the three GPUs with at most two own workers per GPU where resources fit.
Only the allocation is currently authorized. Future 77 commands cannot be
tested until the user reauthorizes that route.

After those checks, provide the user the exact reviewed roster, costs and
one-command worker entrypoints. The giant campaign then awaits explicit GO.
