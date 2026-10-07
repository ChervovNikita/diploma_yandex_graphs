Two narrow auxiliary/selection attribution controls
==================================================

Source-only preparation. No code/numerical imports, server, model, data, score,
checkpoint, TEST, launch or campaign execution. No seeds, scientific experiment
or current-family changes are adopted. Both controls use the sealed public V2
complete driver; runtime, costs and predictive behavior remain unverified.

be_init_alignment_only
----------------------
Construct the exact public be_init_contrastive Session: same initialized shared
four-member BE body, native Adam, member streams and two full own views. Its
update is source mean-own +0.05 alignment, omitting ONLY residual_member_contrast
and its addend. Full source-active parameter permissions remain unrestricted
total.backward; there is no private-only steering, block masking, upstream
detach, projector, new parameter or changed target. Alignment gradients reach
parameters where connected, exactly as in the original full package. The same
source finite predictions/representations/active gradients/model/Adam checks,
bounded deterministic auxiliary index and task label/edge-identity targets stay.

This isolates the added residual package term when compared with the original
be_init_contrastive on a separately matched cohort. Alignment-only versus be_init
asks about alignment, not graph evidence or novel diversity. Private-alignment
steering is a different control with different gradient permissions and is not
substituted here. Shared own reduction stays .5*(own_A+own_B).mean(),8 member/view
forwards, one scalar backward and one native Adam call per completed update.

untied4_own_joint
----------------
Construct the exact ordinary independent4 Session: four native bodies, original
seed+1009*m constructors, four persistent member streams, two complete own views
per member, original per-body Adam/groups and mean-member serving. Delegate its
unchanged own-only train_step. The source sum of four member own losses preserves
unscaled per-body gradients, not the shared mean reduction. Alignment/residual
are zero. Eight member/view forwards, one source scalar backward across four
disjoint bodies and four native Adam calls remain; equal backward API count is
not equal FLOPs, capacity, memory or serving cost.

The distinct control ID makes the original full driver use the same joint pool
strict-first selector and WikiCS joint local model/Adam restoration as
independent4_contrastive. Live end-local RNG stays live. It NEVER assembles
individually selected own-best checkpoints or restores own-local checkpoints.
The original driver retains per-body own_best/own_local diagnostic snapshots;
this wrapper labels them explicitly as unused for selection/transition. Ordinary
own-selected independent4 remains separate, unchanged and a valid comparator.

Compared with independent4_contrastive, this isolates its auxiliary package with
architecture, initialization, supervision scale, member streams and joint
selection held fixed. Compared with a shared bank it controls the selection
rule but still changes parameter capacity, sharing, initialization/private maps,
gradient scale interpretation and compute. WikiCS joint versus own local reset
also changes the subsequent global trajectory; do not reuse an ordinary
own-selected trajectory as this new fit or interpret it as only posthoc pooling.

Full callable/CLI
-----------------
run_complete(task, control, train, valid, output, seed, device='cpu',
             polynormer=None, ncn_model=None, ncn_utils=None, public_root=None)

Tasks: wikics, collab, molhiv. Explicit --seed is required; no default study seed
or roster is adopted. Standard source --train/--valid numeric NPZ, native source
paths and fresh --output apply. CLI template, not executed:
  python controls.py --task molhiv --control untied4_own_joint --seed CALLER_SEED \
    --train TRAIN.npz --valid VALID.npz --output FRESH_OUTPUT
WikiCS also requires --polynormer; Collab --ncn-model/--ncn-utils. --public-root
can relocate the exact same sealed public V2. No depth, strength, horizon,
batch/role/selector overrides, TEST, resume, retry, queue or automatic campaign.
Calls must be serial per process because original train.main reads sys.argv.

The small in-memory interface substitutes Session/recipe and only annotates
original snapshot/write/evaluate helpers, then calls unchanged train.main. It
copies no epoch loop or data/evaluation/selection implementation. All original
full populations and source horizons remain: WikiCS split0 TRAIN580/VALID5274,
1100 epochs with100 local then1000 global and1 update/epoch; Collab all1179052
TRAIN events and60084 VALID positives +100000 authentic negatives,100 epochs
with18 updates; Molhiv TRAIN32901/VALID4113 with original atom/bond scaffold
roles,100 epochs with258 updates. Original input checks do not invent official
provenance or author-runtime parity. VALID selection optimism remains.

Identity, effective recipe and costs
-----------------------------------
RUN/TRACE/selected and diagnostic snapshots/COMPLETE/FAILURE visibly identify
the control, original constructor and source digests. Wrapper config reports
effective A=.05/R=0 for shared alignment-only and A=R=0 for untied own-only.
The unchanged frozen base contrastive recipe and its digest are credited
separately. Native raw Session config and own-only step remain unchanged.

Counters observe actual native Adam calls/completions and completed full TRAIN
bank views without additional numerical work. COMPLETE checks all source
steps/VALID events and8 forwards with1/4 Adam calls. Scalar reverse count at
completed updates is explicitly a source contract; partial failed reverse calls
are not instrumented/claimed. Validation dispatch seconds and original inclusive
cost/parameter counts remain recorded. No new cost padding, parity fixture,
gradient audit or resource qualification framework is added.

Source FAILURE and partial artifacts are preserved with identities; pre-driver
source failures also receive a fresh labelled receipt. Exact resume is not
supported by the full driver and is not added. Original sources/families stay
immutable. Neither control proves novelty, causal graph specialization, useful
residual separation, efficiency or a population effect. Any later comparison
must be separately fixed/adopted and preserve failures and method-specific costs.

Design source
-------------
cross_task_internal_BE_design_critique_20261007_v1/REPORT.md identified these two
gaps. This packet implements those narrow source controls, not a new prior
search or adoption of the critique's broader cross-task GNCL experiment.
