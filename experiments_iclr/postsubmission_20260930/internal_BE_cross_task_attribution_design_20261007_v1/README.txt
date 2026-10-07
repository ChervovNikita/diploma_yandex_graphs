Cross-task internal BE attribution design — prospective, immutable
================================================================

WikiCS, Collab and MolHIV ONLY. No new dataset, split, initializer-by-strength
grid, literature search, numerical import, model/data/score access, execution or
family change. Main pending Wiki24 remains unchanged and closed. This document
defines future comparisons; it makes no novelty, performance or efficiency claim.
TASK_COMPARISONS.json contains exact task-specific contracts and paid-work counts.

Freeze one complete protocol BEFORE outcomes
--------------------------------------------
Root must freeze one constructor, one explicit fixed lambda, paired development
and confirmation seed IDs, negative-sampling/runtime contract, one primary
confirmation contrast, uncertainty method and multiplicity rule. None are picked
here. The complete source-ready roster is anchored to be_init: the implemented
alignment-only comparator is initialization-matched be_init_alignment_only.
There is no constructor/strength grid. This design uses one frozen constructor
and one lambda across the three O/I/P/G task rosters, with identical values in
I/P/G. O records lambda but uses own supervision. Lambda0 makes supervision
allocation degenerate; lambda1 removes phi's individual-risk mixture component.
Neither endpoint nor any value is recommended/adopted.

Fresh matched blocks preserve original model configuration, full TRAIN labels,
two existing views/eight full member-view forwards, persistent dropout streams,
actual pool, native optimizer, original full horizon and strict first-max complete
VALID selector. No warm/copied/resource weights. Reuse an existing result only
if its entire source/init/data/pool/update/transition/selector contract matches.

Exact task rules
----------------
WikiCS: official split0 accuracy. TRAIN580 / VALID5274.1100 epochs, local100
then global1000; one full TRAIN update/epoch. Own risk is mean member CE over
all TRAIN nodes. Serving is mean MEMBER SOFTMAX probabilities. Pool risk is
stable -log(mean_m p_m(y)), averaged over all labels. For each view form J=(1-
lambda)*mean-own+lambda*actual-pool risk, THEN average the two view mixtures.
True-class responsibilities r_m=p_m(y)/sum_j p_j(y) yield the cotangent coefficient
(1-lambda)/4+lambda*r_m (plus object/view reductions). This ordinary mixture credit
varies by node/member; it is no learned router or competence guarantee.

Collab: official temporal TRAIN through2017 / VALID2018; repeated different-year
pair records preserved. Primary Hits@50 on60084 positives against100000 original
negatives. TRAIN1179052 positive records with unchanged sampled TRAIN negatives;
100 epochs/18 complete batches per epoch, batch65536 including full tail. Own
member risk AND actual pool risk are BCE_positive.mean() PLUS BCE_negative.mean().
There is NO half factor and NO mean over concatenated labels. The pool is BCE
on mean RAW logits; serving/ranking also uses mean raw logits, never mean sigmoid.
Freeze the exact TRAIN-only negative sampler/runtime and audit paired seed/epoch
negative/query-order hashes. Preserve source historical support, both directions,
all duplicate target removals, and no future-positive rejection.

MolHIV: official scaffold ROC AUC, TRAIN32901 / VALID4113.100 epochs/258 complete
batches, batch128 with TRAIN tail5; VALID33 batches with tail17. Own member BCE
and pool BCE.mean() retain EVERY original finite target, with no missing-target
mask, class balancing or replacement. Pool BCE is on mean raw logits; raw mean
logits feed the original AUC evaluator. Preserve full atom/bond fields and graph
order. Source batch means/tail weighting are not changed into epoch-global means.

In BOTH binary tasks, the pool cotangent is the SAME residual
(sigmoid(mean_j z_j)-y)/4 for every member (with exact batch/group/view weights),
subsequently pulled through different member Jacobians. There is no Wiki-style
winning-member responsibility mechanism. Better BCE alone establishes neither
Hits/AUC utility nor different graph-evidence use.

Alignment targets (unchanged; no loss change)
---------------------------------------------
Source cap=min(objects,512), evenly spaced original TRAIN positions; temperature
0.2, coefficient0.05. WikiCS/MolHIV opposite-view positives are ALL same TRAIN-
label objects WITHIN that selected set, including the identical example and
distinct examples. MolHIV binary equality is coarse, not chemical equivalence.
Collab positives are ALL same canonical target-edge identities (sorted endpoints)
in the selected opposite view; other identities are instance negatives, not
asserted semantic class equivalents. No VALID/TEST auxiliary labels are used.
Own/pool supervision still retains ALL labels; the auxiliary cap does not shorten it.

Three separate attribution questions
------------------------------------
1. Internal allocation, identical internal-only alignment A:
   O theta own / psi own / phi own+.05A.
   I theta own / psi own / phi J+.05A.
   P theta own / psi J   / phi J+.05A.
   G theta J   / psi J   / phi J+.05A.
   I-O: internal ensemble-risk exposure. I-P: boundary protection. P-G: core own
   protection. I-G: candidate versus ordinary GNCL supervision on the same tied
   model, with separately fixed phi-only A. G does not reproduce untied GNCL or
   global full-aux architecture. Actual reverse collections are2/2/3/2, followed
   by one native shared Adam transition at the old state; no upstream detach.

2. Original package, chosen initialized shared bank:
   shared own-only; shared alignment-only (Lown+.05A); original shared full
   package (Lown+.05A+.05R). A/R retain ORIGINAL ALL-CONNECTED-PARAMETER permissions,
   no block mask. Naturally unused gradients may be None; “all permissions” does
   not assert every classifier has a nonzero auxiliary gradient. Private alignment
   is NOT a substitute. Full minus alignment-only tests residual utility; full
   minus own-only tests only the complete package. These source scalar updates
   use one backward/one shared Adam and the same eight forwards.

3. Untied training/selection controls:
   untied4_own_joint versus ordinary independent4 own-selected bank. Both use the
   ORIGINAL sum of four two-view own risks, unscaled per-body gradients, one
   scalar backward and four original Adam calls. Packed own uses one jointly
   selected epoch/bank. Ordinary selects each body independently and evaluates
   the resulting bank, possibly combining epochs; it is a practical baseline.
   WikiCS also restores joint versus own local model/Adam checkpoints after100
   local epochs, so later trajectories can differ: this is NOT terminal selection
   alone or unchanged whole acquisition. Collab/Mol have no stage transition;
   exact matched own trajectories can isolate selection if equivalence is established.
   Original independent4_contrastive is coupled sum-own+4*(.05A+.05R), joint-selected;
   compare it to packed own for its package effect, not ordinary own-selected.
   Shared own versus packed own matches selector for sharing/capacity context.
   I versus packed own is a practical FULL-method comparison, not sharing alone.
   Four untied bodies have greater/distinct capacity; paired seeds do not make
   their initial functions identical to shared factors. No hardware packing gain is assumed.

Readouts, common errors and net repair
-------------------------------------
Use exact selected joint states or the original selected own bank. Official
served task metric is primary; never average accuracy/Hits/AUC into a composite.
Show all member metrics/losses at those states, paired seed deltas, member mean/
minimum/distribution, and full-population repair/break accounting. NLL/BCE and
overlap reductions are secondary; they cannot rescue nonpositive served utility.

Wiki: on all5274 nodes report O wrong->I correct and O correct->I wrong, plus
both correct and same/different wrong. Net repairs/5274 equals accuracy change.
Predefine the O common-competitor cohort: some SAME non-target class strictly
outranks y in every member. This guarantees mean-probability pool error. Keep
all-members-top1-wrong separately; that alone need not imply a pool error.

Collab: for all60084 positive targets compute each state's official Hits50 mask
against its OWN complete100000-negative threshold, with original strict ties.
O miss->I hit minus O hit->I miss /60084 equals Hits50 change. Fixed O common-miss
cohort requires all members miss AND O pool miss. Member misses alone do not
guarantee pool misses when negative thresholds/orderings differ. Do not quietly
claim a same-negative competitor witness from this cheaper common-miss readout.

Mol: score ALL finite VALID positive-negative pairs with win/tie/loss credit
1/.5/0. Record all9 O->I ordering transitions; net credit/pairs equals AUC change.
O-defined common inversion means every member strictly misorders the SAME pair,
which mean raw logits also misorder. Charge the pair analysis. Logit0 threshold
error repairs may be secondary; they do not substitute for AUC ranking repair.

Freeze cohort/tie rules before outcomes; O defines cohorts, I/P/G cannot choose
favourable members/items. Full-population net repair remains decisive. If I matches
P/G, restriction adds no demonstrated utility. If alignment-only matches full,
residual utility is unsupported. If packed untied explains a gain, sharing-specific
attribution weakens. These are prospective interpretations, not observed findings.

Finite confirmation and statistical scope
------------------------------------------
Use the existing finite3-development/5-confirmation block budget, with IDs still
to be frozen. Three paired seed blocks are DESCRIPTIVE/EXPLORATORY; show paired
seed uncertainty. Nodes/edges/molecules/pairs in one split are not independent
model replicates. No intervals or numerical tests have been computed here.
At most ONE conditional confirmation round per task: a complete frozen pilot
with positive mean paired official I-O, I-P and I-G differences can advance the
internal-restriction claim. This directional gate is not significance evidence.
Root must predeclare ONE primary confirmation contrast before ANY outcomes;
other contrasts remain diagnostics unless included in a frozen confirmatory
family. Any confirmatory multi-task/multi-contrast claim requires a predeclared
family, multiplicity rule and error level. Fresh paired confirmation IDs are
pre-frozen; no best-seed selection, secondary-loss replacement, strength/initializer
rescue grid or repeated seed expansion. Same fixed dataset/split/VALID selector
supports conditional optimizer-repeat utility, not graph-population significance
or independent held-out TEST confirmation. Report all failed/nonpositive tasks.

Separate practical-superiority goal
------------------------------------
The broader accuracy/utility claim requires I to beat BOTH unchanged publicV2
native single (--arm single) AND ordinary own-selected4 (--arm independent4)
on the predeclared official paired metric. Policy wins alone do not satisfy
that goal. These are required existing baselines, not an initializer/dataset
grid. Include their contrasts in any frozen confirmatory multiplicity family;
show their original budgets and actual costs. Single uses2 member-view forwards
and1Adam/update, ordinary4 uses8 and4. Report narrower internal attribution
separately if practical superiority is not established.

Cost and source readiness
--------------------------
Full updates/arm: Wiki1100, Collab1800, Mol25800. Training member-view forwards
are8x those counts. O/I/P/G reverse calls are2/2/3/2x; source shared package and
untied own controls use1x. Untied has4Adam calls versus shared1, greater parameters/
optimizer state, and an extra complete final own-bank evaluation for ordinary4.
Charge support/negative sampling/batching, auxiliary and reverse work, complete
VALID, snapshots/reload/storage, peak RSS/GPU and selected-state serving/readouts.
Cached exact complete predictions may avoid duplicate forwards, not readout cost.
Equal epochs or reverse API counts do not imply equal FLOPs, memory or time.

Exact sealed publicV2/OIPG/private-pool and new auxiliary-selection controls are
bound in SOURCE_BINDINGS.json. New be_init_alignment_only/untied4_own_joint controls
are sealed/staged and CLI help passed according to root; runtime remains pending.
This immutable design points to their actual seal, not a hypothetical control.
No source packet is changed by this document. Root's separate readiness handoff
will carry host/runtime qualifications; this design itself qualifies no execution.
