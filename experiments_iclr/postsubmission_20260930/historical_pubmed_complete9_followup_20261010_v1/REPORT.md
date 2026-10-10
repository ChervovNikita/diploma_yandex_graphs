# Complete PubMed9: history update and next action

2026-10-10. Frozen criteria were read before the complete comparison. This uses closed `COMPARISON.json`/`COSTS.json`, the subsequently completed stored-prediction `COUNTS.json`, and source/protocol text. No original scores are recomputed, raw arrays/checkpoints/models/server are opened, current IMDB qualifier files or initializer-six outcomes are read, or fits/continuations are launched. Numerical entries are saved observations or explicit unit/arithmetic comparisons of saved scalars.

## Keep CORE closed

All nine declared records are complete in the released comparison. CORE fails four unchanged gates: positive accuracy gain in every seed, mean gain≥.20pp, mean class macro accuracy no worse, and mean pooled NLL no worse. The aggregate mean/worst-member degradation guards pass; so does the fallback independent-level gate. That last pass does not mean a positive independent gap was closed: plain shared already exceeds this I4 reference. `continuation_eligible=false` and `further18_activated=false` remain the decision. All18 continuation records stay disabled.

| Seed | CORE−own accuracy pp | Macro pp | NLL | Class0/1/2 accuracy pp |
|---|---:|---:|---:|---|
|9101|−.2790|−.2775|+.013124|−.2439 / −.9696 / +.3810|
|9203|−.0761|−.3623|+.239930|−1.9512 / +2.1978 / −1.3333|
|9307|+.1776|+.2823|−.018085|+.8537 / −.3878 / +.3810|

The saved mean accuracy contrast is−.0591916pp. Mean macro accuracy is .903863 for CORE versus .905055 for own; pooled NLL is .366020 versus .287697. Class effects change sign across seeds. In9203 a large class1 gain trades against class0/2 losses, and every class has worse NLL (+.371627/+.149132/+.260548). The positive9307 result does not rescue the two negative accuracy seeds or the fixed mean gates. No stable class-specific benefit is established.

This rejects the fixed masked-CE plus reconstruction/contrastive combination at this screen. Since the masked-CE-only, shuffled/rewired ownership, four-view single and untied common-decoder conditions remain unexecuted, the screen cannot attribute the failure uniquely to one component. It also cannot certify a universal failure of persistent contexts, contrastive learning or ordinary upstream gradients. No strength sweep, favourable-class selection or Stage2 launch follows.

## The positive signal belongs to the plain shared control

I4−plain shared accuracy is−1.1669/−.6088/−.3044pp, mean−.693387pp. CORE also exceeds I4, but contributes no mean gain over the stronger plain control. Thus this family supplies contrary evidence to an unconditional “sharing always weakens members” story; it does not supply CORE's claimed incremental benefit.

| Condition | Pooled accuracy | Mean member accuracy | Pooled NLL |
|---|---:|---:|---:|
|Plain shared|.909014|.908803|.287697|
|Shared CORE|.908422|.908211|.366020|
|Common-selected native I4|.902080|.890178|.355818|

From these saved means, shared pool-minus-mean-member lift is only about .0211pp for either shared condition; I4 lift is about1.1902pp. Stronger shared members coexist with almost no average inference pooling gain. The subsequently completed error counts resolve the previously missing overlap discriminator below; the small lift alone would not have established it.

The I4 bodies are genuinely disjoint and independently seeded/trained, but **one common pooled-accuracy-selected bank epoch** supplies the reported reference. This is not an ensemble assembled from individually selected members. Plain shared's positive practical comparison is therefore source-compatible evidence against this reference, with selection qualification; it is not a causal estimate of tying or superiority over ordinary own-selected I4/capable singles. The selected epochs differ substantially (own424/390/322; CORE227/212/350; I4149/128/579).

All three optimizer seeds share one encountered PubMed graph and the same amended stratified split (TRAIN11829, VALID3942; authoritative VALID class order820/1547/1575). Repeated VALID selection, dependent nodes/classes and three seeds prevent treating seed/node counts as independent confirmation or inventing a precise generalization interval. Accuracy selection also does not establish an NLL/calibration mechanism. The stronger shared control could reflect competence/regularization/optimization and selection; this screen does not isolate their causes.

Smaller parameter counts do not establish a resource win: the saved allocated peaks are3,621,505,536 bytes for own,7,590,887,424 for CORE and1,947,582,464 for I4. Those are fit/diagnostic peaks, not isolated serving residency. Inclusive times span unequal stopping horizons and should not be interpreted as matched throughput.

## The completed stored9 error counts resolve overlap

Root reports owned diagnostic completion in4.31 seconds with all original signatures/integer counts unchanged and no new scores. This audit reads its compact count report only, without rerunning the diagnostic or reopening owner receipts. Each row is the same3942 VALID nodes; seed repetitions are dependent, not new populations.

| Bank |9101 no-correct / pool errors / lost alternative|9203|9307|
|---|---|---|---|
|Plain shared|343 /347 /4|356 /364 /8|363 /365 /2|
|Shared CORE|350 /358 /8|357 /367 /10|349 /358 /9|
|Common-selected I4|229 /393 /164|244 /388 /144|241 /377 /136|

Every no-correct event in either shared bank has a strict common false rival above truth in all members; no actual FP32 pool crossing occurs. Thus almost all shared pool errors have a witnessed obstruction to an unchanged convex mixture of those member probabilities, and only4/8/2 plain-shared errors lose existing correct alternatives. This is a precise scarcity/obstruction finding, not proof that features/scores lack learnable information or that a new predictor cannot repair them. Different wrong argmaxes are still possible: CORE9307 has349 strict-blocked events but348 unanimous wrong argmaxes.

The next-action implication is sharper than a small average lift: CORE9101 has350 strict-blocked nodes versus plain shared's347 total errors. Under the stated real-arithmetic convex-mixture obstruction, recovering every existing correct CORE alternative still cannot give the required positive gain in that seed. No oracle accuracy or alternative pool was scored, and floating-point boundary chasing is not a remedy admitted by the source.

I4 has broader any-correct coverage (3713/3698/3701 versus plain3599/3586/3579) but far more pool-lost alternatives. Its strict-blocked counts are223/241/239, leaving6/3/2 no-correct events without this sufficient obstruction; none is rescued by its actual fixed pool. Absence of that obstruction remains a non-proof of feasible weighting. This explains why the plain control can win total pool accuracy despite narrower useful coverage. It does not establish a calibration cause or prescribe a new pool.

CORE changes real factual decisions, unlike the closed IMDB source-credit bank's zero pool/coverage changes versus own-only:

| Seed | CORE repairs / introduced pool errors | Clears / introduces no-correct strict-rival events |
|---|---|---|
|9101|37 /48|36 /43|
|9203|69 /72|70 /71|
|9307|48 /41|50 /36|

The learning task acts, but its new useful alternatives/common-error repairs do not dominate introduced harms consistently. In9203 class1 clears45 and introduces10 no-correct events, while class0 clears12/introduces27 and class2 clears13/introduces34. The class gain is a redistribution, not a generally useful specialization. Third-seed coverage improvement survives as a scoped positive observation; it does not erase the frozen failures.

## Three actionable assumptions for new-method/combination work

**1. Strong members and useful alternative decisions are separate targets.** PubMed plain shared has stronger mean members but narrower coverage and negligible average pool lift; IMDB role3's higher mean F1 also coexists with narrower correct-member coverage; centered Tolokers improves competence while leaving little useful coverage. QK shows the converse difficulty: both competence and coverage remain deficient. A new learning mechanism should state which deficit it addresses and measure net correct-alternative gains/introduced harms, rather than count geometry, gradients or source ablation utility as success. CORE's36/70/50 common-error repairs show that changed full learning can act, but43/71/36 newly blocked events prevent a consistent net remedy. Arbitrary new end-to-end predictors and typed mass conditioning have no learning result in this audit.

**2. Credit the incremental mechanism against the strongest matched baseline.** PubMed's positive plain-shared signal is not a CORE effect. NCNC improves over N64 at all five seeds but loses to I4 at four; QK's small shared own-reference gain widens the matched I4 gap; Amazon proper-score gains fail against the processed single. A new combination cannot claim complementary benefit by adding cross-family aggregate rescues. Here capable singles and individually selected I4 are missing, and no exact starts×contexts interaction has been tested. Preserve the positive control signal for later reference work without reopening CORE's failed screen.

**3. Diagnose errors in the actual task/serving space before proposing a pooling fix.** IMDB binary common-wrong events provide a fixed-threshold obstruction, with only three shared lost alternatives and no lost correct three-member majorities. PubMed now supplies the separate strict common-false-rival witness for every shared no-correct event; all-member argmax wrong alone remains insufficient in multiclass prediction. NCNC pooled hit overlaps do not supply member-level common-negative identities. Direct12 supplies no affirmative head-only gain, with six converged/six nonconverged endpoints; it is not an IMDB or arbitrary end-to-end head verdict. IMDB confidence/ranking/feature-readout and NCNC internal common negatives remain untested. Neither stored-error diagnostic authorizes weighting/threshold tuning on these reused selected cohorts.

| Already tested/closed idea | Avoid repeating this inference | What remains untested or outside its scope |
|---|---|---|
|Wiki15/SupCon and Wiki24 initializer/auxiliary package|Representation separation or live upstream gradients automatically create useful ensemble decisions|Arbitrary end-to-end heads; current native-scorer initializer-six outcomes were not read|
|Context9; completed IMDB source-credit/view supervision; PubMed CORE9|Persistence, source usefulness or an auxiliary task alone establishes specialization|PubMed masked-CE-only/shuffle/rewire/four-view-single/untied-decoder controls remain disabled; typed mass modulation has no learned result read here|
|Relation18/QK36; Tolokers private sheaf/centering|Relocating credit, changing attention coordinates or identity centering has already solved the capable-reference gap|Different complete training tasks/initial functions may still work; current pending outcomes are not substituted for evidence|
|Amazon99 aggregation; shared-message products study|A pooled proper-score gain or improvement over the exact-member recipe establishes practical superiority|The strong processed single/competent native singles remain the relevant references|
|NCNC private completion; Citeseer interaction frames|Positive native/control comparisons establish the new private mechanism or cross-family complementary errors|NCNC internal member/common-negative evidence and a matched combination are absent; frame failure does not erase the plain shared-control signal|

MolHIV's completed candidate contrasts remain negative with its missing control limiting the whole-family verdict. DDI/fixedCollab and other previously recorded closure gaps remain unknown, not negatives. Excluded wrong-allocation evidence stays excluded. This map reuses preserved completed conclusions; it does not re-audit administrative receipts.

## Next action

Keep the18 closed and retire rescue by member selection or unchanged convex probability mixing for these shared PubMed banks: the observed opportunity is tiny and most remaining errors have the strict common-rival witness. Preserve the plain-control positive signal for a later, separately designed comparison with capable singles/individually selected I4 on unused populations. New-method roles should pursue useful member learning with competence/coverage/harm controls, rather than repeat persistence/geometry/source-utility premises or combine aggregate cross-task repairs. No new fit, threshold/weighting rule or source-strength search is admitted by this report.

The completed counts explain the closed screen and qualify the plain-control signal. They cannot reverse its gates, identify a training cause or supply independent confirmation. The separately sealed source assessment remains the reader's source-only review; actual completion is root's later observation. This update adds zero new primary/full-paper reads and no manuscript verdict.
