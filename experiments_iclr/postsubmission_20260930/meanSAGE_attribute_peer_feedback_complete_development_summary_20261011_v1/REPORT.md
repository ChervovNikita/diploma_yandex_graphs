# Complete meanSAGE peer-feedback development result

11 October 2026. **The full scientific cohort completed successfully; the fixed candidate failed its development rule.** All30 banks,48 selected optimizer trajectories,57 physical backbones and30 label-free bank exports closed before all30 fixed CAL endpoints/15,000 scalar updates, followed by one whole development release. Acquisition, CAL and score owners each exited0, waited/reaped their worker and recorded worker/CUDA absence. `WATCH_END` is complete. No survivor subset, model retry, duplicate reader or partial predictive read occurred.

The30-bank study was admitted at12:12:37.042436UTC with `dropout_results_opened_before_decision=false`; the complete dropout fetch followed at12:13:58.830464UTC. This study was independently selected before that result. The first earlier zero-step qualification failure is retained in the root qualification/admission history; the scientific cohort itself has no acquisition/CAL/score failure.

## Population and unchanged scientific procedure

Three fixed paired mask/optimizer blocks are10103/10427/10949 on the complete encountered Physics graph:34,493 nodes,8,415 binary features,5 classes and530,417 support entries. Each block uses100 TRAIN/150 SELECT/150 CAL/34,093 DEVELOPMENT_EVAL nodes. The raw undropped feature incidence has1,137,311 entries; the exact X/support/C/D digests are retained in SUMMARY.json. These are repetitions on one graph with overlapping masks, not independent graph domains or unused confirmation.

The source meanSAGE backbone retains hidden32/depth2/dropout.4, native aggregate/skip and row clipping, TF1 Adam.001, first-layer effective-weight L2, latest tied minimum total SELECT CE+L2, patience50 and100,000-update ceiling. Genuine I4/IA4 have four own optimizers, full own objectives and own selectors. UX4 is one live coupled learner with four physical bodies and a joint selector. Every fit restores its complete selected state; no qualification or old fitted state supplies a scientific model.

Raw primary serving is the archived native FP32 arithmetic mean of member probabilities. CAL learns one common positive scalar **inside each member before that probability mean**, using CPU float64 Adam.05 for exactly500 updates, clamped logT[-5,5], with the latest tied minimum among step0..500. Fitted CAL quality is not interpreted. Raw/calibrated decisions, NLL and Brier remain separate.

## Every arm

Values are three-block means. Parameters describe one served bank; I4/IA4 counts include all four independent models. Full raw/calibrated accuracies, member losses, Brier, every block and costs are in the CSVs and SUMMARY.json.

| Arm | Raw accuracy % | Mean / worst member % | Raw NLL | Calibrated NLL | Parameters |
| --- | ---: | ---: | ---: | ---: | ---: |
| P1: exact source single |93.212683|93.212683 /93.212683|.939005|.218619|538917|
| F1: factor single |92.996607|92.996607 /92.996607|.941020|.226692|555885|
| I4: genuine factor ensemble |93.137399|93.010295 /92.719913|.938344|.219192|2223540|
| S4: shared factors, no feedback |93.110023|92.914479 /92.660272|.947117|.221643|606900|
| PA1: source single, self feedback |93.006384|93.006384 /93.006384|.944966|.221272|539941|
| IA4: genuine factor ensemble, self feedback |93.100245|92.991963 /92.747289|.942064|.219482|2227636|
| SA4: shared self feedback |93.058204|92.872682 /92.359135|.952779|.220211|607924|
| SM4: peer mean before nonlinear map |93.248859|93.053315 /92.853861|.947511|.216036|607924|
| **SX4: candidate nonlinear donor mean** |**93.091446**|**92.866571 /92.312205**|**.952094**|**.218273**|**607924**|
| UX4: coupled unshared donor feedback |93.146198|93.028872 /92.811819|.939659|.217158|2227636|

## Quality, competence and attribution decisions

**Factor competence passes.** F1−P1 mean accuracy is−.216076pp, with worst block−.425307; I4 mean members−P1 is−.202388pp, with worst block−.341712. Both meet the declared−.25 mean/−.5 block tolerances. The source single and ordinary shared bank are competent in this cohort.

S4 mean members are only−.095816pp below I4 here. The earlier approximately five-point deficit under the different native E recipe is absent in this source-backbone cohort. This supports a recipe-dependent competence account; it does not show that a particular ingredient causally solved that deficit. The competence and member-retention caps remain practical tolerances, not statistical equivalence.

**Every worthwhile candidate gain fails.** SX4 must gain at least.25pp on average with every block positive against S4/P1/PA1/I4/IA4. Its differences are:

| Reference | SX4 raw delta, pp | Block10103 /10427 /10949 |
| --- | ---: | --- |
| S4 |−.018577|−.014666 /−.158390 /+.117326|
| P1 |−.121237|−.202388 /−.114393 /−.046930|
| PA1 |+.085061|−.366644 /−.014666 /+.636494|
| I4 |−.045953|−.316781 /−.020532 /+.199454|
| IA4 |−.008799|−.231719 /+.017599 /+.187722|
| UX4 |−.054752|−.061596 /+.026398 /−.129059|

The coupled-unshared quality clause also fails. SX4 mean members−S4 is−.047908pp, failing the required+.25/all-positive gain. Its mean-member allowance versus IA4 passes at−.125392pp, although worst members are−.435084pp below IA4. Passing a harm allowance does not establish improved acquisition.

**Risk allowances pass.** All five calibrated-NLL and three raw-NLL clauses meet their frozen practical caps. SX4 calibrated-NLL differences are−.000346 versus P1,−.000918 versus I4,−.001208 versus IA4 and+.001115 versus UX4. Raw NLL is worse by+.004978 versus S4 and+.010030 versus IA4. These caps are practical protection screens, not statistical noninferiority or universal proper-loss wins. Calibration does not rescue the failed raw primary rule.

**Peer and nonlinear-ordering attribution fail.** SX4−SA4 is only+.033242pp, with one negative block, below the.10/all-nonnegative peer-access requirement. SX4−SM4 is−.157413pp and negative in all three blocks. The declared donor-before-mean explanation earns no positive empirical support. SM4 has the highest raw mean in this panel; it is retained as a control outcome and earns no replacement-candidate or confirmation admission from this release.

**Retain the limited mean-first clue.** SM4−S4 gains+.138836pp raw accuracy in all three blocks[+.067463,+.029332,+.319714], with mean members also+.138836pp in all three[+.019799,+.071129,+.325580]. SM4−I4 is+.111460pp with mixed signs and SM4−P1 only+.036176pp with mixed signs. This is a modest development clue for the existing control; it supplies no supported broad advantage or unused-confirmation priority. The exact SX4 mapping-before-averaging hypothesis failed; a statement that peer feedback in general failed would exceed the evidence.

All accuracy intervals in SX4_CONTRASTS.csv are descriptive three-block intervals; the I4 interval is[−.689480,.597575]pp and IA4[−.532860,.515261]. The full90 ordered contrasts preserve all12 metric vectors and their frozen sensitivity summaries. No new inferential test or node-iid uncertainty was added.

## Useful alternatives and actual serving

Across the102,279 dependent readouts, SX4 versus S4 adds61 covered alternatives and80 pooling losses, yielding−19 served correct. Versus IA4 it adds317 coverage,325 losses and−1 aggregation-only correction, yielding−9 correct. Versus I4 it adds419 coverage,467 losses and+1 aggregation-only correction, yielding−47 correct. Both pooling-retention clauses fail. Gross repairs/harms are419/438 versus S4,557/566 versus IA4 and534/581 versus I4.

On the fresh S4 common-wrong-argmax cohort, SX4 supplies554 new correct-member alternatives across5,147 readouts but serves48 repairs. Outside I4 coverage it supplies299 alternatives and serves5 repairs; outside IA4 coverage it supplies302 and serves9. These are factual reference cohorts after separate training/selection; they do not isolate a causal pooling effect or show absent information. The predicate is agreement on the same wrong argmax, not a new strict-margin condition. All paired integer identities verify.

## Paid work, custody and interpretation boundary

Acquisition paid23,598 Adam steps/backwards,55,791 TRAIN and55,791 SELECT route evaluations,223,350 meanSAGE layer calls,158,454 incidence SPMM calls,8,252 checkpoint writes and93 selected routes. Direct acquisition owner wall is1562.610s; recorded runtime scope1557.985s. CAL owner wall is14.713s and score owner4.798s; scalar fitting itself is14.242s and scoring4.488s. The watcher includes waiting and must not be added to these walls. Whole acquisition allocated CUDA peak is1.367GiB and lifetime RSS2.511GiB; no per-arm peak is inferred.

Three-block fit-wall sums are29.433s P1,246.928s SX4,172.316s IA4 and189.701s UX4. The candidate stores fewer weights than four equipped independent models, but takes more recorded fitting time and loses raw accuracy to the smaller source single. These scopes include differing selected updates/stopping; they establish no equal-budget causal effect or isolated throughput claim. FIT_COSTS_AND_SELECTION.csv preserves all48 trajectories and their selection, complete-update and work counts.

The full4,033,882-byte server report is bound by SHA256 `4a02e2090e80fb2c8e0db2c538ea0cee278e6d92c8304d6d43137f413d1ea2e5`. All per-class evidence, checkpoints, prediction arrays and truth stay remote. The local compact summary preserves every arm, all90 ordered comparisons/error counts, all22 frozen clauses, fixed CAL parameters, native body costs, owner/watch closures and the exact source manifest. No scientific code or source was altered by this acquisition audit/readout interpretation.

Generic incidence propagation and stream communication have CONN/TextGCN and continuing-stream/set ancestry. This experiment tests the fixed interlayer peer composition. Its failure closes this exact rank16/scale.25/site/operator/learning recipe; it does not refute all attribute diffusion, prove ensemble expressivity separation, or supply a novelty/manuscript verdict.

## At most three next actions

1. Close the exact SX4 recipe and its claimed peer/ordering/pooling improvement; retain all failures, allowances and control outcomes. No rank, scale, site, serving or replacement-arm rescue follows.
2. Keep the competent source single, factor single and genuine equipped I4 as mandatory references in any separately chosen architecture study. A substantive new mechanism needs its own prospective decision and closest-primary comparison; this release selects none.
3. Keep the conditional Computers confirmation unadmitted for these failed fixed methods. A future complete supported mechanism must earn that priority and pass root custody/overlap/source/resource clearance before new graph scoring. Root owns final decision and publication.
