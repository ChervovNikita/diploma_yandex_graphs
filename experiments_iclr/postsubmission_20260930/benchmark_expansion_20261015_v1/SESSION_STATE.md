# Continuation state — release 10 preparation, 15 October 2026

## Scope and immediate objective

Read phase AGENTS.md. Use only the authorized one-GPU allocation through
remote_control.py, which verifies literal anogena-2 destination, hostname and
sole GPU UUID before project operations. No 18.77/MacLink, seven-GPU science,
sudo, host changes, unrelated data, original-five reruns, original-score changes,
GENLINK, Desktop output or PDF compilation. Ordinary caches are allowed.

Prepare the expanded executable study and final review package. Bounded pilots
and engineering checks are authorized. The full campaign requires user review
and GO; no such GO has been received. Preserve failures and costs, and do not
repeat native checks solely for small floating-point replay differences.

## Publication

Remote branch: codex/postsubmission-research-20260930.
Last exact-ref verified commit: ba2612a5d6fad4c5ce412b8b37ad50159679d58c (release09).
Push09 completed at10:15:39UTC with matching remote branch hash. Read
COMMIT_RECEIPT_10.json/PUSH_RECEIPT_10.json for the next publication status. Root owns Git. Use explicit
byte-pinned selections and reviewed source/compact results only. Exclude datasets,
checkpoints, logits, tensor contexts, full generated descriptors and fake fixture
trees. Each publication file must be below 2,000,000 bytes. Diff-data files require
their explicit whitespace exclusions. Named-key push uses the existing interactive
helper and ssh.github.com:443; keep credentials private. Never reuse a release ID.

README, SESSION_STATE and BENCHMARK_OVERVIEW updates must verify the previous
published hashes in DOC_PREVIOUS_PINS_09.json. The root README previously had SHA
2502cb81af84fbb84d457cc1b2fcd1436863bd49dc50b67bde83e22aedfc6909.
Replace only the current research block; preserve prior dated history.

## Completed scientific evidence

MolBACE24: official scaffold0, native GIN5/width300/dropout0.5, Adam LR.001/WD0,
TRAIN1210/VALID151, seeds101/202/303. Per seed four independent bodies, G0,
alignment+variance, nonlinear private correction and their combination.
Minimum100/cap1000/patience50; all24 stopped by patience at149–210epochs.
3,680epochs/139,840updates,7.81elapsedh. No failed/cap-stopped fit. TEST closed.

Selected VALID AUC mean±sampleSD(%), mean BCE:
M1 73.32±1.17/.779; I4 74.85±.50/.890; G0 67.78±4.39/.719;
geometry71.17±2.66/.551; correction71.76±3.91/.941;
combination69.28±1.80/.882. Complete results and selection/convergence audit:
pilot_20261015_molbace_v1/complete_analysis_v1.
Combination interaction negative in all3, mean−5.873pp, approximate paired-t95%
[−12.251,+.505]; selected VALID only. Retain negative outcomes.

PrivateBN3: only route running buffers are private; learned native affine weights
stay shared. AUC73.61416±1.05742%, BCE1.085523. Improvement overG0 in3/3,
mean+5.83639pp; belowI4 by1.23321pp.464epochs/17,632updates, no cap stop.
pilot_20261015_bn_state_v1/complete_analysis_v1. Known normalization control,
not novelty. Different selected epochs preclude buffer-only causal attribution.

Route diagnosis uses all24 parent saved banks plus3 reconstructedBN banks.
Mean member→pool AUC: I4 73.40965→74.84737; G0 62.00549→67.77778;
geometry64.24→71.17; correction66.07→71.76; both61.00→69.28;
BN72.62821→73.61416. Diverse errors alone did not improve the combination.
Remaining BN–I4 gap equals0.78144pp mean-member gap+0.45177pp smaller pooling
gain, descriptive arithmetic only. See analysis/member_pool_gap_decomposition_v1.

New source/algebra note analysis/bn_state_mechanism_scope_v1 states that private
buffers also change EMA time scale and route-order weighting. At illustrative
native momentum.1, shared four-call update replaces34.39% of its state versus10%
per private bank. No new data, training, literature read or diagnostic is implied.

Complete fixed aggregation analysis now covers all27completed fits/21banks
(7methods×3seeds), CPU only,7.810s,0newfits/forwards. Probability averaging:
M1AUC73.321,I474.896,G069.011,geometry70.256,correction71.587,both70.317,
BN73.468. I4BCE.820,correction.638,BN1.007. The simple rule does not close
I4rankinggap; no original or campaign serving rule changed. See
analysis/complete_fixed_aggregation_v1/REPORT.md and all paired scalar outcomes.

## Running scientific work

Nine-fit pilot: pilot_20261015_bn_geometry_correction_v1.
SPEC SHA eacd22be19a1cc108d816cd062f87dde902a4ad67aab6fe31301730c663c2d6b.
Manifest SHA8b0e915319b77b64e9ed17f7f995a020ef9492f67697ed7d4ee3ddfa394769d2.
Launched09:36:19UTC, owner840429/study840431, one new scientific fit at a time,
12h queue cap, no automatic retries. Output results/bn_geometry_followup_v1;
owner results/bn_geometry_followup_v1_owner/OWNER.json.
Qualification/BN_GEOMETRY_ROOT_ADMISSION_v1.json and BN_GEOMETRY_LAUNCH_v1.json
record admission. Three new arms (BN+geometry, BN+correction, BN+both)×three seeds,
15 completed factorial controls reused. Same native settings/controller as
MolBACE24. Frozen reporting covers all cells, AUC/BCE, conditional effects,
interaction moderation, seeds and costs. Selected VALID exploration only.
At10:21UTC first fit000_c53897a759a16a32 completed149epochs by patience;
second fit001_559b911a4356905a is running. No partial quality results inspected. Do not promote partial scores. Nine fits are separate from6460.

Older GRU acquisition remains separate:
phase/private_GRU_acquisition_root_20261011_v1/acquisition_v1.
Owner640426/worker640428,7-day cap. At09:49UTC29/30fitbanks and44/48trajectories,
no acquisition terminal. Do not restart, change criteria, expose partialDEV or
revive excluded arms. Read actual terminal files before interpreting outcomes.

## Expanded study

All6,460 source descriptors prepared by expanded_job_binding_v1 overlayv3:
100new/6360reused;90remaining source-context blocks closed;2480ancestor contracts
retained. READINESS SHA dacf157ffa7bebc52acafbe1021cac1eaeffc1eac2d1326c9c821b0bb48eb374.
VERIFICATION SHA9a9a192a6cd30701f72f58ecb9a10bf49e4557bed4415bc77227e9a42484acb8.
Full descriptors stay on allocation. This is source readiness, not full launch
admission. Handoff expanded_job_binding_v1/HANDOFF_v3.json.

Counts: node1920 (80reference+320native+1520shared), molecule3840
(160+640+3040), link700 (30+120+550). Seven datasets; eight architecture families;
27source presets;25recipe definitions including controls. M1 reuses I4body0.
NoHPO, dummy backbones or original-five reruns. Exact native settings and
applicability in expanded_study_design_v1; plain explanation BENCHMARK_OVERVIEW.md.
Nodes use ten paired split/seed rows; OGB/link use fixed official splits with
independent seeds. Do not call them ten independent dataset samples.

The dispatcher has38passed fake-subprocess tests, permanent claims, resume,
completed skipping, I4 dependencies, STOP/drain and two-slot resource admission.
Its original full release remains allocation-only, all6730tasks blocked.
A metadata-only successor is being built from compiler v3. Reuse source-family
functional coverage; do not manufacture literal per-seed native PASS receipts.
Missing branch checks, resource profiles, reporting/access freeze and humanGO
remain explicit. Both future77workers are disabled/unprobed.

## Engineering evidence, no repeats needed

BUDDY production conditional seed1:2updates/3wholeVALID/2constructions/32forwards,
111.36sCPU2. Actual production provider and same-call mean passed; no full epoch.
qualification/BUDDY_PRODUCTION_SEED1_QUALIFICATION_07.json.
Other production recipes have source-equivalence evidence, not independent runs.

NCNC runtime_link_v6:2epochs/34updates/4wholeVALID/3constructions/2meanexports,
181.32sCUDA. Completed-epoch interruption recovery and stopped finalization passed.
Peakallocated12,673,890,304B/reserved26,698,842,112B.
qualification/NCNC_RUNTIME_V6_QUALIFICATION_07.json. v5 blocker is preserved;
it never had a native run. Do not rerun either generation.

Molecular runtime_v4:2epochs/76updates/4wholeVALID/3constructions/2meanexports,
45.760sCPU2, noCUDA initialization. Actual selected mean, own checkpoint authority,
completed-epoch recovery and zero-update stopped finalization passed.
qualification/MOLECULAR_RUNTIME_V4_QUALIFICATION_09.json pins fetched RESULT
SHA3038c0e76fb8f3c863684560cb2e4e8c0364d20f4e96fc05f91a4b5c0783441d.
Raw checkpoints stay allocation. Alternative CUDA gate not run and not required
as an automatic repeat. The BN pilot cannot replace this mean-finalizer evidence.

Earlier node runtime, NCN context, BUDDYBN/context/interactions and NCNC
context/interactions passed with costs under their exact original receipts.
Coverage audit:38evidence units/511bindings/28source signatures/25methods.
Its19obligations are not19newruns.73old selected-smoke failures were recorded
floating-replay tolerance failures; exact initial state/RNG restoration and
post-replay RNG checks passed. Preserve statuses, do not mass-rerun them.

## Literature and reporting

Saved flow-method readings: FlowMatching2210.02747v2 methods/theorem statements;
CARD2206.07275v4 method/setup; ContrastiveFlowMatchingICCV2025 method/setup.
One-step endpoint residual regression is not novel supervision merely by naming
it flow matching. Conditional one-step loss algebra and deferred method are in
literature/flow_matching_shared_backbone_method_v1. No additional method entered
6460. Raw sources were hash-verified offloaded,51.84MB local duplicates removed.
Compact offload/publication receipts need release09. XBM scoped source read and
rare-class support limitations remain in their existing notes; no queue method
is implemented. Do not inflate reading counts or reread without a concrete need.

The broad matrix is exploratory. No separated SD bars or selected validation
win establishes superiority. The existing separate confirmation proposal of
20freshpairedcohorts/120fits remains a proposal requiring its own reviewed budget.
Exact sign tests there concern repeat-training win probability, not mean effect
or graph-population uncertainty. Historical exposure must be explicit. A fresh
Codex manuscript review through the requested skill follows substantial supported
paper progress, never each check and never with a requested verdict.

Exploratory reporting now root-reviewed in benchmark_reporting_protocol_v1:
ROOT_REVIEW.json accepts the paired descriptive protocol, all31sourcepinschecked.
EXPOSURE_ADDENDUM_01.json adds completed all27bank aggregation use. Historical
Physics TEST exposure verified; other6historiesunknown. Newseeds do not erase
prior exposure. HumanGO remains absent; later fullGOmayauthorize finalscoring
prospectively without a redundant later approval prompt for the same action.

## Current division and next work

- Root: collect terminal evidence, scientific interpretation, publication09,
  monitor the two running scientific queues without partial favorable claims.
- combination_study_reuse_20261010: metadata-only v3 dispatcher binding and minimal
  real remaining functional branch plan. No native work or Git.
- massive_preparation_resource_admission_20261012: metadata-only cost/scenario and
  resource ledger. Earlier gates finished; do not revive stale task ownership.
- buddy_completion_20261015: readable exact methods/applicability/cohort review
  package and reporting/statistics/access proposals. No native work or Git.

Resource budget complete: matched minimum237,211,000optimizer maps; source caps
plus matched minima247,526,000; allcaps2,364,025,000. No full ETA. Staged approval
option646+5814fits preserves roster. Resource agent now prepares one MolHIV
GIN-VN GPU full-epoch cost gate using unchanged runtime, two realVALIDcalls,
no replay or newtraining study, unlaunched pending root resource review.
Dispatcher agent found/fixes genuine noninteraction-extension metadata and
lexicalvenv execution issues in successors, originalsources/tests preserved.
Reporting agent prepares explicit descriptive paired protocol and initial
exposure ledger; future confirmation requirements do not block exploration.

Collect agent handoffs, verify source/policy scope, then publish09 with completed
molecular evidence, v3 compiler, pilot admission and compact literature receipts.
Deliver the final count/settings/methods, measured-versus-estimated cost and
one-command launch preparation for user review before giantGO. Do not silently
shrink the roster to avoid revealing its compute cost.

## Release10 continuation addendum

Root accepted runtime_link_v8 declaration/mean protocols and dispatcher2 lexical
venv execution; six v6 runtime core files and dispatcher engine bytes unchanged.
ROOT_SOURCE_REVIEW_10.json records exact source review. MolHIV GIN-VN qualifier
source is accepted, unlaunched: the two existing scientific CUDA workers still
occupy both slots. BUDDY production descriptors contain body:null and old provider
casts int(None); combination agent is implementing provider2/runtime9/compiler4
source successors plus final finite qualifiers. No failed production fit occurred.

benchmark_reporting_implementation_v1 implements the accepted descriptive report,
1170prescribedcontrasts,9focusedmetadata checks and unchanged MolBACE3seed extraction.
Resource agent now integrates full registry/binding/cost/mean sidecars; no tensors,
training or77. Buddy agent builds disabled host portability; do not call77ready.

Completed new analysis/bn_bank_assignment_v1: all3selectedBNcheckpoints, own/tied/
all3cyclicbankassignments,15fullVALIDcalls,300nativeforwards,0TRAIN/updates/TEST.
CPU2threads,30.82077selapsed,maxRSS628108KiB. MeanAUC own73.61416%,tied59.05983%,
meanofshift-scores57.97721%. Own beats tied in3/3 and average shifts in3/3, but
shift1 improves seed101AUC. BCE own1.08552,tied1.25036,shifts1.77466; tied improves
BCE for2/3,so no uniform calibration claim. Inputs/planfrozen beforelaunch; first
resource API call failed before modelconstruction; preserved and corrected keys.
Models/checkpointsunchanged; rawtensorsremote. This diagnostic does not identify
original EMA/time/order/selection effects or establish I4superiority. Prior-art
note literature/switchable_norm_scope_v1 reads Slimmable abstract plus exact author
code, including shared-affine/private-running-state USBatchNorm implementation.
No full-paper read or new mechanism novelty claim.
