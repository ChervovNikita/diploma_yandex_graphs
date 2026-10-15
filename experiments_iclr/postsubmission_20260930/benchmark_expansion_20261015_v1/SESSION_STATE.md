# GNNM continuation state — release 11 preparation, 15 October 2026

## Scope

Read phase AGENTS.md. Use only the literal authorized one-GPU allocation through
remote_control.call, verifying anogena-2-0 and sole UUID
GPU-44039938-fd82-41d2-fefd-de71514e2fac before project operations. No77/MacLink,
seven-GPU science, sudo, host changes, unrelated data, Desktop, PDF compilation,
GENLINK, original-five reruns or original-score changes. Ordinary caches allowed.
Bounded representative pilots and engineering checks are authorized. Giant
campaign GO is absent. Do not launch or call it fully ready.

## Publication and files

Branch codex/postsubmission-research-20260930; exact-ref verified release10 commit
f0c521dda59ee70e2122bae22c25ad072ac6a938, pushed10:39:06UTC.
PUSH_RECEIPT_10.json is local. Root owns Git. Release11 is being prepared; consult
COMMIT_RECEIPT_11.json and PUSH_RECEIPT_11.json if present for its actual status.
Use explicit byte-pinned selections, expected parent and files below2MB. Raw
checkpoints, data, logits, full generated queues and fixtures remain allocation.
Never reuse a release ID or overwrite frozen source. Unified diff artifacts
need their explicit whitespace exclusions. Private named-key push helper exists.

DOC_PREVIOUS_PINS_11.json pins prior published mutable documents. Update remote
README/SESSION_STATE by checking those bytes. Root README exists remotely only;
replace ROOT_README_BLOCK_10.md with ROOT_README_BLOCK_11.md after verifying full
root README SHA725d8053e942df71ab1a57763e1efe7358cd985a6fcfd28ccb8ca6a4cf56f86b.
Preserve older dated sections. BENCHMARK_OVERVIEW.md is the plain design.

## Completed scientific evidence

MolBACE24: native OGB GIN5/width300/dropout.5, official scaffold0, full features,
Adam LR.001/WD0, TRAIN1210/VALID151, seeds101/202/303. Four independent bodies,
G0, geometry, private correction and their combination per seed. Minimum100,
cap1000, patience50;149–210epochs,3680epochs/139840updates,7.81elapsedh. No cap
stops or failed fits. TEST closed. All outcomes retained.

Selected VALID AUC mean±SD(%), mean BCE:
M1 73.32±1.17/.779; I4 74.85±.50/.890; G0 67.78±4.39/.719;
geometry71.17±2.66/.551; correction71.76±3.91/.941; both69.28±1.80/.882.
Interaction negative all3, mean−5.873pp. Complete reports in
pilot_20261015_molbace_v1/complete_analysis_v1.

Separate privateBN3 uses private running buffers with all learned native affine
weights shared. AUC73.61416±1.05742%, BCE1.085523, +5.83639pp vsG0 in3/3,
−1.23321pp vsI4.464epochs/17632updates, no cap stops. Known normalization control,
not novelty. Different selected epochs limit causal attribution.

Mean member→pool AUC: I4 73.40965→74.84737; G0 62.00549→67.77778;
BN72.62821→73.61416. Remaining BN gap equals0.78144pp member-quality gap plus
0.45177pp smaller pooling gain, descriptive arithmetic only. Geometry+correction
has greater error diversity but weaker routes. Complete route diagnostics retained.

Complete fixed aggregation uses all27saved fits/21banks, zero forwards/training:
probability-average AUC M173.321,I474.896,G069.011,geometry70.256,correction71.587,
both70.317,BN73.468. No fixed rule closes I4 gap. Original serving unchanged.
analysis/complete_fixed_aggregation_v1.

Fixed-weight bank assignment: analysis/bn_bank_assignment_v1. Same3selectedBN
checkpoints, own/tied/all3cyclic shifts, all151VALID,15fullVALIDcalls/300CPUforwards,
zeroTRAIN/updates/TEST.30.82077s, maxRSS628108KiB. Mean ownAUC73.61416,
tied59.05983, mean shifted-score57.97721. Own beats tied and average shifts3/3;
shift1 improves seed101 to77.22. BCE own1.085523,tied1.250362,shifts1.774657;
tied improves BCE seeds202/303, so no uniform calibration claim. Banks restored.
Preflight argument failure occurred before model construction and is preserved.
Does not isolate original EMA clock/order/selection effects. Tied variance is
arithmetic mean, not pooled-population variance. Raw tensors remain remote.

Literature: switchable_norm_scope_v1 read Slimmable abstract and exact author
SwitchableBatchNorm2d/USBatchNorm2d classes; USBatchNorm uses shared affine and
private statistics. Implementation precedent, not a full-paper read or novelty.
FlowMatching/CARD/ContrastiveFlowMatching scoped conclusions and offload receipts
are already saved. No flow arm added; no need to reread without a new question.

## Running science — no partial quality promotion

Nine-fit pilot pilot_20261015_bn_geometry_correction_v1:
SPEC eacd22be19a1cc108d816cd062f87dde902a4ad67aab6fe31301730c663c2d6b;
manifest8b0e915319b77b64e9ed17f7f995a020ef9492f67697ed7d4ee3ddfa394769d2.
Started09:36:19UTC, owner840429/study840431. Remote results/bn_geometry_followup_v1;
owner results/bn_geometry_followup_v1_owner/OWNER.json. Three new arms×three seeds:
BN+geometry, BN+correction, BN+both;15completed factorial controls reused.
Same native settings/controller. One fit at a time;12hqueue cap, no retries.
At11:01UTC3/9finished (149/152/149epochs, patience), fit003has progress metadata. No partial quality
scores inspected. Read full terminal and frozen all-cell plan before interpreting.
This pilot remains separate from6460.

Older phase/private_GRU_acquisition_root_20261011_v1/acquisition_v1 remains stage2:
owner640426/worker640428,29/30banks and44/48trajectories at11:01UTC;7-daycap.
Do not restart, alter criteria, expose partialDEV or revive excluded arms.
At11:01UTC soleGPU98%,2002MiB; both scientific queues remain unfinished.
PROGRESS_METADATA_11.json records the scalar-only status observation.
Do not launch a third GPU check while both occupied slots remain.

## Expanded benchmark

6460physical fits,7datasets,8families,27fixed source presets,25recipe definitions.
Node1920=80source+320native+1520shared; molecule3840=160+640+3040;
link700=30+120+550. Total270source+1080native+5110shared. M1 reuses I4body0.
270I4exports give6730queue tasks. Ten paired split/seed rows for nodes; ten seeds
on official split for molecules/links. No HPO or removed F1/internal-placement/
untied/equipped/capacity arms. Contrasts are exploratory, not universal superiority.

Budget resource_budget_draft_v1: matched minimum237211000optimizer transitions;
source caps+matched minima247526000; allcaps2364025000. Not FLOPs or ETA. Only60
exact BACE intentions have conditional timing priors. Staged review option646fits
+27exports then5814fits+243exports keeps complete roster. No outcome-driven pruning.

Compiler4 expanded_job_binding_v1/generated_v4; runtime_link_v9; dispatcher
expanded_dispatcher_v2/generated_v4. Handoff expanded_dispatcher_v2/HANDOFF_v4.json.
102files/880243B source selection plus handoff/selection. ROOT_SOURCE_REVIEW_11
accepts source; all102pins verified and runtime core/functions unchanged from8.
BUDDY provider2 maps null shared-body metadata to0, retaining scientific body:null.
100descriptors rebound,6360reused;6460prepared/0source blocks/6730candidates,
0fully admitted. READINESS80c1084e1d5f07c9df92bbee4933a269910d1b2bb8637fe43fe325e7c34c3fd7;
runtime manifestfd6b9c5202a948a4cabb86005571369c6e012c4feb474f097f7cfaffd3ddc755.

## Remaining finite native checks

Final qualification/runtime_link_v9_combined_lifecycle_v1, manifest
06734e50a5c2be41b866acff155c6fbdcb116d9d2dc136287c6b3e2d07702511.
NOT LAUNCHED: NCN34updates/2epochs/1800s and BUDDY3170updates/2epochs/3600s.
Each4wholeVALID/3constructions/2existingexports. Frozen observer encloses same
calls with no extra forwards/RNG/tensor deserialization. Engineering binding_spec
is None, so cannot certify campaign comparison bindings.98metadata receipts checked.
Actual root resource/peer admission and available slot needed. Do not run obsolete8
qualifier. Reuse prior node/molecular/NCNC evidence; no repeated numerical tests.

MolHIV qualification/molhiv_gin_vn_cuda_resource_v1 source accepted/published10,
NOT LAUNCHED. GIN-VN/dropout.5, interaction_input_conditionalJ seed101,
1fullTRAIN32901graphs/1029updates/4116forwards plus2wholeVALID/1032forwards,
5148forwards total,1800swatchdog. Need actual allocation resource/slot admission.
Do not introduce train.pt/valid.pt: original CSV role selection must stay exact.

## Active agents and integration

Resource agent: benchmark_reporting_integration_v2. Actual compiler4 registry has
6460verified descriptors,27presets,6190expected predictor rows and1170contrasts.
Real empty campaign collector retains all missing rows; all intervals withheld.
Full generated registry/reports remote; compact receipts local. Observer source
2a992dcf68f3b229f001f88779acb061036c0d1c974ffab613f1e3c1f588ff9a frozen.
Root accepted final reporting source in ROOT_REPORTING_REVIEW_11.json. The pure
argv transform preserves backend spawn ownership; repaired cost collection keeps
retained histories with uncertain overlap. Final manifest669ad7031c75a65cde1ada29998f2f595ecc64736602b9f6f8525104dfd73f26.
Two older unpublished staged sources were preserved and CAS-updated;
REPORTING_STAGED_REPAIR_11.json records them. Do not infer observed bindings from expected hashes.

Completed portable_dispatcher_v2 source, manifest
158dbd047d0d929d68e4930f5544f9bc071f8d06c973158a1f5caaeaf0f22ade.
ROOT_PORTABLE_REVIEW_11 verifies41packagefiles and16runtime module comparisons;
only guards/CLI and3newguardhelpers differ.30local metadata/sourcechecks passed.
Host-wide flock covers reservation through solePopen/PIDcommit; cgroup-aware RAM,
permanent cohort ownership, retained fit failures, detached one-command runner,
delivered STOP acknowledgment and explicit offline STOP limits. Native/host
qualification remains pending,77disabled. No namespace/mount/host changes.
Guarded metadata projection prepare_projection.py is next afterpublication11;
no scientific work. Default all6730tasks allocated locally; three-host future
placement waits reviewed costs/actualhostauthority. Do not mistake default map
for an efficient final three-GPU assignment.

Resource agent now prepares benchmark_reporting_portable_rebind_v1 source.
It must wait for root's exact generated allocation projection receipts; preserve
original registry and all native/data/controller/evaluator fingerprints, replacing
only concrete runtime/descriptor pins and I4 runtime-derived readout authority.
No native work or77. Reporting transform remains under reserved Popen.

Completed research_next_decision_v1: four compact files,20sourcepins. Two
conditional follow-ups are matched conditional aggregation and structured
responsibility. Defer dependent run choice until nine-fit terminal; no new native
work or paper read. benchmark_review_map_v1 gives exact25method applicability:
19node/19molecule/18NCN-NCNC/19BUDDY, derived from unchanged roster.

## Next actions

Review/freeze exact current selections and publish11
with explicit sources/compact outcomes and prior commit/push receipts. Avoid
broad new engineering test matrices. Preserve source-only/incomplete status.
When a slot frees, admit only needed published finite checks. Integrate nine-fit
results only when terminal under frozen all-cell plan. Final review package must
show exact methods/applicability/settings, measured/unknown costs and one-command
GPU launch preparation. User GO remains required before giant launch.
Fresh Codex manuscript review through supplied skill follows substantial supported
paper progress, never every experiment and never with a requested verdict.
