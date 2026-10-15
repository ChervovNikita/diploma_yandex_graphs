# Continuation state — release 06, 15 October 2026

## Scope and Git

Read phase AGENTS.md. Only the dedicated one-GPU allocation is authorized.
Use remote_control.py's literal anogena-2 route and hostname/sole-GPU guard
before project operations. No 18.77/MacLink, seven-GPU science, sudo, host or
mount changes, unrelated data, original-five reruns, original-score changes,
GENLINK, Desktop output or PDF compilation. Normal incidental caches are allowed.
Bounded representative pilots and engineering checks are authorized. The full
campaign still requires the user's review and GO.

The remote branch is codex/postsubmission-research-20260930. The verified
release05 parent is 7f83932007fc4d7d7502c9032a7592ce03ca4be0. This document is
prepared for release06. Read COMMIT_RECEIPT_06.json and PUSH_RECEIPT_06.json to
establish whether publication finished. Missing receipts mean pending, not
success. Never reuse a published release identity. Root owns Git. Exclude raw
datasets, checkpoints, logits, tensor contexts and full generated descriptors.
Use explicit byte-pinned source selections. Publisher limit is 2,000,000 bytes
per file. Exact unified-diff data needs its declared whitespace exclusion.

## Completed scientific work

MolBACE24 completed at 06:56:18 UTC. Official scaffold0, native GIN5/width300,
dropout0.5, Adam LR0.001/WD0, seeds101/202/303. Each seed has four independent
bodies, G0, alignment+variance, private input correction and their combination.
TRAIN1210/VALID151, TEST closed. Minimum100/cap1000/patience50. All24 fits stopped
by patience at149–210epochs. Total3,680epochs/139,840updates,7.81elapsedhours.
Parent source/spec/output immutable. Complete compact audit is under
pilot_20261015_molbace_v1/complete_analysis_v1.

Selected VALID AUC mean±SD(%): M1 73.32±1.17, I4 74.85±0.50, G0 67.78±4.39,
geometry71.17±2.66, correction71.76±3.91, combination69.28±1.80. Corrections improve
G0 in3/3 but do not beat I4 on average and worsen BCE. Combination is weaker
than either ingredient's mean AUC. Retain all outcomes.

Private-BN follow-up3 also completed. Frozen SPEC
 df726ef6f9fdfc01a859f9d3fe536120113d5e0e10da3bde609b5f9898d69986.
AUC73.61416±1.05742%, meanBCE1.085523. Deltas vsG0 +12.0513,+3.47985,+1.97802pp,
mean+5.83639. Mean delta vsI4−1.23321pp.464epochs/17,632updates, no cap stop.
Only running buffers are private. Learned native affine parameters stay shared.
Known normalization control, no novelty claim. Complete result and limitations:
pilot_20261015_bn_state_v1/complete_analysis_v1.

Complete parent route diagnosis is under analysis/complete_route_diagnostic_v1.
It used24 saved route banks, no forwards/training/TEST. I4 meanmember→pool AUC
73.41→74.85, G0 62.01→67.78, geometry64.24→71.17, correction66.07→71.76,
combination61.00→69.28. Combination reduced common ranking failures while
weakening route quality. Diverse errors alone are insufficient.

New analysis/bn_route_diagnostic_v1 completed all3 selected BN checkpoints:
3constructions/60VALIDrouteforwards/0updates,21.70seconds,0sanitywarnings.
MeanmemberAUC72.63→pool73.61. Member AUC improves overG0 in3/3 but remains below
I4 in3/3. Meanmember improvement+10.62pp. MeanmemberBCE1.122, pool1.086. Different
selected epochs preclude a buffer-only causal claim. A BN×geometry×correction
follow-up is proposed in REPORT.md (nine new fits, existing matchedBN control),
but no source/plan or launch for that future study is implied.

No heldout/confirmed/general/novelty claim follows from these selected-VALID
studies. Raw curves, checkpoints and tensors stay inside the allocation repo.

## Remaining old training

private_GRU_acquisition_root_20261011_v1 has owner640426/worker640428 and a finite
7day cap. Last compact observation08:04:52UTC:28/30fitbanks,40trajectories,
block10949_GI16/fit3 update590, selected587, stale3. At08:46:16UTC it had advanced
to update762, selected756, stale6, still28/30banks. It was the only
scientific GPU worker at08:32UTC. Do not restart, change frozen criteria, expose
partial DEV or reinstate previously excluded arms. Verify current terminal files
before describing its status. It is separate from the expanded campaign.

## Expanded design and compiler

6,460 proposed fits: node1,920, molecule3,840, link700. Seven datasets, eight
architecture families,27source presets,25recipe definitions including controls.
270reference fits +1,080 independent native bodies +5,110 shared fits. M1 is I4
body0. No HPO, dummy backbones or original-five paper reruns. Native source
settings and ambiguities are in expanded_study_design_v1/NATIVE_SETTINGS.json.
BENCHMARK_OVERVIEW.md explains counts and methods.

expanded_job_binding_v1/HANDOFF.json: v1 had6366concrete/94blocked. V2 has6370
concrete/90blocked, all intentions retained. All2480ancestor contracts and270
M1/I4ownership groups verified.270half-pool aliases:261concrete/9blocked.
Remaining blocks:45BUDDY C6/responsibility,36BUDDY interaction,9BUDDYBN, all
nonzero-seed contexts. Full descriptors stay allocation-only. Do not overwrite
v1/v2 or count a source port as full runtime qualification.

expanded_launch_preparation_v1 is dry-run preparation:6460fits+270exports,
540cohortgroups,0children. Its worker refuses execution. Actual production
dispatch is now being implemented separately in expanded_dispatcher_v1.
Only allocation/gpu0 is enabled in principle. Both77workers disabled/unprobed.
At most two owned compatible jobs/GPU requires measured admission. Cost priors
cover only60 exactly matched MolBACE intentions. No campaign ETA supported.

## New engineering results

- NCNC C3:4updates/6wholeVALID/4constructions,780.73s,peak12.89GiB.
- NCNC interactions:4cases/8updates/12VALID/8constructions/12backwards,
  1571.54s,peak13.55GiB. Sharedcgroup pressure recorded, no new OOM.
- BUDDY C3:4updates/6VALID/4constructions/8backwards,200.898s.
- BUDDYBN:2updates/3VALID/3constructions/3backwards,115.425s.
- BUDDY interactions:8updates/12VALID/8constructions/12backwards,436.094s.
- NCN C3 CUDA v2:3updatesincludingreplay/4VALID/2constructions,24.697s,
  peakallocated8.451GB,0numericalwarnings. Exact checkpoint/RNG identity retained.
- BUDDY compact provider seed1 parity: corrected gate passed2constructions,
  3170sampler calls,6TRAIN+4VALIDhydrations,0forward/update,43.775s. First gate
  failed because it omitted runner global seeding. Both attempts and86.796s
  total retained. Provider source unchanged. Only native-base input parity is
  established. Candidate production binding/lifecycle still pending.

BUDDY release handoff: qualification/BUDDY_RELEASE06_HANDOFF_v1.json.
Source manifests and compact evidence are pinned. No ten duplicate1.230GB
contexts. Seed0materialized context remains a smoke oracle on the allocation.

runtime_link_v5 frozen manifest
bfb796eef480d20662ad66db4f56ce6fce241f6f561e25e2dfff59fbca695831.
29contract checks passed. Four producer files unchanged fromv4. New code
preserves extension/policy/dependency authority and query identities and exports
actual+uniformmean from the same selected VALID call. Explicit engineering
policy permits finite2epochs without weakening production source policy.
Native lifecycle plan frozen:2epochs/34updates/3constructions/4wholeVALID/
2selectedexports, NCNC conditionalJ, interruptedcompletedepoch+stoppedresume.
No native lifecycle launch yet. Authentic metadata review found a v5 blocker:
native adapter_manifest_receipt includes adapter_hashes_verified:true, so whole
dict equality with a plain file receipt rejects it. Synthetic fixtures missed
this annotation. Source-only blocker evidence is retained. Runtime_link_v6 and
successor finite lifecycle bindings are now being prepared. Keep v5 unchanged.

## Active responsibilities and next work

- root: scientific interpretation, publication06, source/plan review and resource
  admission. No giant launch or77access. Finished brief BN route diagnosis.
- link_completion_20261015: narrow runtime_link_v6 receipt repair, authentic
  metadata contract check and qualifier/supervisor against unchanged work bounds.
  Source-only until publication and explicit root admission.
- buddy_completion_20261015: compact-provider callback successors for production
  C6/responsibility/BN/interactions. Source/contract checks only. Propose bounded
  nonzero-seed lifecycle before running it. Do not rewrite existing descriptors.
- combination_study_reuse_20261010: real guarded dispatcher successor with fake
  child tests only. Reuse claims/resume/GO/resource gates. No native training.

Collect immutable handoffs before publication. Then admit bounded native link
lifecycle on the free GPU slot after resource review. Integrate qualified BUDDY
successors in a new compiler overlay. Complete remaining real full-epoch,
selected-serving,provider and resource gates. Deliver exact final counts,
settings, measured budget and one-command workers for user review before GO.

Literature: literature/xbm_support_followup_v1 records a scoped primary read of
Cross-Batch Memory (CVPR2020),Sections3.1–3.3/Eq1–6/Algorithm1, not full-paper or
proof/code review. Detached memory can supply previous comparisons without
changing native sampler, but does not create live positive anchors/variance
support. A possible256molecule queue is unimplemented and absent from6460.

confirmation_protocol_proposal_v1 remains a separate PROPOSAL: at most one
candidate/setting,20freshpairedcohorts/120fits, its own reviewed budget and GO.
Preserve historical exposure. Sign-test claim concerns repeat-training win
probability, not mean effect or graph-population uncertainty. Fresh Codex paper
review through supplied skill follows substantial supported manuscript progress,
not each engineering check. Never request a preferred verdict.
