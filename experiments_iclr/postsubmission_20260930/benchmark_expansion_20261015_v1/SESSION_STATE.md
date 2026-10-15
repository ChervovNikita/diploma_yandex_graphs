# Continuation state — 15 October 2026

## Scope

Read phase AGENTS.md. Only the dedicated one-GPU allocation is authorized.
Use remote_control.py's literal route/hostname/sole-GPU guard before remote
project operations. No77/MacLink, seven-GPU science, sudo, host/mount changes,
unrelated files, original-five reruns, original-score changes, GENLINK or PDF
compilation. Ordinary caches are allowed. Bounded representative pilots and
engineering checks are authorized. The full campaign still requires user GO.

## Scientific results and running work

Parent MolBACE24 completed successfully at2026-10-15 06:56:18UTC. All24 fits,
three paired seeds101/202/303, official scaffold0, native GIN are complete.
Frozen SPEC8a1a004369e5c15de2254323a9f4e17d7b5c4e648566e15afaef519fc3bfbb1c.
Keep all parent runtime/spec/watchers unchanged. No rerun.

Complete results: pilot_20261015_molbace_v1/complete_analysis_v1/REPORT.md/.json
and SELECTED_SCORES.csv. Remote raw results/curves stay under
results/molbace_pilot_v1; full root audit under
analysis_outputs/molbace_complete_pilot_v1. The audit checked all3680epochs,
139840updates and each own first-best AUC selection.24fits lasted149–210epochs,
all stopped by patience, none bycap. Elapsed7.81hours; summedworker7.75hours.

Mean selected VALID AUC(%): M1 73.32; I4 74.85; G0 67.78; geometry71.17;
private input71.76; combination69.28. Private input improves G0 in3/3seeds,
mean+3.98points, but remains−3.09belowI4 and worsens meanBCE. Geometry improves
AUC2/3 and meanBCE; combination underperforms each ingredient's meanAUC.
No heldout, significance, generality or novelty claim follows. TEST closed.

BN follow-up supervisor826667/analysis826681 now started after parent24.
Frozen3fits, SPECdf726ef6f9fdfc01a859f9d3fe536120113d5e0e10da3bde609b5f9898d69986.
Remote results/bn_state_followup_v1. IMPORTANT: supervisor PROGRESS remains
waiting_for_parent24 during its first subprocess; inspect actual fits/*/PROGRESS
and result/log files. It updates supervisor PROGRESS only after child completion.
Do not mistake stale display for inactivity or restart it. At07:11UTC firstfit
had3192completedupdates (~84epochs); allthree completion/analysis pending.

Older private-GRU worker640428/owner640426 remains live and finite,7daycap.
Prior excluded arms remain history. It plus BN reserve two own GPU slots.
No third GPU training/model worker. C3NCNv2 CUDA gate waits for a free slot
and resource recheck. Do not modify existing frozen jobs.

## Design and scientific audit

BENCHMARK_OVERVIEW.md gives the plain-language design.
expanded_study_design_v1 retains6460 proposed fits across7datasets/8families/
27presets:270source-reference,1080matchednativebodies,5110shared.25recipes
include controls. No HPO. The old2480 descriptors are preserved and metadata
valid; they do not make the6460 matrix executable.77 remains undeployed.

expanded_binding_audit_v1 accounts for511method/preset bindings. Its frozen
snapshot had494prepared/17missing entries (4940/170sharedfits). Later ports
have closed some source gaps; create a new audit rather than rewrite v1.

confirmation_protocol_proposal_v1 is a PROPOSAL: select at most one candidate/
setting on VALID, then20freshpairedcohorts =120additionalfits, separate from6460.
Requires budget/review/GO/access-ledger freeze. Exact IUT sign-test claim is win
probability vsM1/G0/I4, not mean effect or graph-population uncertainty. No false
untouched-TEST claim; historical exposure must be retained.

molecular/TRAIN_SUPPORT_REPORT.json and RARE_CLASS_SUPPORT_NOTE.md use actual
TRAIN labels only, no model/VALID/TEST labels. MolHIV TRAIN32901 (positive1232),
nativebatch32:29.47%expected zero-positive,36.72%one-positive,33.81%>=2positive.
Thus class auxiliary support differs greatly from MolBACE. Keep native sampler;
no future queue/memory hypothesis is added to the frozen roster by this note.

## Completed new engineering work

- node_all_split_provider_audit_v1 passed40actual dataset/split role providers,
  covering80Mono/Cheb bindings, no model/TEST scoring. Remote role caches stay
  on server and do not replace native runtime transforms.
- interaction_node_molecular_v1:8CPUcases/16updates/16fullVALID/restore passed.
- link_expansion_ports_NCNC_v1:8CPUcases/16updates/24wholeVALID passed.
- link_expansion_ports_BUDDY_v2: exact native year2007 context passed, then
  8CPUcases/16updates/24wholeVALID passed. Original joint mixed-batch BCE and
  installed datasketch1.5.9 preserved.1.230GB context remains server-only.
- interaction_link_ncn_v2:4CPUcases/8updates/12wholeVALID passed under actual
  autograd-call ownership/assignment check. V1 manifest-format bug preserved.
  The initial v2 qualifier demanded repeat-backward bit equality and failed
  before any Adam step; preserve its47.13s/1construction/4reversecollections.
  Revised qualifier uses actual returned gradients,0extra reversecollections,
  unchanged method source and no tolerance search. Total both attempts8updates/
  9constructions/16reversecollections/263.41s. No scientificfit or TEST.
- runtime_v4 frozenMF6ea4735aac075314e66d0f18a46873d430f98b1803f8452ad24ba4e39ce91d98.
  Reporting move + same-call selectedmean export;4producerfiles matchv3,
  remainingAST unchanged except declared additions. Saved-tensor regression
  passed5fixtures with0native work, retaining2harnessfailures.
  Native CiteSeer/Cheb conditional-interaction gate passed2wholeTRAINepochs/
  2Adamsteps/3constructions/4wholeVALID, deliberate post-PROGRESS interruption,
  recovery and stoppedfinalization with0furtherTRAIN. Correct own selected
  actual/mean export and authority. Launcher attempt01 lacked/usr/bin/time
  and did0native work; corrected explicitattempt02 used Python accounting.
  Native molecular/link/GPU/partial-epoch failure scopes remain pending.

## Active agent ownership

- combination_study_reuse_20261010: completed NCNC/BUDDY8case gates; now BUDDY
  private-BN source/minimum smoke plan. No new fit; source-specific bank owner.
- massive_preparation_resource_admission_20261012: completed NCN interactions;
  C3NCNC source frozen1748b647dee6de992261dfc96b8d1b0b210307d786ce61b7da4f7d05a1b0b526.
  Root admitted common/split CPUgate4updates/6VALID/4constructions/2AUXprobes,
  >=18GiBheadroom/CPU2 afterreserves. Next NCNC interaction ports. No newGPU.
- conditional_fast_factors_distinct_method_20261011: completed runtimev4;
  now BUDDY C3common/split source+minimumsmokeplan, distinct from privateBN.

Check actual agent messages/terminal files before resuming. Source manifests
are immutable; readiness/results live separately. Keep concise counts and do
not conflate setup, CPU lifecycle, GPU readiness and scientific outcomes.

## Publication and next actions

Remote branch codex/postsubmission-research-20260930 is authoritative. Last
previous exact-ref verified release04commit58882a033d8eeff3a1f8448b14719a889d08013d.
Release05 collects completed scientific analysis and qualified source/notes.
Inspect numbered COMMIT_RECEIPT/PUSH_RECEIPT before any publication; never rerun
an already used release identity. Use the latest exact pushed parent and a new
unused selection/receipt identity. Root handles publication, agents do not.
Exclude datasets, checkpoints, logits, tensor fixtures and whole workspace.
Publisher v2 permits only explicit byte-pinned unified-diff artifacts to skip
whitespace lint (unchanged blank context lines are literal spaces); all source
files still receive the normal whitespace check.

Finish remaining native ports (NCNC/BUDDY context and interactions; BUDDYBN),
expanded source binding and concrete descriptors, full-native epoch/selected
serving/restore gates, and resource/cost measurements. Freeze meaningful
statistical/access decisions before the reviewed massive launch. Provide the
user exact final count, methods, settings, measured budget and one-command
worker entrypoints once ready. No three-GPU readiness or campaign GO is assumed.

Fresh Codex manuscript review through supplied paper-review skill waits for
substantial supported manuscript progress; do not seek a requested verdict or
review every engineering check. Preserve all reviews and unsuccessful results.
