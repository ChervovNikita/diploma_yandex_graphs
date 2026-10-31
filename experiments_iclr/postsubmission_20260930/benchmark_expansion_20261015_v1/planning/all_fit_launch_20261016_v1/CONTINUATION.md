# All-fit smoke tests, then authorized training

The latest user request is explicit conditional GO: run two actual optimizer updates
for every one of the 6,460 training fits, repair technical failures, then run full
training in tmux. No further user approval is required. Preserve the 270 export
tasks, frozen scientific settings, seeds/splits, canonical task IDs and three-GPU
assignments. Smoke states never initialize scientific fits.

## Current state (16 October 2026, Moscow)

No all-fit smoke PASS yet; no massive scientific training has begun.
Allocation all-fit tmux v1 was gracefully stopped via its STOP marker while
waiting, to replace it with the repaired execution package. Its generic handler
labels this supervisor_error; ALLOCATION_QUEUE_SUCCESSOR_STOP.json proves it was
an intentional zero-child stop. The separate tmux prerequisite owner remains
alive, waiting for the original PubMed pilot.
The latter runs the reviewed 38-update MolBACE lifecycle and zero-update BUDDY
selected replay. Do not repeat the paid 1,585-update BUDDY acquisition.

Server51 uses execution v2 (duplicate cgroup mount-view startup fix).
Both worker tmux panes are currently stopped after preserving failed smoke
attempts. Attempt 1 failed at zero updates because two source README files were
not deployed. All nine link-policy binding entries have now been verified and
the two exact missing files restored. Attempt 2 reached the native BUDDY provider
and failed at zero updates because its static dataset receipts hardcode the
allocation's absolute paths. Actual server51 GRAPH/TRAIN/VALID byte counts and
SHA256 hashes, and native fixed settings, all match. See
`SERVER51_BUDDY_PATH_FAILURE_DIAGNOSIS.json`.

Buddy agent owns the narrow source-preserving portable BUDDY provider/callback
repair. Link agent owns source publication and then its producer/descriptor
integration. Resource agent audits the finite source dependency closure.
Do not bypass provenance checks or fake old allocation paths in observed receipts.
Preserve provider v2 for earlier paid replay and preserve both failed attempts.
After an explicit source successor, regenerate any changed descriptors,
inventories/plans/reporting bindings before retrying. Do not reuse old smoke
PASS claims across changed execution sources (currently none exist).

## Verified source identities

- all_fit_smoke_20261016_v1 manifest:
  dd7fb42c26336d1ae96707513f75bbda9e5c9c9aae2decf42ee07d97473479fc
- all_fit_resources_20261016_v1 manifest:
  e41cb2ca51ded8793b83a68b5a1dc1575491813033392e14c711a364544b2148
- execution v1 manifest:
  e52348a34496ca70a840eadae2101c48b3b9b7b6bdcea098652e6249570410dd
- execution v2 manifest:
  76567917ace56c62c800bfa966048808ed49b0f4d5cfd7be1250f9b45aeadb87
- original deployed port v1 manifest:
  5107ebd694b1333629ec306808f64eb3e63c2b9fc24e4f4e3cbab7f0649b9209

Exact inventories and plan pins are in SMOKE_PREPARE_* and TMUX_SMOKE_* launch
receipts. Active original inventories contain 2,154 allocation fits and 4,306
server51 fits, split 2,153 per GPU. Keep both live Git HEADs at
48329b19941ea869c439e5a220a3df0e30aa1237 while the original pilot owner exists.

## Routes and monitoring

Use only E/remote_control.py::call for the authorized one-GPU allocation and
D/remote51.py::call for server51 via MacLink. No server77 operations. Seven-GPU
route is forwarding relay only. All deliberate work stays in the project repos;
no sudo, mounts, namespaces, PDFs, GENLINK or unrelated data.

`observe.py` saves exact metadata observations. Read status/counts selectively;
its full failure output can be large. Inspect actual native RESULT/TERMINAL
records, not merely tmux existence. Preserve previous pilot and GRU ownership.
At 02:46 Moscow the original GRU PIDs were no longer present; do not infer a
scientific result without its own terminal evidence. The fixed scheduler detects
original owner liveness itself. Allocation PubMed pilot was still on fit 7 of 8.

## Publication and continuation

Source snapshot bundle publication_all_fit_smoke_training_20261016_v1 was
committed and pushed: b7faa8f5061fa5acfba09e2c1bb5bab9edc3bc9f on
codex/all-fit-smoke-training-20261016; SOURCE_PUSH_RECEIPT.json verifies remote
ref and unchanged live HEAD/index. It contains only compact reviewed source and receipts,
parent2e6ed736554f23901ab6fa74a3bdda061fd719bc, with liveHEAD48329... kept separate.
Use its snapshot-aware publish helper; do not checkout/reset the live tree.
Push via the forthcoming push_source_branch.py, not the older parent==HEAD helper.
Keep credentials out of files and outputs.

Active heartbeat: run-gnnm-smoke-tests-and-training, every30minutes. Continue
repairs and actual execution, staying quiet if state is unchanged. After ALL
three summaries are PASS, retrieve the two inventories and three summaries,
run the exact global smoke validator for all6,460IDs/12,920updates, and deploy
GLOBAL_SMOKE_PASS.json to each active execution package. Tmux owners then launch
full training from fresh seeds automatically. Full training remains unauthorized
until that condition is genuinely satisfied; it is authorized immediately after.

## Repair preparation update

The BUDDY successor provider patchset is independently reviewed PASS and staged
on both hosts: buddy_provider_portability_20261016_v1/MANIFEST.json SHA
2d9c0eca0ddb5c738b477acb63d143f2fe5e864803c326c983f5f79dcd4dd712.
Its four new packages preserve old numerical methods and actual observed paths.
Seventy-six additional finite source-manifest members (238,703 bytes) have also
been staged and verified on server51; see SERVER51_TRANSITIVE_CLOSURE_REPAIR.json.

Link agent is preparing server51_campaign_port_v2 with explicit new provider
bindings, unchanged canonical IDs/settings/assignments and rederived execution
metadata. Root prepared DRAFT successors all_fit_smoke_20261016_v2,
campaign_execution_20261016_v3 and server51_campaign_projection_cache_v2.
They have no final MANIFEST yet and must NOT run until exact port2 SHA is set,
changed source checks pass and manifests are finalized. Port2 projection uses
PORT2_PROJECTION_ORIGINAL_INPUTS.json with the same canonical inputs as port1.
Use cache2 to avoid repeated hashing, then transfer generated server51 view,
verify it, make fresh smoke inventories/plans and start all three tmux workers.
Old failed smoke attempts remain untouched in execution v2.

stage_batch.py provides small compressed immutable-file transfers, avoiding
shell argument-size failures in large staging records. Source batches verify
local and remote hashes and never overwrite differing existing source.

## Frozen repair and active generation

Port2 final independently reviewed source manifest SHA
0be1ec6926b717206f403b5ed2524b7fae8a3594947d06e75b8f075f4ef7fc15
(70 members, 11,633-byte manifest). Both private-BN final-inference attachment
and owned model fingerprint were repaired and reviewed. No known source blocker.
Downstream final source pins (14 aggregate stdlib tests PASS):
- smoke2: 7eda584c1e2b387f3fa9b47c404f49fb90c9b0b553b9a6c2c6e4d2aa23cba9d4
- execution3: b714cad8746e95354e1f5adc0458f997b7c4cff574e0b81274738c0a7d1af2b0
- cache2: b5c05f31e165f5701ea7e90819209f4edbadc327c6d14034cc9bebb4a4f791a6
- cache2/prepare_cached.py: 5452d9bebd6ce780e86aed7f5439823e15daebfda77e7fff7c4088c85dcc0f64

All93 port2/downstream source files staged on allocation. Server51 transfer hit
an OpenSSH multiplex command-size limit; stage_batch.py now transmits compressed
payloads in3,000-byte chunks, verifies and installs exact files, then removes
only its completed temporary compressed payload. Retry is exec session78084;
final receipt will be PORT2_RUNTIME_STAGE_server51.json. It skips identical files.

Projection attempt01 failed in under1second because a globally supplied allocation
host-config SHA was incorrectly applied while reading server51 configs. No output
folder/native work was created. prepare_port2_attempt02.py clears that optional
single-host expected SHA for the multi-host metadata projector (the frozen port
manifest still verifies every host config). It preserves attempt01 and runs with
new CACHE_RECEIPT_02.json / PORT2_PROJECTION_RUN_02.json. Exec session14256 running;
expected fresh output server51_campaign_port_v2/generated_v1. Poll its actual
result before transferring. All original canonical input pins in
PORT2_PROJECTION_ORIGINAL_INPUTS.json remain fixed.

Next: complete generated metadata, transfer server51 archive with the existing
reviewed stream (local L/stream_campaign_metadata_to51.py), verify deployment,
prepare smoke2 inventories and execution3 plans, and restart three tmux owners.
The old allocation owner was intentionally stopped; prerequisites tmux and old
pilot remain untouched. Allocation pilot last observed7/8fits complete.

Global marker coordinator is ready at collect_global_smoke_barrier.py with
GLOBAL_SMOKE_COORDINATOR_README.md (4offline tests PASS). Root must write frozen
binding config with exact final3plans/port2/smoke2/execution3 pins. Its pending
mode creates no marker; after all6460PASS it transfers and validates evidence
then idempotently delivers the exact marker to both execution3roots, allowing
the existing tmux owners to start full training. Configure and invoke it after
launch; preserve its explicit binding SHA in this continuation.

## Live progress 16 October, 03:18 Moscow

Projection attempt02 PASS in276s. Generated server51 metadata archive SHA
b9ee9daaa1e896f5f0f63894513f730edf9104951927fefe3657d8ce2c43d565,
106,220,548bytes/12,489files, streamed without Mac payload copies and deployed
with exact member verification PASS. See PORT2_GENERATED_* receipts.
Allocation port2 verification and smoke2 inventory preparation PASS. Execution3
allocation tmux is alive and waiting prerequisite pilot (no smokes yet).
Exact plan/inventory receipts: TMUX_SMOKE3_LAUNCH_allocation.json.
Server51 deployment verification PASS; inventory/plan/launch pipeline is active
in local exec session90315 (prepare_launch_port2_host.py server51).
Root owns remote launches. observe.py now prefers execution3 when its runs exist.
Link agent is publishing the repaired source as successor of b7faa8... while
preserving live checkout. Resource agent checks coordinator transport-size fixes.

## Execution3 native failure (current blocker)

Both server51 execution3 owners started exactly2smokes each, then correctly
stopped. All4 attempted fits completed ZERO optimizer updates. New failure is
buddy_native_provider_v3/provider.py:99: actual static_tensor_bindings differs
from STATIC_PROVIDER (old path check now passes). Compact retained receipt:
SERVER51_STATIC_TENSOR_SMOKE_FAILURE.json. Buddy agent owns zero-update exact
constructor diagnosis on51 and will report numerical/seed/runtime cause before
any new source repair. Do not blindly retry the same provider. Root intentionally
stopped allocation3 while waiting (EXECUTION3_ALLOCATION_PENDING_STOP.json);
original pilot and prerequisite owner remain untouched.
Frozen3-plan coordinator config now exists: GLOBAL_SMOKE_BINDINGS_PORT2.json,
1688bytes, SHA392f785339b00b64f8ae4ed073037de116fe467e3d2098214c2d048eecad0e23.
It cannot produce PASS while these failures exist. Resource agent is repairing
its nested transport command-size issue and may run only a pending probe now.

## Static-float repair and parallel checks

BUDDY diagnosis proved only float32 HLL cards differed:7,552/471,736 values,
maxabs1.52587890625e-5/maxrel1.137607506753407e-7. Every other field incl6integer
sketch arrays matched exactly, seeds1/2 cards matched each other. No scientific
model/input change needed. Provider4 repair uses finite/shape/dtype checks and
original source/register numerical reconstruction with rtol=atol1e-6; preserves
actual input tensor and canonical historical pin.8actualCPU checks passed,
including rejecting material corruption/NaN/wrong dtype/integer lineage.
Frozen repair E/buddy_provider_numerical_portability_20261016_v1 manifestSHA
 e836777af59cd263c482fc8e4c5c31fb43ce88e6ec071595d7f01181705fa181.
Root staged its exact sources to both hosts (session99234, receipts forthcoming).
Link agent prepares port3; root has DRAFT smoke3/execution4/cache3 WITHOUT manifests,
plus freeze_port3_downstream.py and three port3 deployment helpers; execute only
when exact port3 manifest and focused tests are final.

Current source snapshot pushed ef2f2d4c254e0873c28915d422a5931234c52698, parent
b7faa8..., same codex/all-fit-smoke-training-20261016 branch. LiveHEAD/index intact.
New numerical repair not yet in that commit; follow-up required after launch.

Resource agent launched gnnm-nonbuddy-diagnostic-gpu1-20261016 tmux on51GPU1:
exact small CiteseerCheb GNNM and MolBACEGCN GNNM smoke2 fits sequentially, two
updates/VALID/reload each. These are extra engineering checks, explicitly NOT
current global smoke coverage. Uses original slot/physical/worker locks and
resource/custody code; let it finish before new51GPU1queue acquires same leases.
No full training started and all-fitPASS count remains0.

## Current deployment3 state (03:37 Moscow)

Port3 frozen manifestf8f5a2092b8b8658f7a35e043598e5a1375f287ea9f07494c3f8b391503861b5.
All93runtime files and33numerical-repair files verified on BOTH hosts. Projector3
running allocation via local exec96187 (stage_port3_and_project.py), subprocess
864244 at last observation, expected PORT3_PROJECTION_RUN_01.json/CACHE_RECEIPT_01.
No new4queue launched yet. Final downstream14tests PASS:
 smoke3 3647284234a887a79bf7b04a1827eb9b33192f047bf566a4e7853225b4f0279a
 execution4 66becfe915a06bd335a38344d7fa2198a4f8347dce486b5627adb77495daeddf
 cache3 6321a63bdf25bcfe0b904573436d537c36e0d2c5487ce9ccde5043d88bc5e2cd
 cache3wrapper d28325edab3681afdf30032d118c3a6bf7016242f52e62dd25662988fb2e39c2.
AfterprojectionPASS run deploy_port3_generated.py, then prepare_launch_port3_host.py
on bothhosts (parallelindependenthosts). Exactport3helpers alreadyfilled finalpins.
Then write GLOBAL_SMOKE_BINDINGS_PORT3.json from actualnewplans and invoke
collect_global_smoke_barrier_port3.py (SHA5ace27d9c8bf92baea05fda7befe850eb2cccce79859561d2528f4782444fd25,5testsPASS), updateobserve.pypreferexecution4.

Original allocationpilot END nowcomplete. MolBACE38-updateprerequisite PASS.
Oldpaid BUDDY zero-update selectedreplay failedbeforemodelconstruction: replay.py
passes malformedreceipt to backend.source_authority. Failure saved
BUDDY_SELECTED_PREREQUISITE_FAILURE.json. Buddyagentowns narrow harnessfix and
ZERO-update replay inallocationtmux, authorizedroot; neverredo1585updates. It will
preservefailedprereq snapshot and markALLOCATION_PREREQUISITESPASS onlyafteractual
newreplayPASS. Newallocation4owner may waitthis. RootprojectionCPUonly.
Resourceagentaudits completed24fit/12collector/18bankPubMedpilot under frozencriteria;
no newtraining orTEST and nochange to6460campaign.
Linkagentpublishes latestfixed sources snapshotv3 parentef2f2... with nohead/indexchange.
Two exactnon-BUDDYdiagnosticsPASS (CiteseerCheb/MolBACEGCN), each2updates/wholeVALID/
reload, outsideglobalcoverage. TheirGPU1tmuxexited0 and allleasesreleased.

## Live repair 16 October, 03:50 Moscow

Publication v3 completed and exact remote verified:52b3dfd7500b42a816da347f85c6535a1ec4bfb5 on codex/all-fit-smoke-training-20261016. Live HEAD/index unchanged.
Port3 projection/transfer/deployment PASSED on51. Execution4 first4smokes (2perGPU) failed before any updates: composed/runtime_bridge expects source-v2 but port3 producer manifest says-v3. Both tmux owners exited; all-fitPASS remains0. Failures retained in execution4 and latest L/observations. Link agent owns narrow port4 correction and checks all copied producer/reporting/scoring schema guards together; resource agent owns smoke4/execution5/cache4/coordinator4 successors. Root owns deployment.
Allocation prepare_launch_port3_host exec38199 was stalled at local SSH before remote preparation started. Short guarded probe found no remote preparation process/receipt. Root signalled only own localSSH15304, preserving diagnosis/stop receipts. It exited1; no remote process signalled. Port4 launch/projection helpers now compress command payload to <5000bytes. No duplicate preparation launched.
Both allocation prerequisites nowPASS, including exact BUDDY selected replay (zero training) and MolBACE lifecycle. PubMed24pilot complete; frozen residual criterion FAILS. G0−I4 mean+0.5833pp but uncertainty includes0; no new successful mechanism. Full conclusions saved in PUBMED24_CONCLUSIONS_20261016.md.
Root prepared stage_port4_and_project.py, deploy_port4_generated.py, prepare_launch_port4_host.py. prepare_port4.py has finalSHA placeholders for resource agent to fill at freeze. Next stage/project4, deploy generated4, then launch3execution5owners, set coordinator4bindings from actual plans. No full campaign starts before exact6460PASS/12920updates.

Port4 sources now frozen and staged on both hosts (95files each): L/PORT4_RUNTIME_STAGE_{allocation,server51}.json. Port4manifest226828e2e407a86c9a5691cd1ca4b40796b12e45cf1bce0d6cc172c2dc417fd6; smoke4manifest94e7f84da38ed74de9e808ca88e4fa79255966c9d07b68c703513a017d4ebd5d; execution5manifest1960d3a07693e9813e00789a4c8eb54393f197639cd89ee66cbe91076b99184c; cache4manifest0a8cf1c906ef60caaf7d4ca2ad7a4a99bd7e090699ab66812d9f58971cd1bd46. Exact14downstreamtestsPASS plus9portchecksPASS. Projection4 running via localexec14273 (stage_port4_and_project.py). Then run deploy_port4_generated.py and prepare_launch_port4_host.py allocation/server51. Root bind_global_smoke_port4.py writes bindings only from actual launch receipts; coordinator4 actual helper SHA b066a3f276f2fd7987c4b3312ba927ffcd2dc14242f8a9f1233381386cf11f00. Observer now prefers execution5 if its runs directory exists. Publication v4 being handled by linkagent; parent52b3dfd.

## Execution5 launch, 16 October about04:02 Moscow

Port4projectionPASS in265.43s; unchanged6460fits/270exports. Generatedserver51archive106238719bytesSHAfa63841490d1108ea88263d838b0bac3435f527d0c328a9ccbbc27f20a2e3c1d streamed and12489membersverifiedPASS withoutMacpayloadcopy. Local exec14273(stage/project),3484(deploy) completed0. Allocation launch93531completed0; actualreceipt L/TMUX_SMOKE5_LAUNCH_allocation.json. PlanSHA3745203f77c1f13e9630ea344b375f242d9902177385e5bfe9b49ee469a0bff7, inventorySHA1b0469aabf6281f15637131e3ab3cc90013b8c1d03c5dd1f4b275986e9fc6fc0. Actual tmux alive; latest observation6PASS/2running/2146pending. Full training not started. Server51 prepare/launch localexec97112stillrunning; wait and inspect actualreceipt before globalbinding.
Newsourcepublication pushed2782129a0e89724865eca89bde02aec314797d62, exactremoteverified, parent52b3dfd; publicationv4FINAL_CHECKS receipt available. LiveHEAD/indexintact.

## All three execution5 queues active — 16 October, 04:03 Moscow

Server51 launch session 97112 completed with exit 0. Both launch receipts are
saved as `TMUX_SMOKE5_LAUNCH_allocation.json` and `TMUX_SMOKE5_LAUNCH_server51.json`.
All three tmux owners were alive at the latest observation, with two active
smokes each and 18 PASS: allocation 14, server51 GPU0 2, server51 GPU1 2.
There were no failures in execution5. The BUDDY cells that previously failed
at the source-schema guard now pass. Full training has not started.

Exact server51 plan hashes:
- GPU0: `0d219057695d86a6670a5779d50b13867049f5dc05f3df0c0a876b2c38636c51`
- GPU1: `13197fc3c785fcfb50870e7ed662edf7e03d883ce39eb673aefab3cb615f3129`
- 4,306-fit inventory: `a47e18bf01d0a7d72a94e5d6075dbf1832bdcf9b2ac423af662ee7a099d4ac23`

The exact coordinator binding is `GLOBAL_SMOKE_BINDINGS_PORT4.json` (1,688 bytes),
SHA `151272a149224e3bc1a9a8c20aa1abdceefc592babb6e3116d5f6046bbcef742`.
Initial probe session 37472 completed with exit 0: all three workers pending,
zero markers deployed, zero large files transferred. The receipt is
`GLOBAL_SMOKE_PORT4_PENDING_PROBE_01.json`.

Run the coordinator with:

```sh
python3 -B '/Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/benchmark_expansion_20261015_v1/planning/all_fit_launch_20261016_v1/collect_global_smoke_barrier_port4.py' \
  --bindings '/Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/benchmark_expansion_20261015_v1/planning/all_fit_launch_20261016_v1/GLOBAL_SMOKE_BINDINGS_PORT4.json' \
  --bindings-sha256 151272a149224e3bc1a9a8c20aa1abdceefc592babb6e3116d5f6046bbcef742
```

Continue observing the three queues and repairing actual failures while
preserving all prior outcomes. No further user approval is needed. Only all
6,460 PASS / 12,920 native updates permit the global marker; existing tmux
owners then launch fresh full fits. The active
`run-gnnm-smoke-tests-and-training` automation should use this latest state.

## Live observation — 16 October, 04:22 Moscow

Execution5: allocation 106 PASS, 2 running, 2,046 pending; tmux alive.
Server51 GPU0 and GPU1 each have 3 PASS, 1 failed, 2,149 pending; both
owners exited after the failure. Total 112 PASS / 6,460, 2 failed, 2 running.
The two failures are the same BUDDY configuration at seeds1/2:
`wrapper-boundary_private_half_pool/normalization-route-private-v1`.
Failure occurred before any optimizer updates: `AttributeError: LinkEpochAdapter
object has no attribute policy_receipt` in port4/composed/runtime_bridge.py
`_helper_binding`. Exact traces are preserved in the latest host observations
and remote execution5 failed attempts. Do not blindly retry unchanged code.

Root delegated narrow port5 helper metadata repair to link_completion_20261015,
including a check of all link adapter metadata interfaces to avoid repeated
missing-attribute repairs. Resource agent massive_preparation_resource_admission_20261012
is examining minimal scheduler-only continuation: retain failed receipts,
continue independent pending smokes, and keep full training blocked until
every exact fit passes. Neither agent is authorized to launch remotely in these
subtasks. Root owns launches. Allocation remains running and must not be
interrupted simply because the other workers failed. Frozen port4 global
binding is still valid for its existing evidence and cannot pass with failures.
No full training has started.

## BUDDY repair verified; queues continued — 16 October, about 04:40 Moscow

Port5 fixed the actual LinkEpochAdapter metadata contract, preserving native
training/scientific settings. Final port5 manifest: 3fa47ca1ec206990fd25e37aa2f273e97aae8ec9e37c47800e9cfeda75654d13.
Actual exact failed-cell GPU1 diagnostic PASSED: two TRAIN updates, complete
160,084-record VALID, exact parameter and optimizer reload, no TEST or extra
TRAIN. Diagnostic tmux exited 0. Full source-bound evidence is
BUDDY_POLICY_DIAGNOSTIC_OBSERVATION.json. This extra diagnostic is explicitly
excluded from global all-fit coverage; its temporary checkpoint was removed
only after successful reload.

Both server51 queues resumed under campaign_smoke_continuation_20261016_v1
(manifest 033d567d3ce6c60fc70b854d149b1e2d961d806fbfa2750dc66b1f8c661f56db).
They retain PASS/failed attempts, never retry failed unchanged IDs, and continue
other pending smokes. They cannot launch full training. GPU1 was briefly drained
for the repair diagnostic, then resumed successfully; see
SERVER51_GPU1_AFTER_POLICY_DIAGNOSTIC_LAUNCH.json. Allocation continued throughout.

Port5 downstream sources are final: smoke5 / execution6 / cache5 / coordinator5;
exact pins and 19 focused checks are in PORT5_DOWNSTREAM_SOURCE_FREEZE.json.
Root stage_port5_and_project.py is running in exec46904. Root
finish_port5_rollout.py is running in exec78149: it waits for successful
projection, deploys the exact generated server51 metadata, gracefully drains
all old children without signals, launches the three new execution6 queues
in tmux, binds coordinator5, and performs its pending probe. DO NOT duplicate
these operations while that process is live. Observer now prefers execution6.
The old source attempts remain preserved; the modest early smoke set will be
rechecked under the exact fixed source instead of using a complicated
cross-version evidence transfer. Full campaign remains blocked until all
6,460 required fixed-source smoke passes. No original-paper reruns.

Link agent is publishing compact reviewed source/evidence on the same branch,
parent 2782129a0e89724865eca89bde02aec314797d62. Final publication receipt pending.

## Port5 rollout completed; additional failure causes under repair — 16 October, 04:53 Moscow

Root sessions46904 and78149 completed0. All three execution6 tmux queues launched
from port5/smoke5. Exact coordinator5 binding SHA
22d28bb3f06a083df37e8ee373c05b6b10a46c977036594a3722131b45b00695.
PORT5_ROLLOUT_COMPLETE.json and GLOBAL_SMOKE_PORT5_PENDING_PROBE_01.json confirm
pending, no markers, no full training. Old execution5 queues gracefully drained
without signals: allocation249PASS; server51 GPU0 71PASS/10failed; GPU1 58PASS/9failed.
All these earlier attempts remain preserved.

Read-only audit L/SERVER51_EXECUTION5_FAILURE_INVENTORY_20261016.json found FOUR
causes among19failed attempts:8policy_receipt failures (port5 repair verified);
8missing interaction_BUDDY_production_v5/objectives.py;2private-BN delegated epoch
paths attempt a third smoke update;1MolBACE native-replica source RNG audit carries
allocation absolute paths. Do not claim port5 fixes all19.
Buddy agent owns staging exact omitted objectives.py from the source-v3 manifest
(SHA406d8e8e8e4f55b3e11dbdeca16c6f2540a841918b4a3d9bd4878ac10b7d4740,4076bytes)
and a narrow successor MolBACE projection/audit path repair. Resource agent owns
smoke6 to cap the explicit session.base train_update chain without double-counting
nested wrapper calls and retaining the optimizer guard. Link agent owns DRAFT
execution7 to continue independent pending smoke IDs after failures without
automatically retrying them; every exactPASS/global barrier still required.
Root owns regression runs and next deployment. Do not overwrite frozen sources.

Publicationv5 is pushed and exact verified:fb880e19646cfa4d235710a706b442dd8ac33f65,
parent2782129a0e89724865eca89bde02aec314797d62. LiveHEAD/index unchanged.
finish_port5_rollout.py and latest observations/continuation need later compact
publication. No original-paper reruns or scientific conclusions from these checks.

## Remaining repair release finalized and deploying — 16 October, about05:04 Moscow

Port6 fixes OGB server51 audit receipt paths, preserving24source hashes and all
semantic assumptions. Exact omitted BUDDY objectives.py restored on bothhosts
with additive dependency provenance and included in transitive source closure.
Port6manifest b64f793b9c097497395110e6a2f15ca4ad4ed654388ffc618a53690b9b8a6e24
(13084bytes/79members),26focusedtestsPASS. Smoke6 repairs delegated base epoch
callbacks with reentrancy-aware counting and exact restoration, preserving the
optimizer guard and VALID/reload;8testsPASS. Smoke6manifest
c911fef19b379a130c18718dfe1272fff146ac63fd16041d1715104f7077e4a4(2405bytes).
Execution7 now continues pending independent smokes after failures, never retries
unchanged failedIDs, and preserves full/globalPASS rules;7focusedtestsPASS.
Execution7manifest52a996e6effb77748339c57a716159a80a06272d88914d37beaa2dd680e7e890
(3951bytes). Cache6/coordinator6 six focusedtestsPASS; exact freeze/pins in
PORT6_DOWNSTREAM_SOURCE_FREEZE.json. No fresh native PASS claimed for these repairs.

Root stage_port6_and_project.py running exec33964, then finish_port6_rollout.py
running exec80372 owns deployment, graceful old-owner drain, execution7 launch,
coordinator6 binding/probe. DO NOT duplicate while live. Both source/audit packages
staged by roothelper; buddy already staged objectives/auditprovenance on bothhosts.
Old execution6 allocation continues; server51owners stopped at knownprivateBN and
missing-objectives errors. Observer nowprefers execution7 when its runs exist.
Expected receipts PORT6_PROJECTION_RUN_01.json,PORT6_ROLLOUT_COMPLETE.json,
TMUX_SMOKE7_LAUNCH_{allocation,server51}.json,GLOBAL_SMOKE_BINDINGS_PORT6.json.
Fulltraining has NOT started. All6460 exactpasses stillrequired.

## Execution7 running; eight old failures now genuinely PASS — 16 October,05:18 Moscow

Root exec33964 and80372 completed0. PORT6_ROLLOUT_COMPLETE.json confirms all3
execution7 queues launched in tmux. Actual latest compact observation around
05:17Moscow: allocation16PASS/2running; server51GPU0 8PASS/2running; GPU1
8PASS/2running; no failures. Allownersalive. Fulltraining has NOT started.
Coordinator6binding SHAb605b36e90fe86b7b58cbb366918ea7c7efdba331a33fd897962696d4f4eda96. Initial pendingprobe completed0, zero markers.
Use collect_global_smoke_barrier_port6.py with GLOBAL_SMOKE_BINDINGS_PORT6.json
and this exactSHA; automation must prefer latestexecution7, not oldfailedqueues.

PORT6_PREVIOUS_FAILURE_RECHECK_01.json checks exact old19failedIDs againstnew
execution7 receipts:8nowPASS, including BOTH private-BN seeds and missing-objective
geometry/input-conditional variants. Each verifiedresult has2actualTRAINupdates,
finiteVALID and checkpointreloadPASS. Otherspending/running, including MolBACE
source-audit cell. Do not claim that all19orall6460havepassed. Sources/core fixes
are now genuinely exercised, but continue diagnosing any remaining failures.
Observer updated to include up to4failureexamples even while phase smokecontinues,
with explicittruncation flag. This monitor update and later rollout receipts
postdate sealedpublication6; include them in nextcompactpublication when needed.

Source/publication6pushed and exactverified:68988d18ad4b9b7d28fb1afc7ba0b273e8e4bc3f,
parentfb880e19646cfa4d235710a706b442dd8ac33f65;154selected/147changedfiles.
FINAL_CHECKS PASS. LiveHEAD/index and all5prior snapshots unchanged. Sourcepack
README truthfully reflects its sealed05:08deployment-underway state.

Next: monitorexecution7, repairactualfailureswithoutautomaticunchangedretries,
preserveallpriorattempts, runcoordinator6whenall6460PASS. All3tmuxowners will
startfreshfullfits onlyafterglobalmarker; no furtherapprovalneeded.

## Connection restored and queue health verified — 16 October, about05:24 Moscow

Local MacLink controller stopped after turn interruption; root restarted the
existing paired connect command as detached local process26390 (stdoutdiscarded,
project stderrlog). Existing otherMac reconnected and authenticated server51
SSH socket remained usable. Scientific tmux jobs were never restarted or stopped.
Latest HEALTH_CHECK_20261016T022331Z.json: 106PASS, 0failed, 6running across execution7; all3tmuxownersalive.
Still smokephase, nofulltraining. Continue existing queues/coordinator6.

## Smoke verification and scheduler recovery — 19 October, about 15:50 Moscow

User requested verification everywhere. Direct guarded checks found allocation
745 PASS / 1,409 pending, idle despite live tmux; server51 GPU0 2,152 PASS / one
failed (dead pane); GPU1 2,153 PASS and waiting for the global barrier. Full
training had NOT started. Do not describe the preceding days as continuously
monitored: these conditions remained unresolved until this verification.

SMOKE_RECEIPT_AUDIT_20261019_v1/AUDIT.json independently checks exactly 6,460
unique fit assignments and 270 unique exports, no gaps/overlaps. All initial
5,050 PASS STATE -> ATTEMPT_RESULT -> NATIVE_RESULT chains validate, with
10,100 actual optimizer updates, finite TRAIN/VALID, checkpoint reload PASS,
no TEST access and frozen source bindings. This is engineering evidence.

Allocation diagnosis: memory.current was 255GB of a 261GB cgroup limit,
including 251GB clean file cache; the old scheduler credited only inactive
cache and reported 20.8GB available. No model job was active. Additive wrapper
campaign_memory_admission_20261019_v1 conservatively credits clean active and
inactive file cache, excluding shmem/dirty/writeback/unevictable, bounded by
actual cgroup limits and host MemAvailable. No cache clearing, system limit,
mount or scientific change. Manifest SHA
6905ecec8b840925d2c3e03fb9dd43f06083372a05077579082f468a93aa4c31.
Twelve focused tests PASS. Live probe reports 256.6GB available. Old owner was
gracefully stopped with no active children. Same execution7 plan/state resumed
in original tmux under explicit wrapper provenance, retaining all745 PASS.
REPAIR_LAUNCH_allocation_gpu0_20261019.json records launch. Subsequent observation
confirmed747 PASS and two active BUDDY smoke jobs: backlog is progressing.

Server51 isolated failure is CUDA OOM in Physics PolyFormer mono, long schedule,
split9/seed9 structural_class_quartile_permuted_private_v1. At admission GPU0
had78.8GiB free and no owned active reservations; at failure an unidentified
PID1495596 used59.85GiB, failing child used18.95GiB. PID not found in saved
execution1-7 native receipts; ownership is not established. Preserve attempt0001
(0 optimizer updates). Agent is freezing narrow explicit attempt0002 retry;
no other successful fit is to rerun. Full training still requires all6460 PASS.

MacLink/server51 SSH control authentication restored. Use current guarded
remote51.call; no credentials in receipts. observe.py now tails bounded bytes
and targets failed_fit_ids, avoiding whole-log reads. Memory wrapper keeps
original execution7 run/global barrier/full behavior. All prior failures,
source generations, fit descriptors, inventories and original paper scores
remain preserved. Coordinator6 binding unchanged:
b605b36e90fe86b7b58cbb366918ea7c7efdba331a33fd897962696d4f4eda96.

## Isolated server51 retry PASS — 19 October, about15:53 Moscow

campaign_isolated_smoke_retry_20261019_v1 manifest
6c8427257b9eb1ecbb7d2582fbcdfb09a6f83c53d01fbf74c6207969a3eaf212;
authorization03e13d28f722e632ba16dcf451708e7dc6f91c6883cfd55c590fb0a90ad5dea0.
Sixteen focused tests PASS. Same native descriptor/source/settings; exact failed
attempt0001 preserved. Explicit attempt0002 in original GPU0 tmux completed
PASS with exit0, two optimizer updates, finite complete VALID, exact checkpoint
reload, noTEST;83.0sec wall, peak CUDA reserved55.7GB. Receipt:
SERVER51_ISOLATED_RETRY_PASS_20261019.json. Both server51 workers now2153PASS,
waiting_for_global_all_fit_PASS; no remaining failed smoke on either server.
Allocation observed755PASS+2active while this retry completed; continues same
2154fit assignment. Do not rerun success receipts or start fulltraining early.

Root helper repair_smoke_20261019.py stages only additive checked package files,
compiles guarded remote snippets locally, preserves prior OWNER/STOP history,
and relaunches only drained original tmux owners. Initial transfer attempts hit
local/remote syntax and command-size errors before any model launch; corrected
chunked transfer and source hash verification were then successful. No scientific
attempt or prior source was overwritten. New compact publication pending.

## Verified push and final observation — 19 October,15:59 Moscow

Publication commit6adbcf17b706f73a1cd878c20e046ba3f22fee39 pushed and exact
remote ref verified on codex/all-fit-smoke-training-20261016. Parent68988d18...;
live HEAD/index unchanged. Failed noninteractive push attempt preserved;
interactive attempt02 verified without saving credentials. Receipt in
publication_all_fit_smoke_training_20261019_v1/PUSH_RECEIPT_ATTEMPT02.json.
Final allocation status: {"PASS": 807, "pending": 1345, "running": 2}, tmux alive.
Server51 both2153PASS waitingglobal. Fulltraining remains gated by allocation
completion. Use existing coordinator6 and samebinding, no successfulsmoke reruns.

## Smoke complete; readiness review — 21 October, about06:10 Moscow

Latest user asks to verify smoke completion and report before deciding on full
training. Treat this as deferring the older conditional launch authorization:
do not create/install GLOBAL_SMOKE_PASS or start new full fits yet.

Fresh guarded allocation STATUS:2154PASS, waiting_for_global_all_fit_PASS,
tmux alive, no training acceptance/global marker, about698GiB free. New
planning/readiness_20261021_v1/audit_allocation.py is a reviewed plaintext
successor to the old encoded audit transport. It was transferred via ordinary
SCP and invoked by its visible path. Complete audit verified2154PASS,4308
optimizer updates, all frozen-source/receipt chains and summary, zeroerrors.
Remote summary SHA86ca25327ad0e2fc6bce901d5d395ae2005d890c05a1f3b5bd14d5359dafccd0.
The first metadata-audit invocation lacked the coordinator-binding JSON; this
was preserved as a failed read-only attempt and fixed by copying the exact
readable configuration. No training executed or numerical result changed.

MacLink reports controller running but paired otherMac disconnected. No
connector restart or alternate route attempted. Server51's2153+2153PASS from
October19 remain preserved evidence; live GPU/disk/source state is unverified.
Thus all6460fits have recordedPASS evidence (12920passing-run optimizerupdates),
with freshallocation and historicalserver51 verification. No assignmentoverlaps.

REVIEW.md/READINESS.json in planning/readiness_20261021_v1 document source review:
no demonstrated fresh-start bug; full-epoch/resume/export lifecycle evidence
already exists for important paths. Concrete gaps: queue accepts I4 completion
without verifying bank_receipt/path; active-job disk admission does not implement
planned retention cap/checkpoint-boundary safeguards. Do not claim all-lifetime
storage or future-run failure immunity. No new massive runs or repeat smokegrid.

Readable-command skill now applies to all wrappers. Legacy compressed coordinator
and repair helper are prohibited for future dispatch. Preserve history and prepare
plaintext replacements before a later authorized launch. Direct reviewed SSH
plaintext audit was used today; no encoded executable payloads or connector changes.

User launch authorization (21 October 2026, after MacLink reconnection): start
the verified massive campaign on each of the three assigned GPUs as capacity
becomes available; leave tmux queue owners waiting on busy GPUs. This explicitly
supersedes the earlier readiness-review deferral. Complete exact all-fit PASS
verification and the reviewed operational safeguards before releasing the global
barrier. Preserve other users' jobs; no GPU preallocation, eviction or busy loop.
Keep at most two resource-compatible owned fits per GPU and frozen assignments.

## 21 October launch in progress after explicit GO

All 6,460 exact fit smoke records are independently audited PASS across both
hosts, with 12,920 passing-attempt optimizer updates, no assignment gaps or
overlaps, and 270 separate export tasks. Full global validator PASS candidate
SHA256 is 065b2574147b257ab7aa47163eaaa475f7b8a0f6df1f4a6378b4a43f30d8c4f4.
All three original tmux sessions now run the additive named safeguards wrapper,
manifest b7da088e468530267e0fd2afbaafa927c82fb6b39b5ee5008f0a1222168e9824.
Native settings, source descriptors, fit IDs and prior attempts are unchanged.

Allocation global marker was delivered; two fresh full BUDDY fits are running,
with actual Linux process titles AI Auto Research Pipeline. Server51 transfer
encountered the SSH mux command-length limit before its marker was delivered.
The preserved partial candidate is only 335,608 bytes and cannot release jobs.
A reviewed resume helper campaign_readable_release_resume_20261021_v1 now uses
8,000-byte readable blocks and the SAME v2 proof/candidate. Root exec session
81923 owns this delivery; inspect completion before any retry. Server51 GPU1
was externally busy with only 198 MiB free; the active named owner will wait
for capacity after marker release. No other user's job was modified.

## 21 October global release complete

Root exec81923 completed successfully. The identical global marker
065b2574147b257ab7aa47163eaaa475f7b8a0f6df1f4a6378b4a43f30d8c4f4 is
now delivered on BOTH hosts; do not re-release or restart existing owners.
Coordinator v2 proof retains DELIVERED-allocation.json, DELIVERED-server51.json
and successful RESUME_RELEASE receipt. Readiness folder STARTED_TRAINING.json
records all three owners in full phase with accepted barriers and live tmux.
Allocation has two running BUDDY fits (PIDs1909014,1909017), each at4 complete
epochs/6340 optimizer updates. Server51GPU0 has two fresh BUDDY fits
(PIDs2469460,2469471); GPU1 has zero children and waits for GPU memory.
Exact actual process title is AI Auto Research Pipeline for all four children
and all three supervisors. Only the campaign processes were named.

Monitor via the named plaintext planning/readiness_reconnected_20261021_v1/
monitor_training.py --host allocation|server51 on each guarded route. Source
SHA256 3f32afa82718dd90cf191f6477f73cb4b32753e164f76204b72cf88343c7b73a.
Do not rerun smokes, replace frozen sources or use legacy encoded transports.
Keep other users' jobs intact; max2 compatible owned jobs/GPU. Training starts
automatically on GPU1 when capacity becomes available without either Mac online.

Final verification at about03:58UTC: both server51GPU0 children completed their
first native full epoch (1,585 optimizer updates each), with no queue error.
Allocation was last checked at6 full epochs/9,510 updates per child. GPU1 still
waits for capacity; all three owners remain live in tmux. No fits were rerun.
Source and compact launch evidence committed/pushed and exact remote ref verified:
cdd99d9713f7ca1999fef18187a7e7d57ed29e83 on codex/all-fit-smoke-training-20261016,
parent6adbcf17b706f73a1cd878c20e046ba3f22fee39. LiveHEAD/index unchanged. Receipt:
publication_all_fit_smoke_training_20261021_v1/PUSH_RECEIPT_ATTEMPT02.json.
The first interactive push attachment was interrupted before authentication;
the second succeeded. No credentials are present in source/evidence receipts.
This final continuation and later HEALTH/READY_AND_STARTED summaries postdate
the immutable publication snapshot and can be included with the next update.

## 23 October allocation reporting recovery

Live check at01:07UTC found the allocation idle in full_failed since
21October18:19UTC. Four native BUDDY fits and one I4 aggregation were complete;
two aggregation wrapper fits had finished149epochs but failed after export,
because composed_authority.py rejected a legitimate pinned README.md dependency.
The failure went undetected after launch, leaving about31hours idle.

New additive campaign_reporting_repair_20261023_v1 source manifest
01ff5b1f7377ffa6ec2f9878930424ef9ccc3f6a3bb7e19801ff1b9e1bf085fd
retains all frozen native sources, scientific settings and queue assignments.
It patches only the reporting witness to permit the exact BUDDY production README
and SHA; all other dependency checks remain active.15focused tests PASS on Mac
and allocation, plus both actual failed runtime identities pass preflight.

Explicit recovery config planning/allocation_recovery_20261023_v1/RECOVERY_CONFIG.json
SHAf0ed9d1f5c5f5ad4092610f0e480f4b81be5d08063b9c6242fc0de8211b7999f
binds only the two observed failed attempts. RecoveryWorker resumes their stopped
checkpoints; both now complete with0newTRAINepochs/updates and unchanged selected
epochs32/95. RESUME_AUDIT.json verifies exact equality of old/new VALID logits,
labels,ids and link endpoints. Previous failed attempts and oldexports preserved.

The existing dead allocation tmux pane%11 was resumed (no process signals) via
activate_allocation.py. Journal activation-1792718276409481919. New owner is
1910274; latest newchildPIDs1910452,1910456 run boundary_global_half_pool and
boundary_private_half_pool on collab-buddy/long_convergence/split0/seed0, each
normalization-route-private-v1. At01:19UTC:7complete queue tasks (6fits+I4),
2running,2235pending,no currentfailure. Actual titles AI Auto Research Pipeline.
Use planning/allocation_recovery_20261023_v1/compact_status.py for compact live
status; source SHAc6b9012feff9c7b645dbd46de05439d43267c8a0f10240c4ba95b5cbf6c99441.
Do not restart the old named supervisor or rerun smokes. Server51 unchanged and
not inspected today. No new scientific quality claim; TEST remains closed.

Post-recovery check at01:25UTC: both new fits completed2full epochs/3170optimizer
updates each; queue healthy with7complete,2running,2235pending. RESUME_AUDIT
confirms old/new exported tensors exactly equal. Repair/source/evidence committed
b86f8080fb1197ac159f6f5e2a94f455f7285e1c on fresh branch
codex/reporting-recovery-20261023, parentcdd99d9713f7ca1999fef18187a7e7d57ed29e83;
liveHEAD/index unchanged. GitHubpush NOT completed: ssh.github.com:443 timedout
after keyunlock; github.com:22 connectivity also timedout. PUSH_RECEIPT.json
preserves failure. Source/compactevidence retained locally and on allocation.
Do not claim published or retry the already-completed training recovery.
