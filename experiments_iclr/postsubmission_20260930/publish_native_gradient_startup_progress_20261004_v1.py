"""Record actual diagnostic failure/startup progress without predictive claims."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

P = Path(__file__).resolve().parent
PUB = P / 'publication/native_gradient_startup_progress_20261004_v1'
SNAP = P / 'coordination_snapshots/20261004_native_gradient_startup_before_state_v1'
assert not PUB.exists() and not SNAP.exists()
PUB.mkdir(); SNAP.mkdir()

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')

for name in ('PUBLIC_STATUS.md', 'RESEARCH_STATE.md'):
    (SNAP/name).write_bytes((P/name).read_bytes())
prior_ledger = P/'research_ledger.json'
write(SNAP/'LEDGER_CUSTODY.json', dict(bytes=prior_ledger.stat().st_size,sha256=sha(prior_ledger),
    prior_GitHub_head='31f3f581aed94b6a445aaaecd16f713c0e9ccb79',
    local_history_preserved_by_append_and_prior_publication=True))
E1 = P/'graph_count_conditioned_pattern_minimal_gradient_execution_root_20261004_v1'
E2 = P/'graph_count_conditioned_pattern_minimal_gradient_execution_root_20261004_v2'
failed = E1/'owned_monitor01/supervision/run01/TERMINAL.json'
terminal = json.loads(failed.read_text())
assert terminal['status']=='FAILED_NO_DIAGNOSTIC_ADOPTION' and terminal['physical_exit_code']==1
assert terminal['physical_session_closed'] is True and terminal['VALID_TEST_reads'] is False
launch = json.loads((E2/'DETACHED_LAUNCH.json').read_text())
observation = json.loads((E2/'owned_monitor02/OBSERVATION.json').read_text())
assert launch['optimizer_updates']==0 and launch['VALID_TEST_access'] is False
amazon_dir = P/'amazon_polynormer_paired_family_execution_root_20261003_v3/v6_queue_owned_monitoring_20261003_v1'
amazon = json.loads((amazon_dir/'MONITOR_0025_RESULT.json').read_text())
fits = amazon['registered_fit_progress']
complete = [r for r in fits if (r.get('physical_terminal') or {}).get('status')=='success']
active = [r for r in fits if r.get('trace_progress') and not r.get('physical_terminal')]
assert len(complete)==5 and len(active)==1 and not amazon['failures']
fit = active[0]; updates=fit['trace_progress']['actual_update']
source = P/'graph_count_conditioned_pattern_minimal_gradient_preparation_20261004_v4'
review = P/'graph_count_conditioned_pattern_minimal_gradient_independent_source_review_20261004_v4'
assessment = P/'ncnc_collab_baseline_competitiveness_assessment_20261004_v1'
for d in (source,review,assessment):
    m=json.loads((d/'MANIFEST.json').read_text())
    for row in m.get('files',m.get('payload_files',[])):
        f=d/row['path']; assert f.stat().st_size==row['bytes'] and sha(f)==row['sha256']
assert json.loads((review/'REVIEW.json').read_text())['candidate_manifest_sha256']==sha(source/'MANIFEST.json')
now=datetime.now(timezone.utc).isoformat()
failure_record=dict(UTC=now,status='ADOPTED_PHYSICALLY_COMPLETE_FAILED_STARTUP_NO_DIAGNOSTIC',
    terminal=dict(path=str(failed.relative_to(P)),bytes=failed.stat().st_size,sha256=sha(failed)),
    physical_session_closed=True,automatic_retry=False,numerical_or_predictive_result=False,
    condition='CUDA memory-stat reset preceded CUDA initialization; worker failed before data/model loading.',
    successor='v4 moves reset after authenticated ordinary_runtime; separately reviewed and admitted in fresh execution directory.',
    original_attempt_preserved=True)
write(E1/'ROOT_FAILURE_ADOPTION.json',failure_record)
gradient=P/'graph_count_conditioned_pattern_native_gradient_root_adoption_20261004_v1'
actual=json.loads((gradient/'ROOT_ADOPTION.json').read_text())
assert actual['status']=='ADOPTED_NATIVE_SAME_STATE_GRADIENT_DIAGNOSTIC_ONLY'
event=dict(UTC=now,predictive_evidence_unchanged=True,manuscript_acceptance=False,
    failed_startup=failure_record,corrected_launch=launch,
    latest_diagnostic_observation=dict(path=str((E2/'owned_monitor02/OBSERVATION.json').relative_to(P)),
        bytes=(E2/'owned_monitor02/OBSERVATION.json').stat().st_size,
        sha256=sha(E2/'owned_monitor02/OBSERVATION.json'),
        UTC=observation['UTC'],progress=observation['progress'],terminal=observation['terminal']),
    Amazon=dict(UTC=amazon['UTC'],complete=5,total=15,active_fit=fit['fit_id'],updates=updates,total_updates=2700,
        failures=[],partial_quality_or_TEST_decisions=False),
    NCNC_competitiveness=dict(packet=assessment.name,manifest_sha256=sha(assessment/'MANIFEST.json'),
        official_core_recipe_matched=True,published_score_reproduction=False,broad_SOTA_verified=False,
        new_primary_reads=0,new_paper_identities=0),
    native_gradient_adoption=dict(packet=gradient.name,sha256=sha(gradient/'ROOT_ADOPTION.json'),
        aggregate_disjoint_parameter_L2=actual['aggregate_disjoint_parameter_L2'],
        child_wall_seconds=actual['child_wall_seconds'],predictive_improvement=False),
    Pubmed=dict(source_v1='pubmed_shared4_zero_update_first_batch_repeat_source_20261004_v1',
        review_v1='pubmed_shared4_zero_update_first_batch_repeat_independent_source_review_20261004_v1',
        status='BLOCKED_STORAGE_ALIAS_COMPARISON_COVERAGE',runtime_alias_defect_inferred=False,
        v2_narrow_repair_sealed=True,v2_independent_review_pending=True,scientific_GNNM_fit=False),
    prospective_decision='graph_count_conditioned_pattern_predictive_decision_root_20261004_v1/DECISION.md')
write(PUB/'RESEARCH_EVENT.json',event)
ledger=json.loads(prior_ledger.read_text())
assert 'native_gradient_startup_and_quality_progress_20261004_v1' not in ledger
ledger['native_gradient_startup_and_quality_progress_20261004_v1']=event
ledger['last_updated_utc']=ledger['updated_UTC']=ledger['current_status_update_UTC']=now
ledger['current_priority']='Completed native TRAIN diagnostic shows small connected auxiliary/model-gradient distinction, no predictive benefit yet. Inspect exact-law batching optimization before representative joint/separate/target-only fits. Maintain Amazon15 queue, review Pubmed storage-alias repair and prepare modern PENCIL source plan.'
ledger['status']='active_incomplete_predictive_extension_unconfirmed'
ledger['latest_verified_publication']=dict(commit='31f3f581aed94b6a445aaaecd16f713c0e9ccb79',
    branch='codex/postsubmission-research-20260930',GitHub_exact_ref_verified=True,
    receipt='publication/native_repeat_and_corev3_update_20261004_v1/PUSH_RECEIPT.json',
    receipt_sha256=sha(P/'publication/native_repeat_and_corev3_update_20261004_v1/PUSH_RECEIPT.json'),
    before_current_publication=True)
ledger['publication_state']=dict(branch='codex/postsubmission-research-20260930',
    latest_pushed_commit='31f3f581aed94b6a445aaaecd16f713c0e9ccb79',
    pending_changes='Exact startup failure/repair and actual gradient result, Amazon23–25 metadata, Pubmed blocked/repaired source and NCNC recipe assessment await current explicit publication.')
ledger['goal_turn_classification']=dict(classification='concrete_execution_and_source_progress',
    reason='Actual native TRAIN gradient diagnostic completed, failed startup preserved, exact repair independently reviewed, official NCNC recipe assessment completed. No new predictive advantage or acceptance.')
ledger['resource_state']['current_observation_UTC']=observation['UTC']
ledger['resource_state']['current_stage']='Amazon15 predictive queue on authorized one-GPU allocation; native TRAIN gradient diagnostic physically complete on18.77 GPU1. Pubmed sealed zero-update repair is under independent source review. No new scientific fit.'
ledger['resource_state']['gpu77']='Authorized two-GPU repository, owned native diagnostic reaped and sessionclosed; no new scientific fit claimed.'
prior_ledger.write_text(json.dumps(ledger,indent=2,sort_keys=True,allow_nan=False)+'\n')
current = ('running; no terminal or result yet' if observation['terminal'] is None else
           observation['terminal']['status']+'; see exact terminal before interpreting any diagnostic output')
status=f'''# Current GNNM research status

Updated: {now}. Goal active and incomplete. Original paper scores unchanged.

## Predictive evidence

The strongest verified result remains the five-seed official ogbl-collab TEST comparison: GNNM private completion **67.2909% Hits@50**, native single64 **66.4426%**, independent ensemble4 **67.6298%**. The exploratory single-model gain is **+0.8483 points**, positive in all five seeds; the independent ensemble remains higher.

The frozen private-minus-pooled contrast is **+0.2236 points**, with paired descriptive95% seed interval **[-0.7775,+1.2247]** and exact sign-flip p=.6875. It is inconclusive. TEST is consumed. [Complete results](ncnc_frozen_all25_heldout_root_adoption_20261004_v1/RESULTS_SUMMARY.md).

**No confirmed new methodological advantage or fresh manuscript acceptance exists.**

## Current hypothesis and actual execution

The auxiliary supervises which candidate neighbours belong together after fixing their observed TRAIN counts. J_K shares one member responsibility across both ends of an edge; J_K_sep mixes the ends independently. Serving retains the existing count-free ranker. Conditional Bernoulli likelihoods, mixtures/cardinality inference, MaskGAE degree supervision and GRAN shared components are prior. Predictive transfer beyond the separate-side control is the unresolved question.

The fixed full native TRAIN-batch gradient diagnostic completed on18.77 GPU1: **childexit0, physical sessionclosed and custody matched**. One retained forward graph supplied three reverse evaluations, with **zero optimizer updates and no VALID/TEST access**. The auxiliary reaches encoder/member parameters. Joint gradient norm is2.785% of target norm; joint-minus-separated norm is0.0561% of target norm and2.014% of joint norm. These are a small derivative opportunity at one initialization, not predictive improvement. Per-slot differences remain within the unchanged tolerance, whose absolute scale exceeds mean-reduced derivatives.

The child took87.30s, with10.69s forward and30.13/31.14s auxiliary reverse passes; peak allocator memory was28.63GB allocated/42.42GB reserved. Dispatching1,505 genuine groups and132,447 slot loops is expensive. An exact-law support-bucket optimization is being inspected before broader training. [Actual result and limits](graph_count_conditioned_pattern_native_gradient_root_adoption_20261004_v1/RESULTS_SUMMARY.md).

The first startup attempt failed before data/model loading because memory-stat reset preceded CUDA initialization; failure receipts are preserved. V4 moved that reset after authenticated initialization, with separate source review and a fresh execution directory.

Complete fabricated core-v3 CPU QA passed22,855 comparison reports. That establishes law/gradient implementation consistency, not predictive usefulness. The complete TRAIN census found both sides variable on5.8242% and both genuine on1.3599% of positive queries; negatives supply essentially no cross-side pattern signal. [Prospective decision](graph_count_conditioned_pattern_predictive_decision_root_20261004_v1/DECISION.md).

## Predictive queue and baselines

At {amazon['UTC']}, Amazon training was **5/15 complete fits**; the sixth, `{fit['fit_id']}`, had **{updates}/2700 updates**. No failures, restarts or partial quality/TEST decisions were recorded.

A completed source assessment confirms native NCNC64 matches the pinned official Collab core width/depth/100epoch recipe. Five rather than ten runs reduces replication, not per-fit capacity. Published numerical reproduction and broad recent-method competitiveness remain unverified. Feature-enabled PENCIL is one source-pinned future comparison; its resource cost is unknown. [Assessment](ncnc_collab_baseline_competitiveness_assessment_20261004_v1/REPORT.md).

Six native Pubmed baselines are complete; no GNNM predictive result exists. The earlier shared4 repeat remains failed under its unchanged rule. The native-only252-update repeat passed in a different warm-state context. Fresh review blocked the new zero-update comparison source because it omitted storage alias relationships between distinct tensor views; no runtime alias defect is inferred. V2's narrow repair is sealed and under a different fresh source review; it has not executed.

## History and boundaries

Literature index_v45 retains198 scoped conclusions across147 paper identities and two software identities; these are not full-paper read totals. The new NCNC assessment reuses existing source/literature scopes and adds no primary reads or identities. Failed experiments, original scores, decisions and reviews remain preserved.

Latest verified GitHub head before this update:31f3f581aed94b6a445aaaecd16f713c0e9ccb79. Current changes await explicit publication. Science uses authorized anogena-2 and18.77 project repositories only; the seven-GPU route is forwarding only. No sudo, PDF compilation, GENLINK, server configuration or unrelated-job changes.
'''
(P/'PUBLIC_STATUS.md').write_text(status)
(P/'RESEARCH_STATE.md').write_text(f'''# GNNM post-submission research state

Updated: {now}. Goal active and incomplete. [Current evidence](PUBLIC_STATUS.md). Prior status files preserved in {SNAP.relative_to(P)}; ledger history retained.

## Next actions

1. Preserve the physically complete actual native gradient result: small connected joint/separate parameter distinction, expensive dispatch and no predictive gain evaluated. Inspect exact-law batching before a representative study; no scientific fit follows merely from gradients or code checks.
2. Maintain Amazon's original15-fit queue: monitor25 records5 complete fits and sixth at{updates}/2700. No partial outcome selection or TEST decisions.
3. Obtain fresh independent exact review of sealed Pubmedv2 storage-alias comparison repair, then execute only the fixed zero-update comparison. Do not change tolerance or infer a causal runtime alias defect.
4. Use the prospectively recorded joint/separate/target-only comparison decision after actual native signal/resources. Stronger recent priors and an independent benchmark are required for any methodological claim; Collab TEST is already consumed.
5. Reuse the NCNC official-recipe assessment. Prepare the missing PENCIL launcher/data/query source scopes; do not infer its speed or superiority from NCNC measurements.
6. Keep support-bucket optimization separate and disabled until exact equivalence and native resource evidence. Preserve failed receipts, literature conclusions and full explicit Git inventories.
7. Revise the manuscript from supported predictive outcomes, then use fresh independent skill-based reviewers with immutable evidence and no requested verdict. Acceptance remains unachieved.

## Boundaries

Original paper scores unchanged. Authorized anogena-2 one-GPU and18.77 project repositories only for science; seven-GPU route forwarding only. No sudo, PDF compilation, GENLINK, Desktop writes, server configuration or unrelated-job changes. Small incidental caches allowed.
''')
paths = [P/'PUBLIC_STATUS.md',P/'RESEARCH_STATE.md',prior_ledger,Path(__file__),PUB/'RESEARCH_EVENT.json']
folders = [SNAP,P/'graph_count_conditioned_pattern_predictive_decision_root_20261004_v1',assessment,gradient,
    P/'publication/native_gradient_startup_progress_20261004_v1_preparation_failed',
    P/'coordination_snapshots/20261004_native_gradient_startup_before_state_v1_preparation_failed']
for version in (1,2,3,4):
    folders.extend(P/name for name in (
        f'graph_count_conditioned_pattern_minimal_gradient_preparation_20261004_v{version}',
        f'graph_count_conditioned_pattern_minimal_gradient_independent_source_review_20261004_v{version}'))
folders.extend(P/name for name in ('pubmed_shared4_zero_update_first_batch_repeat_source_20261004_v1',
    'pubmed_shared4_zero_update_first_batch_repeat_source_20261004_v2',
    'pubmed_shared4_zero_update_first_batch_repeat_independent_source_review_20261004_v1'))
for folder in folders:
    assert folder.exists()
    paths.extend(f for f in sorted(folder.rglob('*')) if f.is_file() and f.suffix in ('.py','.md','.json','.diff','.csv'))
paths.extend([P/'build_native_gradient_v4_startup_repair.py',P/'adopt_native_pattern_gradient_20261004_v1.py'])
for execution in (E1,E2):
    paths.extend(f for f in sorted(execution.iterdir()) if f.is_file() and (
        f.suffix=='.py' or f.name in ('CLIENT_PREPARATION.json','SOURCE_STAGE_INVENTORY.json',
        'SOURCE_STAGE_RECEIPT.json','RESOURCE_AND_PREREQUISITE_RECEIPT.json','ROOT_RELEASE.json',
        'ROOT_ADMISSION.json','RELEASE_STAGE_RECEIPT.json','PRENUMERICAL_GATE.json','DETACHED_LAUNCH.json',
        'ROOT_FAILURE_ADOPTION.json')))
    for folder in sorted(execution.glob('owned_monitor*')):
        paths.extend(f for f in sorted(folder.rglob('*')) if f.is_file() and f.suffix in ('.json','.txt'))
for number in (23,24,25):
    paths.extend(f for f in sorted(amazon_dir.rglob(f'MONITOR_{number:04d}*')) if f.is_file())
for name in ('PUSH_RECEIPT.json','ACKNOWLEDGEMENT.json'):
    paths.append(P/'publication/native_repeat_and_corev3_update_20261004_v1'/name)
rows=[]
for f in dict.fromkeys(paths):
    assert f.resolve().is_relative_to(P) and not f.is_symlink() and f.stat().st_size<2_000_000
    relative=str(f.relative_to(P))
    rows.append(dict(source=relative,target='experiments_iclr/postsubmission_20260930/'+relative,
                     bytes=f.stat().st_size,sha256=sha(f)))
write(PUB/'INVENTORY.json',dict(UTC=now,branch='codex/postsubmission-research-20260930',
    expected_head='31f3f581aed94b6a445aaaecd16f713c0e9ccb79',
    message='Record actual native graph-pattern gradients, preserved startup failure and baseline recipe audit',
    files=rows,remove=[]))
print(json.dumps(dict(files=len(rows),bytes=sum(r['bytes'] for r in rows),
    publication_directory=str(PUB.relative_to(P)),predictive_evidence_unchanged=True)))
