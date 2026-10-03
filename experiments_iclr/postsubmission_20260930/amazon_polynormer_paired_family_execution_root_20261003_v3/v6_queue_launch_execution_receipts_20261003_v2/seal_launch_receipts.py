from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
PHASE=ROOT.parent


def read(p):
    return json.loads(p.read_text())


def desc(p):
    assert p.suffix not in ('.pt','.npz','.npy','.pth','.bin')
    b=p.read_bytes();b.decode('utf8')
    return dict(path=p.relative_to(PHASE).as_posix(),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))


def verify(row):
    p=PHASE/row['path'];assert p.resolve().is_relative_to(PHASE) and desc(p)==row
    return p


def write(p,v):
    with p.open('x') as h:json.dump(v,h,indent=2,allow_nan=False);h.write('\n')
    return desc(p)


launch=read(HERE/'LAUNCH_RESULT.json')
assert read(HERE/'LAUNCH_TRANSPORT.json')['exit_code']==0
q=read(verify(launch['published_queue']))
candidate=read(verify(launch['disabled_queue']))
assert q==dict(candidate,execution_authorized=True)
assert launch['changed_fields']==['execution_authorized'] and q['kind']=='all'
assert len(q['fit_releases'])==15 and q['automatic_retry_authorized'] is False and q['test_labels_authorized'] is False
monitor=read(HERE/'MONITOR_0001_RESULT.json')
assert monitor['status']=='running' and monitor['failures']==[]
assert monitor['published_queue']==launch['published_queue']
assert monitor['exact_owned_handles_only'] and monitor['signals_sent'] is False and monitor['retries_or_new_launches'] is False
for row in [*launch['fetched_descriptors'],*monitor['fetched_descriptors']]:verify(row)
handles=[]
for h in monitor['handles']:
    identity=h['current_identity'];assert identity is not None
    handles.append(dict(role=h['role'],fit_id=h.get('fit_id'),identity=identity))
assert [h['identity']['pid'] for h in handles]==[388424,388425,388426,388427]
first=next(r for r in monitor['registered_fit_progress'] if r['output_exists'])
assert first['fit_id']=='split0_gnnm_boundary_4_seed17' and first['claim_exists']
assert first['trace_progress']['actual_update']==13 and first['trace_progress']['stage']=='local'
assert first['trace_progress']['quality_fields_not_reported']
initial_source=desc(HERE/'monitor_owned_queue_initial.py')
assert initial_source['sha256']==read(HERE/'MONITOR_0001_TRANSPORT.json')['client_sha256']
failed=ROOT/'v6_queue_launch_execution_receipts_20261003_v1'
assert read(failed/'DIAGNOSTIC_RESULT.json')['paths']==[
    {'path':str((ROOT/'v6_releases/all_v1.json').relative_to(PHASE)),'exists':False},
    {'path':str((ROOT/'v6_full_schedule_v1').relative_to(PHASE)),'exists':False},
    {'path':str((ROOT/'v6_queue_detached_launch_20261003_v1').relative_to(PHASE)),'exists':False}]
monitor_argv=['/usr/bin/python3','-I','-S','-B',str((HERE/'monitor_owned_queue.py').resolve()),
              '--sequence','2','--receipt-dir',str((ROOT/'v6_queue_owned_monitoring_20261003_v1').resolve())]
handle_row=write(HERE/'MONITOR_HANDLES.json',dict(
    schema='amazon_polynormer_owned_queue_monitor_handles_v1',UTC=monitor['UTC'],route=monitor['route'],
    source=q['source'],published_queue=launch['published_queue'],remote_launch_metadata_root=launch['remote_launch_metadata_root'],
    queue_output=q['output'],registry=q['registry'],fixed15_fit_release_descriptors=q['fit_releases'],
    closure_release=q['closure_release'],owned_process_handles=handles,
    exact_read_only_monitor_source=desc(HERE/'monitor_owned_queue.py'),next_snapshot_argv=monitor_argv,
    use_fresh_snapshot_sequence_and_separate_monitor_receipt_directory=True,
    no_process_signals_or_restarts=True,notify_on_physical_failure_completion_or_required_action=True,
    ordinary_progress_and_quality_ranking_not_utility_gate=True))
closure=write(HERE/'LAUNCH_CLOSURE.json',dict(
    schema='amazon_polynormer_original15_detached_queue_initial_launch_closure_v1',
    UTC=datetime.now(timezone.utc).isoformat(),status='running_first_registered_fit',route=launch['route'],
    disabled_queue=launch['disabled_queue'],published_queue=launch['published_queue'],changed_fields=['execution_authorized'],
    resources_before_launch=launch['resources_before_launch'],ordinary_detached_unchanged_supervision=True,
    runner_source=launch['runner_source'],monitor_handles=handle_row,
    exact_owned_process_handles_at_snapshot=handles,first_fit_progress_metadata=first,
    observation_UTC=monitor['UTC'],failed_preexecution_adapter_transport_preserved=dict(
        seal=desc(failed/'SEAL.json'),diagnosis=desc(failed/'DIAGNOSTIC_RESULT.json'),
        metadata_guard_repair=desc(HERE/'ADAPTER_REPAIR_NOTE.json'),
        no_queue_release_output_or_runner_existed_after_failed_transport=True),
    physical_queue_launch_count=1,predictive_fit_cohort_original15_only=True,
    no_extra_fits_TEST_control_scoring_registry_or_replacement=True,
    automatic_retry_or_numerical_restart=False,
    complete_original15_family_closure_and_scoring_pending=True,
    no_quality_ranking_or_predictive_competence_gate_inferred_from_launch=True,
    source_neural_recipe_seeds_selectors_caps_registered_paths_and_admitted_attempt_evidence_unchanged=True))
manifest=write(HERE/'MANIFEST.json',dict(
    schema='amazon_polynormer_detached_original15_initial_launch_receipts_manifest_v1',
    UTC=datetime.now(timezone.utc).isoformat(),execution_authorized=False,
    payload=[dict(relative=p.relative_to(HERE).as_posix(),descriptor=desc(p))
             for p in sorted(HERE.rglob('*')) if p.is_file() and p.name not in ('MANIFEST.json','SEAL.json')]))
seal=write(HERE/'SEAL.json',dict(
    schema='amazon_polynormer_detached_original15_initial_launch_receipts_seal_v1',
    UTC=datetime.now(timezone.utc).isoformat(),manifest=manifest,execution_authorized=False))
print(json.dumps(dict(status='running',closure=closure,monitor_handles=handle_row,manifest=manifest,
                     seal=seal,next_read_only_monitor_argv=monitor_argv),indent=2))
