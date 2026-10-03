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
    raw=p.read_bytes();raw.decode('utf8')
    return dict(path=p.relative_to(PHASE).as_posix(),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))


def verify(row):
    p=PHASE/row['path'];assert p.resolve().is_relative_to(PHASE)
    assert desc(p)==row
    return p


def write(p,value):
    with p.open('x') as h:
        json.dump(value,h,indent=2,allow_nan=False);h.write('\n')
    return desc(p)


r=read(HERE/'PUBLICATION_RESULT.json')
assert read(HERE/'PUBLICATION_TRANSPORT.json')['exit_code']==0
for row in r['fetched_descriptors']:
    verify(row)
before=read(verify(r['resource_original_candidate']))
admitted=read(verify(r['resource_admitted']))
allowed={'status','execution_authorized','full_15_fit_schedule_authorized','root_inspection_and_release_required','root_admission_decision'}
assert {k:v for k,v in admitted.items() if k not in allowed}=={k:v for k,v in before.items() if k not in allowed}
assert admitted['execution_authorized'] and admitted['full_15_fit_schedule_authorized']
assert admitted['root_inspection_and_release_required'] is False
assert admitted['quota_status']==admitted['physical_allocation_expiry_status']=='UNKNOWN_UNCERTIFIED'
assert admitted['observed_physical_allocation_expiry_UTC'] is None
assert admitted['prospective_queue_time_budget_seconds']==691200
assert admitted['prospective_storage_budget_bytes']==32*2**30
plan=read(verify(r['packet']['plan']))
registry=read(verify(plan['original_registry']))
assert len(registry['physical_fits'])==len(r['published_fits'])==15
for registered,row in zip(registry['physical_fits'],r['published_fits']):
    initial=read(verify(row['original_disabled_candidate']))
    rebound=read(verify(row['rebound_disabled_candidate']))
    published=read(verify(row['published']))
    assert row['fit_id']==registered['id']==published['fit_id']
    assert published==dict(rebound,execution_authorized=True)
    assert {k:v for k,v in rebound.items() if k not in ('resource_admission','custody_inputs')}=={
        k:v for k,v in initial.items() if k not in ('resource_admission','custody_inputs')}
    assert rebound['custody_inputs']==initial['custody_inputs']+[r['resource_admitted']]
    assert published['resource_admission']==r['resource_admitted']
    assert published['attempt_registry']==admitted['attempt_registry']
    assert published['self_path']==registered['release_path'] and published['output']==registered['output']
    assert published['registered_claim_path']==registered['claim_path']
    assert published['caps']==admitted['caps']
    assert published['test_labels_authorized'] is False and published['automatic_retry_authorized'] is False
original_close=read(verify(plan['closure_original_candidate']))
disabled_close=read(verify(plan['closure_rebound_disabled']))
published_close=read(verify(plan['closure_published']))
assert published_close==dict(disabled_close,execution_authorized=True)
assert {k:v for k,v in disabled_close.items() if k not in ('resource_admission','custody_inputs')}=={
    k:v for k,v in original_close.items() if k not in ('resource_admission','custody_inputs')}
assert disabled_close['custody_inputs']==original_close['custody_inputs']+[r['resource_admitted']]
queue=read(verify(r['disabled_queue']))
original_queue=read(verify(plan['queue_original_candidate']))
allowed_queue={'resource_admission','fit_releases','closure_release','custody_inputs'}
assert {k:v for k,v in queue.items() if k not in allowed_queue}=={
    k:v for k,v in original_queue.items() if k not in allowed_queue}
assert queue['execution_authorized'] is False and queue['resource_admission']==r['resource_admitted']
assert queue['fit_releases']==[row['published'] for row in r['published_fits']]
assert queue['closure_release']==r['closure_published']
assert queue['attempt_registry']==admitted['attempt_registry']
assert queue['caps']['wall_seconds']==673200<admitted['prospective_queue_time_budget_seconds']
assert all(row in queue['custody_inputs'] for row in [r['resource_admitted'],r['closure_published'],*queue['fit_releases']])
packet_dir=verify(r['packet']['manifest']).parent
for entry in read(packet_dir/'MANIFEST.json')['payload']:
    verify(entry['descriptor'])
assert read(verify(r['packet']['seal']))['manifest']==r['packet']['manifest']
assert r['queue_release_output_absent'] and r['all15_fit_outputs_claims_absent']
assert r['queue_or_fit_TEST_control_scoring_or_new_registration_launched'] is False
closure=write(HERE/'PUBLICATION_CLOSURE.json',dict(
    schema='amazon_polynormer_admitted_fixed15_metadata_publication_closure_v1',
    UTC=datetime.now(timezone.utc).isoformat(),status='passed',route=r['route'],resources=r['current_resources'],
    source=admitted['source'],original_registry=plan['original_registry'],original_master_source_claim=plan['original_master_source_claim'],
    resource_candidate=r['resource_original_candidate'],resource_admitted=r['resource_admitted'],
    resource_changed_fields=r['resource_changed_fields'],root_decision=admitted['root_admission_decision'],
    published_fits=r['published_fits'],closure_published=r['closure_published'],disabled_queue=r['disabled_queue'],
    preparation_packet=r['packet'],exact_after_root_queue_authorization_argv=r['argv'],argv_cwd=r['argv_cwd'],
    identical_other_resource_fields=True,neural_recipe_seed_selector_registered_paths_and_attempt_evidence_unchanged=True,
    all15_fit_outputs_claims_and_queue_actual_release_output_absent_at_publication=True,
    observation_UTC=r['UTC'],metadata_publication_only=True,queue_execution_authorized=False,
    numerical_fit_TEST_control_scoring_registration_or_queue_launches_by_this_publication=0,
    stage_created_file_count=len(r['staging']['created_exact_files']),
    stage_existing_identical_file_count=len(r['staging']['existing_identical_files']),
    existing_files_overwritten=False,arrays_checkpoints_runtime_binaries_not_opened=True,automatic_retry=False))
manifest=write(HERE/'MANIFEST.json',dict(
    schema='amazon_polynormer_admitted_resource_publication_receipts_manifest_v1',
    UTC=datetime.now(timezone.utc).isoformat(),execution_authorized=False,
    payload=[dict(relative=p.relative_to(HERE).as_posix(),descriptor=desc(p))
             for p in sorted(HERE.rglob('*')) if p.is_file() and p.name not in ('MANIFEST.json','SEAL.json')]))
seal=write(HERE/'SEAL.json',dict(
    schema='amazon_polynormer_admitted_resource_publication_receipts_seal_v1',
    UTC=datetime.now(timezone.utc).isoformat(),manifest=manifest,execution_authorized=False))
print(json.dumps(dict(status='passed',closure=closure,manifest=manifest,seal=seal,
                     disabled_queue=r['disabled_queue'],resource=r['resource_admitted'],published_fits=15),indent=2))
