"""Render only after every fixed new-reference owner actually closes."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,socket,subprocess
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
P=R/'experiments_iclr/postsubmission_20260930';H=Path(__file__).resolve().parent
GPU='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=5).splitlines()==[GPU]
O=P/'pubmed_strong_reference_root_execution_20261010_v1'
S=P/'combination_pubmed_strong_reference_source_20261010_v2'
assert H==P/'pubmed_strong_reference_complete12_root_20261010_v1'
assert not (H/'RELEASE.json').exists() and not (O/'science/FAMILY_FAILURE.json').exists()
family=json.loads((O/'science/FAMILY_COMPLETE.json').read_text())
plan=json.loads((O/'SCIENCE_OWNER_PLAN.json').read_text())
assert family['complete'] and family['all_records_directly_waited'] and family['records']==[r['record_id'] for r in plan['records']]
def bind(q):return dict(path=str(q),sha256=hashlib.sha256(q.read_bytes()).hexdigest())
records={}
for row in plan['records']:
    terminal=O/'owners'/row['owner_id']/'RAW_OWNER_TERMINAL.json'
    value=json.loads(terminal.read_text());absent=value['owned_absence']
    assert value['complete'] and value['directly_waited'] and value['child_exit_code']==0 and value['cap_or_owner_failure'] is None
    assert absent['owned_process_absence_verified'] and absent['owned_CUDA_absence_verified'] and row['release_sha256'] in value['argv']
    records[row['record_id']]=dict(terminal_custody=bind(terminal))
for row in plan['records']:
    records[row['record_id']]['complete']=bind(O/'science/cells'/row['record_id']/'COMPLETE.json')
spec=json.loads((S/'COMPARISON_RELEASE_TEMPLATE_DISABLED.json').read_text())
for name in ('enabled','root_authorized','source_review_approved','all_nine_new_owners_closed','all_three_anchor_owners_closed','external_hard_bound_confirmed','fresh_resource_readiness_confirmed'):
    spec[name]=True
spec.update(source_manifest_sha256=hashlib.sha256((S/'SOURCE_MANIFEST.json').read_bytes()).hexdigest(),source_review=bind(O/'ROOT_SOURCE_REVIEW.json'),new_records=records,anchors_sha256=hashlib.sha256((S/'ANCHORS.json').read_bytes()).hexdigest(),output=str(P/'pubmed_strong_reference_comparison_execution_20261010_v1'))
for condition in ('single_native','single_mean4_dropout','independent4_own'):
    record='seed9101__'+condition
    spec['engineering_records'][condition]=dict(complete=bind(O/'engineering/cells'/record/'COMPLETE.json'),terminal_custody=bind(O/'owners'/('engineering__'+record)/'RAW_OWNER_TERMINAL.json'))
free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())*1024**2
assert free>spec['limits']['GPU_bytes']
with (H/'READINESS.json').open('x') as f:json.dump(dict(UTC=datetime.now(timezone.utc).isoformat(),GPU_free_bytes=free,all_new_family_closed=True,TEST_access=False),f,indent=2)
owner=json.loads((S/'FINITE_OWNER_CONTRACT_TEMPLATE_DISABLED.json').read_text())
owner.update(enabled=True,record_id='strong_reference_complete12',limits=spec['limits'],separate_process_group=True,direct_wait_required=True,resource_caps_enforced=True,output_and_log_caps_enforced=True,owner_source=bind(H/'run_owned.py'))
with (H/'OWNER_CONTRACT.json').open('x') as f:json.dump(owner,f,indent=2)
spec.update(resource_readiness_evidence=bind(H/'READINESS.json'),external_owner_release=bind(H/'OWNER_CONTRACT.json'))
with (H/'RELEASE.json').open('x') as f:json.dump(spec,f,indent=2)
print(json.dumps(dict(complete12_reader=bind(H/'RELEASE.json'),TEST_access=False,new_fits=0)))
