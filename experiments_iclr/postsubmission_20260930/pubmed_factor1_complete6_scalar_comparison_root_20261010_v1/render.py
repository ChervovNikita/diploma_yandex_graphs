"""Bind the complete M1 family and the reviewed finite scalar reader."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, socket, subprocess

P=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930')
H=Path(__file__).resolve().parent
S=P/'pubmed_factor1_complete6_scalar_comparison_source_20261010_v1'
M=P/'pubmed_factor1_controls_source_20261010_v1'
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
assert H==P/'pubmed_factor1_complete6_scalar_comparison_root_20261010_v1'
def sha(q):return hashlib.sha256(q.read_bytes()).hexdigest()
def bind(q):return dict(path=str(q),sha256=sha(q))
def write(q,v):
    with q.open('x') as f:json.dump(v,f,indent=2);f.write('\n')

assert sha(S/'SOURCE_MANIFEST.json')=='00167812cf20528e32b53b1d236f014e1c4c838390cfab3c3c7fae2866eb6b63'
assert not (M/'science/FAMILY_FAILURE.json').exists()
family=json.loads((M/'science/FAMILY_COMPLETE.json').read_text())
plan=json.loads((M/'SCIENCE_OWNER_PLAN.json').read_text())
assert family['complete'] and family['all_records_directly_waited'] and family['records']==[r['record_id'] for r in plan['records']]
records={}
# Confirm every actual owner before opening completion quality.
for row in plan['records']:
    q=M/'owners'/row['owner_id']/'RAW_OWNER_TERMINAL.json'
    t=json.loads(q.read_text());absence=t['owned_absence']
    assert t['complete'] and t['directly_waited'] and t['child_exit_code']==0 and t['cap_or_owner_failure'] is None
    assert absence['owned_process_absence_verified'] and absence['owned_CUDA_absence_verified'] and not t['automatic_retry']
    assert row['release_sha256']==t['argv'][-1]
    records[row['record_id']]=dict(raw_terminal=bind(q))
for row in plan['records']:records[row['record_id']]['complete']=bind(M/'science/cells'/row['record_id']/'COMPLETE.json')
spec=json.loads((S/'RELEASE_TEMPLATE_DISABLED.json').read_text())
for flag in ('enabled','root_authorized','source_review_approved','all_six_owned_closed','all_six_roster_frozen','closed_reference12_verified','external_hard_bound_confirmed','ordinary_runtime_confirmed','post_screen_exploration'):spec[flag]=True
review=dict(approved=True,UTC=datetime.now(timezone.utc).isoformat(),reviewer='root source review',source_manifest_sha256=sha(S/'SOURCE_MANIFEST.json'),checks=['full-six admission before quality','exact own-selected frozen scalars','no arrays or new model calls','all seven prospective contrasts','costs retained; selected validation remains exploration'],manuscript_review=False)
write(H/'SOURCE_REVIEW.json',review)
write(H/'OWNER_REVIEW.json',dict(approved=True,owner_sha256=sha(H/'run_owned.py'),reviewer='root finite-owner review',ordinary_execution=True,checks=['120s active/10s cleanup','512MiB RSS/GPU zero','output/log caps','separate owned group and direct wait','no retry or signals to other jobs']))
owner=dict(enabled=True,record_id='factor1_complete6_scalar_comparison',limits=spec['limits'],automatic_retry=False,separate_process_group=True,direct_wait_required=True,resource_caps_enforced=True,output_and_log_caps_enforced=True,owner_source=bind(H/'run_owned.py'),owner_review=bind(H/'OWNER_REVIEW.json'))
write(H/'FINITE_OWNER_CONTRACT.json',owner)
spec.update(source_manifest_sha256=sha(S/'SOURCE_MANIFEST.json'),source_review=bind(H/'SOURCE_REVIEW.json'),science_owner_plan=bind(M/'SCIENCE_OWNER_PLAN.json'),science_family_complete=bind(M/'science/FAMILY_COMPLETE.json'),external_owner_release=bind(H/'FINITE_OWNER_CONTRACT.json'),records=records)
for c in ('factor1_native','factor1_mean4_dropout'):
    r='seed9101__'+c
    spec['engineering_records'][c]=dict(complete=bind(M/'engineering/cells'/r/'COMPLETE.json'),raw_terminal=bind(M/'owners'/('engineering__'+r)/'RAW_OWNER_TERMINAL.json'))
write(H/'RELEASE.json',spec)
print(json.dumps(dict(reader_release=bind(H/'RELEASE.json'),all_six_closed=True,TEST_access=False)))
