"""Render the fixed comparison release only after all nine owned completions."""
import hashlib
import json
from pathlib import Path
import socket
import subprocess

REPO=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE=REPO/'experiments_iclr/postsubmission_20260930'
HERE=Path(__file__).resolve().parent
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
assert HERE==PHASE/'masked_context_pubmed_stage1_root_execution_20261010_v1'
assert not (HERE/'FAMILY_FAILURE.json').exists()
family=json.loads((HERE/'FAMILY_COMPLETE.json').read_text())
plan=json.loads((HERE/'OWNER_PLAN.json').read_text())
assert family['complete'] and family['all_nine_directly_waited']
assert family['records']==[r['record_id'] for r in plan['records']]
assert len(plan['records'])==9

def descriptor(path):
    path=path.resolve(strict=True)
    assert path.is_relative_to(PHASE)
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())

records={}
# Validate every terminal before opening any report containing quality values.
for row in plan['records']:
    identity=row['record_id']
    terminal_path=HERE/'owners'/identity/'TERMINAL_CUSTODY.json'
    terminal=json.loads(terminal_path.read_text())
    assert terminal['record_id']==identity and terminal['directly_waited']
    assert terminal['child_exit_code']==0 and terminal['cap_or_owner_failure'] is None
    assert terminal['owned_process_absence_verified'] and terminal['owned_CUDA_absence_verified']
    raw=json.loads(Path(terminal['raw_owner_terminal']['path']).read_text())
    assert raw['directly_waited'] and raw['complete'] and raw['child_exit_code']==0
    records[identity]=dict(terminal_custody=descriptor(terminal_path))
for row in plan['records']:
    identity=row['record_id']
    records[identity]['complete']=descriptor(HERE/'cells'/identity/'COMPLETE.json')

output=PHASE/'masked_context_pubmed_stage1_complete9_comparison_execution_20261010_v1'
release=HERE/'COMPLETE_COMPARISON_RELEASE.json'
assert not release.exists() and not output.exists()
spec=dict(schema='masked-context-stage1-complete-nine-comparison-release-v1',enabled=True,
    root_comparison_authorized=True,whole_nine_owned_completion_verified=True,
    TEST_access=False,automatic_retry=False,source_manifest_sha256='4ea24b47715777fee4f00b2c0d96a57657249c24d7d0541039645f4e3dd4aea9',
    records=records,output=str(output))
with release.open('x') as stream:
    json.dump(spec,stream,indent=2)
    stream.write('\n')
print(json.dumps(dict(release=str(release),sha256=hashlib.sha256(release.read_bytes()).hexdigest(),all_nine_owned_completion=True,TEST_access=False)),flush=True)
