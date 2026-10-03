"""Root-authorized publication: change only execution authorization."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import subprocess

REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = REPO/'experiments_iclr/postsubmission_20260930'
ROOT = PHASE/'amazon_polynormer_paired_family_execution_root_20261003_v3'
PACKET = ROOT/'v6_execution_metadata_preparation_v1/final_ready_disabled_v1'
os.chdir(REPO)
gpu = subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,timeout=15)
assert Path.cwd() == REPO and gpu.returncode == 0 and gpu.stdout.strip() == 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'


def desc(p):
    raw = p.read_bytes()
    return dict(path=str(p.relative_to(PHASE)),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))


def verify(row):
    p = PHASE/row['path']
    assert not Path(row['path']).is_absolute() and '..' not in p.parts and not p.is_symlink()
    assert desc(p) == row
    return p


plan = json.loads((PACKET/'PLAN.json').read_text())
assert desc(PACKET/'PLAN.json')['sha256'] == 'e48b579fc5b767d6deee77d81b15d3172c2b4289f7081dc111fdf62ca5ddbc95'
assert desc(ROOT/'v6_execution_metadata_preparation_v1/prepare_disabled_metadata.py') == plan['metadata_helper']
source = plan['source']
assert source['manifest']['sha256'] == 'd4160d8aeec2b770274875f9a1fb137facc9f93efd4eca6ad91e58802521ff44'
manifest = json.loads(verify(source['manifest']).read_text())
assert json.loads(verify(source['seal']).read_text())['manifest'] == source['manifest']
for row in manifest['payload']:
    p = verify(source['manifest']).parent/row['path'];actual = desc(p)
    assert (actual['sha256'],actual['bytes']) == (row['sha256'],row['bytes'])
review = json.loads(verify(plan['source_review_for_runtime_gate']).read_text())
assert review['status'] == 'passed' and review['source'] == source
assert review['independent_review_report'] == plan['independent_source_review_evidence']
consumer = json.loads((PACKET/'CONSUMER_RELEASE_DISABLED.json').read_text())
assert consumer['source'] == source and consumer['execution_authorized'] is False and consumer['test_labels_authorized'] is False
prepared = [(PACKET/'CONSUMER_RELEASE_DISABLED.json', ROOT/'V6_CONSUMER_RELEASE_v1.json', consumer)]
for mode,device in (('runtime_cpu','cpu'),('runtime_gpu','cuda:0')):
    path = PACKET/'disabled_releases'/(mode+'.json')
    value = json.loads(path.read_text())
    assert value['source'] == source and value['source_review'] == plan['source_review_for_runtime_gate']
    assert value['execution_authorized'] is False and value['independent_source_review_passed'] is True
    assert value['kind'] == 'runtime_capture' and value['device'] == device
    assert value['caps'] == dict(wall_seconds=900,rss_bytes=16*2**30,cuda_peak_allocated_bytes=16*2**30,cuda_peak_reserved_bytes=16*2**30)
    assert value['self_path'] == plan['root_release_paths'][mode] and value['output'] == plan['outputs'][mode]
    assert not (PHASE/value['output']).exists()
    prepared.append((path,PHASE/value['self_path'],value))
for before,target,value in prepared:
    assert target.is_relative_to(ROOT) and not target.exists()
    assert not any(p.is_symlink() for p in (target,*target.parents) if p.is_relative_to(PHASE))
published = []
for before,target,value in prepared:
    old = dict(value);value = dict(value,execution_authorized=True)
    assert [k for k in value if value[k] != old[k]] == ['execution_authorized']
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    actual = json.loads(target.read_text())
    assert dict(actual,execution_authorized=False) == old
    published.append(dict(candidate=desc(before),published=desc(target),changed_fields=['execution_authorized']))
print(json.dumps({'UTC':datetime.now(timezone.utc).isoformat(),'route':{'repository':str(Path.cwd()),'GPU_UUID':gpu.stdout.strip()},'source':source,'source_review':plan['source_review_for_runtime_gate'],'independent_review':plan['independent_source_review_evidence'],'published':published,'root_explicit_authorization':'Publish approved V6 consumer and CPU/GPU runtime bodies by execution authorization change only; execute each once supervised; stop on failure. No qualifier/fits/scoring.','new_registration':False,'numerical_launch':False,'immutable_files_overwritten':False},indent=2))
