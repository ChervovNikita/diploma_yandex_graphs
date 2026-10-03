"""Copy explicit prepared sources and dispatch one reviewed, score-free witness."""
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parent
PHASE = ROOT.parent
LOGIN = 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
MANIFEST_SHA = 'ee2cc8623c8752e1fc5a56dde5968c624d2e9e9b735f9534672fae6549c592d4'
REPO = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'

REMOTE = r'''
import base64,hashlib,json,os,pathlib,subprocess,sys
repo=pathlib.Path(sys.argv[1]);phase=repo/'experiments_iclr/postsubmission_20260930'
root=phase/'graph_mixed_block_execution_root_20261003_v1'
payload=json.load(sys.stdin)
assert subprocess.run(['git','rev-parse','--show-toplevel'],cwd=repo,capture_output=True,text=True,check=True).stdout.strip()==str(repo)
assert subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
for row in payload['files']:
 rel=pathlib.PurePosixPath(row['path']);assert not rel.is_absolute() and '..' not in rel.parts
 target=phase.joinpath(*rel.parts);assert target.resolve().is_relative_to(phase)
 data=base64.b64decode(row['data']);assert hashlib.sha256(data).hexdigest()==row['sha256']
 if target.exists():assert target.is_file() and target.read_bytes()==data,'Conflicting existing source: '+str(target)
 else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
packet=phase/'graph_mixed_block_training_preparation_20261003_v2'
assert hashlib.sha256((packet/'MANIFEST.json').read_bytes()).hexdigest()==payload['manifest_sha256']
review=json.loads((phase/'mixed_calibration_source_review_20261003_v2/REVIEW.json').read_text())
assert review==payload['review'] and payload['final_review_observed_by_root'] is True
check=subprocess.run([str(repo/'.venv/bin/python'),'-B',str(packet/'verify_source.py')],cwd=repo,capture_output=True,text=True,check=True)
with (root/'REMOTE_STATIC_CHECK_run02.json').open('x') as stream:stream.write(check.stdout)
runner=root/'qualification_supervisor_run02.py'
run=root/'transport/root_focused_qualification_run02';run.mkdir(parents=True,exist_ok=False)
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
with (run/'supervisor.stdout.log').open('x') as out,(run/'supervisor.stderr.log').open('x') as err:
 child=subprocess.Popen([str(repo/'.venv/bin/python'),'-B',str(runner)],cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
receipt=dict(status='focused_qualification_dispatched',PID=child.pid,repository=str(repo),manifest_sha256=payload['manifest_sha256'],qualification_only=True,scientific_training_started=False,validation_or_test_scores_requested=False)
with (run/'STARTED.json').open('x') as stream:json.dump(receipt,stream,indent=2);stream.write('\n')
print(json.dumps(receipt))
'''


def main():
    review_path = PHASE/'mixed_calibration_source_review_20261003_v2/REVIEW.json'
    review = json.loads(review_path.read_text())
    final = review.get('final_rebinding_check', {})
    # Root supplies the independently observed disposition after the reviewer
    # finishes; source hashes alone never stand in for that review.
    observed_path = ROOT/'FINAL_SOURCE_REVIEW_OBSERVED_v2.json'
    observed = json.loads(observed_path.read_text())
    assert observed['review_sha256'] == hashlib.sha256(review_path.read_bytes()).hexdigest()
    assert observed['final_manifest_sha256'] == MANIFEST_SHA
    assert observed['blocking_source_findings'] == []
    packet = PHASE/'graph_mixed_block_training_preparation_20261003_v2'
    assert hashlib.sha256((packet/'MANIFEST.json').read_bytes()).hexdigest() == MANIFEST_SHA
    sources = [packet/row['path'] for row in json.loads((packet/'MANIFEST.json').read_text())['payload']]
    sources += [packet/'MANIFEST.json', packet/'SEAL.json', packet/'STATIC_CHECK_RECEIPT.json']
    sources += [ROOT/name for name in ('FROZEN_STUDY.json','QUALIFICATION_RELEASE_run02.json','qualification_supervisor_run02.py','FINAL_SOURCE_REVIEW_OBSERVED_v2.json')]
    sources += [review_path, review_path.with_name('REPORT.md')]
    sources += [PHASE/'graph_mixed_block_objectives_execution_root_v1'/name for name in ('FROZEN_SCIENTIFIC_DESIGN.json','FROZEN_SCIENTIFIC_DESIGN_before_class_schema_correction.json','PREEXECUTION_CLASS_SCHEMA_CORRECTION.json')]
    sources += [PHASE/'graph_mixed_block_objectives_scout_20261003_v1'/name for name in ('REPORT.md','READ_SCOPES.json')]
    rows=[]
    for path in sorted(set(sources)):
        assert path.resolve().is_relative_to(PHASE) and not path.is_symlink() and path.stat().st_size < 2_000_000
        data=path.read_bytes()
        rows.append(dict(path=str(path.relative_to(PHASE)),sha256=hashlib.sha256(data).hexdigest(),data=base64.b64encode(data).decode()))
    payload=dict(files=rows,manifest_sha256=MANIFEST_SHA,review=review,final_review_observed_by_root=True)
    ssh=['ssh','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','UpdateHostKeys=no','-o','StrictHostKeyChecking=yes',LOGIN]
    result=subprocess.run([*ssh,shlex.join(['/usr/bin/python3','-I','-S','-B','-c',REMOTE,REPO])],input=json.dumps(payload),capture_output=True,text=True,timeout=60)
    receipt=dict(UTC=datetime.now(timezone.utc).isoformat(),destination=LOGIN,exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr,dispatch_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),credential_value_recorded=False,scientific_training_started=False)
    with (ROOT/'QUALIFICATION_TRANSPORT_RECEIPT_run02.json').open('x') as stream:json.dump(receipt,stream,indent=2);stream.write('\n')
    print(json.dumps(receipt))
    return result.returncode


if __name__=='__main__':
    raise SystemExit(main())
