"""One guarded data-only stage/export/audit; transport JSON metadata only."""
import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

HERE=Path(__file__).resolve().parent
REPO='/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
PHASE=REPO+'/experiments_iclr/postsubmission_20260930'
REMOTE=PHASE+'/'+HERE.name
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','-p','2222','-i',
    '/Users/alex/.ssh/mlspace__private_key_anogena.txt',
    'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru']
GUARD="""import socket,subprocess,pathlib,hashlib,json,os,sys,time,resource
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['/usr/bin/nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
"""


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def call(code,payload=None):
    result=subprocess.run(SSH+['/usr/bin/python3 -c '+shlex.quote(GUARD+code)],input=payload if payload is not None else b'',capture_output=True,timeout=1900)
    if result.returncode:raise RuntimeError(result.stdout.decode()+result.stderr.decode())
    return json.loads(result.stdout)


def verify_local():
    manifest=json.loads((HERE/'ACTIVATION_SOURCE_MANIFEST.json').read_text())
    for row in manifest['files']:
        path=HERE/row['path']
        assert sha(path)==row['sha256'] and path.stat().st_size==row['bytes'],row['path']
    return manifest


def stage():
    manifest=verify_local();buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w') as archive:
        for name in ['ACTIVATION_SOURCE_MANIFEST.json']+[row['path'] for row in manifest['files']]:
            archive.add(HERE/name,arcname=name,recursive=False)
    code="""import io,tarfile
root=pathlib.Path(ROOT_VALUE);phase=root.parent
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:') as archive:
 members=archive.getmembers()
 assert all(x.isfile() and not pathlib.PurePosixPath(x.name).is_absolute() and '..' not in pathlib.PurePosixPath(x.name).parts for x in members)
 assert len(members)==len({x.name for x in members})
 data={x.name:archive.extractfile(x).read() for x in members}
manifest=json.loads(data['ACTIVATION_SOURCE_MANIFEST.json'])
assert set(data)=={'ACTIVATION_SOURCE_MANIFEST.json'}|{r['path'] for r in manifest['files']}
for row in manifest['files']:assert hashlib.sha256(data[row['path']]).hexdigest()==row['sha256'] and len(data[row['path']])==row['bytes']
assert not root.exists(),'Fresh activation required; no overwrite'
source=phase/'learnable_internal_be_contrastive_multitask_suite_20261007_v6'
job=json.loads(data['jobs/collab.json']);review=json.loads(data['SOURCE_ELIGIBILITY_REVIEW.json'])
assert hashlib.sha256((source/'MANIFEST.json').read_bytes()).hexdigest()==job['source_manifest_sha256']
assert hashlib.sha256((source/'export_roles.py').read_bytes()).hexdigest()==job['source_program_sha256']
assert review['approved'] is True and review['source_manifest_sha256']==job['source_manifest_sha256']
assert hashlib.sha256(data['SOURCE_ELIGIBILITY_REVIEW.json']).hexdigest()==job['source_review']['sha256']
root.mkdir()
for name,value in data.items():
 path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(value)
for name in ['receipts','outputs','logs']:(root/name).mkdir()
print(json.dumps({'staged':True,'hostname':socket.gethostname(),'physical_GPU_UUID':'GPU-44039938-fd82-41d2-fefd-de71514e2fac','remote_directory':str(root),'activation_source_manifest_sha256':hashlib.sha256(data['ACTIVATION_SOURCE_MANIFEST.json']).hexdigest(),'files':len(manifest['files']),'source_manifest_sha256':job['source_manifest_sha256'],'source_review_sha256':job['source_review']['sha256'],'scientific_fits':0}))
""".replace('ROOT_VALUE',repr(REMOTE))
    result=call(code,buffer.getvalue())
    (HERE/'ACTIVATION_STAGE_RECEIPT.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result))


def run(action):
    verify_local()
    code="""root=pathlib.Path(ROOT_VALUE);repo=pathlib.Path(REPO_VALUE);phase=root.parent
manifest=json.loads((root/'ACTIVATION_SOURCE_MANIFEST.json').read_text())
for row in manifest['files']:
 p=root/row['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'] and p.stat().st_size==row['bytes']
action=ACTION_VALUE
receipt=root/'receipts'/str(action+'_CONTROL_EXECUTION.json')
assert not receipt.exists(),'No execution retry/overwrite'
program='execute_exports.py' if action=='export' else 'audit_collab_costed.py'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(phase/'native_ncn_dependency_overlay_20261005_v1')+':'+str(repo/'.venv/lib/python3.11/site-packages'))
started=time.monotonic();before=resource.getrusage(resource.RUSAGE_CHILDREN)
result=subprocess.run([str(phase/'native_ncn_runtime_20261005_v1/.venv/bin/python'),'-B',str(root/program),'--task','collab'],cwd=str(repo),env=env,capture_output=True,timeout=1800)
after=resource.getrusage(resource.RUSAGE_CHILDREN)
log=root/'logs'/str(action+'_control.log');assert not log.exists();log.write_bytes(result.stdout+result.stderr)
summary={'action':action,'exit_code':result.returncode,'wall_seconds':time.monotonic()-started,'user_CPU_seconds':after.ru_utime-before.ru_utime,'system_CPU_seconds':after.ru_stime-before.ru_stime,'peak_RSS_bytes':int(after.ru_maxrss)*1024,'input_blocks':after.ru_inblock-before.ru_inblock,'output_blocks':after.ru_oublock-before.ru_oublock,'log_path':str(log.relative_to(phase)),'log_sha256':hashlib.sha256(log.read_bytes()).hexdigest(),'log_bytes':log.stat().st_size,'CPU_only':True,'scientific_fits':0,'predictive_scores_computed':False,'binaries_server_only':True,'automatic_retry':False}
receipt.write_text(json.dumps(summary,indent=2,sort_keys=True)+'\\n')
print(json.dumps(summary))
""".replace('ROOT_VALUE',repr(REMOTE)).replace('REPO_VALUE',repr(REPO)).replace('ACTION_VALUE',repr(action))
    result=call(code)
    (HERE/(action.upper()+'_CONTROL_RECEIPT.json')).write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result))
    if result['exit_code']!=0:raise SystemExit(1)


def metadata():
    verify_local()
    code="""root=pathlib.Path(ROOT_VALUE)
names=['outputs/collab/ROLE_MANIFEST.json','outputs/collab/EXPORT_RECEIPT.json','outputs/collab/FAILURE.json','receipts/collab_EXECUTION.json','receipts/collab_OUTPUT_AUDIT_COSTED.json','receipts/export_CONTROL_EXECUTION.json','receipts/audit_CONTROL_EXECUTION.json']
files={}
for name in names:
 path=root/name
 if path.exists():
  payload=path.read_bytes();assert len(payload)<131072;json.loads(payload)
  files[name]={'sha256':hashlib.sha256(payload).hexdigest(),'text':payload.decode('utf-8')}
print(json.dumps({'files':files,'binaries_transferred':False}))
""".replace('ROOT_VALUE',repr(REMOTE))
    result=call(code);index={}
    for name,row in result['files'].items():
        payload=row['text'].encode();assert hashlib.sha256(payload).hexdigest()==row['sha256']
        path=HERE/'metadata'/name;path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():assert path.read_bytes()==payload
        else:path.write_bytes(payload)
        index[name]=row['sha256']
    target=HERE/'METADATA_TRANSPORT_INDEX.json'
    if target.exists():assert json.loads(target.read_text())==index
    else:target.write_text(json.dumps(index,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'exact_JSON_files':len(index),'all_hashes_matched':True,'NPZ_model_raw_transfer':False,'role_manifest':index.get('outputs/collab/ROLE_MANIFEST.json'),'audit':index.get('receipts/collab_OUTPUT_AUDIT_COSTED.json')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['stage','audit','metadata']);args=parser.parse_args()
    if args.action=='stage':stage()
    elif args.action=='metadata':metadata()
    else:run(args.action)
