"""Stage a reviewed fixed pilot and launch only its prospective singleton block."""
import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
SOURCE = PHASE / 'shared_private_transfer_paired_pilot_preparation_20261005_v2'
REMOTE_REPO = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
REMOTE = r'''
import base64, hashlib, json, os, socket, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
payload=json.load(sys.stdin)
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
phase=repo/'experiments_iclr/postsubmission_20260930'
assert Path.cwd().resolve()==repo and socket.gethostname()=='anogena-2-0'
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True,timeout=20).strip().splitlines()
assert len(gpu)==1 and gpu[0].split(',')[0].strip()=='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
decoded=[]
for row in payload['files']:
 rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts
 path=phase/rel;assert path.resolve().is_relative_to(phase.resolve()) and not path.is_symlink()
 raw=base64.b64decode(row['base64'],validate=True)
 assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 if path.exists():assert path.is_file() and path.read_bytes()==raw
 else:decoded.append((path,raw))
source=phase/payload['source_name'];root=phase/payload['receipt_name']
execution=phase/payload['execution_name']
assert not execution.exists() and not (root/'LAUNCH_RECEIPT.json').exists()
for path,raw in decoded:
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('xb') as f:f.write(raw)
assert sha(source/'SOURCE_MANIFEST.json')==payload['manifest_sha256']
for row in json.loads((source/'SOURCE_MANIFEST.json').read_text())['files']:
 assert sha(source/row['path'])==row['sha256']
release_path=root/'ROOT_RELEASE.json';release=json.loads(release_path.read_text())
assert release['execution_blocks']==['b0'] and release['TEST_access'] is False and release['retry'] is False
assert release['execution_directory_relative']==payload['execution_name']
for key in ['root_numeric_cohort_approved','pilot_source_review_approved','cost_completeness_approved','selection_budget_fairness_approved']:
 assert release[key] is True
py=release['python_executable']
assert Path(py).is_file() and Path(py).is_relative_to(repo)
env=dict(os.environ,**release['environment_overrides']);env.pop('PYTHONHOME',None)
argv=[py,'-B',str(source/'freeze_queue.py'),'--release',str(release_path)]
frozen=subprocess.run(argv,cwd=repo,env=env,capture_output=True,text=True,timeout=120)
freeze_receipt={'UTC':datetime.now(timezone.utc).isoformat(),'argv':argv,'exit_code':frozen.returncode,'stdout':frozen.stdout,'stderr':frozen.stderr,'fits_started':0}
with (root/'FREEZE_RECEIPT.json').open('x') as f:json.dump(freeze_receipt,f,indent=2);f.write('\n')
assert frozen.returncode==0, frozen.stderr
queue=execution/'QUEUE.json';q=json.loads(queue.read_text())
assert len(q['entries'])==10 and q['execution_blocks']==['b0']
assert len(json.loads((execution/'COHORT_PLAN.json').read_text())['cells'])==30
donor=json.loads((source/'COLLECTION_RELEASE_TEMPLATE.json').read_text())['donor_blocks'][0]
donor.update(donor_directory_relative=execution.name,replica_directory_relative=execution.name,
 registered_before_provider_first_fit=True,queue_sha256=sha(queue),root_release_sha256=sha(execution/'ROOT_RELEASE.json'),
 provider_admission={'path':str((root/'PROVIDER_ADMISSION.json').relative_to(phase)),'sha256':sha(root/'PROVIDER_ADMISSION.json')})
with (root/'DONOR_REGISTRATION.json').open('x') as f:json.dump(donor,f,indent=2,sort_keys=True);f.write('\n')
argv=[py,'-B',str(source/'run_queue.py'),'--queue',str(queue)]
with (root/'queue.stdout.log').open('xb') as stdout,(root/'queue.stderr.log').open('xb') as stderr:
 child=subprocess.Popen(argv,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
time.sleep(.2)
p=Path('/proc')/str(child.pid);raw=(p/'stat').read_text();v=raw[raw.rfind(')')+2:].split()
actual=[part.decode() for part in (p/'cmdline').read_bytes().split(bytes([0])) if part]
assert actual==argv and int(v[2])==int(v[3])==child.pid and (p/'cwd').resolve()==repo
receipt={'UTC':datetime.now(timezone.utc).isoformat(),'hostname':socket.gethostname(),'GPU_metadata':gpu,
 'queue_identity':{'PID':child.pid,'start_ticks':int(v[19]),'argv':actual,'pgid':int(v[2]),'sid':int(v[3]),'state':v[0],'cwd':str(repo)},
 'source_manifest_sha256':sha(source/'SOURCE_MANIFEST.json'),'release_sha256':sha(release_path),'cohort_plan_sha256':sha(execution/'COHORT_PLAN.json'),
 'queue_sha256':sha(queue),'provider_admission_sha256':sha(root/'PROVIDER_ADMISSION.json'),
 'donor_registration_sha256':sha(root/'DONOR_REGISTRATION.json'),
 'prospective_full_scientific_fits':30,'singleton_block_fits':10,'detached_launches':1,'TEST_access':False,'scores_read':False}
with (root/'LAUNCH_RECEIPT.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps({'receipt':receipt,'canonical_plan_utf8':(execution/'COHORT_PLAN.json').read_text(),'donor_registration_utf8':(root/'DONOR_REGISTRATION.json').read_text()}))
'''


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest-sha256',required=True)
    parser.add_argument('--audit-manifest',type=Path,required=True)
    args=parser.parse_args()
    manifest=SOURCE/'SOURCE_MANIFEST.json'
    assert sha(manifest)==args.manifest_sha256
    for row in json.loads(manifest.read_text())['files']:
        path=SOURCE/row['path']
        assert not path.is_symlink() and path.resolve().is_relative_to(SOURCE.resolve())
        assert sha(path)==row['sha256'] and path.stat().st_size==row['bytes']
        if path.suffix=='.py':ast.parse(path.read_text())
    ast.parse(REMOTE)
    audit=args.audit_manifest.resolve(strict=True)
    assert audit.is_relative_to(PHASE.resolve())
    # Root review must name and resolve the independent findings before execution.
    assert (HERE/'ROOT_REVIEW.md').is_file()
    release=json.loads((SOURCE/'ROOT_RELEASE_TEMPLATE.json').read_text())
    release.update(schema='root_released_paired_private_transfer_pilot_v2',
        root_numeric_cohort_approved=True,pilot_source_review_approved=True,
        cost_completeness_approved=True,selection_budget_fairness_approved=True,
        pilot_source_manifest_sha256=sha(manifest),
        execution_directory_relative='shared_private_transfer_paired_pilot_execution_root_20261005_v2',
        root_review_evidence=[{'path':str((HERE/'ROOT_REVIEW.md').relative_to(PHASE)),'sha256':sha(HERE/'ROOT_REVIEW.md')},
            {'path':str(audit.relative_to(PHASE)),'sha256':sha(audit)}])
    with (HERE/'ROOT_RELEASE.json').open('x') as f:json.dump(release,f,indent=2,sort_keys=True);f.write('\n')
    provider=json.loads((SOURCE/'PROVIDER_ADMISSION_TEMPLATE.json').read_text())
    science=json.loads((SOURCE/'SCIENCE_CONTRACT.json').read_text())
    base=json.loads((SOURCE/'QUALIFIED_BASE_JOB.json').read_text())
    provider.update(schema='root_admitted_singleton_private_transfer_provider_v1',approved=True,
        admitted_before_provider_first_fit=True,provider='authorized_one_GPU_allocation',
        hostname='anogena-2-0',GPU_UUID='GPU-44039938-fd82-41d2-fefd-de71514e2fac',
        authorized_repository=REMOTE_REPO,python_executable=release['python_executable'],
        runtime_versions=base['runtime_versions'],numerical_science_files=science['numerical_science_files'],
        provider_source_manifest={'path':'shared_backbone_private_transfer_training_source_20261005_v2/SOURCE_MANIFEST.json','sha256':base['source_manifest_sha256']},
        program={'path':'shared_backbone_private_transfer_training_source_20261005_v2/run.py','sha256':base['program_sha256']},
        qualification_evidence=[{'path':base['training_step_gate']['path'],'sha256':base['training_step_gate']['sha256']}]+base['runtime_qualification']['evidence'],
        root_source_review_evidence=release['root_review_evidence']+base['source_review']['evidence'])
    with (HERE/'PROVIDER_ADMISSION.json').open('x') as f:json.dump(provider,f,indent=2,sort_keys=True);f.write('\n')
    paths=[SOURCE/'SOURCE_MANIFEST.json']+[SOURCE/r['path'] for r in json.loads(manifest.read_text())['files']]
    paths += [HERE/'ROOT_REVIEW.md',HERE/'ROOT_RELEASE.json',HERE/'PROVIDER_ADMISSION.json',Path(__file__),audit]
    paths += [PHASE/r['path'] for r in release['complete_cycle_cost_evidence']]
    paths += [PHASE/release['root_numeric_decision']['path']]
    for row in release['complete_cycle_cost_evidence']+[release['root_numeric_decision']]:
        assert sha(PHASE/row['path'])==row['sha256']
    files=[]
    for path in sorted(set(paths)):
        assert path.resolve().is_relative_to(PHASE.resolve()) and not path.is_symlink()
        raw=path.read_bytes()
        files.append({'path':str(path.relative_to(PHASE)),'bytes':len(raw),'sha256':sha(path),'base64':base64.b64encode(raw).decode()})
    payload={'files':files,'source_name':SOURCE.name,'receipt_name':HERE.name,
        'execution_name':release['execution_directory_relative'],'manifest_sha256':sha(manifest)}
    # Keep a compact transfer inventory, without duplicating all transported bytes locally.
    with (HERE/'TRANSFER_INVENTORY.json').open('x') as f:
        json.dump([{k:v for k,v in row.items() if k!='base64'} for row in files],f,indent=2);f.write('\n')
    command=['ssh','-T','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt',
        '-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=20',
        'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',
        'cd '+shlex.quote(REMOTE_REPO)+' && exec python3 -I -B -c '+shlex.quote(REMOTE)]
    result=subprocess.run(command,input=json.dumps(payload),capture_output=True,text=True,timeout=150)
    with (HERE/'TRANSPORT.json').open('x') as f:
        json.dump({'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr,
            'remote_source_sha256':hashlib.sha256(REMOTE.encode()).hexdigest()},f,indent=2);f.write('\n')
    if result.returncode:raise RuntimeError(result.stderr)
    value=json.loads(result.stdout);receipt=value['receipt']
    raw=value['canonical_plan_utf8'].encode()
    assert hashlib.sha256(raw).hexdigest()==receipt['cohort_plan_sha256']
    execution=PHASE/release['execution_directory_relative'];execution.mkdir()
    with (execution/'COHORT_PLAN.json').open('xb') as f:f.write(raw)
    raw=value['donor_registration_utf8'].encode()
    assert hashlib.sha256(raw).hexdigest()==receipt['donor_registration_sha256']
    with (HERE/'DONOR_REGISTRATION.json').open('xb') as f:f.write(raw)
    with (HERE/'LAUNCH_RECEIPT.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
