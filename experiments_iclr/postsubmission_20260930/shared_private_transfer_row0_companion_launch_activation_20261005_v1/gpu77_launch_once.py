"""Launch selected companion blocks after root reviews the full spec and their jobs."""
import argparse
import ast
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import zlib

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
PROVIDER = 'gpu77'
REMOTE_REPO = '/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git'
REMOTE_PHASE = REMOTE_REPO + '/experiments_iclr/postsubmission_20260930'
CONTROL = HERE / ('control_' + PROVIDER)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def packet(expected):
    assert sha(HERE/'MANIFEST.json') == expected
    for row in read(HERE/'MANIFEST.json')['files']:
        path = HERE/row['path']
        assert not path.is_symlink() and path.resolve().is_relative_to(HERE.resolve())
        assert sha(path) == row['sha256'] and path.stat().st_size == row['bytes']
    activation = read(HERE/'ACTIVATION.json')
    assert activation['freeze_executed'] is False and activation['launch_executed'] is False
    return activation


def transfer_row(path, relative=None):
    path = Path(path)
    assert not path.is_symlink() and path.resolve(strict=True).is_relative_to(PHASE.resolve())
    raw = path.read_bytes()
    return dict(path=relative or str(path.relative_to(PHASE)), bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(), base64=base64.b64encode(raw).decode())


def transport(code, payload, identity):
    ast.parse(code)
    source = PHASE/'shared_private_transfer_gpu77_environment_execution_20261005_v1/remote_transport.py'
    assert sha(source)=='8c7626be279d3861c575f05fa38420e4e27b68a15c0e7dd517f30a3774655bf7'
    spec = importlib.util.spec_from_file_location('row0_77_existing_transport', source)
    t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t); t.HERE = CONTROL
    assert str(t.REPO) == REMOTE_REPO and str(t.REMOTE_PHASE) == REMOTE_PHASE
    encoded = base64.b64encode(zlib.compress(json.dumps(payload).encode(), 9)).decode()
    remote = 'import base64,json,zlib\npayload=json.loads(zlib.decompress(base64.b64decode('+repr(encoded)+',validate=True)))\n'+code
    ast.parse(remote)
    # Existing relay keeps the single raw transport receipt; no duplicate encoded response is saved here.
    return t.run(identity, remote)


REMOTE = r'''
from pathlib import Path
from datetime import datetime, timezone
import base64, hashlib, importlib.util, json, os, socket, subprocess, sys, time, zlib

repo=Path(payload['repo']);phase=repo/'experiments_iclr/postsubmission_20260930'
assert Path.cwd().resolve()==repo.resolve() and socket.gethostname()=='peptide'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
 with Path(path).open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write(chr(10))
def stage(rows):
 decoded=[]
 for row in rows:
  rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts
  path=phase/rel;assert path.resolve().is_relative_to(phase.resolve())
  for parent in [path,*path.parents]:
   if parent==phase.parent:break
   assert not parent.is_symlink()
  raw=base64.b64decode(row['base64'],validate=True)
  assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
  if path.exists():assert path.is_file() and path.read_bytes()==raw
  decoded.append((path,raw))
 for path,raw in decoded:
  if not path.exists():
   path.parent.mkdir(parents=True,exist_ok=True)
   with path.open('xb') as stream:stream.write(raw)

receiptroot=phase/payload['control_relative'];prep=phase/payload['source_relative'];block=payload['block'];expected=payload['expected']
assert receiptroot.is_dir() and not (receiptroot/(block+'.LAUNCH_RECEIPT.json')).exists()
stage(payload['review_files'])
assert sha(phase/payload['provider_job_review']['path'])==payload['provider_job_review']['sha256']
review=json.loads((phase/payload['provider_job_review']['path']).read_text())
assert review['root_provider_jobs_reviewed'] is True and review['full_nine_cell_spec_reviewed'] is True
assert review['provider']==payload['provider'] and review['scores_read'] is False and review['TEST_access'] is False
assert review['activation_manifest_sha256']==payload['activation_manifest_sha256']
row=payload['freeze_authentication']
assert sha(phase/row['path'])==row['sha256']==review['freeze_authentication_sha256']
frozen=json.loads((phase/row['path']).read_text())
assert frozen['provider']==payload['provider'] and frozen['cohort_plan_sha256']==payload['cohort_plan_sha256']
assert frozen['blocks'][block]==expected
sealed=json.loads((prep/'STUDY_SPEC.json').read_text())
assert review['companion_spec_sha256']==hashlib.sha256((prep/'STUDY_SPEC.json').read_bytes()).hexdigest()
assert review['reviewed_cell_ids']==[cell for cell in sealed['execution_order'] if cell.split('_',1)[0] in frozen['blocks']]
process=None;owner=None;raw_owner=None;observed_cwd=None
try:
 spec=importlib.util.spec_from_file_location('row0_companion_common',prep/'pilot_common.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
 c.bind_block([block])
 c.physical_host();c.verify_packet(payload['operational_manifest_sha256']);c.verify_training();s=c.supervisor()
 execution=phase/expected['execution_directory_relative'];queue=execution/'QUEUE.json';q=c.read(queue);release=c.read(execution/'ROOT_RELEASE.json')
 assert sha(queue)==expected['queue_sha256'] and sha(execution/'ROOT_RELEASE.json')==expected['root_release_sha256']
 assert sha(execution/'COHORT_PLAN.json')==payload['cohort_plan_sha256']==q['cohort_plan_sha256']
 assert q['execution_blocks']==release['execution_blocks']==[block] and release['new_fit_release'] is True
 assert q['retry'] is False and q['TEST_access'] is False and q['queue_hard_seconds']==95700
 assert release['root_provider_block_approved'] is True
 assert sha(execution/'PROVIDER_ADMISSION.json')==expected['provider_admission_sha256'] and sha(execution/'DONOR_REGISTRATION.json')==expected['donor_registration_sha256']
 assert q['environment_overrides']['CUDA_VISIBLE_DEVICES']==c.GPU_UUID
 assert c.GPU_UUID==expected['physical_GPU_UUID']
 assert not (execution/'QUEUE_START.json').exists() and not (execution/'QUEUE_FAILURE.json').exists() and not (execution/'CURRENT_PROCESS.json').exists()
 assert [r['cell_id'] for r in q['entries']]==expected['ordered_cell_ids'] and len(q['entries'])==3
 for entry in q['entries']:
  assert sha(c.phase_file(entry['job_relative']))==entry['job_sha256']==expected['job_sha256_by_cell_id'][entry['cell_id']]
 env=c.check_environment(q['environment_overrides']);assert Path(q['python_executable']).is_file()
 gpu=s.query(['--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],q['resource_limits']['telemetry_timeout_seconds'])
 assert tuple(r.split(',')[0].strip() for r in gpu)==c.GPU_UUIDS
 selected=next(r for r in gpu if r.split(',')[0].strip()==c.GPU_UUID);free=int(selected.split(',')[1].strip())*1024**2
 assert free>=q['resource_limits']['minimum_fresh_GPU_free_bytes']
 preflight=dict(UTC=s.now(),hostname=socket.gethostname(),block=block,GPU_UUID=c.GPU_UUID,GPU_free_bytes=free,
   minimum_fresh_GPU_free_bytes=q['resource_limits']['minimum_fresh_GPU_free_bytes'],GPU_metadata=gpu,
   exact_source_and_raw_gate_verified=True,queue_and_release_and_jobs_verified=True,full_nine_spec_and_selected_jobs_root_review_bound=True,
   provider_job_review=payload['provider_job_review'],queue_start_absent=True,scientific_children_started=0)
 save(receiptroot/(block+'.LAUNCH_PREFLIGHT.json'),preflight)
 argv=[q['python_executable'],'-B',str(prep/'run_queue.py'),'--queue',str(queue)]
 with (receiptroot/(block+'.queue.stdout.log')).open('xb') as out,(receiptroot/(block+'.queue.stderr.log')).open('xb') as err:
  process=subprocess.Popen(argv,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
  save(receiptroot/(block+'.POPEN_STARTED.json'),dict(UTC=s.now(),PID=process.pid,argv=argv,cwd=str(repo),block=block,queue_launches=1,retry=False))
 time.sleep(.2)
 raw_owner=s.identity(process.pid);observed_cwd=(Path('/proc')/str(process.pid)/'cwd').resolve(strict=True)
 assert raw_owner is not None and raw_owner['argv']==argv and raw_owner['pgid']==process.pid and raw_owner['sid']==process.pid and observed_cwd==repo.resolve()
 owner=raw_owner
 receipt=dict(UTC=s.now(),status='ONE_COMPLETE_ROOT_REVIEWED_COMPANION_BLOCK_QUEUE_LAUNCHED',hostname=socket.gethostname(),block=block,
   physical_GPU_UUID=c.GPU_UUID,queue_identity=owner,raw_identity_observation=raw_owner,identity_admitted_for_signals=True,
   observed_cwd=str(observed_cwd),queue_sha256=sha(queue),root_release_sha256=sha(execution/'ROOT_RELEASE.json'),
   pilot_source_manifest_sha256=sha(prep/'SOURCE_MANIFEST.json'),training_source_manifest_sha256=c.SOURCE_SHA,
   qualifier_sha256=c.GATE_SHA,cohort_plan_sha256=payload['cohort_plan_sha256'],provider_job_review=payload['provider_job_review'],
   physical_fits=3,full_family_fits=9,queue_hard_seconds=q['queue_hard_seconds'],resource_limits=q['resource_limits'],
   GPU_free_bytes_at_preflight=free,detached_launches=1,attempts=1,retry=False,scores_read=False,TEST_access=False)
 receipt.update(provider_admission_sha256=sha(execution/'PROVIDER_ADMISSION.json'),donor_registration_sha256=sha(execution/'DONOR_REGISTRATION.json'))
except Exception as e:
 receipt=dict(UTC=datetime.now(timezone.utc).isoformat(),status='OPERATIONAL_BLOCK_LAUNCH_FAILURE',block=block,
   error=type(e).__name__+': '+str(e),queue_Popen_started=process is not None,queue_PID=process.pid if process is not None else None,
   queue_identity=owner,raw_identity_observation=raw_owner,observed_cwd=str(observed_cwd) if observed_cwd is not None else None,
   identity_admitted_for_signals=owner is not None,exit_code=process.poll() if process is not None else None,
   exit_code_authority='subprocess.Popen.poll' if process is not None and process.returncode is not None else None,
   partial_artifacts_preserved=True,retry=False,scores_read=False)
save(receiptroot/(block+'.LAUNCH_RECEIPT.json'),receipt);print(json.dumps(receipt))

'''


def reviewed_provider(args, activation):
    assert activation['providers'][PROVIDER]['execution_currently_authorized'] is True
    auth=CONTROL/'FROZEN_AUTHENTICATION.json'
    assert sha(auth)==args.freeze_sha256
    frozen=read(auth)
    assert frozen['status']=='EXACT_COMPANION_BLOCKS_FROZEN_BEFORE_FITS' and frozen['provider']==PROVIDER
    assert frozen['activation_manifest_sha256']==args.activation_manifest_sha256 and frozen['scientific_children_started']==0
    assert set(frozen['blocks'])==set(activation['providers'][PROVIDER]['blocks'])
    for rel,row in frozen['artifact_inventory'].items():
        path=CONTROL/'authenticated_remote'/rel
        assert path.resolve(strict=True).is_relative_to((CONTROL/'authenticated_remote').resolve())
        assert not path.is_symlink() and sha(path)==row['sha256'] and path.stat().st_size==row['bytes']
    sealed=read(HERE/'STUDY_SPEC.json')
    ordered=[cell for b in activation['providers'][PROVIDER]['blocks'] for cell in frozen['blocks'][b]['ordered_cell_ids']]
    selected=[cell for cell in sealed['execution_order'] if cell.split('_',1)[0] in activation['providers'][PROVIDER]['blocks']]
    assert ordered==selected and len(ordered)==3*len(activation['providers'][PROVIDER]['blocks'])
    for block in activation['providers'][PROVIDER]['blocks']:
        execution=frozen['blocks'][block]['execution_directory_relative']
        plan=CONTROL/'authenticated_remote'/execution/'COHORT_PLAN.json'
        assert sha(plan)==frozen['cohort_plan_sha256']
        plan_value=read(plan)
        assert len(plan_value['cells'])==9 and all(plan_value[k]==sealed[k] for k in ('cells','blocks','execution_order','resource_assignment','resource_limits'))
    review=args.provider_job_review.resolve(strict=True)
    assert review.is_relative_to(PHASE.resolve()) and sha(review)==args.provider_job_review_sha256
    decision=read(review)
    assert decision['root_provider_jobs_reviewed'] is True and decision['full_nine_cell_spec_reviewed'] is True
    assert decision['provider']==PROVIDER and decision['scores_read'] is False and decision['TEST_access'] is False
    assert decision['activation_manifest_sha256']==args.activation_manifest_sha256 and decision['freeze_authentication_sha256']==args.freeze_sha256
    assert decision['companion_spec_sha256']==frozen['companion_spec_sha256'] and decision['reviewed_cell_ids']==ordered
    return frozen, review


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--activation-manifest-sha256',required=True)
    parser.add_argument('--freeze-sha256',required=True)
    parser.add_argument('--provider-job-review',type=Path,required=True)
    parser.add_argument('--provider-job-review-sha256',required=True)
    args=parser.parse_args(); activation=packet(args.activation_manifest_sha256)
    frozen,review=reviewed_provider(args,activation)
    provider=activation['providers'][PROVIDER]
    auth=transfer_row(CONTROL/'FROZEN_AUTHENTICATION.json')
    review_files=[auth,transfer_row(review)]
    for block in provider['blocks']:
        assert not (CONTROL/(block+'.LAUNCH_RECEIPT.json')).exists()
        payload=dict(repo=REMOTE_REPO,block=block,provider=PROVIDER,expected=frozen['blocks'][block],review_files=review_files,
            source_relative=provider['source_relative'],control_relative=str(CONTROL.relative_to(PHASE)),
            operational_manifest_sha256=provider['operational_manifest_sha256'],activation_manifest_sha256=args.activation_manifest_sha256,
            cohort_plan_sha256=frozen['cohort_plan_sha256'],freeze_authentication=dict(path=auth['path'],sha256=auth['sha256']),
            provider_job_review=dict(path=str(review.relative_to(PHASE)),sha256=args.provider_job_review_sha256))
        result=transport(REMOTE,payload,'row0_companion_'+PROVIDER+'_'+block+'_launch_once_20261005_v1')
        save(CONTROL/(block+'.LAUNCH_RECEIPT.json'),result)
        print(json.dumps(result))
        assert result['status']=='ONE_COMPLETE_ROOT_REVIEWED_COMPANION_BLOCK_QUEUE_LAUNCHED', result


if __name__=='__main__':main()
