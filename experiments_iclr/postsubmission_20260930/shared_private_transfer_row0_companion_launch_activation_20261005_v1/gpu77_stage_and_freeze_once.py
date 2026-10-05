"""Stage and freeze exact companion metadata; start no fits."""
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

receiptroot=phase/payload['control_relative'];prep=phase/payload['source_relative']
assert not receiptroot.exists()
for row in payload['releases'].values():assert not (phase/row['execution_directory_relative']).exists()
stage(payload['files']);receiptroot.mkdir()
assert sha(prep/'SOURCE_MANIFEST.json')==payload['operational_manifest_sha256']
assert sha(phase/payload['root_source_review']['path'])==payload['root_source_review']['sha256']
spec=importlib.util.spec_from_file_location('row0_companion_common',prep/'pilot_common.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
c.verify_packet(payload['operational_manifest_sha256']);c.verify_training()
blocks={};artifacts={};plan_sha=None
try:
 for block in payload['blocks']:
  c.bind_block([block])
  c.physical_host()
  declared=payload['releases'][block];release_path=phase/declared['path'];release=c.read(release_path)
  assert sha(release_path)==declared['sha256'] and release['execution_blocks']==[block]
  assert release['new_fit_release'] is True and release['TEST_access'] is False and release['retry'] is False
  assert release['root_numeric_cohort_approved'] is True and release['pilot_source_review_approved'] is True
  assert release['root_provider_block_approved'] is True
  execution=phase/declared['execution_directory_relative'];assert not execution.exists()
  env=c.check_environment(release['environment_overrides'])
  argv=[release['python_executable'],'-B',str(prep/'freeze_queue.py'),'--release',str(release_path)]
  frozen=subprocess.run(argv,cwd=repo,env=env,capture_output=True,text=True,timeout=120)
  freezer=dict(UTC=datetime.now(timezone.utc).isoformat(),argv=argv,exit_code=frozen.returncode,
    stdout=frozen.stdout,stderr=frozen.stderr,fits_started=0,attempts=1,retry=False)
  save(receiptroot/(block+'.FREEZE_RECEIPT.json'),freezer);assert frozen.returncode==0,frozen.stderr
  q=c.read(execution/'QUEUE.json');plan=c.read(execution/'COHORT_PLAN.json');sealed=c.read(prep/'STUDY_SPEC.json')
  expected_order=[r for r in sealed['execution_order'] if r.startswith(block+'_')]
  assert sha(prep/'STUDY_SPEC.json')==c.COMPANION_SPEC_SHA
  assert len(plan['cells'])==9 and all(plan[key]==sealed[key] for key in ('cells','blocks','execution_order','resource_assignment','resource_limits'))
  actual_plan_sha=sha(execution/'COHORT_PLAN.json')
  if plan_sha is None:plan_sha=actual_plan_sha
  assert actual_plan_sha==plan_sha==q['cohort_plan_sha256']
  assert q['execution_blocks']==[block] and len(q['entries'])==3 and [r['cell_id'] for r in q['entries']]==expected_order
  assert q['root_release_sha256']==sha(execution/'ROOT_RELEASE.json')==sha(release_path)
  assert q['queue_hard_seconds']==95700 and q['retry'] is False and q['TEST_access'] is False
  job_hashes={}
  for entry in q['entries']:
   path=c.phase_file(entry['job_relative']);job=c.read(path);cell=next(r for r in sealed['cells'] if r['cell_id']==entry['cell_id'])
   assert sha(path)==entry['job_sha256'] and all(job[k]==v for k,v in cell.items())
   assert job['fits_authorized'] is True and job['VALID_values_access'] is True and job['TEST_access'] is False and job['retry'] is False
   assert job['source_manifest_sha256']==c.SOURCE_SHA and job['program_sha256']==sha(c.SOURCE/'run.py')
   assert job['physical_gpu_uuid']==c.GPU_UUID and job['training_step_gate']==dict(approved=True,path=c.GATE_RELATIVE,sha256=c.GATE_SHA)
   assert job['cohort_plan_sha256']==plan_sha and job['cohort_plan_relative']==str((execution/'COHORT_PLAN.json').relative_to(phase))
   assert job['soft_seconds']==plan['fit_bounds_by_cell'][job['cell']]['soft_seconds'] and entry['hard_seconds']==plan['fit_bounds_by_cell'][job['cell']]['hard_seconds']
   for key in ('source_review','runtime_qualification','feature_authority','negative_pool_authority'):
    assert job[key]['approved'] is True and job[key]['evidence']
    for evidence in job[key]['evidence']:assert sha(c.phase_file(evidence['path']))==evidence['sha256']
   job_hashes[entry['cell_id']]=entry['job_sha256']
  assert not (execution/'QUEUE_START.json').exists() and not (execution/'CURRENT_PROCESS.json').exists()
  block_result=dict(execution_directory_relative=str(execution.relative_to(phase)),queue_sha256=sha(execution/'QUEUE.json'),
    root_release_sha256=sha(execution/'ROOT_RELEASE.json'),physical_GPU_UUID=c.GPU_UUID,ordered_cell_ids=expected_order,
    job_sha256_by_cell_id=job_hashes,physical_fits=3,scientific_children_started=0)
  names=['QUEUE.json','ROOT_RELEASE.json','COHORT_PLAN.json']
  adm=c.read(execution/'PROVIDER_ADMISSION.json');reg=c.read(execution/'DONOR_REGISTRATION.json')
  assert reg['queue_sha256']==sha(execution/'QUEUE.json') and reg['root_release_sha256']==sha(execution/'ROOT_RELEASE.json')
  assert reg['provider_admission']['sha256']==sha(execution/'PROVIDER_ADMISSION.json')
  assert reg['registered_before_provider_first_fit'] is True and adm['admitted_before_provider_first_fit'] is True and adm['approved'] is True
  assert adm['GPU_UUID']==c.GPU_UUID and adm['provider_source_manifest']['sha256']==c.SOURCE_SHA
  assert adm['python_executable']==release['python_executable'] and adm['training_step_gate']==dict(approved=True,path=c.GATE_RELATIVE,sha256=c.GATE_SHA)
  block_result.update(provider_admission_sha256=sha(execution/'PROVIDER_ADMISSION.json'),donor_registration_sha256=sha(execution/'DONOR_REGISTRATION.json'))
  names+=['PROVIDER_ADMISSION.json','DONOR_REGISTRATION.json']
  blocks[block]=block_result
  for path in [execution/n for n in names]+[phase/r['job_relative'] for r in q['entries']]+[receiptroot/(block+'.FREEZE_RECEIPT.json')]:
   raw=path.read_bytes();rel=str(path.relative_to(phase))
   artifacts[rel]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),zlib_base64=base64.b64encode(zlib.compress(raw,9)).decode())
 compact=dict(UTC=datetime.now(timezone.utc).isoformat(),status='EXACT_COMPANION_BLOCKS_FROZEN_BEFORE_FITS',provider=payload['provider'],
   activation_manifest_sha256=payload['activation_manifest_sha256'],root_source_review=payload['root_source_review'],
   source_manifest_sha256=c.SOURCE_SHA,pilot_source_manifest_sha256=sha(prep/'SOURCE_MANIFEST.json'),
   companion_spec_sha256=c.COMPANION_SPEC_SHA,cohort_plan_sha256=plan_sha,qualifier_sha256=c.GATE_SHA,blocks=blocks,
   scientific_children_started=0,fits=0,scores_read=False,retry=False,
   artifact_inventory={rel:{k:v for k,v in row.items() if k!='zlib_base64'} for rel,row in artifacts.items()})
 save(receiptroot/'FROZEN_AUTHENTICATION.json',compact)
 result=dict(compact,artifacts=artifacts)
except Exception as e:
 result=dict(UTC=datetime.now(timezone.utc).isoformat(),status='OPERATIONAL_FREEZE_FAILURE',
   error=type(e).__name__+': '+str(e),partial_artifacts_preserved=True,scientific_children_started=0,fits=0,retry=False)
 save(receiptroot/'FREEZE_CONTROL_FAILURE.json',result)
print(json.dumps(result))

'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--activation-manifest-sha256', required=True)
    parser.add_argument('--root-source-review', type=Path, required=True)
    parser.add_argument('--root-source-review-sha256', required=True)
    args=parser.parse_args(); activation=packet(args.activation_manifest_sha256)
    assert activation['providers'][PROVIDER]['execution_currently_authorized'] is True
    review=args.root_source_review.resolve(strict=True)
    assert review.is_relative_to(PHASE.resolve()) and sha(review)==args.root_source_review_sha256
    assert not CONTROL.exists()
    provider=activation['providers'][PROVIDER]; source=PHASE/provider['source_relative']
    paths=[HERE/'ACTIVATION.json',HERE/'STUDY_SPEC.json',HERE/'NUMERIC_PREREQUISITES.json',source/'SOURCE_MANIFEST.json',review]
    paths += [source/r['path'] for r in read(source/'SOURCE_MANIFEST.json')['files']]
    paths += [PHASE/r['path'] for r in provider['releases'].values()]
    rows=[transfer_row(path) for path in sorted(set(paths))]
    for row in activation['metadata_copy_map']:
        path=PHASE/row['local_relative']; assert sha(path)==row['sha256']
        rows.append(transfer_row(path,row['server_relative']))
    unique={}
    for row in rows:
        if row['path'] in unique:assert unique[row['path']]['sha256']==row['sha256']
        unique[row['path']]=row
    rows=[unique[key] for key in sorted(unique)]
    payload=dict(provider=PROVIDER,repo=REMOTE_REPO,files=rows,blocks=provider['blocks'],releases=provider['releases'],
        source_relative=provider['source_relative'],control_relative=str(CONTROL.relative_to(PHASE)),
        operational_manifest_sha256=provider['operational_manifest_sha256'],activation_manifest_sha256=args.activation_manifest_sha256,
        root_source_review=dict(path=str(review.relative_to(PHASE)),sha256=args.root_source_review_sha256))
    CONTROL.mkdir()
    save(CONTROL/'TRANSFER_INVENTORY.json',[{k:v for k,v in r.items() if k!='base64'} for r in rows])
    result=transport(REMOTE,payload,'row0_companion_gpu77_stage_freeze_once_20261005_v1')
    assert result['status']=='EXACT_COMPANION_BLOCKS_FROZEN_BEFORE_FITS', result
    assert set(result['blocks'])==set(provider['blocks']) and result['scientific_children_started']==0
    for rel,row in result['artifacts'].items():
        raw=zlib.decompress(base64.b64decode(row['zlib_base64'],validate=True))
        assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
        relative=Path(rel); assert not relative.is_absolute() and '..' not in relative.parts
        target=CONTROL/'authenticated_remote'/relative; target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream:stream.write(raw)
    compact={k:v for k,v in result.items() if k!='artifacts'}
    save(CONTROL/'FROZEN_AUTHENTICATION.json',compact)
    print(json.dumps(compact))


if __name__=='__main__':main()
