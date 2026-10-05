"""Exactly one root-conditionally-released GPU0 qualifier after complete TRAIN equality."""
from datetime import datetime,timezone
from pathlib import Path
import base64
import hashlib
import importlib.util
import json
import zlib

HERE=Path(__file__).resolve().parent
PHASE=HERE.parent
ENV=PHASE/'shared_private_transfer_gpu77_environment_execution_20261005_v1'
PREP=PHASE/'shared_private_transfer_gpu77_qualification_preparation_20261005_v3'
SAMPLING=PHASE/'shared_private_transfer_sampling_execution_20261005_v1'
REVIEW=PHASE/'shared_private_transfer_gpu77_qualification_root_release_20261005_v2/ROOT_REVIEW.md'
spec=importlib.util.spec_from_file_location('qualifier77_transport',ENV/'remote_transport.py')
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t);t.HERE=HERE
REPO=t.REPO;REMOTE_PHASE=t.REMOTE_PHASE;REMOTE=REMOTE_PHASE/HERE.name
GPU='GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998'
SOURCE='shared_backbone_private_transfer_training_source_gpu77_20261005_v1'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(name,value):
    if (HERE/name).exists():
        assert json.loads((HERE/name).read_text())==value
        return
    with (HERE/name).open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True);stream.write('\n')

def row(path,target=None):
    raw=path.read_bytes()
    return {'path':target or str(path.relative_to(PHASE)),'bytes':len(raw),'sha256':sha(path),'base64':base64.b64encode(raw).decode()}

def main():
    failure=json.loads((HERE/'PRE_MODEL_CONTROL_FAILURE.json').read_text())
    assert failure['execution_root_absent'] is True and failure['numerical_children_started']==0 and failure['owned_supervisor_PID_absent'] is True
    assert sha(REVIEW)=='a579832479b0458b1cec88066814a27fd6ee3f1b3afedceead63365387490c17'
    assert sha(PREP/'MANIFEST.json')=='1c88896203ec52b7c960abf61a90025a533ec613bad933acc88503af73cf5a45'
    assert sha(PREP/'physical_qualifier_supervisor.py')=='0c946cd910fa104fa5b97a9d9b80872e13fffd5f623f1321f5bfa7e319034e78'
    assert sha(ENV/'MANIFEST.json')=='14b6a9106a018d578f7a815fc5ab77c0d0b7eef415cee86baf2d7a2f3d1928c6'
    comparison=json.loads((SAMPLING/'COMPARISON.json').read_text())
    assert comparison['status']=='PASS' and comparison['all_component_hashes_equal'] is True and comparison['mismatches']==[]
    assert comparison['paired_seed_cycle_draws']==240 and comparison['compared_component_hashes']==1680
    assert comparison['cycles']==list(range(60)) and comparison['seeds']==[20261005,0,1,2]
    for host in ('singleton','gpu77'):
        root=SAMPLING/host;receipt=json.loads((root/'EXECUTION_RECEIPT.json').read_text())
        assert receipt['exit_code']==0 and receipt['terminal_wait_observed'] is True and receipt['owned_PID_absent'] is True
        assert receipt['reason'] is None and receipt['signals_sent']==[] and receipt['attempts']==1 and receipt['retry'] is False
        assert comparison['provider_bindings'][host]['result_sha256']==sha(root/'result/RESULT.json')
        assert comparison['provider_bindings'][host]['execution_receipt_sha256']==sha(root/'EXECUTION_RECEIPT.json')
        closure=json.loads((root/'PHYSICAL_TERMINAL.json').read_text())
        assert closure['status']=='PASS' and closure['owned_child_PID_absent'] is True and closure['owned_supervisor_PID_absent'] is True
    bindings={'UTC':datetime.now(timezone.utc).isoformat(),'root_review_sha256':sha(REVIEW),
        'sampling_comparison_sha256':sha(SAMPLING/'COMPARISON.json'),'providers_sha256':sha(ENV/'PROVIDERS.json'),
        'source_manifest_sha256':'7f274c09bb317e6976d0c8b8636e779dc3ffcd2f2b76493d00f008b8bd08cebc',
        'qualifier_sha256':'38fcec87c05d744135797546c819eb4e203e4a58c0273796ef7f60676194198e',
        'wrapper_manifest_sha256':sha(PREP/'MANIFEST.json'),'supervisor_program_sha256':sha(PREP/'physical_qualifier_supervisor.py'),
        'physical_gpu_uuid':GPU,'fits_authorized':False,'retry':False,'minimum_GPU0_free_bytes':12*1024**3}
    if (HERE/'ROOT_BINDINGS.json').exists():
        previous=json.loads((HERE/'ROOT_BINDINGS.json').read_text())
        assert {k:v for k,v in previous.items() if k!='UTC'}=={k:v for k,v in bindings.items() if k!='UTC'}
        bindings=previous
    save('ROOT_BINDINGS.json',bindings)
    job=json.loads((PHASE/'shared_private_transfer_gpu77_qualification_preparation_20261005_v1/QUALIFICATION_JOB_DISABLED.json').read_text())
    job.update(schema='root_released_exact_provider_gpu77_finite_qualification_v2',source_review_approved=True,
        source_manifest_sha256=bindings['source_manifest_sha256'],physical_gpu_uuid=GPU,
        output_directory=str(REMOTE/'result'),external_hard_bound_confirmed=True,
        scope='One root-released unchanged discarded FP32/native-Adam qualification after exact full-horizon TRAIN transcript equality. No fits.',
        source_review={'approved':True,'evidence':[{'path':str(REVIEW.relative_to(PHASE)),'sha256':sha(REVIEW)},
            {'path':'shared_backbone_private_transfer_source_independent_review_20261005_v3/REVIEW_MANIFEST.json','sha256':'ade060b5ad4d380d4b026d9970f9194f0285997e96a4a1b548573f991e21e05f'}]},
        runtime_qualification={'approved':True,'evidence':[{'path':str((ENV/'PROVIDERS.json').relative_to(PHASE)),'sha256':sha(ENV/'PROVIDERS.json')},
            {'path':str((ENV/'ENVIRONMENT_VERIFICATION.json').relative_to(PHASE)),'sha256':sha(ENV/'ENVIRONMENT_VERIFICATION.json')},
            {'path':str((SAMPLING/'COMPARISON.json').relative_to(PHASE)),'sha256':sha(SAMPLING/'COMPARISON.json')}]})
    save('JOB.json',job)
    target=PHASE/'shared_private_transfer_gpu77_qualification_root_release_20261005_v2'
    metadata=[row(REVIEW),row(SAMPLING/'COMPARISON.json')]
    for host in ('singleton','gpu77'):
        metadata.extend(row(SAMPLING/host/name) for name in ('EXECUTION_RECEIPT.json','PHYSICAL_TERMINAL.json','ROOT_RELEASE.json','result/RESULT.json'))
    metadata.append(row(HERE/'ROOT_BINDINGS.json',str(target.relative_to(PHASE))+'/ROOT_BINDINGS.json'))
    blob=zlib.compress(json.dumps(metadata).encode(),9);encoded=base64.b64encode(blob).decode()
    inventory=[{k:v for k,v in r.items() if k!='base64'} for r in metadata]
    save('CONDITIONAL_ADMISSION_STAGE_INVENTORY.json',inventory)
    chunks=[encoded[start:start+40000] for start in range(0,len(encoded),40000)]
    chunk_root=REMOTE_PHASE/target.name/'conditional_admission_chunks'
    for number,chunk in enumerate(chunks,1):
        chunk_code=f'''
from pathlib import Path
import hashlib,json,socket
repo=Path({str(REPO)!r});root=Path({str(chunk_root)!r})
assert Path.cwd()==repo and socket.gethostname()=='peptide' and not root.is_symlink()
root.mkdir(parents=True,exist_ok=True);path=root/{('part%03d.b64'%number)!r};raw={chunk!r}.encode()
with path.open('xb') as stream:stream.write(raw)
print(json.dumps(dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())))
'''
        transferred=t.run(f'private_transfer77_qualifier_v3_admission_chunk{number:03d}_20261005_v1',chunk_code)
        assert transferred=={'bytes':len(chunk),'sha256':hashlib.sha256(chunk.encode()).hexdigest()}
    code=f'''
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,socket,zlib
repo=Path({str(REPO)!r});phase=Path({str(REMOTE_PHASE)!r})
assert Path.cwd()==repo and socket.gethostname()=='peptide'
chunk_root=Path({str(chunk_root)!r})
blob=base64.b64decode(b''.join((chunk_root/('part%03d.b64'%i)).read_bytes() for i in range(1,{len(chunks)+1})),validate=True)
assert len(blob)=={len(blob)} and hashlib.sha256(blob).hexdigest()=={hashlib.sha256(blob).hexdigest()!r}
rows=json.loads(zlib.decompress(blob));decoded=[]
for row in rows:
 rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts
 path=phase/rel;assert path.resolve().is_relative_to(phase)
 for parent in [path,*path.parents]:
  if parent==phase.parent:break
  assert not parent.is_symlink()
 raw=base64.b64decode(row['base64'],validate=True);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 if path.exists():assert path.is_file() and path.read_bytes()==raw
 decoded.append((path,raw))
for path,raw in decoded:
 if not path.exists():
  path.parent.mkdir(parents=True,exist_ok=True)
  with path.open('xb') as stream:stream.write(raw)
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),status='EXACT_CONDITIONAL_ADMISSION_EVIDENCE_STAGED',files=[{{k:v for k,v in row.items() if k!='base64'}} for row in rows],models_launched=False)))
'''
    # ROOT_BINDINGS is staged in a distinct release folder so the actual owned
    # execution root remains fresh for the one-shot supervisor's guard.
    staged=t.run('private_transfer77_qualifier_v3_conditional_admission_stage_20261005_v1',code)
    save('CONDITIONAL_ADMISSION_STAGE_RECEIPT.json',staged)
    payload={'qualification_execution_enabled':True,'fits_authorized':False,'retry':False,
        'remote_execution_root':str(REMOTE),'training_source_name':SOURCE,'physical_gpu_uuid':GPU,
        'wrapper_manifest_sha256':sha(PREP/'MANIFEST.json'),'supervisor_program_sha256':sha(PREP/'physical_qualifier_supervisor.py'),
        'files':[row(HERE/'JOB.json')]}
    save('STAGING_PAYLOAD.json',payload)
    launch_rows=[row(HERE/'STAGING_PAYLOAD.json',str(target.relative_to(PHASE))+'/STAGING_PAYLOAD.json')]
    providers=json.loads((ENV/'PROVIDERS.json').read_text())
    expected_inputs=json.loads((ENV/'INPUT_STAGE_INVENTORY.json').read_text())['files']
    code=f'''
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,os,socket,subprocess
repo=Path({str(REPO)!r});phase=Path({str(REMOTE_PHASE)!r});root=Path({str(REMOTE)!r});release_root=phase/{target.name!r};prep=phase/{PREP.name!r}
assert Path.cwd()==repo and socket.gethostname()=='peptide' and not root.exists()
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  while part:=stream.read(1024*1024):h.update(part)
 return h.hexdigest()
assert sha(prep/'MANIFEST.json')=={sha(PREP/'MANIFEST.json')!r} and sha(prep/'physical_qualifier_supervisor.py')=={sha(PREP/'physical_qualifier_supervisor.py')!r}
assert sha(phase/{(SOURCE+'/SOURCE_MANIFEST.json')!r})=={bindings['source_manifest_sha256']!r}
for row in json.loads((phase/{(SOURCE+'/SOURCE_MANIFEST.json')!r}).read_text())['files']:
 assert sha(phase/{SOURCE!r}/row['path'])==row['sha256']
assert sha(phase/{str((ENV/'PROVIDERS.json').relative_to(PHASE))!r})=={sha(ENV/'PROVIDERS.json')!r}
for provider in {providers['providers']!r}.values():
 for item in [provider['module'],*provider['native_files']]:
  path=Path(item['path']);assert path.resolve().is_relative_to(repo/'.gnnm_runtime/private_transfer_cp311_cu118_20261005_v1')
  assert path.is_file() and path.stat().st_size==item['bytes'] and sha(path)==item['sha256']
for row in {expected_inputs!r}:
 path=phase/row['path'];assert path.is_file() and not path.is_symlink() and path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True,timeout=20).strip().splitlines()
assert [r.split(',')[0].strip() for r in gpu]==['GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998','GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced']
assert int(gpu[0].split(',')[1].strip())*1024**2>=12*1024**3
rows={launch_rows!r}
for row in rows:
 path=phase/row['path'];raw=base64.b64decode(row['base64'],validate=True);assert hashlib.sha256(raw).hexdigest()==row['sha256']
 with path.open('xb') as stream:stream.write(raw)
argv=['/usr/bin/python3','-I','-S','-B',str(prep/'physical_qualifier_supervisor.py')]
env=dict(os.environ);env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
with (release_root/'supervisor.stdout.log').open('xb') as out,(release_root/'supervisor.stderr.log').open('xb') as err,(release_root/'STAGING_PAYLOAD.json').open('rb') as payload:
 child=subprocess.Popen(argv,cwd=repo,env=env,stdin=payload,stdout=out,stderr=err,start_new_session=True)
 stat=Path('/proc',str(child.pid),'stat').read_text().split(') ',1)[1].split()
 receipt=dict(UTC=datetime.now(timezone.utc).isoformat(),status='ONE_UNCHANGED_FINITE_QUALIFIER_LAUNCHED',hostname=socket.gethostname(),supervisor_PID=child.pid,supervisor_start_ticks=int(stat[19]),supervisor_pgid=int(stat[2]),supervisor_sid=int(stat[3]),source_manifest_sha256={bindings['source_manifest_sha256']!r},program_sha256={bindings['qualifier_sha256']!r},supervisor_sha256={sha(PREP/'physical_qualifier_supervisor.py')!r},job_sha256={sha(HERE/'JOB.json')!r},staging_payload_sha256={sha(HERE/'STAGING_PAYLOAD.json')!r},physical_gpu_uuid={GPU!r},GPU_metadata=gpu,soft_seconds=600,hard_seconds=720,attempts=1,retry=False,fits=0,VALID_TEST_values_access=False)
 (release_root/'DETACHED_LAUNCH.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\\n')
print(json.dumps(receipt))
'''
    launched=t.run('private_transfer77_qualifier_launch_v3_20261005_v1',code)
    save('DETACHED_LAUNCH.json',launched);print(json.dumps(launched))

if __name__=='__main__':main()
