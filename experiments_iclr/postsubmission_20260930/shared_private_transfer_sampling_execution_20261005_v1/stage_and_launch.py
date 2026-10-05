"""One exact root-approved CPU transcript on each admitted host; no models/fits."""
from datetime import datetime,timezone
from pathlib import Path
import base64
import hashlib
import importlib.util
import json
import shlex
import subprocess
import sys

HERE=Path(__file__).resolve().parent
PHASE=HERE.parent
PREP=PHASE/'shared_private_transfer_sampling_preparation_20261005_v2'
ENV=PHASE/'shared_private_transfer_gpu77_environment_execution_20261005_v1'
spec=importlib.util.spec_from_file_location('sampling77_transport',ENV/'remote_transport.py')
transport=importlib.util.module_from_spec(spec);spec.loader.exec_module(transport);transport.HERE=HERE
SSH=['ssh','-T','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=20','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru']
HOSTS={
 'singleton':{'repo':'/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs','hostname':'anogena-2-0','source':'shared_backbone_private_transfer_training_source_20261005_v2','source_sha':'db7102df30491be8809ea4295b9ce0d5c48f3608129f09dcfef74b8f7aa2233f','execution':'shared_private_transfer_singleton_sampling_execution_root_20261005_v1'},
 'gpu77':{'repo':'/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git','hostname':'peptide','source':'shared_backbone_private_transfer_training_source_gpu77_20261005_v1','source_sha':'7f274c09bb317e6976d0c8b8636e779dc3ffcd2f2b76493d00f008b8bd08cebc','execution':'shared_private_transfer_gpu77_sampling_execution_root_20261005_v1'}
}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(name,value):
 with (HERE/name).open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True);stream.write('\n')

def call(host,identity,code):
 if host=='gpu77':return transport.run(identity,code)
 repo=HOSTS[host]['repo'];command=SSH+['cd '+shlex.quote(repo)+' && exec /usr/bin/python3 -I -S -B -']
 result=subprocess.run(command,input=code,capture_output=True,text=True,timeout=60)
 save(identity+'_TRANSPORT.json',{'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'remote_code_sha256':hashlib.sha256(code.encode()).hexdigest()})
 assert result.returncode==0,result.stderr+result.stdout
 return json.loads(result.stdout.strip())

def release(host):
 h=HOSTS[host];repo=h['repo'];phase=repo+'/experiments_iclr/postsubmission_20260930'
 python=phase+'/native_ncn_runtime_20261005_v1/.venv/bin/python' if host=='singleton' else repo+'/.gnnm_runtime/private_transfer_cp311_cu118_20261005_v1/bin/python'
 overlay=phase+'/native_ncn_dependency_overlay_20261005_v1:'+repo+'/.venv/lib/python3.11/site-packages' if host=='singleton' else ''
 return {'sampling_execution_enabled':True,'models_authorized':False,'fits_authorized':False,'VALID_values_access':False,'TEST_access':False,'program_path':phase+'/'+PREP.name+'/sampling_transcript.py','program_sha256':sha(PREP/'sampling_transcript.py'),'supervisor_sha256':sha(PREP/'sampling_supervisor.py'),'sampling_source_manifest_sha256':sha(PREP/'SOURCE_MANIFEST.json'),'seeds':[20261005,0,1,2],'cycles':list(range(60)),'outer_size':64,'inner_size':256,'members':4,'repository':repo,'expected_hostname':h['hostname'],'training_source_name':h['source'],'source_manifest_sha256':h['source_sha'],'available_manifest_relative':'citeseer_heart_official_acquisition_server_20261005_v1/AVAILABLE_MANIFEST.json','available_manifest_sha256':'1b9c8bb57278d91b0f6212136225afcfd6b067c6b17e0dfed7ee36dd6316efdc','runtime_versions':{'CUDA':'11.8','numpy':'1.26.4','torch':'2.1.2+cu118','torch_geometric':'2.7.0','torch_scatter':'2.1.2+pt21cu118','torch_sparse':'0.6.18+pt21cu118'},'output_directory':phase+'/'+h['execution']+'/result','python_executable':python,'declared_PYTHONPATH':overlay,'soft_seconds':1800,'hard_seconds':2100,'retry':False,'root_authorization_path':phase+'/'+PREP.name+'/ROOT_AUTHORIZATION.json','root_authorization_sha256':sha(PREP/'ROOT_AUTHORIZATION.json')}

def stage_launch(host):
 h=HOSTS[host];repo=h['repo'];phase=repo+'/experiments_iclr/postsubmission_20260930';root=phase+'/'+h['execution']
 rel=release(host);save(host+'_ROOT_RELEASE.json',rel);local_release=HERE/(host+'_ROOT_RELEASE.json')
 rows=[]
 for path in sorted(PREP.iterdir()):
  assert path.is_file();raw=path.read_bytes();rows.append({'path':PREP.name+'/'+path.name,'bytes':len(raw),'sha256':sha(path),'data':base64.b64encode(raw).decode()})
 raw=local_release.read_bytes();rows.append({'path':h['execution']+'/ROOT_RELEASE.json','bytes':len(raw),'sha256':sha(local_release),'data':base64.b64encode(raw).decode()})
 code=f'''
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,os,socket,subprocess
repo=Path({repo!r});phase=Path({phase!r});root=Path({root!r})
assert Path.cwd()==repo and socket.gethostname()=={h['hostname']!r}
assert phase.is_dir() and not phase.is_symlink() and not root.exists()
assert hashlib.sha256((phase/{(h['source']+'/SOURCE_MANIFEST.json')!r}).read_bytes()).hexdigest()=={h['source_sha']!r}
assert hashlib.sha256((phase/'citeseer_heart_official_acquisition_server_20261005_v1/AVAILABLE_MANIFEST.json').read_bytes()).hexdigest()=='1b9c8bb57278d91b0f6212136225afcfd6b067c6b17e0dfed7ee36dd6316efdc'
rows={rows!r};decoded=[]
for row in rows:
 rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts
 path=phase/rel;assert path.resolve().is_relative_to(phase)
 for parent in [path,*path.parents]:
  if parent==phase.parent:break
  assert not parent.is_symlink()
 raw=base64.b64decode(row['data'],validate=True);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 if path.exists():assert path.is_file() and path.read_bytes()==raw
 decoded.append((path,raw))
root.mkdir()
for path,raw in decoded:
 if not path.exists():
  path.parent.mkdir(parents=True,exist_ok=True)
  with path.open('xb') as stream:stream.write(raw)
prep=phase/{PREP.name!r}
argv=['/usr/bin/python3','-I','-S','-B',str(prep/'sampling_supervisor.py'),'--release',str(root/'ROOT_RELEASE.json'),'--release-sha256',{sha(local_release)!r}]
env=dict(os.environ);env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
with (root/'supervisor.stdout.log').open('xb') as stdout,(root/'supervisor.stderr.log').open('xb') as stderr:
 child=subprocess.Popen(argv,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
 stat=Path('/proc',str(child.pid),'stat').read_text().split(') ',1)[1].split()
 receipt=dict(UTC=datetime.now(timezone.utc).isoformat(),hostname=socket.gethostname(),repository=str(repo),root=str(root),supervisor_PID=child.pid,supervisor_start_ticks=int(stat[19]),supervisor_pgid=int(stat[2]),supervisor_sid=int(stat[3]),release_sha256={sha(local_release)!r},program_sha256={sha(PREP/'sampling_transcript.py')!r},supervisor_sha256={sha(PREP/'sampling_supervisor.py')!r},soft_seconds=1800,hard_seconds=2100,CUDA_VISIBLE_DEVICES='',models_authorized=False,fits=0,VALID_TEST_values_access=False,attempts=1,retry=False)
 (root/'DETACHED_LAUNCH.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\\n')
print(json.dumps(receipt))
'''
 save(host+'_STAGE_INVENTORY.json',[{k:v for k,v in row.items() if k!='data'} for row in rows])
 result=call(host,f'private_transfer_{host}_sampling_launch_20261005_v1',code)
 save(host+'_DETACHED_LAUNCH.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
 assert sha(PREP/'SOURCE_MANIFEST.json')=='19114f6a6c050e8da0ea056719e8d3734cace7ca64f0751c34c708d0a9d2fff5'
 stage_launch('singleton')
 stage_launch('gpu77')
