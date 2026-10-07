import os,socket,sys,json,hashlib,base64,subprocess,datetime
from pathlib import Path
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';os.chdir(R)
GPU='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['/usr/bin/nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==[GPU]
for row in json.loads(sys.stdin.read()):
 q=P/row['path'];assert q.resolve().is_relative_to(P)
 data=base64.b64decode(row['data']);assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
 if q.exists():assert q.is_file() and q.read_bytes()==data,row['path']
 else:q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(data)
S=P/'learnable_internal_be_WikiCS_scientific_family_driver_source_20261007_v1';D=P/'learnable_internal_be_WikiCS_scientific_family_activation_20261007_v1'
assert not (D/'LAUNCH_OWNER.json').exists()
assert not (P/'learnable_internal_be_WikiCS_scientific_family_execution_root_20261007_v1').exists()
for row in json.loads((S/'MANIFEST.json').read_text())['files']:
 q=S/row['path'];assert hashlib.sha256(q.read_bytes()).hexdigest()==row['sha256'] and q.stat().st_size==row['bytes']
PYTHON=P/'native_ncn_runtime_20261005_v1/.venv/bin/python'
env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,PYTHONPATH=str(P/'native_ncn_dependency_overlay_20261005_v1')+':'+str(R/'.venv/lib/python3.11/site-packages'));env.pop('PYTHONHOME',None)
with (D/'DRIVER.log').open('x') as f:
 child=subprocess.Popen([str(PYTHON),'-B',str(S/'driver.py'),'--release',str(D/'RELEASE.json')],cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 stat=(Path('/proc')/str(child.pid)/'stat').read_text();ticks=int(stat[stat.rfind(')')+2:].split()[19])
 receipt={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'PID':child.pid,'start_ticks':ticks,'driver_manifest_sha256':hashlib.sha256((S/'MANIFEST.json').read_bytes()).hexdigest(),'release_sha256':hashlib.sha256((D/'RELEASE.json').read_bytes()).hexdigest(),'fixed_scientific_family_launched':True,'source_staged':True,'cell_count':24,'TEST_access':False,'GPU77access':False}
 (D/'LAUNCH_OWNER.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(receipt))
