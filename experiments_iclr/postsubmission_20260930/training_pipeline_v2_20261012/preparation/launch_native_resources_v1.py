"""Launch finite graph-only resource qualification, not benchmark fitting."""
import json,shlex,subprocess
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent
REMOTE=r"""
import json,os,socket,subprocess,time,hashlib
from pathlib import Path
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
phase=repo/'experiments_iclr/postsubmission_20260930'
packet=phase/'massive_native_packet_20261012_v2'
owner=packet/'resource_owner_v1';assert not owner.exists();owner.mkdir()
command=[str(repo/'.venv/bin/python'),'-B','-u',str(packet/'qualification/profile_native_resources.py'),'--project-root',str(phase),'--runtime-package','massive_native_packet_20261012_v2/runtime','--native-model-source','massive_native_packet_20261012_v2/native/models.py','--output','massive_native_packet_20261012_v2/resources_v1','--expected-datasets','8','--worker-capacities','massive_native_packet_20261012_v2/metadata/PLANNING_CAPACITIES.json','--case-timeout-seconds','300','--warmup-iterations','1','--measured-iterations','1','--torch-threads','2']
for name in ('cora','citeseer','pubmed','wikics','amazon-computers','amazon-photo','coauthor-cs','coauthor-physics'):
 command+=['--custody','training_pipeline_v2_20261012/prepared_data/'+name+'/CUSTODY.json']
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONPATH=str(packet/'runtime'),PYTHONDONTWRITEBYTECODE='1')
with (owner/'child.log').open('x') as log:
 child=subprocess.Popen(command,cwd=phase,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 time.sleep(.2)
 if child.poll() is not None:raise RuntimeError('Resource child exited at startup: '+(owner/'child.log').read_text())
 fields=(Path('/proc')/str(child.pid)/'stat').read_text().rsplit(')',1)[1].split()
 v={'schema':'finite-native-resource-actual-launch-v1','child':{'pid':child.pid,'start_ticks':int(fields[19]),'pgid':int(fields[2])},'command':command,'environment':{'CUDA_VISIBLE_DEVICES':'0','PYTHONPATH':str(packet/'runtime')},'expected_resource_cases':216,'optimizer_updates':0,'scientific_fits':0,'predictive_scores':0,'GO_created':False,'source_sha256':hashlib.sha256((packet/'qualification/profile_native_resources.py').read_bytes()).hexdigest(),'survives_connection_loss':True}
 (owner/'ACTUAL_LAUNCH.json').write_text(json.dumps(v,indent=2)+'\n')
 print(json.dumps(v))
"""
a=['ssh','-tt','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',shlex.join(['/usr/bin/python3','-I','-S','-B','-c',REMOTE])]
f=HERE/'NATIVE_RESOURCE_LAUNCH_TRANSPORT_V1.json';assert not f.exists()
r=subprocess.run(a,capture_output=True,text=True,timeout=45)
f.write_text(json.dumps({'UTC':datetime.now(timezone.utc).isoformat(),'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr},indent=2)+'\n')
assert r.returncode==0,r.stdout+r.stderr
print(r.stdout)
