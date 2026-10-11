import json,os,socket,subprocess,time
from pathlib import Path
h=socket.gethostname();g=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).splitlines();assert h=='anogena-2-0' and g==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
R='/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs';os.chdir(R);base=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/graph_walk_acquisition_root_20261011_v1');assert not (base/'LAUNCH.json').exists() and not (base/'owner_v1').exists()
argv=[R+'/.venv/bin/python','-B','-u','/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/native_meanSAGE_tour_scan_execution_source_20261011_v1/owner.py',str(base/'CONFIG.json')]
with (base/'owner_dispatch.log').open('xb') as log:
 process=subprocess.Popen(argv,cwd=R,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
value=dict(launched=True,owner_pid=process.pid,owner_command=argv,detached_session=True,finite_wall_seconds=43200,source_manifest_sha256='ce7d2f5dbabd922a2722afe2f401f65b73c973e783ad6f863d5cf66c5f6dc94d',new_banks=18,new_trajectories=27,prior_banks=12,prior_trajectories=21,route=dict(hostname=h,sole_gpu=g))
(base/'LAUNCH.json').write_text(json.dumps(value,indent=2)+'\n')
for unused in range(20):
 if (base/'owner_v1/OWNER_START.json').is_file():break
 time.sleep(.1)
if (base/'owner_v1/OWNER_START.json').is_file():value['owner_start']=json.loads((base/'owner_v1/OWNER_START.json').read_text())
value['early_dispatch_exit_code']=process.poll();print(json.dumps(value),flush=True)
