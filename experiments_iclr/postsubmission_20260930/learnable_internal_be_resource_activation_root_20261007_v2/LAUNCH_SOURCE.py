import json,base64,hashlib,socket,subprocess,sys,os,datetime
from pathlib import Path
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930'
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['/usr/bin/nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
for row in json.loads(sys.stdin.read()):
 p=P/row['path'];assert p.resolve().is_relative_to(P)
 data=base64.b64decode(row['data']);assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
 if p.exists():assert p.is_file() and p.read_bytes()==data
 else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
D=P/'learnable_internal_be_resource_activation_root_20261007_v2'
assert hashlib.sha256((D/'CONFIG_ADOPTED.json').read_bytes()).hexdigest()=='c9da38f9d5ea5219928247406f561e18b552c7e0d07038b05f39ebc0d8807913'
assert not (D/'LAUNCH_OWNER.json').exists()
assert not (P/'learnable_internal_be_WikiCS_resource_execution_root_20261007_v2').exists()
with (D/'DRIVER.log').open('x') as log:
 child=subprocess.Popen(['/usr/bin/python3','-I','-S','-B',str(D/'execute.py')],cwd=R,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 stat=(Path('/proc')/str(child.pid)/'stat').read_text();ticks=int(stat[stat.rfind(')')+2:].split()[19])
 receipt={'PID':child.pid,'start_ticks':ticks,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'resource_queue_launched':True,'scientific_fits':0,'predictive_scores_closed':True,'GPU77access':False,'config_sha256':hashlib.sha256((D/'CONFIG_ADOPTED.json').read_bytes()).hexdigest()}
 (D/'LAUNCH_OWNER.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(receipt))
