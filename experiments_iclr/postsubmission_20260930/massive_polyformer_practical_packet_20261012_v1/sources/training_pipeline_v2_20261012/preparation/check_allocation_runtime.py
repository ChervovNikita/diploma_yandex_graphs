import json,shlex,subprocess
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent
REMOTE=r'''
import json,socket,subprocess,shutil
from pathlib import Path
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
r=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
code="import json,torch,torch_geometric,sys;print(json.dumps({'python':sys.version,'torch':torch.__version__,'torch_geometric':torch_geometric.__version__,'cuda_runtime':torch.version.cuda}))"
p=subprocess.run([str(r/'.venv/bin/python'),'-B','-c',code],capture_output=True,text=True,check=True,timeout=45)
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.total,memory.free','--format=csv,noheader'],text=True,timeout=10)
print(json.dumps({'runtime':json.loads(p.stdout),'GPU':gpu.strip(),'disk':dict(zip(('total','used','free'),shutil.disk_usage(r))),'current_research_head':subprocess.check_output(['git','-C',str(r),'rev-parse','HEAD'],text=True).strip(),'scientific_training_runs':0}))
'''
cmd=['ssh','-tt','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',shlex.join(['/usr/bin/python3','-I','-S','-B','-c',REMOTE])]
p=HERE/'RUNTIME_V1.json';assert not p.exists()
r=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
p.write_text(json.dumps({'UTC':datetime.now(timezone.utc).isoformat(),'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr},indent=2)+'\n')
assert r.returncode==0;print(r.stdout)
