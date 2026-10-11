import base64,importlib.util,json,shlex,zlib,hashlib
from pathlib import Path
W=Path(__file__).resolve().parent;P=W.parent
remote=r'''
import base64,json,socket,subprocess,sys,zlib,pathlib,time
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
payload=json.loads(zlib.decompress(base64.b64decode(sys.stdin.read())))
assert sys.argv[1:4]==['/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs','GPU-44039938-fd82-41d2-fefd-de71514e2fac','probe']
start=time.monotonic()
mem=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,memory.total','--format=csv,noheader,nounits'],text=True,timeout=15).strip().split(',')
gpu_free,gpu_total=[int(a.strip())*1024**2 for a in mem]
host={line.split(':')[0]:int(line.split()[1])*1024 for line in pathlib.Path('/proc/meminfo').read_text().splitlines() if line.split()[1].isdigit()}
assert gpu_free>=20*1024**3 and host['MemAvailable']>=20*1024**3,'Insufficient free GPU or host RAM for delegated finite caps'
R=pathlib.Path(sys.argv[1]);runtime=subprocess.check_output([str(R/'.venv/bin/python'),'-B','-c','import json,numpy,torch;print(json.dumps(dict(numpy=numpy.__version__,torch=torch.__version__)))'],text=True,timeout=90)
active=subprocess.run(['ps','-p','635864,635866,636243','-o','pid=,stat=,comm='],capture_output=True,text=True,timeout=15)
print(json.dumps(dict(route_verified=True,hostname=socket.gethostname(),gpu_uuid=sys.argv[2],GPU_free_bytes=gpu_free,GPU_total_bytes=gpu_total,host_MemAvailable_bytes=host['MemAvailable'],runtime_versions=json.loads(runtime),protected_pids_readonly_observation=active.stdout,seconds=time.monotonic()-start,overlap_with_sign_dropout=True,no_array_model_or_fit_work=True)))
'''
(W/'PROBE_REMOTE_CODE.py.txt').write_text(remote)
spec=importlib.util.spec_from_file_location('reviewed_peer_pty',P/'publication/publish_exact_inventory_v19.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
command=shlex.join(['/usr/bin/python3','-I','-S','-B','-c',remote,'/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs','GPU-44039938-fd82-41d2-fefd-de71514e2fac','probe'])
argv=['ssh','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',command]
r=m.terminal_transport(argv,input=base64.b64encode(zlib.compress(b'{}')).decode(),capture_output=True,text=True,timeout=150)
(W/'PROBE_TRANSPORT.json').write_text(json.dumps(dict(exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr),indent=2)+'\n')
if r.returncode:print(json.dumps(dict(exit_code=r.returncode,stdout=r.stdout[-2000:],stderr=r.stderr[-2000:])));raise SystemExit(r.returncode)
v=json.loads(r.stdout);(W/'RESOURCE_RUNTIME_PROBE.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v,indent=2))
