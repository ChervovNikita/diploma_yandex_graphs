"""Read only the authorized host, GPU and already-known PENCIL processes."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
CODE = r'''
from datetime import datetime, timezone
import json, socket, subprocess
from pathlib import Path
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
assert Path.cwd().resolve()==repo and socket.gethostname()=='anogena-2-0'
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.total,memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=20).strip().splitlines()
assert len(gpu)==1 and gpu[0].split(',')[0].strip()=='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
identities=[]
for pid,expected_ticks in [(407356,None),(407358,5998031229)]:
 p=Path('/proc')/str(pid)
 try:
  raw=(p/'stat').read_text(); f=raw[raw.rfind(')')+2:].split()
  ticks=int(f[19]); argv=[x.decode() for x in (p/'cmdline').read_bytes().split(bytes([0])) if x]
  if expected_ticks is not None: assert ticks==expected_ticks
  assert f[0]!='Z' and str((p/'cwd').resolve())==str(repo)
  assert any('pencil_citeseer_native300' in a for a in argv)
  identities.append(dict(PID=pid,start_ticks=ticks,state=f[0],cwd=str((p/'cwd').resolve()),argv=argv,physically_live=True))
 except FileNotFoundError: identities.append(dict(PID=pid,physically_live=False))
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),host=socket.gethostname(),GPU_metadata=gpu,known_PENCIL_processes=identities,scientific_values_read=False,signals_sent=False)))
'''

def main():
    command=['ssh','-T','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt',
        '-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=20',
        'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',
        'cd /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs && exec python3 -I -']
    result=subprocess.run(command,input=CODE,text=True,capture_output=True,timeout=50)
    with (HERE/'PREFLIGHT_TRANSPORT.json').open('x') as f:
        json.dump(dict(UTC=datetime.now(timezone.utc).isoformat(),exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr),f,indent=2); f.write('\n')
    if result.returncode: raise RuntimeError(result.stderr)
    observation=json.loads(result.stdout)
    with (HERE/'PREFLIGHT.json').open('x') as f: json.dump(observation,f,indent=2); f.write('\n')
    print(json.dumps(observation))

if __name__=='__main__': main()
