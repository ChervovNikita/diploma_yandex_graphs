"""Local authorized SSH launcher; root CPU audit only after exact full40 exit."""
import argparse,hashlib,json,shlex,subprocess
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');args=p.parse_args()
route=HERE.parent/'protocols/AUTHORIZED_ALLOCATION_ROUTE_20260930_v1.json'
b=route.read_bytes();assert hashlib.sha256(b).hexdigest()=='7d255ec7a43bd00781e842e0f564698fef26fb28e42f80ba53b303dba6acafda'
d=json.loads(b);login='anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru';assert d['ssh_destination']==login and d['port']==2222
code=(HERE/'materialize_and_evaluate_remote.py').read_text();stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
ssh=['ssh','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','UpdateHostKeys=no','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2',login]
command=[*ssh,shlex.join(['/usr/bin/python3','-I','-S','-B','-c',code,'--execute' if args.execute else '--materialize'])]
# No timeout, signal, restart or isolation wrapper is added to an admitted audit.
result=subprocess.run(command,capture_output=True,text=True,timeout=None if args.execute else 55)
receipt={'UTC':datetime.now(timezone.utc).isoformat(),'destination':login,'port':2222,'remote_code_sha256':hashlib.sha256(code.encode()).hexdigest(),'exit_code':result.returncode,'stderr':result.stderr,'stdout_sha256':hashlib.sha256(result.stdout.encode()).hexdigest(),'audit_execution_requested':args.execute,'private_key_contents_inspected':False}
with (HERE/('READY_TRANSPORT_'+stamp+'.json')).open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
if result.stdout:
    value=json.loads(result.stdout)
    with (HERE/('READY_OBSERVATION_'+stamp+'.json')).open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    print(json.dumps(value))
else:print(json.dumps(receipt))
raise SystemExit(result.returncode)
