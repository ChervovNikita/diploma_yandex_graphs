"""Execute exactly one root-admitted V6 runtime capture on the authorized route."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
LOGIN = 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
CODE = '''from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess,sys
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
phase=repo/'experiments_iclr/postsubmission_20260930'
root=phase/'amazon_polynormer_paired_family_execution_root_20261003_v3'
source=phase/'amazon_polynormer_paired_family_source_preparation_20261003_v6'
mode=sys.argv[1];assert mode in ('runtime_cpu','runtime_gpu')
os.chdir(repo)
g=subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,timeout=15)
assert Path.cwd()==repo and g.returncode==0 and g.stdout.strip()=='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
release=root/'v6_releases'/(mode+'_v1.json')
value=json.loads(release.read_text());assert value['execution_authorized'] is True and value['kind']=='runtime_capture'
assert value['source']['manifest']['sha256']=='d4160d8aeec2b770274875f9a1fb137facc9f93efd4eca6ad91e58802521ff44'
assert value['self_path']==str(release.relative_to(phase))
expected=root/('v6_runtime_cpu_v1' if mode=='runtime_cpu' else 'v6_runtime_cuda0_v1')
assert phase/value['output']==expected and not expected.exists()
assert value['device']==('cpu' if mode=='runtime_cpu' else 'cuda:0')
assert value['caps']==dict(wall_seconds=900,rss_bytes=16*2**30,cuda_peak_allocated_bytes=16*2**30,cuda_peak_reserved_bytes=16*2**30)
assert hashlib.sha256((source/'MANIFEST.json').read_bytes()).hexdigest()==value['source']['manifest']['sha256']
review=json.loads((phase/value['source_review']['path']).read_text());assert review['status']=='passed' and review['source']==value['source']
command=['/usr/bin/python3','-B',str(source/'supervise.py'),'--kind','runtime_capture','--release',str(release),'--output',str(expected)]
r=subprocess.run(command,cwd=phase,capture_output=True,text=True,stdin=subprocess.DEVNULL)
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),mode=mode,route=dict(repository=str(repo),single_GPU_UUID=g.stdout.strip()),supervisor_command=command,physical_supervisor_exit_code=r.returncode,supervisor_stdout=r.stdout,supervisor_stderr=r.stderr,automatic_retry=False,qualifier_fit_or_scoring_launched=False)))
sys.exit(r.returncode)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('runtime_cpu','runtime_gpu'),required=True)
    args = parser.parse_args()
    paths = {key: HERE/(args.mode+'_'+key+'.json') for key in ('TRANSPORT','RESULT')}
    assert not any(p.exists() for p in paths.values()), 'Once-only client receipt already exists'
    ssh = ['ssh','-T','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt',
        '-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','UpdateHostKeys=no',
        '-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15',
        '-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2',LOGIN]
    started = datetime.now(timezone.utc).isoformat()
    result = subprocess.run([*ssh,shlex.join(['/usr/bin/python3','-I','-S','-B','-c',CODE,args.mode])],capture_output=True,text=True)
    receipt = dict(start_UTC=started,terminal_UTC=datetime.now(timezone.utc).isoformat(),
        destination=LOGIN,port=2222,mode=args.mode,exit_code=result.returncode,
        stderr=result.stderr,stdout_sha256=hashlib.sha256(result.stdout.encode()).hexdigest(),
        client_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),remote_code_sha256=hashlib.sha256(CODE.encode()).hexdigest(),
        ordinary_supervision=True,automatic_retry=False,private_key_contents_inspected=False)
    with paths['TRANSPORT'].open('x') as stream:
        json.dump(receipt,stream,indent=2);stream.write('\n')
    if result.stdout:
        value = json.loads(result.stdout)
        with paths['RESULT'].open('x') as stream:
            json.dump(value,stream,indent=2);stream.write('\n')
        print(json.dumps(value),flush=True)
    else:
        print(json.dumps(receipt),flush=True)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
