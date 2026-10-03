"""Launch one explicitly authorized V5 runtime capture; preserve client terminals."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
REPO = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
LOGIN = 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
SSH = ['ssh','-T','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt',
    '-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','UpdateHostKeys=no','-o','StrictHostKeyChecking=yes',LOGIN]
CODE = '''from pathlib import Path
import hashlib,json,os,subprocess,sys
repo=Path(sys.argv[1]);phase=repo/'experiments_iclr/postsubmission_20260930'; mode=sys.argv[2]
assert mode in ('cpu','gpu')
assert subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()=='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
root=phase/'amazon_polynormer_paired_family_execution_root_20261003_v3'
packet=phase/'amazon_polynormer_paired_family_source_preparation_20261003_v5'
release=root/'releases'/('runtime_'+mode+'.json'); output=root/('runtime_'+mode)
a=json.loads(release.read_text()); assert a['kind']=='runtime_capture' and a['execution_authorized'] is True
assert a['device']==('cpu' if mode=='cpu' else 'cuda:0')
assert a['interpreter']['path']==str(repo/'.venv/bin/python')
assert a['source']['manifest']['sha256']=='25606a662be16d39219c9ef1fb75f13433f6d76b43559624b8607d1a5a5642bb'
assert not output.exists(), 'Existing runtime output; no retry'
os.chdir(phase)
sys.exit(subprocess.run(['/usr/bin/python3','-B',str(packet/'supervise.py'),'--kind','runtime_capture','--release',str(release),'--output',str(output)],cwd=phase).returncode)
'''


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',required=True,choices=('cpu','gpu'))
    parser.add_argument('--id',required=True)
    args=parser.parse_args()
    assert Path(args.id).name==args.id and args.id not in ('','.','..')
    paths=[HERE/(args.id+s) for s in ('_stdout.log','_stderr.log','_TERMINAL.json')]
    assert not any(p.exists() for p in paths), 'Immutable client attempt already exists'
    command=shlex.join(['/usr/bin/python3','-I','-S','-B','-c',CODE,REPO,args.mode])
    started=datetime.now(timezone.utc).isoformat()
    with paths[0].open('x') as so, paths[1].open('x') as se:
        result=subprocess.run([*SSH,command],stdout=so,stderr=se,check=False)
    terminal=dict(start_UTC=started,terminal_UTC=datetime.now(timezone.utc).isoformat(),exit_code=result.returncode,
        route=LOGIN,kind='runtime_capture',mode=args.mode,source_client_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        remote_code_sha256=hashlib.sha256(CODE.encode()).hexdigest(),source_packet='amazon_polynormer_paired_family_source_preparation_20261003_v5',
        automatic_restart=False,filesystem_isolation=False,qualifier_or_fit_launched=False)
    with paths[2].open('x') as f:
        json.dump(terminal,f,indent=2);f.write('\n')
    print(json.dumps(terminal),flush=True)
    raise SystemExit(result.returncode)


if __name__=='__main__':
    main()
