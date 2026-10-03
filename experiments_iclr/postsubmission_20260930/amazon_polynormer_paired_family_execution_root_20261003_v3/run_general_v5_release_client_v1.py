"""Dispatch metadata preparation or one admitted V5 register/qualify release."""
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
repo=Path(sys.argv[1]);phase=repo/'experiments_iclr/postsubmission_20260930'
action,release,output=sys.argv[2:];root=phase/'amazon_polynormer_paired_family_execution_root_20261003_v3'
assert subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()=='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
def verify(row):
 p=phase/row['path'];assert not Path(row['path']).is_absolute() and '..' not in Path(row['path']).parts
 assert p.resolve().is_relative_to(phase) and not p.is_symlink()
 b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==row['sha256'] and len(b)==row['bytes'];return p
policy=json.loads((root/'CONDITIONAL_REGISTER_QUALIFIER_SPEC_v1.json').read_text())
if action in ['after-runtime','after-register']:
 helper=verify(policy['metadata_helper'])
 command=['/usr/bin/python3','-I','-S','-B',str(helper),'--stage',action]
else:
 assert action in ['register','qualify']
 for value in [release,output]:
  assert not Path(value).is_absolute() and '..' not in Path(value).parts and (phase/value).is_relative_to(root)
 a=json.loads((phase/release).read_text());assert a['kind']==action and a['execution_authorized'] is True
 assert a['source']==policy['source'] and a['source_review']==policy['source_review']
 assert a['output']==output and a['self_path']==release and not (phase/output).exists()
 manifest=verify(a['source']['manifest']);verify(a['source']['seal'])
 assert manifest.parent.name=='amazon_polynormer_paired_family_source_preparation_20261003_v5'
 assert a['source']['manifest']['sha256']=='25606a662be16d39219c9ef1fb75f13433f6d76b43559624b8607d1a5a5642bb'
 review=json.loads(verify(a['source_review']).read_text());assert review['status']=='passed' and review['source']==a['source']
 command=['/usr/bin/python3','-B',str(manifest.parent/'supervise.py'),'--kind',action,'--release',str(phase/release),'--output',str(phase/output)]
os.chdir(phase)
sys.exit(subprocess.run(command,cwd=phase).returncode)
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    actions=parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--prepare-stage',choices=('after-runtime','after-register'))
    actions.add_argument('--kind',choices=('register','qualify'))
    parser.add_argument('--release')
    parser.add_argument('--output')
    parser.add_argument('--id',required=True)
    args=parser.parse_args()
    assert Path(args.id).name==args.id and args.id not in ('','.','..')
    assert bool(args.kind)==bool(args.release and args.output)
    action=args.prepare_stage or args.kind
    paths=[HERE/(args.id+s) for s in ('_stdout.log','_stderr.log','_TERMINAL.json')]
    assert not any(p.exists() for p in paths), 'Immutable client attempt already exists'
    command=shlex.join(['/usr/bin/python3','-I','-S','-B','-c',CODE,REPO,action,args.release or '',args.output or ''])
    started=datetime.now(timezone.utc).isoformat()
    with paths[0].open('x') as so,paths[1].open('x') as se:
        result=subprocess.run([*SSH,command],stdout=so,stderr=se,check=False)
    terminal=dict(start_UTC=started,terminal_UTC=datetime.now(timezone.utc).isoformat(),exit_code=result.returncode,
        route=LOGIN,action=action,release=args.release,output=args.output,
        source_client_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        remote_code_sha256=hashlib.sha256(CODE.encode()).hexdigest(),
        supervisor_selected_from_admitted_source=True,automatic_restart=False,filesystem_isolation=False,
        predictive_fits_launched=False)
    with paths[2].open('x') as f:
        json.dump(terminal,f,indent=2);f.write('\n')
    print(json.dumps(terminal),flush=True)
    raise SystemExit(result.returncode)


if __name__=='__main__':
    main()
