"""Launch one root-admitted PubMed stage and preserve its exact owner identity."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

HERE=Path(__file__).resolve().parent
REMOTE=r'''
import hashlib,json,os,socket,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==sys.argv[2]
os.chdir(repo)
phase=repo/'experiments_iclr/postsubmission_20260930'
stage=sys.argv[1];assert stage in ('prepare','qualify','acquire')
attempt=sys.argv[4];assert attempt==stage or attempt==stage+'_recovery01'
root=phase/'pubmed_private_landmark_memory_activation_root_20261011_v2'/attempt
source=phase/'pubmed_private_landmark_memory_execution_source_20261011_v2'
config_path=root/'CONFIG.json'
assert hashlib.sha256(config_path.read_bytes()).hexdigest()==sys.argv[3]
config=json.loads(config_path.read_text())
assert config['execution_enabled'] is True and config['stage']==stage
assert Path(config['output']).resolve().is_relative_to(root) and Path(config['owner_output']).resolve().is_relative_to(root)
assert not Path(config['output']).exists() and not Path(config['owner_output']).exists()
assert not (root/'launch.log').exists() and not (root/'LAUNCH.json').exists(),'Retain existing launch; never duplicate a stage'
item=config['source_manifest'];manifest_path=Path(item['path'])
assert manifest_path==source/'MANIFEST.json'
assert manifest_path.stat().st_size==item['bytes'] and hashlib.sha256(manifest_path.read_bytes()).hexdigest()==item['sha256']
for rel,row in json.loads(manifest_path.read_text())['files'].items():
    path=source/rel
    assert path.resolve().is_relative_to(source)
    assert path.stat().st_size==row['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
argv=[str(repo/'.venv/bin/python'),'-B','-u',str(source/'owner.py'),str(config_path)]
env=dict(os.environ);env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
env['PYTHONDONTWRITEBYTECODE']='1'
with (root/'launch.log').open('xb') as log:
    child=subprocess.Popen(argv,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,close_fds=True)
    deadline=time.monotonic()+15;start_path=Path(config['owner_output'])/'OWNER_START.json'
    while not start_path.exists() and child.poll() is None and time.monotonic()<deadline:time.sleep(.1)
def identity(pid):
    path=Path('/proc')/str(pid)/'stat'
    if not path.exists():return None
    fields=path.read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,start_ticks=int(fields[19]),pgid=int(fields[2]),state=fields[0])
start=json.loads(start_path.read_text()) if start_path.exists() else None
value=dict(UTC=datetime.now(timezone.utc).isoformat(),stage=stage,owner_spawn_pid=child.pid,owner_observed=identity(child.pid),source_commit=sys.argv[2],source_manifest=item,config_sha256=sys.argv[3],owner_start=start,scientific_child_observed=identity(start['child']['pid']) if start else None,no_retry=True)
end_path=Path(config['owner_output'])/'OWNER_END.json'
if end_path.exists():value['owner_end']=json.loads(end_path.read_text())
with (root/'LAUNCH.json').open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
print(json.dumps(value))
assert start and start['stage']==stage and start['owner']['pid']==child.pid,'Owner did not enter the admitted stage; retain logs and inspect the exact launch'
assert value['owner_observed'] is not None or value.get('owner_end',{}).get('status')=='COMPLETE_EXIT_0','Owner ended unsuccessfully; no automatic restart'
'''

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','qualify','acquire'))
    parser.add_argument('expected_commit')
    parser.add_argument('--attempt')
    args=parser.parse_args();assert re.fullmatch(r'[0-9a-f]{40}',args.expected_commit)
    attempt=args.attempt or args.stage;assert attempt==args.stage or attempt==args.stage+'_recovery01'
    root=HERE/attempt;target=root/'LAUNCH_TRANSPORT.json';assert not target.exists()
    digest=hashlib.sha256((root/'CONFIG.json').read_bytes()).hexdigest()
    argv=['ssh','-tt','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',shlex.join(['/usr/bin/python3','-I','-S','-B','-c',REMOTE,args.stage,args.expected_commit,digest,attempt])]
    result=subprocess.run(argv,capture_output=True,text=True,timeout=45)
    with target.open('x') as stream:json.dump(dict(UTC=datetime.now(timezone.utc).isoformat(),exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr),stream,indent=2);stream.write('\n')
    assert result.returncode==0,'Observe the existing exact stage before any further action; no retry on transport failure'
    value=json.loads(result.stdout)
    with (root/'ACTUAL_LAUNCH.json').open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
    print(json.dumps(value))

if __name__=='__main__':main()
