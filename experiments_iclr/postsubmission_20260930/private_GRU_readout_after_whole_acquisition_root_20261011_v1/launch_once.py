"""Launch one finite GRU whole-family watcher from an exact published commit."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
REMOTE = r'''
import hashlib,json,os,socket,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==sys.argv[1]
assert subprocess.check_output(['git','branch','--show-current'],cwd=repo,text=True).strip()=='codex/postsubmission-research-20260930'
os.chdir(repo)
phase=repo/'experiments_iclr/postsubmission_20260930'
root=phase/'private_GRU_readout_after_whole_acquisition_root_20261011_v1'
source=phase/'private_GRU_whole_readout_watcher_source_20261011_v1'
config_path=root/'CONFIG.json'
assert hashlib.sha256(config_path.read_bytes()).hexdigest()==sys.argv[2]
config=json.loads(config_path.read_text())
assert config['execution_enabled'] is True
assert Path(config['output_directory'])==root/'watch_v1'
assert not (root/'watch_v1').exists() and not (root/'LAUNCH.json').exists() and not (root/'watcher_launcher.log').exists(),'Retain previous launch; no duplicate watcher'
item=config['watcher_manifest'];manifest_path=Path(item['path'])
assert manifest_path==source/'MANIFEST.json'
assert manifest_path.stat().st_size==item['bytes'] and hashlib.sha256(manifest_path.read_bytes()).hexdigest()==item['sha256']
for rel,row in json.loads(manifest_path.read_text())['files'].items():
    path=source/rel
    assert path.resolve().is_relative_to(source)
    assert path.stat().st_size==row['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
argv=[str(repo/'.venv/bin/python'),'-B','-u',str(source/'watcher.py'),str(config_path)]
env=dict(os.environ);env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
env['PYTHONDONTWRITEBYTECODE']='1'
with (root/'watcher_launcher.log').open('xb') as log:
    child=subprocess.Popen(argv,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,close_fds=True)
    deadline=time.monotonic()+15;start=root/'watch_v1/WATCH_START.json'
    while not start.exists() and child.poll() is None and time.monotonic()<deadline:time.sleep(.1)
def identity(pid):
    path=Path('/proc')/str(pid)/'stat'
    if not path.exists():return None
    fields=path.read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,start_ticks=int(fields[19]),pgid=int(fields[2]),state=fields[0])
value=dict(UTC=datetime.now(timezone.utc).isoformat(),watcher_pid=child.pid,watcher_observed=identity(child.pid),source_commit=sys.argv[1],command=argv,config_sha256=sys.argv[2],launch_exit=child.poll(),watch_start=json.loads(start.read_text()) if start.exists() else None,no_model_retry=True,no_readout_retry=True,partial_predictions_opened=False)
with (root/'LAUNCH.json').open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
print(json.dumps(value))
assert value['watcher_observed'] and value['watch_start'] and value['watch_start']['pid']==child.pid and value['watch_start']['stage']=='waiting_for_whole_acquisition','Watcher did not enter the admitted wait; retain logs and inspect exact launch before any action'
'''

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('expected_commit')
    args=parser.parse_args()
    assert re.fullmatch(r'[0-9a-f]{40}',args.expected_commit)
    config_sha=hashlib.sha256((HERE/'CONFIG.json').read_bytes()).hexdigest()
    argv=['ssh','-tt','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',shlex.join(['/usr/bin/python3','-I','-S','-B','-c',REMOTE,args.expected_commit,config_sha])]
    path=HERE/'LAUNCH_TRANSPORT.json';assert not path.exists()
    run=subprocess.run(argv,capture_output=True,text=True,timeout=45)
    value=dict(UTC=datetime.now(timezone.utc).isoformat(),exit_code=run.returncode,stdout=run.stdout,stderr=run.stderr)
    with path.open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
    assert run.returncode==0,'Transport or startup failed: inspect the existing exact watcher before any further action'
    result=json.loads(run.stdout)
    with (HERE/'ACTUAL_LAUNCH.json').open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result))

if __name__=='__main__':main()
