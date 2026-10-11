"""Read lifecycle metadata; fetch diagnostic only after whole successful ownership closure."""
import argparse
import base64
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shlex
import zlib

H = Path(__file__).resolve().parent
P = H.parent
REMOTE = r'''
import base64,zlib,json,sys,socket,subprocess,pathlib,datetime
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
payload=json.loads(zlib.decompress(base64.b64decode(sys.stdin.read())))
H=pathlib.Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/WikiCS_E_source_context_association_diagnostic_root_20261011_v1')
def read(n):
    path=H/n;return json.loads(path.read_text()) if path.exists() else None
def process(pid):
    path=pathlib.Path('/proc')/str(pid)/'stat'
    if not path.exists():return dict(pid=pid,present=False)
    b=path.read_text().rsplit(')',1)[1].split();return dict(pid=pid,present=True,state=b[0],parent_pid=int(b[1]),start_ticks=int(b[19]))
launch,start,end=read('SCIENTIFIC_LAUNCH_V1.json'),read('OWNER_START_V1.json'),read('OWNER_END_V1.json')
value=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),launch=launch,owner_start=start,owner_end=end,progress=read('complete_v1/PROGRESS.json'),failure=read('complete_v1/FAILURE.json'),partial_predictive_metrics_read=False,TEST_access=False)
value['processes']={}
if launch:value['processes']['owner']=process(launch['owner_pid'])
if start:value['processes']['child']=process(start['child_pid'])
if end and end['exit_code']==0 and end['complete_roster'] and end['child_absent']:
    complete=read('complete_v1/COMPLETE_ASSOCIATION_DIAGNOSTIC.json')
    assert complete['complete'] and complete['served_banks']==36 and complete['native_trajectories']==108
    value['complete_diagnostic']=complete
elif end and end['exit_code']!=0:
    value['failure_log']=(H/'child_v1.log').read_text()[-5000:]
elif launch and not value['processes']['owner']['present'] and not end:
    value['owner_failure_log']=(H/'owner_outer_v1.log').read_text()[-5000:]
print(json.dumps(value))
'''

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--version',type=int,required=True);args=parser.parse_args()
    output=H/f'METADATA_OBSERVATION_V{args.version}.json';assert not output.exists()
    spec=importlib.util.spec_from_file_location('association_metadata_transport',P/'publication/publish_exact_inventory_v19.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    argv=['ssh','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',shlex.join(['/usr/bin/python3','-I','-S','-B','-c',REMOTE,'head-lifecycle-metadata','literal-route','complete-roster-only'])]
    result=module.terminal_transport(argv,input=base64.b64encode(zlib.compress(b'{}')).decode(),capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr[-2000:]+result.stdout[-2000:]
    value=json.loads(result.stdout)
    complete=value.pop('complete_diagnostic',None)
    if complete:
        path=H/'COMPLETE_ASSOCIATION_DIAGNOSTIC_V1.json'
        if path.exists():assert json.loads(path.read_text())==complete
        else:path.write_text(json.dumps(complete,indent=2)+'\n')
        value['complete_diagnostic_fetched']=str(path)
    output.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))

if __name__=='__main__':main()
