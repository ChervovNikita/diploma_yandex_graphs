"""Observe this exact readout watcher and closure counts without opening scores."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess

HERE=Path(__file__).resolve().parent
REMOTE=r'''
import json,socket,subprocess
from pathlib import Path
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
root=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/pubmed_LM4_whole_readout_watcher_root_20261011_v1')
launch=json.loads((root/'LAUNCH.json').read_text());expected=launch['watcher_observed']
stat=Path('/proc')/str(expected['pid'])/'stat';observed=None
if stat.exists():
    fields=stat.read_text().rsplit(')',1)[1].split()
    observed=dict(pid=expected['pid'],start_ticks=int(fields[19]),pgid=int(fields[2]),state=fields[0])
    assert all(observed[k]==expected[k] for k in ('pid','start_ticks','pgid'))
value=dict(watcher_observed=observed,partial_scores_opened=False)
for rel in ('WATCH_START.json','WATCH_PROGRESS.json','WATCH_END.json','calibrate_v1/CAL_PROGRESS.json','score_v1/DEV_PROGRESS.json'):
    path=root/'watch_v1'/rel
    if path.exists():value[rel]=json.loads(path.read_text())
if value.get('WATCH_END.json',{}).get('complete') is False:
    path=root/'watcher_launcher.log'
    if path.exists():value['failure_log_tail']=path.read_text()[-10000:]
print(json.dumps(value))
'''

def main():
    parser=argparse.ArgumentParser();parser.add_argument('label');args=parser.parse_args()
    assert args.label.replace('_','').isalnum()
    target=HERE/('OBSERVATION_'+args.label+'.json');assert not target.exists()
    argv=['ssh','-tt','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',shlex.join(['/usr/bin/python3','-I','-S','-B','-c',REMOTE])]
    result=subprocess.run(argv,capture_output=True,text=True,timeout=45)
    with target.open('x') as stream:json.dump(dict(UTC=datetime.now(timezone.utc).isoformat(),exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr),stream,indent=2);stream.write('\n')
    assert result.returncode==0,'Observation failed; do not restart the watcher'
    print(result.stdout.strip())

if __name__=='__main__':main()
