"""Observe an existing exact stage; fetch only complete engineering reports."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
REMOTE = r'''
import hashlib,json,socket,subprocess,sys
from pathlib import Path
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
phase=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930')
stage=sys.argv[1];assert stage in ('prepare','qualify','acquire')
attempt=sys.argv[2];assert attempt==stage or attempt==stage+'_recovery01'
root=phase/'pubmed_private_landmark_memory_activation_root_20261011_v1'/attempt
launch=json.loads((root/'LAUNCH.json').read_text())
def read(path):return json.loads(path.read_text())
def receipt(path):return dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
def observed(expected):
    path=Path('/proc')/str(expected['pid'])/'stat'
    if not path.exists():return None
    f=path.read_text().rsplit(')',1)[1].split()
    value=dict(pid=expected['pid'],start_ticks=int(f[19]),pgid=int(f[2]),state=f[0])
    assert all(value[k]==expected[k] for k in ('pid','start_ticks','pgid'))
    return value
value=dict(stage=stage,partial_quality_read=False,documents={})
start=root/'owner/OWNER_START.json';end=root/'owner/OWNER_END.json'
if start.exists():
    started=read(start);assert started['owner']['pid']==launch['owner_spawn_pid']
    value['owner_observed']=observed(started['owner']);value['child_observed']=observed(started['child'])
    value['documents']['OWNER_START.json']=dict(receipt=receipt(start),value=started)
if end.exists():
    ended=read(end);value['documents']['OWNER_END.json']=dict(receipt=receipt(end),value=ended)
    if stage in ('prepare','qualify'):
        name={'prepare':'CUSTODY.json','qualify':'QUALIFICATION.json'}[stage]
        report=root/'output'/name
        if report.exists():
            assert receipt(report)==ended['produced_artifact']
            if ended['status']=='COMPLETE_EXIT_0':assert read(report)['complete']
            value['documents'][name]=dict(receipt=receipt(report),value=read(report))
    if ended['status']!='COMPLETE_EXIT_0':
        log=root/'owner/child.log'
        value['failure_log_tail']=log.read_text()[-12000:] if log.exists() else None
if stage=='acquire' and (root/'output/PANEL.json').exists():
    panel=read(root/'output/PANEL.json')
    value['progress']=dict(complete=panel['complete'],status=panel['status'],
        groups_complete=sum(g['status']=='selected_complete' for g in panel['groups']),
        trajectories_complete=sum(t['status']=='selected_complete' for t in panel['trajectories']),
        current=[dict(seed=g['seed'],arm=g['arm'],stage=g['stage'],updates=[f['completed_updates'] for f in g['fits']]) for g in panel['groups'] if g['status']=='running'])
print(json.dumps(value))
'''

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','qualify','acquire'));parser.add_argument('label');parser.add_argument('--attempt');args=parser.parse_args()
    assert re.fullmatch(r'[A-Za-z0-9_]+',args.label)
    attempt=args.attempt or args.stage;assert attempt==args.stage or attempt==args.stage+'_recovery01'
    root=HERE/attempt;target=root/('OBSERVATION_'+args.label+'.json');assert not target.exists()
    argv=['ssh','-tt','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',shlex.join(['/usr/bin/python3','-I','-S','-B','-c',REMOTE,args.stage,attempt])]
    result=subprocess.run(argv,capture_output=True,text=True,timeout=45)
    with target.open('x') as stream:json.dump(dict(UTC=datetime.now(timezone.utc).isoformat(),exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr),stream,indent=2);stream.write('\n')
    assert result.returncode==0,'Failed observation; do not restart a stage'
    value=json.loads(result.stdout)
    for name,item in value['documents'].items():
        path=root/name
        if path.exists():assert json.loads(path.read_text())==item['value']
        else:
            with path.open('x') as stream:json.dump(item['value'],stream,indent=2);stream.write('\n')
        authority=root/(name+'.REMOTE_RECEIPT.json')
        if not authority.exists():authority.write_text(json.dumps(item['receipt'],indent=2)+'\n')
    summary={k:v for k,v in value.items() if k!='documents'}
    if 'OWNER_END.json' in value['documents']:summary['owner_end']=value['documents']['OWNER_END.json']['value']['status']
    for name in ('CUSTODY.json','QUALIFICATION.json'):
        if name in value['documents']:
            report=value['documents'][name]['value'];summary['report']=dict(complete=report['complete'],status=report['status'],costs=report['costs'])
    print(json.dumps(summary))

if __name__=='__main__':main()
