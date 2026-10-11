"""Execute one separately admitted finite readout stage and preserve receipts."""
import argparse
import datetime
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
REPO = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
PHASE = REPO + '/experiments_iclr/postsubmission_20260930'
REMOTE = r'''
import hashlib,json,os,socket,subprocess
from pathlib import Path
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
repo=__REPO__;os.chdir(repo);base=Path(__ROOT__);stage=__STAGE__
config=json.loads((base/__CONFIG__).read_text())
assert config['stage']==stage
result=subprocess.run([repo+'/.venv/bin/python','-B','-u',__OWNER__,str(base/__CONFIG__)],cwd=repo,stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=config['limits']['wall_seconds']+70)
def fetch(path):
    path=Path(path);body=path.read_bytes()
    return dict(receipt=dict(path=str(path),bytes=len(body),sha256=hashlib.sha256(body).hexdigest()),value=json.loads(body))
value=dict(stage=stage,exit_code=result.returncode,owner_stdout=result.stdout,owner_stderr=result.stderr)
owner=Path(config['owner_output'])/'OWNER_END.json'
if owner.exists():value['owner_end']=fetch(owner)
artifact=Path(config['output'])/('TEMPERATURES.json' if stage=='calibrate' else 'DEVELOPMENT_REPORT.json')
if artifact.exists():
    item=fetch(artifact)
    if stage=='score' and result.returncode==0:
        # Preserve every class, contrast and cost. Omit only redundant exhaustive
        # eight-assignment tables from the transferred compact copy; the sealed
        # full report remains on this server with its exact receipt.
        for contrast in item['value'].get('comparisons',[]):
            for metric in contrast['metric_deltas'].values():metric.pop('eight_sign_flip_assignments',None)
        for rows in ('accuracy_screens','confidence_screens'):
            for gate in item['value'].get('gates',{}).get(rows,[]):
                gate.get('contrast',{}).pop('eight_sign_flip_assignments',None)
        item['value'].update(full_report_receipt=item['receipt'],all_per_class_results_retained=True,exhaustive_sign_assignments_retained_in_full_remote_report=True,full_checkpoints_logits_and_targets_remain_remote=True)
    value['artifact']=item
if result.returncode:
    log=Path(config['owner_output'])/'child.log'
    if log.exists():value['child_failure_log']=log.read_text()[-12000:]
print(json.dumps(value,separators=(',',':')),flush=True)
'''

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('calibrate','score'))
    args=parser.parse_args()
    config_name='CALIBRATE_CONFIG.json' if args.stage=='calibrate' else 'SCORE_CONFIG.json'
    config=json.loads((HERE/config_name).read_text())
    remote=REMOTE.replace('__REPO__',repr(REPO)).replace('__ROOT__',repr(PHASE+'/'+HERE.name)).replace('__STAGE__',repr(args.stage)).replace('__CONFIG__',repr(config_name)).replace('__OWNER__',repr(PHASE+'/native_meanSAGE_tour_scan_execution_source_20261011_v1/owner.py'))
    compile(remote,'readout_dispatch','exec')
    (HERE/(args.stage+'_DISPATCH_SOURCE.py')).write_text(remote)
    command=['ssh','-tt','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',shlex.join(['/usr/bin/python3','-I','-S','-B','-c',remote])]
    result=subprocess.run(command,capture_output=True,text=True,timeout=config['limits']['wall_seconds']+100)
    (HERE/(args.stage+'_TRANSPORT.json')).write_text(json.dumps(dict(exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr),separators=(',',':'))+'\n')
    if result.returncode:raise RuntimeError('Readout transport failed; inspect preserved transport')
    value=json.loads(result.stdout.strip());value['observed_UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (HERE/(args.stage+'_ACTUAL_RESULT.json')).write_text(json.dumps(value,separators=(',',':'))+'\n')
    if 'owner_end' in value:(HERE/(args.stage+'_OWNER_END.json')).write_text(json.dumps(value['owner_end']['value'],indent=2)+'\n')
    if 'artifact' in value:
        name='TEMPERATURES.json' if args.stage=='calibrate' else 'SUMMARY.json'
        (HERE/name).write_text(json.dumps(value['artifact']['value'],separators=(',',':'))+'\n')
    artifact=value.get('artifact',{}).get('value',{})
    print(json.dumps(dict(stage=args.stage,exit_code=value['exit_code'],complete=artifact.get('complete'),owner_status=value.get('owner_end',{}).get('value',{}).get('status'),artifact_receipt=value.get('artifact',{}).get('receipt'),failure=artifact.get('failure'),child_failure_log=value.get('child_failure_log'))))
    if value['exit_code']:raise SystemExit(value['exit_code'])

if __name__=='__main__':main()
