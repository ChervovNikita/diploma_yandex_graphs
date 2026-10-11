
import hashlib,json,os,socket,subprocess
from pathlib import Path
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
repo='/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs';os.chdir(repo);base=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/graph_walk_complete_readout_root_20261011_v1');stage='score'
config=json.loads((base/'SCORE_CONFIG.json').read_text())
assert config['stage']==stage
result=subprocess.run([repo+'/.venv/bin/python','-B','-u','/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/native_meanSAGE_tour_scan_execution_source_20261011_v1/owner.py',str(base/'SCORE_CONFIG.json')],cwd=repo,stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=config['limits']['wall_seconds']+70)
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
