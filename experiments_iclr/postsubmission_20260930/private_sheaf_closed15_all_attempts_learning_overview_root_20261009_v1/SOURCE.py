from pathlib import Path
import json,subprocess,socket,datetime
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';A=P/'private_sheaf_baseline_screen_activation_root_20261009_v1';D=P/'private_sheaf_baseline_screen_execution_root_20261009_v1'
assert socket.gethostname()=='anogena-2-0' and subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
assert not Path('/proc/545332').exists() and not Path('/proc/545335').exists()
terminal=json.loads((A/'TERMINAL.json').read_text());assert terminal['reaped'] and terminal['actual_worker_CUDA_absent']
records=[]
for f in sorted(D.glob('*/RESULT.json')):
 x=json.loads(f.read_text());assert x['status'] in ('complete','failed');h=[json.loads(s) for s in (f.parent/'HISTORY.jsonl').read_text().splitlines()];r={k:x[k] for k in ('run_id','family','config_id','seed','status','epochs_completed','selected_epoch','complete_attempt_seconds') if k in x};r['scores']=x.get('scores');r['failure']=x.get('failure');r['TRAIN_first_last_NLL']=[h[0]['train']['nll'],h[-1]['train']['nll']];r['VALID_first_last_AUROC']=[h[0]['valid']['auroc'],h[-1]['valid']['auroc']];records.append(r)
assert len(records)==15
summary=json.loads((D/'SCREEN_SUMMARY.json').read_text());assert summary['architecture_frozen'] is False and summary['panel_complete'] is False
print(json.dumps({'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'all15_terminal_before_read':True,'screen_status':summary,'records':records,'means_or_winner_selected':False,'TEST_truth':False,'original_paper_scores_changed':False}))
