from pathlib import Path
import socket,subprocess,json,datetime
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';A=P/'private_sheaf_baseline_screen_activation_root_20261009_v1';D=P/'private_sheaf_baseline_screen_execution_root_20261009_v1';B=P/'bsnn_full_input_qualification_activation_root_20261009_v1';E=P/'bsnn_full_input_qualification_execution_root_20261009_v1'
assert socket.gethostname()=='anogena-2-0' and subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
def ident(pid):
 try:
  s=Path('/proc',str(pid),'stat').read_text();v=s[s.rfind(')')+2:].split();return {'pid':pid,'start_ticks':int(v[19]),'state':v[0]}
 except FileNotFoundError:return None
terminal=json.loads((A/'TERMINAL.json').read_text());assert terminal['reaped'] and terminal['actual_worker_absent'] and terminal['actual_worker_CUDA_absent'] and ident(545332) is None and ident(545335) is None
out={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'all15_attempts_terminal':True,'comparative_scores_read':False,'failures':{},'bsnn':{}}
for f in D.glob('*/RESULT.json'):
 x=json.loads(f.read_text())
 if x['status']=='failed':out['failures'][f.parent.name]={k:v for k,v in x.items() if k in ('failure','last_attempted_epoch','selected_epoch','epochs_completed','restore_max_abs_role_logp_difference','restore_auroc_absolute_difference','restore_role_prediction_changes','complete_attempt_seconds','forward_counts','fit_cuda_peak_allocated_bytes')}
launch=json.loads((B/'LAUNCH.json').read_text());parent=ident(launch['parent']['pid']);out['bsnn']['parent']=parent
if parent:assert parent['start_ticks']==launch['parent']['start_ticks']
if (B/'WORKER_OWNER.json').exists():
 h=json.loads((B/'WORKER_OWNER.json').read_text());child=ident(h['child']['pid']);out['bsnn']['child']=child
 if child:assert child['start_ticks']==h['child']['start_ticks']
for name in ('COMPLETE.json','RESULT.json'):
 f=E/name
 if f.exists():
  x=json.loads(f.read_text());out['bsnn'][name]={k:x[k] for k in ('status','qualification_passed','stage','counters','failure','reconstruction_diagnostics','complete_attempt_seconds','peak_CUDA_allocated_bytes') if k in x}
if (B/'TERMINAL.json').exists():out['bsnn']['terminal']=json.loads((B/'TERMINAL.json').read_text())
if parent is None:out['bsnn']['owner_log']=(B/'OWNER.log').read_text()[-5000:];out['bsnn']['worker_log']=(B/'WORKER.log').read_text()[-5000:] if (B/'WORKER.log').exists() else None
print(json.dumps(out))
