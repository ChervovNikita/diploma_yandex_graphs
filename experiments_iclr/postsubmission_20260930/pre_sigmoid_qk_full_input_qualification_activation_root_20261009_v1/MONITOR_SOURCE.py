from pathlib import Path
import json,socket,subprocess,datetime
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';A=P/'pre_sigmoid_qk_full_input_qualification_activation_root_20261009_v1';D=P/'pre_sigmoid_qk_full_input_qualification_output_20261009_v1'
assert socket.gethostname()=='anogena-2-0' and subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
def ident(pid):
 try:r=Path('/proc',str(pid),'stat').read_text();f=r[r.rfind(')')+2:].split();return dict(pid=pid,start_ticks=int(f[19]),state=f[0])
 except FileNotFoundError:return None
launch=json.loads((A/'LAUNCH.json').read_text());parent=ident(launch['parent']['pid'])
if parent:assert parent['start_ticks']==launch['parent']['start_ticks']
x=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),parent=parent,cells=[],science=False,VALID_scores=False)
f=A/'WORKER_OWNER.json'
if f.exists():
 h=json.loads(f.read_text());child=ident(h['child']['pid']);x['child']=child
 if child:assert child['start_ticks']==h['child']['start_ticks']
f=A/'TERMINAL.json'
if f.exists():x['terminal']=json.loads(f.read_text())
f=D/'REPORT.json'
if f.exists():
 v=json.loads(f.read_text());x['cells']=[dict(operator=c['operator'],kind=c['kind'],passed=c['passed'],error=c.get('error'),stages=[dict(stage=z['stage'],passed=z['passed'],max_abs_difference=z.get('identity_max_abs_logit_difference')) for z in c['stages']]) for c in v['cells']];x['report_complete']=v['complete'];x['report_error']=v.get('error')
if 'terminal' in x:x['worker_log']=(A/'WORKER.log').read_text()[-5000:]
print(json.dumps(x))
