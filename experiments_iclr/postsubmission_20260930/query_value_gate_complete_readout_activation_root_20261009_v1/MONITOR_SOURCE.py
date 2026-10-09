from pathlib import Path
import json,datetime,socket,subprocess
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';A=P/'query_value_gate_complete_readout_activation_root_20261009_v1'
assert socket.gethostname()=='anogena-2-0' and subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
def ident(pid):
 try:
  s=Path('/proc',str(pid),'stat').read_text();v=s[s.rfind(')')+2:].split();return dict(pid=pid,start_ticks=int(v[19]),state=v[0])
 except FileNotFoundError:return None
launch=json.loads((A/'LAUNCH.json').read_text());out=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),handles={},terminal={})
for h in launch['handles']:
 z=ident(h['identity']['pid'])
 if z:assert z['start_ticks']==h['identity']['start_ticks']
 out['handles'][h['name']]=z
for name,folder,terminal in [('readout','query_value_gate_complete_readout_execution_root_20261009_v1','ROUTINE_TERMINAL.json')]:
 D=P/folder;f=D/terminal
 if f.exists():out['terminal'][name]=json.loads(f.read_text())
 f=D/'PROGRESS.json'
 if f.exists():q=json.loads(f.read_text());out[name+'_progress']={k:q[k] for k in ('completed_bank_count','native_calls') if k in q}
 if (D/'FAILURE.json').exists():out['terminal'][name+'_failure']=json.loads((D/'FAILURE.json').read_text())
 if f.exists() or name in out['terminal']:
  log=A/('reconstruct15.log');out[name+'_log']=log.read_text()[-3000:]
print(json.dumps(out))
