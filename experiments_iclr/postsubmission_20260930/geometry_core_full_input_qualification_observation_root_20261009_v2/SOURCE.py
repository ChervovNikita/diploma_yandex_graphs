from pathlib import Path
import json,socket,subprocess,datetime,hashlib
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';A=P/'geometry_only_core_full_input_qualification_activation_root_20261009_v1';D=P/'geometry_only_core_full_input_qualification_execution_root_20261009_v1'
assert socket.gethostname()=='anogena-2-0' and subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
def ident(h):
 if h is None:return None
 try:s=Path('/proc',str(h['pid']),'stat').read_text()
 except FileNotFoundError:return None
 f=s[s.rfind(')')+2:].split();x=dict(pid=h['pid'],start_ticks=int(f[19]),state=f[0]);assert x['start_ticks']==h['start_ticks'];return x
out=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),metadata={},files={},scores_read=False)
for name in ('LAUNCH.json','WORKER_OWNER.json','RESOURCE_ADMISSION.json','TERMINAL.json'):
 f=A/name
 if f.exists():
  b=f.read_bytes();out['metadata'][name]=json.loads(b);out['files'][str(f.relative_to(P))]=dict(sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),data=b.decode())
out['actual_parent']=ident(out['metadata']['LAUNCH.json']['parent']);out['actual_worker']=ident(out['metadata'].get('WORKER_OWNER.json',{}).get('child'))
f=D/'QUALIFICATION.json'
if f.exists():
 x=json.loads(f.read_text());out['qualification_progress']={k:v for k,v in x.items() if k in ('status','qualification_passed','failure','counters','stage','source_only_preparation','actual_TRAIN_forwards','all_native_forwards')}
if out['metadata'].get('TERMINAL.json',{}).get('error'):
 out['owner_log_tail']=(A/'OWNER.log').read_text()[-3000:];out['worker_log_tail']=(A/'WORKER.log').read_text()[-3000:] if (A/'WORKER.log').exists() else None
print(json.dumps(out))
