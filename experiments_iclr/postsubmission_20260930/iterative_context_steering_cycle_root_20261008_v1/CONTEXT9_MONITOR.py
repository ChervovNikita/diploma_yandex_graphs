from pathlib import Path
import socket,json,datetime,subprocess
P=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930');A=P/'context_positive_stage1_scientific_activation_root_20261008_v1';D=P/'context_positive_stage1_scientific_execution_root_20261008_v1';assert socket.gethostname()=='anogena-2-0'
def ident(pid):
 try:
  s=Path('/proc/'+str(pid)+'/stat').read_text();f=s[s.rfind(')')+2:].split();return dict(pid=pid,start_ticks=int(f[19]),state=f[0])
 except FileNotFoundError:return None
out={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':{},'progress':{},'scores_read':False}
for name in ['PARENT_OWNER.json','CURRENT.json','PROGRESS.json','FAMILY_CLOSURE.json','OWNER_FAILURE.json']:
 f=A/name
 if f.is_file():out['files'][name]=json.loads(f.read_text())
for f in D.glob('*/PROGRESS.json'):out['progress'][str(f.relative_to(D))]=json.loads(f.read_text())
for f in [A/'owner.stderr.log']+list((A/'logs').glob('*.stderr.log'))+list(D.glob('*/FAILURE.json')):
 if f.is_file() and f.stat().st_size:out['files'][str(f.relative_to(P))]=f.read_text()[-4000:]
owner=out['files'].get('PARENT_OWNER.json');out['actual_parent']=ident(owner['pid']) if owner else None
current=out['files'].get('CURRENT.json');out['actual_child']=ident(current['child']['pid']) if current else None
out['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip();print(json.dumps(out))
