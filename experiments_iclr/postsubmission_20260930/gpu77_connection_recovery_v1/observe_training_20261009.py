"""Read only exact owned training handles and metric-free endpoint presence."""
from pathlib import Path
import datetime,json,socket,subprocess
R=Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
P=R/'experiments_iclr/postsubmission_20260930'
GPUS={'GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998','GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced'}
assert set(subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines())==GPUS
def identity(pid,birth):
 try:s=Path('/proc',str(pid),'stat').read_text()
 except FileNotFoundError:return dict(pid=pid,present=False,expected_birth=birth)
 f=s[s.rfind(')')+2:].split();assert int(f[19])==birth,'Reused PID'
 return dict(pid=pid,present=True,start_ticks=int(f[19]),group=int(f[2]),session=int(f[3]),state=f[0])
d=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),hostname=socket.gethostname(),
 boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
 head=subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip(),
 GPU=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True).splitlines(),
 processes=[identity(3819571,1766266039),identity(3819585,1766266049),identity(3713404,1760762180)],
 scores_opened=False,TEST_truth_accessed=False,other_processes_changed=False)
d['qk']=[]
for route in ['gpu77_a998','gpu77_8ced']:
 q=P/'pre_sigmoid_qk_distributed_F_execution_root_20261009_v1'/route
 v=dict(route=route,present=q.exists())
 if q.exists():
  v['completed_endpoints']=[str(x.relative_to(q)) for x in sorted(q.glob('**/COMPLETE.json'))]
  v['active_handles']=[]
  for path in sorted(q.glob('handles/*.json')):
   h=json.loads(path.read_text());child=h.get('child')
   if not child:continue
   observed=identity(child['pid'],child['start_ticks'])
   if observed['present']:v['active_handles'].append(dict(identity=observed,spec=h.get('spec'),output=h.get('output')))
  for name in ['QUEUE_COMPLETE.json','QUEUE_CLOSED.json','CLOSURE.json','COMPLETE.json']:
   v[name+'_present']=(q/name).exists()
 d['qk'].append(v)
if d['processes'][2]['present']:
 d['relation_owner_command']=Path('/proc/3713404/cmdline').read_bytes().replace(b'\0',b' ').decode()
print(json.dumps(d))
