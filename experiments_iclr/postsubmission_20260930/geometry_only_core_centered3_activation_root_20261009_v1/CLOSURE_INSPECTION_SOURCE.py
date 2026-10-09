from pathlib import Path
import json,socket,subprocess,hashlib,datetime
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930'
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()=='24c315a7-3c08-471f-b550-b9a3e1faf75d'
for pid in (563273,563276):assert not Path('/proc',str(pid)).exists()
for d in Path('/proc').iterdir():
 if not d.name.isdigit():continue
 try:s=(d/'stat').read_text()
 except (FileNotFoundError,PermissionError,ProcessLookupError):continue
 fields=s[s.rfind(')')+2:].split();assert int(fields[2]) not in (563273,563276)
for row in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).splitlines():assert row.strip() not in ('563273','563276')
a=P/'geometry_only_core_scientific18_activation_root_20261009_v1'
def bound(f):return dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),bytes=f.stat().st_size)
t=json.loads((a/'TERMINAL.json').read_text());launch=json.loads((a/'LAUNCH.json').read_text())
assert t['complete'] and t['exit_code']==0 and t['reaped'] and t['actual_worker_absent'] and t['actual_worker_CUDA_absent']
assert t['child']['pid']==563276 and t['child']['start_ticks']==6035067565
assert launch['parent']['pid']==563273 and launch['parent']['start_ticks']==6035067559
complete=P/'geometry_only_core_scientific18_execution_root_20261009_v1/COMPLETE.json';c=json.loads(complete.read_text())
expected={('shared_fit',s,-1) for s in (7409,8501,9607)}|{('independent_pool',s,-1) for s in (7409,8501,9607)}|{('independent_member',s,m) for s in (7409,8501,9607) for m in range(4)}
assert c['complete'] and len(c['records'])==18 and {(r['kind'],r['base_seed'],r.get('member',-1)) for r in c['records']}==expected and all(r['status']=='complete' for r in c['records'])
custody=dict(release_owner='root',all18_complete_verified=True,comparative_outcomes_opened=False,parent_pid=563273,parent_birth=6035067559,parent_group=563273,worker_pid=563276,worker_birth=6035067565,worker_group=563276,boot_id='24c315a7-3c08-471f-b550-b9a3e1faf75d',actual_parent_absent=True,actual_parent_group_absent=True,actual_worker_absent=True,actual_worker_group_absent=True,actual_worker_CUDA_absent=True,reaped=True,terminal=bound(a/'TERMINAL.json'),launch=bound(a/'LAUNCH.json'))
print(json.dumps(dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),custody=custody,complete_binding=bound(complete),terminal=t,launch=launch,head=subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip(),available_GPU_MiB=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip(),scores_opened=False)))
