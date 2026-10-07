import os,socket,signal,time,json,hashlib,subprocess,datetime
from pathlib import Path
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';os.chdir(R)
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['/usr/bin/nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
D=P/'learnable_internal_be_WikiCS_scientific_family_activation_20261007_v1';D.mkdir(exist_ok=True)
output=D/'RESOURCE_QUEUE_SUPERSESSION.json';assert not output.exists()
OLD=P/'learnable_internal_be_resource_activation_root_20261007_v2';E=P/'learnable_internal_be_WikiCS_resource_execution_root_20261007_v2'
sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
assert sha(OLD/'execute.py')=='7f07490358501979ffd83756e1f46157c0b0565c4c225fc6a93d538b6778e153'
assert sha(OLD/'CONFIG_ADOPTED.json')=='c9da38f9d5ea5219928247406f561e18b552c7e0d07038b05f39ebc0d8807913'
PID=506790;TICKS=6015082416
def ident(pid):
 q=Path('/proc')/str(pid)
 try:
  stat=(q/'stat').read_text();f=stat[stat.rfind(')')+2:].split();args=(q/'cmdline').read_bytes().split(b'\0')
 except (FileNotFoundError,ProcessLookupError):return None
 return {'PID':pid,'start_ticks':int(f[19]),'state':f[0],'ppid':int(f[1]),'args':[x.decode(errors='replace') for x in args if x]}
before=ident(PID);assert before and before['start_ticks']==TICKS and str(OLD/'execute.py') in before['args']
assert ident(PID)['start_ticks']==TICKS
os.kill(PID,signal.SIGSTOP)
stopped=False
for _ in range(30):
 a=ident(PID)
 if a and a['start_ticks']==TICKS and a['state'] in ('T','t'):stopped=True;break
 time.sleep(.05)
if not stopped:
 os.kill(PID,signal.SIGCONT);raise RuntimeError('Owner did not quiesce')
active=[];allprocs=[]
for q in Path('/proc').iterdir():
 if not q.name.isdecimal():continue
 a=ident(int(q.name))
 if not a:continue
 allprocs.append(a)
 if any(x.startswith(str(P)+'/') and 'learnable_internal_be_resource_qualifier_source_' in x and Path(x).name in ('supervise.py','worker.py') for x in a['args']):active.append(a)
children=[a for a in allprocs if a['ppid']==PID]
if active or children:
 os.kill(PID,signal.SIGCONT)
 print(json.dumps({'superseded':False,'resumed_owner':True,'active_resource_children':active,'direct_children':children}))
 raise SystemExit(2)
uptime=float(Path('/proc/uptime').read_text().split()[0]);age=max(0.,uptime-TICKS/os.sysconf('SC_CLK_TCK'))
rows=[]
for f in sorted((E/'receipts').glob('*_DRIVER.json')):
 a=json.loads(f.read_text());rows.append(a)
assert ident(PID)['start_ticks']==TICKS
os.kill(PID,signal.SIGTERM);os.kill(PID,signal.SIGCONT)
for _ in range(60):
 if ident(PID) is None:break
 time.sleep(.1)
assert ident(PID) is None,'Owner still present; do not launch'
receipt={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'PID':PID,'start_ticks':TICKS,'config_sha256':sha(OLD/'CONFIG_ADOPTED.json'),'source_driver_sha256':sha(OLD/'execute.py'),'superseded':True,'idle_boundary_verified':True,'no_active_resource_child':True,'scientific_jobs_signalled':False,'owner_before':before,'resource_children_at_boundary':active,'direct_children_at_boundary':children,'prior_completed_measurements':rows,'cost_bound':{'kind':'measured process-lifetime upper bound on cumulative queue wait; exact wait onset not recorded','cumulative_wait_lower_seconds':0,'cumulative_wait_upper_seconds':age,'resource_compute_driver_seconds':sum(r['inclusive_driver_seconds'] for r in rows)},'prior_files_preserved':True,'signals_sent_to_exact_engineering_owner_only':['SIGSTOP','SIGTERM','SIGCONT']}
with output.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps(receipt))
