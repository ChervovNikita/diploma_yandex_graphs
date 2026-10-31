
import hashlib,json,os,signal,socket,subprocess,sys,time
from pathlib import Path
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');phase=repo/'experiments_iclr/postsubmission_20260930';out=Path(__file__).resolve().parent
def save(n,v):
 p=out/n;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def ident(pid):
 f=Path('/proc')/str(pid)/'stat'
 if not f.exists():return None
 a=f.read_text().rsplit(')',1)[1].split();return {'pid':pid,'start_ticks':int(a[19]),'pgid':int(a[2]),'state':a[0]}
def alive(expected):
 now=ident(expected['pid'])
 return now is not None and now['state']!='Z' and all(now[k]==expected[k] for k in ['pid','start_ticks','pgid'])
def custody():
 p=phase/'private_GRU_acquisition_root_20261011_v1/owner_v1';start=json.loads((p/'OWNER_START.json').read_text());seen={}
 for k in ['owner','child']:
  expected=start[k];now=ident(expected['pid']);seen[k]=now
  if now is not None:assert all(now[z]==expected[z] for z in ['pid','start_ticks','pgid'])
  else:assert (p/'OWNER_END.json').exists()
 return seen
def pin(p):
 p=Path(p);raw=p.read_bytes();return {'path':str(p.relative_to(phase)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
runtime=json.loads((phase/'shared_replay_epoch1_root_20261012_v1/RUNTIME_BINDING.json').read_text())
template=phase/'shared_link_replay_epoch1_admission_20261012_v1/ROOT_RELEASE_TEMPLATE_WITH_SOURCE_PINS.json'
release=json.loads(template.read_text())
for row in release['additional_source_files'].values():assert pin(phase/row['path'])==row
program=phase/'shared_link_replay_epoch1_admission_20261012_v1/qualify_epoch1.py';assert {k:v for k,v in pin(program).items() if k!='path'}==release['program_pin']
source_data=phase/'training_pipeline_v2_20261012/qa/link_official_cache_v4/fit_data'
forbidden=[str(p.relative_to(phase)) for p in sorted(source_data.iterdir()) if p.is_file() and p.name not in ['GRAPH.pt','TRAIN.pt']]
assert any(p.endswith('/SELECT.pt') for p in forbidden)
before=custody()
free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
assert free>=34816
assert __import__('shutil').disk_usage(out).free>=8*1024**3
decision={'scientific_GO':False,'scientific_fits':0,'optimizer_updates':0,'model_construction':False,'source_epoch':1,'native_source_draws':2,'source_contexts':51,'finite_scope':'One complete GRAPH/TRAIN integer replay equivalence and two shared proof-only readers; no SELECT/CAL/EVALUATE, no training or scoring.','wall_limit_seconds':1800,'RSS_limit_bytes':64*1024**3,'CUDA_limit_mib':32768,'external_headroom_mib':2048,'output_limit_bytes':2*1024**3,'retry':False,'next_epochs_or_remaining_queue_authorized':False,'Physics_before':before,'root_reviewed_program':pin(program),'runtime_binding':pin(phase/'shared_replay_epoch1_root_20261012_v1/RUNTIME_BINDING.json')}
save('ROOT_ENGINEERING_DECISION.json',decision)
release.update(execution_enabled=True,runtime=runtime['runtime'],forbidden_target_paths=forbidden,output=out.name+'/proof',root_authorization_reference=str((out/'ROOT_ENGINEERING_DECISION.json').relative_to(phase)),supervisor_identity=ident(os.getpid()),resource_limits=decision)
save('RELEASE.json',release)
command=[str(repo/'.venv/bin/python'),'-B','-u',str(program),'--release',str(out/'RELEASE.json')]
started=time.monotonic()
with (out/'child.log').open('x') as log:
 child=subprocess.Popen(command,cwd=out,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 owned=ident(child.pid);assert owned is not None and owned['pgid']==child.pid
 save('OWNER_START.json',{'owner':ident(os.getpid()),'child':owned,'command':command,'scientific_GO':False,'Physics_before':before})
 peak={'RSS_bytes':0,'CUDA_mib':0,'written_bytes':0};reason=None;observations=0
 while child.poll() is None:
  assert alive(owned)
  # Traverse only the admitted process's children; no other jobs are controlled.
  pending=[child.pid];members=[]
  while pending:
   pid=pending.pop();now=ident(pid)
   if now is None:continue
   assert now['pgid']==owned['pgid'];members.append(pid)
   f=Path('/proc')/str(pid)/'task'/str(pid)/'children'
   if f.exists():pending.extend(map(int,f.read_text().split()))
  rss=0
  for pid in members:
   f=Path('/proc')/str(pid)/'status'
   if f.exists():rss+=sum(int(x.split()[1])*1024 for x in f.read_text().splitlines() if x.startswith('VmRSS:'))
  gpu=0
  query=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=10)
  for line in query.splitlines():
   parts=[x.strip() for x in line.split(',')]
   if len(parts)==2 and parts[0].isdigit() and int(parts[0]) in members and parts[1].isdigit():gpu+=int(parts[1])
  written=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
  peak={k:max(peak[k],v) for k,v in dict(RSS_bytes=rss,CUDA_mib=gpu,written_bytes=written).items()}
  seen=custody();observations+=1
  elapsed=time.monotonic()-started
  save('PROGRESS.json',{'elapsed_seconds':elapsed,'child':ident(child.pid),'peak':peak,'Physics':seen,'observations':observations,'scientific_fits':0})
  if elapsed>1800:reason='wall_limit'
  if rss>64*1024**3:reason='RSS_limit'
  if gpu>32768:reason='CUDA_limit'
  if written>2*1024**3:reason='output_limit'
  if reason:
   assert alive(owned);os.killpg(owned['pgid'],signal.SIGTERM)
   try:child.wait(timeout=10)
   except subprocess.TimeoutExpired:
    assert alive(owned);os.killpg(owned['pgid'],signal.SIGKILL);child.wait(timeout=10)
   break
  time.sleep(5)
 code=child.wait()
 end={'exit_code':code,'termination_reason':reason,'child_reaped':True,'child_absent':ident(child.pid) is None,'Physics_after':custody(),'elapsed_seconds':time.monotonic()-started,'peak':peak,'observations':observations,'scientific_fits':0,'GO':False,'failure_outputs_preserved':True}
 result=out/'proof/RESULT.json'
 if result.exists():
  value=json.loads(result.read_text());assert value['scientific_fits']==value['optimizer_updates']==value['SELECT_CAL_EVALUATE_reads']==0
  end['result']=pin(result);end['result_status']=value['status']
 else:end['failure_log_tail']=(out/'child.log').read_text()[-8000:]
 save('OWNER_END.json',end);print(json.dumps(end),flush=True)
