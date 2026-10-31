
import base64,hashlib,json,os,shutil,socket,subprocess,sys,time,zlib
from pathlib import Path
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
phase=repo/'experiments_iclr/postsubmission_20260930'

import signal
out=Path(__file__).resolve().parent;source=phase/'bsnn_bayesian_resource_admission_20261012_v4'
program=source/'qualify_bsnn.py';assert hashlib.sha256(program.read_bytes()).hexdigest()=='0e80cd69acc61e79d09701a2382be7afca048834c964b7401a623bbb61a68519'
assert hashlib.sha256((source/'MANIFEST.json').read_bytes()).hexdigest()=='695952865cbb09ff1ce5401876887676da5c69c3d956ec7f1dd718b6f13bfdf4'
template=json.loads((source/'ROOT_RELEASE_TEMPLATE_WITH_SOURCE_PINS.json').read_text());assert template['execution_enabled'] is False
binding=phase/'shared_replay_epoch1_root_20261012_v1/RUNTIME_BINDING.json';runtime=json.loads(binding.read_text())['runtime'];assert template['runtime']==runtime
casefile=source/'CASES.json';all_cases=json.loads(casefile.read_text())['cases'];assert len(all_cases)==4 and len(template['forbidden_target_paths'])==30
cases={k:v for k,v in all_cases.items() if k.endswith('__native_M1')};assert len(cases)==2
prior=json.loads((phase/'bayesian_bsnn_resource_results_root_20261012_v1/CLOSURE.json').read_text());assert prior['all_children_closed']
assert {r['case_id'] for r in prior['attempts'] if r['exit_code']!=0}==set(cases)
def pin(p):
 p=Path(p);raw=p.read_bytes();return {'path':str(p.relative_to(phase)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def save(n,v):
 p=out/n;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def ident(pid):
 f=Path('/proc')/str(pid)/'stat'
 if not f.exists():return None
 a=f.read_text().rsplit(')',1)[1].split();return {'pid':pid,'start_ticks':int(a[19]),'pgid':int(a[2])}
def custody():
 p=phase/'private_GRU_acquisition_root_20261011_v1/owner_v1';start=json.loads((p/'OWNER_START.json').read_text());seen={}
 for k in ('owner','child'):
  expected=start[k];now=ident(expected['pid']);seen[k]=now
  if now is not None:assert all(now[x]==expected[x] for x in now)
  else:assert (p/'OWNER_END.json').exists()
 return seen
decision={'exact_case_ids':sorted(cases),'complete_TRAIN_callbacks':4,'scientific_fits':0,'optimizer_updates':0,'scientific_GO':False,'no_automatic_retry':True,'concurrent_children':1,'explicit_prior_failure_resolution':'Prior repeated sparse CUDA logits equality test was stronger than the source law; prove first4 complete sampling streams and fixed source inputs instead. Successful GNNM cases are not repeated.','maximum_wall_seconds_per_child':600,'maximum_RSS_bytes':64*1024**3,'maximum_owned_CUDA_mib':32768,'minimum_external_headroom_mib':2048,'source_program':pin(program),'source_cases':pin(casefile),'runtime_binding':pin(binding),'scope':'Four native fixed CayleyBSNN source geometries: denied-Adam NLL+KL callbacks, finite gradients, source streams/plain aliases/modes/flags/Adam serialization, label-free4MC andM1 MC16 control; no native optimizer kernels, selectors, heldout targets or scores.'}
save('ROOT_ENGINEERING_DECISION.json',decision);records=[]
for case_id in sorted(cases):
 before=custody();free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>=34816
 release=dict(template);release.update(execution_enabled=True,case_id=case_id,output=out.name+'/case_'+case_id,root_authorization_reference=str((out/'ROOT_ENGINEERING_DECISION.json').relative_to(phase)),runtime=runtime,root_supervisor_identity=ident(os.getpid()),resource_limits=decision)
 save(case_id+'_RELEASE.json',release)
 command=[str(repo/'.venv/bin/python'),'-B','-u',str(program),'--release',str(out/(case_id+'_RELEASE.json')),'--case',case_id]
 started=time.monotonic();RSS_peak=0;CUDA_peak=0;termination=None
 with (out/(case_id+'.log')).open('x') as log:
  child=subprocess.Popen(command,cwd=out,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1'))
  owner=ident(child.pid);assert owner is not None and owner['pgid']==child.pid;save(case_id+'_OWNER.json',{'owner':owner,'Physics_before':before,'command':command})
  while child.poll() is None:
   now=ident(child.pid);assert now==owner;elapsed=time.monotonic()-started
   f=Path('/proc')/str(child.pid)/'status';rss=sum(int(x.split()[1])*1024 for x in f.read_text().splitlines() if x.startswith('VmRSS:')) if f.exists() else 0
   gpu=0
   for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=10).splitlines():
    parts=[x.strip() for x in line.split(',')]
    if len(parts)==2 and parts[0]==str(child.pid) and parts[1].isdigit():gpu=int(parts[1])
   RSS_peak=max(RSS_peak,rss);CUDA_peak=max(CUDA_peak,gpu);current=custody()
   save('ACTIVE.json',{'case_id':case_id,'elapsed_seconds':elapsed,'RSS_peak_bytes':RSS_peak,'CUDA_peak_mib':CUDA_peak,'Physics':current,'child':now,'scientific_fits':0})
   if elapsed>600:termination='wall_limit'
   if rss>64*1024**3:termination='RSS_limit'
   if gpu>32768:termination='CUDA_limit'
   if termination:
    assert ident(child.pid)==owner;os.killpg(owner['pgid'],signal.SIGTERM)
    try:child.wait(timeout=10)
    except subprocess.TimeoutExpired:
     assert ident(child.pid)==owner;os.killpg(owner['pgid'],signal.SIGKILL);child.wait(timeout=10)
    break
   time.sleep(3)
  code=child.wait()
 row={'case_id':case_id,'exit_code':code,'child_absent':ident(child.pid) is None,'Physics_before':before,'Physics_after':custody(),'elapsed_seconds':time.monotonic()-started,'RSS_peak_bytes':RSS_peak,'CUDA_peak_mib':CUDA_peak,'termination_reason':termination,'no_automatic_retry':True}
 result=phase/release['output']/'RESULT.json'
 if result.exists():
  v=json.loads(result.read_text());assert v['scientific_fits']==v['optimizer_updates']==0 and v['real_SELECT_CAL_EVALUATE_read'] is False
  row['result']=pin(result);row['resource']={k:v[k] for k in ['status','cuda_peak_allocated_bytes','cuda_peak_reserved_bytes','CPU_peak_RSS_bytes','complete_checkpoint_bytes','callback_seconds','source_used_leaves_with_synthetic_mature_Adam_moments','additional_unstepped_Adam_workspace_bytes','posterior_source_serving_draws','native_M1_MC16_extra_count_control_draws']}
 else:row['failure_log_tail']=(out/(case_id+'.log')).read_text()[-7000:]
 records.append(row);save('PROGRESS.json',{'attempts':records,'scientific_fits':0,'GO':False})
closure={'attempts':records,'all_children_closed':all(r['child_absent'] for r in records),'scientific_fits':0,'optimizer_updates':0,'GO':False,'automatic_remaining_queue':False};save('CLOSURE.json',closure);print(json.dumps(closure),flush=True)
