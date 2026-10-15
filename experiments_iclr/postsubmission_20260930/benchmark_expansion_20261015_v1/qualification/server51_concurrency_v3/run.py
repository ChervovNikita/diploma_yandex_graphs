"""One detached root-admitted command; exact prior slot held for three phases."""
import argparse,fcntl,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
from qualifier_contracts import HERE,PHASES,preflight,capacity,pin,read,write,identity,rss,require
from summarize import summary
from terminal_cost import complete_terminal_cost

def main():
 p=argparse.ArgumentParser();p.add_argument('--admission',required=True);p.add_argument('--admission-sha256',required=True);p.add_argument('--owner',action='store_true');args=p.parse_args()
 a,host,output,slot,ap=preflight(args.admission,args.admission_sha256)
 if not args.owner:
  output.parent.mkdir(parents=True,exist_ok=True);output.mkdir(exist_ok=False)
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=a['physical_gpu_uuid'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
  command=[a['python_executable'],'-B','-u',str(HERE/'run.py'),'--admission',str(ap),'--admission-sha256',args.admission_sha256,'--owner']
  with (output/'owner.log').open('xb') as log:child=subprocess.Popen(command,cwd=host['project_root'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True,env=env)
  record={'schema':'server51-throughput-qualifier-detached-launch-v1','owner':identity(child.pid),'command':command,'admission':pin(ap),'source_manifest':pin(HERE/'MANIFEST.json'),'scientific_fits':0,'TEST_accessed':False}
  write(output/'LAUNCH.json',record);print(json.dumps(record,sort_keys=True));return 0
 require(os.environ.get('CUDA_VISIBLE_DEVICES')==a['physical_gpu_uuid'],'Exact UUID mask for owner')
 require(output.is_dir() and not (output/'OWNER_CLAIM.json').exists(),'Detached starter must create fresh output')
 started=time.monotonic();active={};lock=None;report={'schema':'server51-native-structured-concurrency-qualification-v1','status':'preflight',
  'worker':a['worker'],'hostname':'olig','physical_GPU_UUID':a['physical_gpu_uuid'],'admission':pin(ap),'source_manifest':pin(HERE/'MANIFEST.json'),
  'prior_terminal':a['prior_terminal'],'prior_owner_claim':a['prior_owner_claim'],'slot_lock':str(slot),'phases':{},'capacity_observations':[],
  'children_started':0,'retained_child_terminals':[],'automatic_retries':0,'scientific_fits':0,'accuracy_claim':False,'TEST_accessed':False,'owner':identity(os.getpid())}
 def save():report['owner_wall_seconds']=time.monotonic()-started;write(output/'PROGRESS.json',report)
 def interrupted(*_):raise InterruptedError('Owned qualifier interrupted; retain costs and no retry')
 for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,interrupted)
 def retain_terminal(arm,entry,reason):
  child=entry['child'];path=entry['folder']/'RESULT.json'
  terminal={'phase':entry['folder'].parent.name,'arm':arm,'identity':entry.get('identity'),'reason':reason,'exit_code':child.poll(),
   'forced_KILL':entry.get('forced_KILL',False),'partial_costs_may_be_unrecorded':True}
  if path.is_file():
   terminal['RESULT']=pin(path)
   try:
    value=read(path);terminal['reported_authorities']={k:value.get(k) for k in ('child','owner','phase','arm','admission','source_manifest')}
    terminal['reported_actual_costs']={k:value.get(k) for k in ('status','TRAIN_epochs_completed','VALID_calls_completed','optimizer_calls','startup_seconds','warmup_wall_seconds','invocation_wall_seconds','own_peak_RSS_bytes','GPU_peak_allocated_bytes','GPU_peak_reserved_bytes','failure')}
    terminal['terminalization_confirmed']=complete_terminal_cost(value,child.poll(),
     expected={'child':entry.get('identity'),'owner':report['owner'],'phase':entry['folder'].parent.name,'arm':arm,'admission':pin(ap),'source_manifest':pin(HERE/'MANIFEST.json')},
     forced_KILL=entry.get('forced_KILL',False),cleanup_error=entry.get('cleanup_error'))
    terminal['partial_costs_may_be_unrecorded']=not terminal['terminalization_confirmed']
   except (OSError,ValueError) as error:terminal['result_read_error']=str(error)
  report['retained_child_terminals'].append(terminal)
 def cleanup():
  errors=[]
  for arm,entry in list(active.items()):
   child=entry['child']
   try:
    if child.poll() is None:
     observed=identity(child.pid)
     if observed is None:
      try:child.wait(timeout=1)
      except subprocess.TimeoutExpired:raise ValueError('Own child identity unavailable; no signal sent')
     else:
      require(observed==entry['identity'],'Child PID identity changed; never signal a reused/unrelated process')
      child.send_signal(signal.SIGTERM)
      try:child.wait(timeout=20)
      except subprocess.TimeoutExpired:
       require(identity(child.pid)==entry['identity'],'Child identity changed before owned KILL');child.kill();child.wait(timeout=10);entry['forced_KILL']=True
    else:child.wait()
   except BaseException as error:
    entry['cleanup_error']=str(error);errors.append({'arm':arm,'error':str(error),'identity':entry.get('identity')})
   finally:
    retain_terminal(arm,entry,'owner_cleanup');entry['log'].close()
  active.clear()
  if errors:report['cleanup_errors']=errors;report['unreaped_owned_work_possible']=True
 def wait_capacity(arms):
  while True:
   require(time.monotonic()-started<a['maximum_owner_wall_seconds'],'Finite owner wall bound while waiting for capacity')
   ok,observed=capacity(a,arms);report['admission_state']='admitted' if ok else 'waiting_resources';report['latest_capacity']=observed;save()
   if ok:report['capacity_observations'].append({'arms':arms,'observation':observed});return
   time.sleep(5)
 def phase(name,arms):
  wait_capacity(arms);folder=output/'phases'/name;folder.mkdir(parents=True,exist_ok=False)
  entry={'status':'starting','arms':{},'wall_started_monotonic_ns':time.monotonic_ns()};report['phases'][name]=entry;save()
  for arm in arms:
   childfolder=folder/arm;childfolder.mkdir()
   command=[a['python_executable'],'-B','-u',str(HERE/'measure_child.py'),'--admission',str(ap),'--admission-sha256',args.admission_sha256,'--phase',name,'--arm',arm]
   log=(childfolder/'child.log').open('xb');child=subprocess.Popen(command,cwd=host['project_root'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
   childid=identity(child.pid)
   active[arm]={'child':child,'identity':childid,'log':log,'folder':childfolder};report['children_started']+=1
   require(childid is not None,'Child identity unavailable immediately after Popen')
   entry['arms'][arm]={'identity':childid,'command':command,'status':'running'};save()
  released=False
  while active:
   require(time.monotonic()-started<a['maximum_owner_wall_seconds'],'Finite owner wall bound')
   require((time.monotonic_ns()-entry['wall_started_monotonic_ns'])/1e9<a['maximum_child_wall_seconds']+30,'Finite phase wall bound')
   ready={arm:row['folder']/'WARMUP_READY.json' for arm,row in active.items()}
   if not released and len(active)==len(arms) and all(path.exists() for path in ready.values()):
    for arm,path in ready.items():
     value=read(path);require(value['child']==active[arm]['identity'] and value['admission']==pin(ap) and value['phase']==name and value['arm']==arm,'Exact own warmup witnesses required')
    write(folder/'START_MEASUREMENT.json',{'schema':'root-owned-throughput-measurement-barrier-v1','phase':name,'owner':report['owner'],'admission':pin(ap),'released_monotonic_ns':time.monotonic_ns(),'ready':{arm:pin(path) for arm,path in ready.items()}})
    released=True;entry['barrier']=pin(folder/'START_MEASUREMENT.json');save()
   apps=subprocess.check_output(['nvidia-smi','--id='+a['physical_gpu_uuid'],'--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=10)
   memory={int(pid.strip()):int(mib.strip())*1024**2 for pid,mib in (line.split(',') for line in apps.splitlines() if line.strip())}
   for arm,row in list(active.items()):
    child=row['child'];code=child.poll()
    if code is None:
     require(identity(child.pid)==row['identity'],'Own child process identity changed')
     require(rss(child.pid)<=a['resources'][arm]['RAM_bytes'],'Own child RSS exceeded reviewed envelope')
     require(memory.get(child.pid,0)<=a['resources'][arm]['GPU_bytes'],'Own child VRAM exceeded reviewed envelope')
     continue
    child.wait();row['log'].close();del active[arm]
    retain_terminal(arm,row,'ordinary_child_terminal')
    require(code==0 and (row['folder']/'RESULT.json').exists(),'Owned timing child failed; no retry')
    result=read(row['folder']/'RESULT.json');require(result['status']=='PASS' and result['child']==row['identity'] and result['phase']==name and result['arm']==arm and result['admission']==pin(ap),'Exact child result authority required')
    entry['arms'][arm].update(status='PASS',exit_code=code,result=pin(row['folder']/'RESULT.json'));save()
   require(sum(path.stat().st_size for path in output.rglob('*') if path.is_file())<=a['resources']['disk_reserved_bytes'],'Own timing output exceeds reviewed disk budget')
   if active:time.sleep(.5)
  require(released,'Both complete workloads require the owner measurement barrier');entry.update(status='PASS',wall_end_monotonic_ns=time.monotonic_ns());save()
 save()
 try:
  lock=slot.open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)  # Never steal a live prior pilot slot.
  preflight(args.admission,args.admission_sha256,full=True)
  write(output/'OWNER_CLAIM.json',{'owner':report['owner'],'admission':pin(ap),'source_manifest':pin(HERE/'MANIFEST.json'),'source_data_verified':True,'slot_lock':str(slot)})
  report['status']='running';report['source_verification_and_lock_seconds']=time.monotonic()-started;save()
  for name,arms in PHASES.items():phase(name,arms)
  results={name:{arm:read(entry['result']['path']) for arm,entry in phaseentry['arms'].items()} for name,phaseentry in report['phases'].items()}
  report.update(status='PASS',timing=summary(results),all_actual_costs={name:{arm:{k:row.get(k) for k in ('invocation_wall_seconds','startup_seconds','warmup_wall_seconds','barrier_wait_seconds','TRAIN_epochs_completed','VALID_calls_completed','optimizer_calls','own_peak_RSS_bytes','GPU_peak_allocated_bytes','GPU_peak_reserved_bytes')} for arm,row in arms.items()} for name,arms in results.items()})
 except BaseException as error:report.update(status='failed',failure={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc(limit=12)})
 finally:
  for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,signal.SIG_IGN)
  cleanup()
  if report.get('cleanup_errors'):report['status']='failed'
  if lock is not None:lock.close()
  save();write(output/'END.json',report)
 print(json.dumps({k:report.get(k) for k in ('status','worker','children_started','timing','failure')},sort_keys=True))
 return 0 if report['status']=='PASS' else 1
if __name__=='__main__':raise SystemExit(main())
