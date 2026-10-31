"""Root-reviewed finite13 HGT engineering owner; sequential children, no retry."""
import argparse,hashlib,json,os,signal,socket,subprocess,sys,time,threading
from pathlib import Path
HELPER='native_hgt_resource_helper_20261013_v1/runtime/hgt_resource.py'
HELPER_SHA='84e066543a095f49d0de9172ad95710ac8545f5a4f8630f546dd987daaafccd0'
PROPOSAL='native_hgt_resource_helper_20261013_v1/RESOURCE_PROPOSAL.json'
PROPOSAL_SHA='d33b1655b3e6e44b8f021e0f1ed0e501e758dd833732c14ed05477b8e633ffa4'
UUID='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
GIB=1024**3
def guard():
 if socket.gethostname()!='anogena-2-0' or subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()!=[UUID]:raise ValueError('Exact one-GPU allocation route required')
def identity(pid):
 fields=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split();return {'pid':pid,'start_ticks':int(fields[19]),'pgid':int(fields[2])}
def alive(owner):
 try:
  current=identity(owner['pid']);fields=(Path('/proc')/str(owner['pid'])/'stat').read_text().rsplit(')',1)[1].split();return current==owner and fields[0]!='Z'
 except FileNotFoundError:return False
def write(path,value):
 with path.open('x') as stream:stream.write(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
 path.chmod(0o444)
def receipt(root,path):
 raw=Path(path).read_bytes();return {'path':str(Path(path).relative_to(root)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def rss_bytes(pid):
 try:
  rows=(Path('/proc')/str(pid)/'status').read_text().splitlines();return next(int(line.split()[1])*1024 for line in rows if line.startswith('VmRSS:'))
 except (FileNotFoundError,StopIteration):return 0
def gpu_status(owner_pid=None):
 guard()
 raw=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True,timeout=10).strip().split(',')
 if len(raw)!=2 or raw[0].strip()!=UUID:raise ValueError('Allocation free-memory query identity differs')
 result={'gpu_uuid':UUID,'free_cuda_bytes':int(raw[1].strip())*1024**2}
 if owner_pid is not None:
  rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=10).splitlines();used=0
  for line in rows:
   fields=[v.strip() for v in line.split(',')]
   if len(fields)==3 and fields[0].isdigit() and int(fields[0])==owner_pid and fields[1]==UUID and fields[2].isdigit():used+=int(fields[2])*1024**2
  result['owned_driver_cuda_bytes']=used
 return result
def run_child(command,folder,label,env,limits,*,measure_gpu=True):
 with (folder/(label+'_STDOUT.log')).open('x') as stdout,(folder/(label+'_STDERR.log')).open('x') as stderr:
  child=subprocess.Popen(command,stdout=stdout,stderr=stderr,env=env,start_new_session=True);owner=identity(child.pid);start=time.monotonic()
  write(folder/(label+'_START.json'),{'command':command,'owner':owner,'limits':limits,'models_are_engineering_only':True,'scientific_fits':0,'GO':False})
  maximum_rss=maximum_cuda=0;violation=None;next_gpu=0.;samples=0;watchdog_state={'fired':False}
  def deadline():
   watchdog_state['fired']=True
   if alive(owner):os.killpg(owner['pgid'],signal.SIGTERM)
  watchdog=threading.Timer(limits['wall_seconds'],deadline);watchdog.daemon=True;watchdog.start()
  try:
   while child.poll() is None:
    now=time.monotonic();memory=rss_bytes(child.pid);maximum_rss=max(maximum_rss,memory);samples+=1
    if now-start>limits['wall_seconds']:violation='child_wall_limit'
    elif memory>limits['rss_bytes']:violation='child_RSS_limit'
    if measure_gpu and now>=next_gpu:
     value=gpu_status(child.pid);maximum_cuda=max(maximum_cuda,value['owned_driver_cuda_bytes']);next_gpu=now+1.
     if value['owned_driver_cuda_bytes']>limits['cuda_bytes']:violation='child_CUDA_limit'
    if violation:
     if alive(owner):os.killpg(owner['pgid'],signal.SIGTERM)
     try:child.wait(timeout=5)
     except subprocess.TimeoutExpired:
      if alive(owner):os.killpg(owner['pgid'],signal.SIGKILL)
      child.wait(timeout=5)
     break
    time.sleep(.25)
  except BaseException as error:
   violation='monitor_error:'+type(error).__name__+':'+str(error)
   if alive(owner):os.killpg(owner['pgid'],signal.SIGTERM)
   try:child.wait(timeout=5)
   except subprocess.TimeoutExpired:
    if alive(owner):os.killpg(owner['pgid'],signal.SIGKILL)
    child.wait(timeout=5)
  finally:
   watchdog.cancel()
   if watchdog_state['fired'] and violation is None:violation='child_wall_watchdog'
  result={'label':label,'owner':owner,'exit_code':child.returncode,'wall_seconds':time.monotonic()-start,'limit_violation':violation,
   'maximum_observed_RSS_bytes':maximum_rss,'maximum_observed_owned_driver_CUDA_bytes':maximum_cuda,'RSS_samples':samples,'limits':limits,
   'stdout_path':str(folder/(label+'_STDOUT.log')),'stderr_path':str(folder/(label+'_STDERR.log')),'partials_preserved':True,'automatic_retry':False,'scientific_fits':0,'GO':False}
  write(folder/(label+'_TERMINAL.json'),result);return result
def main():
 p=argparse.ArgumentParser();p.add_argument('--project-root',required=True);p.add_argument('--engineering-admission',required=True);p.add_argument('--output',required=True);a=p.parse_args();guard();root=Path(a.project_root).resolve()
 def safe(relative):
  rel=Path(relative)
  if rel.is_absolute() or '..' in rel.parts:raise ValueError('Exact phase-relative owner paths required')
  path=root/rel
  if not path.resolve().is_relative_to(root):raise ValueError('Owner path escapes phase')
  return path
 helper=safe(HELPER);proposal_path=safe(PROPOSAL)
 if hashlib.sha256(helper.read_bytes()).hexdigest()!=HELPER_SHA or hashlib.sha256(proposal_path.read_bytes()).hexdigest()!=PROPOSAL_SHA:raise ValueError('Root-adopted exact source helper/proposal changed')
 proposal=json.loads(proposal_path.read_text());cases=proposal['cases'];admission_path=safe(a.engineering_admission);admission=json.loads(admission_path.read_text());limits={'wall_seconds':600,'rss_bytes':64*GIB,'cuda_bytes':32*GIB,'minimum_free_cuda_bytes':34*GIB}
 case_ids=[c['case_id'] for c in cases]
 if len(cases)!=13 or len(set(case_ids))!=13 or any(c['repetitions']!=2 for c in cases):raise ValueError('Exact13 case/26 callback roster required')
 if admission.get('root_review_complete') is not True or admission.get('helper_sha256')!=HELPER_SHA or admission.get('proposal_sha256')!=PROPOSAL_SHA or sorted(admission.get('admitted_case_ids',[]))!=sorted(case_ids) or admission.get('scientific_fits_allowed') is not False or admission.get('owner_limits')!=limits:raise ValueError('Root-reviewed exact13 case engineering admission and declared limits required')
 folder=safe(a.output)
 if folder.exists():raise FileExistsError('Prior owner/results must be preserved; no automatic retry')
 folder.mkdir(parents=True);write(folder/'OWNER_START.json',{'owner':identity(os.getpid()),'hostname':socket.gethostname(),'helper':receipt(root,helper),'proposal':receipt(root,proposal_path),
  'admission':receipt(root,admission_path),'limits':limits,'case_ids':case_ids,'root_authorization':'engineering only; no scientific GO','Physics_identity_or_state_mutated':False,'scientific_fits':0,'GO':False})
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2');records=[];status='failed';failure=None
 try:
  for index,case in enumerate(cases):
   guard();hardware=gpu_status()
   if hardware['free_cuda_bytes']<limits['minimum_free_cuda_bytes']:raise MemoryError('Pre-case free GPU below34 GiB; owner stops without launching or changing other owners')
   case_folder=folder/('case_'+str(index).zfill(2));case_folder.mkdir();case_relative=str((case_folder/'resource').relative_to(root));command=[sys.executable,'-B',str(helper),'--project-root',str(root),'--proposal',PROPOSAL,'--proposal-sha256',PROPOSAL_SHA,
    '--engineering-admission',a.engineering_admission,'--case-id',case['case_id'],'--output',case_relative,'--device','cuda:0']
   write(case_folder/'PRESTART_HARDWARE.json',hardware);terminal=run_child(command,case_folder,'SOURCE_CASE',env,limits);terminal['case_id']=case['case_id'];records.append(terminal)
   if terminal['exit_code'] or terminal['limit_violation']:raise RuntimeError('Source case failed; terminal/log/partial receipts retained; owner stops without retry')
   result_path=root/case_relative/'RESULT.json';result=json.loads(result_path.read_text())
   if result['case_id']!=case['case_id'] or result['optimizer_denials']!=2 or result['scientific_optimizer_updates']!=0 or result['source_scheduler_denials']!=0 or result['predictive_scores']!=0 or result['real_SELECT_CAL_evaluation_targets_opened'] is not False or not result['full_source_model_optimizer_OneCycle_private_global_RNG_modes_flags_and_grad_state_restored']:raise ValueError('Exact zero-update/no-heldout source callback contract failed')
   if len(result['callbacks'])!=2 or any(c['scientific_optimizer_updates'] or c['peak_cuda_allocated_bytes']>limits['cuda_bytes'] for c in result['callbacks']):raise ValueError('Callback count or exact measured CUDA bound failed')
   terminal['source_result']=receipt(root,result_path);terminal['callbacks_complete']=2;write(case_folder/'VERIFIED_ZERO_UPDATE_CASE.json',{'case_id':case['case_id'],'result':terminal['source_result'],'callbacks':2,'optimizer_updates':0,'scheduler_updates':0,'scores':0,'resource_profile_qualified':False})
  status='complete'
 except BaseException as error:failure={'type':type(error).__name__,'message':str(error)}
 result={'schema':'root-reviewed-finite-HGT13-engineering-owner-v1','status':status,'limits':limits,'helper_sha256':HELPER_SHA,'proposal_sha256':PROPOSAL_SHA,'case_records':records,
  'source_cases_completed':sum(r.get('callbacks_complete',0)==2 for r in records),'source_callbacks_completed':sum(r.get('callbacks_complete',0) for r in records),'failure':failure,
  'stop_on_first_failure':True,'automatic_retry':False,'one_child_at_a_time':True,'successful_cases_retained_for_root_resource_merge':True,'resource_profile_qualified':False,
  'Physics_identity_or_state_mutated':False,'scientific_fits':0,'optimizer_updates':0,'scheduler_updates':0,'heldout_targets_read':False,'predictive_scores':0,'server77_access':False,'GO':False}
 write(folder/'OWNER_RESULT.json',result);print(json.dumps({k:v for k,v in result.items() if k!='case_records'},sort_keys=True),flush=True);return 0 if status=='complete' else 1
if __name__=='__main__':sys.exit(main())
