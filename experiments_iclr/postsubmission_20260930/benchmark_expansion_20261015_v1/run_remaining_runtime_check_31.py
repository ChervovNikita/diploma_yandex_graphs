"""Root-reviewed one-shot NCNC/BUDDY launcher; no waiting, retries or Git."""
import argparse
import hashlib
import json
from pathlib import Path
from remote_control import call, REMOTE_EXPANSION, REMOTE_ROOT

CODE = r'''import fcntl,hashlib,json,os,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
R=Path(@ROOT@);E=Path(@EXP@);case=@CASE@;GPU='GPU-44039938-fd82-41d2-fefd-de71514e2fac';GiB=1024**3
nc=case=='NCNC_C4';rss_cap,gpu_cap,disk_cap,wall=(8*GiB,32*GiB,GiB,610) if nc else (16*GiB,40*GiB,8*GiB,3610)
output=E/('qualification/remaining_'+case+'_runtime_31');output.mkdir(exist_ok=False)
report={'case':case,'status':'preflight','native_child_started':False,'automatic_retries':0,'TEST_accessed':False,'full_campaign_GO':False,'launcher_sha256':@LAUNCHER@}
child=None;lock=None;started=time.monotonic();launched=None;peak_rss=peak_gpu=0;checked=[]
def require(ok,message):
 if not ok:raise ValueError(message)
def pin(p):
 p=Path(p).resolve();require(p.is_relative_to(R),'Path escapes repository');b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def verify(row,base=R):
 p=base/row['path'];actual=pin(p);require(all(actual[k]==row[k] for k in ('bytes','sha256')),'Pinned bytes changed: '+str(p));checked.append(actual);return p
def fixed(name,sha):
 p=E/name;actual=pin(p);require(actual['sha256']==sha,'Reviewed source changed: '+name);checked.append(actual);return p
def read(p):return json.loads(Path(p).read_text())
def write(p,value):
 tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)
def manifest(name,sha):
 p=fixed(name,sha)
 for row in read(p)['files']:verify(row,p.parent)
 bindings=p.parent/'SOURCE_BINDINGS.json'
 if bindings.exists():
  require(any(r['path']=='SOURCE_BINDINGS.json' for r in read(p)['files']),'Unpinned source bindings')
  for row in read(bindings)['files']:verify({**row,'path':row['path_relative_to_phase']},E.parent)
 return p
def stop(*_):raise InterruptedError('Supervisor interrupted; preserve attempt without retry')
def deadline_signal(*_):report['timed_out']=True;raise TimeoutError('Outer wall bound exceeded')
for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,stop)
try:
 lock=(E/'analysis_outputs/selected_VALID_collection_gpu_slot.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 qmanifest=manifest('qualification/structural_complete_collection_queue_v1/MANIFEST.json','071ee8c0d137c241536b239346d3786bb4d24c80ded3bd4405e9aa68c1a47f54')
 launch=read(fixed('STRUCTURAL_COLLECTION_QUEUE_LAUNCH_28.json','ada674b2576c11a97c55b6006aeeaba140eb0568bc9b46be44a056dd2a987449'))
 qpath=E/'qualification/structural_complete_collection_queue_run_v1/QUEUE.json';q=read(qpath)
 require(q.get('status')=='complete' and q.get('source')==pin(qmanifest) and q.get('admission')==launch['admission'],'Successful exact collection queue required')
 require(all(q['owner'][k]==launch['owner'][k] for k in ('pid','start_ticks')),'Collection owner differs');verify(q['admission'])
 require(q.get('TEST_accessed') is False and q.get('automatic_retries')==0 and (q.get('new_collections'),q.get('reused_banks'),q.get('complete_banks'),q.get('paid_fits'))==(6,33,39,45),'Collection counts/scope differ')
 rows=q['invocations'];require([r['name'] for r in rows]==['commands',*[f'collect_{i}' for i in range(6)],'analyze'] and all(r.get('status')=='passed' and r.get('exit_code')==0 and not r.get('timed_out') and not r.get('failure') for r in rows),'Collection invocation failed/incomplete')
 end=read(verify(q['collection_END']));analysis=read(verify(q['complete_analysis']))
 require(end['status']=='complete' and end.get('failure') is None and len(end['children'])==6 and end['TEST_accessed'] is False,'Collection END failed')
 require(analysis['schema']=='structural-private-risk-selected-VALID-39-bank-addition-v1' and analysis['status'] in ('complete','complete_with_warnings') and analysis['TEST_accessed'] is False and len(analysis['rows'])==39 and len(analysis['training_fit_outcomes'])==45 and analysis['new_models']==analysis['new_native_forwards']==analysis['new_optimizer_steps']==0,'Complete analysis differs')
 for key,value in q['prerequisites'].items():
  for row in value if isinstance(value,list) else [value]:verify(row)
 report['collection_terminal']=pin(qpath)
 helper=E/'qualification/portablev4_link_queued_parent_authority_repair_v1'
 fixed(str((helper/'supervise.py').relative_to(E)),'a8056e0e29045031ae2302d94f67328273274c9084f98f6c163e6eeb3677a5d6')
 fixed(str((helper/'support.py').relative_to(E)),'19ca77d9b69f0d49e932b5d0d14e5774328e9e4e0533d31ecc648d1e89bf6254')
 sys.path.insert(0,str(helper));from supervise import process,preflight,gpu_state
 review=fixed('ROOT_CPU_PYTHON_REVIEW_15.json','e26f362304281c884d902ede1d8ad54e9515e7e09a4cc4d1ec35a0d273edf79a');python=str(R/'.venv/bin/python')
 probe='import sys,json;print(json.dumps(dict(executable=sys.executable,prefix=sys.prefix,base_prefix=sys.base_prefix,version=sys.version)))'
 interpreter=json.loads(subprocess.check_output([python,'-B','-c',probe],text=True,timeout=15));require(interpreter==read(review)['runtime'],'Original qualified interpreter differs');report['interpreter']=interpreter
 if nc:
  package=E/'qualification/final_selected_node_NCNC_VALID_v3';source=manifest(str((package/'MANIFEST.json').relative_to(E)),'8fc619d48cc82b1106c1ca6e3534376e661a1dd1c9987f6b335d477fc94f0f97')
  inputs=read(package/'INPUTS.json')
  for row in [inputs['mapping'],inputs['scorer'],inputs['node_provider_authority'],*inputs['reused_harness']]:verify({**row,'path':row['path_relative_to_expansion']},E)
  manifest('final_selected_scoring_v5/MANIFEST.json','e0b86279593c9f20e49b615c57fce41360babc40201b3172ae76db6eab6f9659')
  request=read(package/'ROOT_ADMISSION_TEMPLATE_NCNC_C4.json')
 else:
  package=E/'composable_normalization_runtime_v1';source=manifest(str((package/'MANIFEST.json').relative_to(E)),'518784a9e83f4ba92c4f7bff55705999a787c7a67ff69c0cc9c09b5530550d38')
  request=read(fixed('analysis/composed_buddy_qualification_handoff_v1/ROOT_ADMISSION_TEMPLATE.json','2c0dd9123e040ad6ba63b35f20fca84c1b2d4b2087856031f38c743dc87811f9'))
  parent=read(verify(request['parent_descriptor']));verify(request['VALID_role_contract'])
  require(parent['recipe_id']=='collab-buddy' and parent['family']=='link' and parent['scientific_method_id']==parent['procedure']=='interaction_input_conditionalJ' and parent['seed']==1 and parent['TEST_access'] is False and parent.get('torch_threads',2)==2,'BUDDY parent scope differs')
 peers=[];old=process(640428)
 if old:
  require(old['start_ticks']=='6055039165','GRU PID reused');peers=[{'pid':640428,'start_ticks':'6055039165','peak_host_RSS_bytes':16*GiB,'peak_GPU_bytes':32*GiB}]
 resources=preflight({'device':'cpu','peers':peers},{'minimum_host_headroom_after_peer_growth_bytes':rss_cap,'host_RSS_reservation_bytes':rss_cap,'disk_reservation_bytes':disk_cap})
 state=gpu_state();require(state['uuid']==GPU and len(state['processes'])<=1,'Sole allocation/at most GRU peer required');growth=0
 for row in state['processes']:
  require(peers and row['pid']==640428 and row['start_ticks']=='6055039165' and row['gpu_memory_bytes']<=32*GiB,'Unadmitted GPU peer');growth+=32*GiB-row['gpu_memory_bytes']
 require(state['free_bytes']-growth>=gpu_cap,'Insufficient GPU headroom after GRU growth')
 resources.update(GPU=state,GPU_peer_growth_bytes=growth,GPU_reservation_bytes=gpu_cap,host_RSS_reservation_bytes=rss_cap,disk_reservation_bytes=disk_cap,wall_seconds=wall)
 write(output/'RESOURCES.json',resources);report.update(resources=resources,source_manifest=pin(source),source_receipts=checked.copy())
 request['output']=str(output/'native')
 if nc:request.update(status='ADMITTED',harness_manifest_sha256=pin(source)['sha256'],INPUTS_sha256=pin(package/'INPUTS.json')['sha256'],python_executable=python,qualified_python_review=pin(review),peer_resource_review=pin(output/'RESOURCES.json'))
 else:request.update(root_reviewed_source=True,root_verified_two_worker_ceiling=True,root_resource_evidence=pin(output/'RESOURCES.json'))
 admission=output/'ADMISSION.json';write(admission,request)
 command=[python,'-B','-u',str(package/('replay.py' if nc else 'qualify_native.py')),'--root' if nc else '--project-root',str(R),'--admission',str(admission)]+(['--case',case] if nc else [])
 require('torch' not in sys.modules,'Supervisor must remain metadata-only')
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU if nc else '0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
 with (output/'child.log').open('xb') as log:child=subprocess.Popen(command,cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
 launched=time.monotonic();signal.signal(signal.SIGALRM,deadline_signal);signal.alarm(wall)
 identity=process(child.pid);report.update(status='running',native_child_started=True,child=identity,command=command,admission=pin(admission))
 while child.poll() is None:
  if time.monotonic()-launched>=wall:report['timed_out']=True;raise TimeoutError('Outer wall bound exceeded')
  own=[]
  for p in Path('/proc').iterdir():
   if p.name.isdigit():
    try:
     if os.getpgid(int(p.name))==child.pid:
      current=process(int(p.name))
      if current:own.append(current)
    except ProcessLookupError:pass
  peak_rss=max(peak_rss,sum(p['rss_bytes'] for p in own));require(peak_rss<=rss_cap,'Own process-group RSS reservation exceeded')
  old=process(640428)
  if old:require(peers and old['start_ticks']=='6055039165' and old['rss_bytes']<=16*GiB,'GRU identity/host envelope changed')
  try:state=gpu_state()
  except ValueError as error:
   if str(error)=='GPU process could not be authenticated' and child.poll() is not None:break
   raise
  require(state['uuid']==GPU and len(state['processes'])<=2,'Unexpected GPU/third worker')
  own_ids={(p['pid'],p['start_ticks']) for p in own};own_gpu=0
  for row in state['processes']:
   if (row['pid'],row['start_ticks']) in own_ids:own_gpu+=row['gpu_memory_bytes']
   else:require(peers and row['pid']==640428 and row['start_ticks']=='6055039165' and row['gpu_memory_bytes']<=32*GiB,'Unadmitted GPU process')
  peak_gpu=max(peak_gpu,own_gpu);require(peak_gpu<=gpu_cap,'Own process-group GPU reservation exceeded')
  require(sum(p.stat().st_size for p in output.rglob('*') if p.is_file())<=disk_cap,'Own output disk reservation exceeded')
  report.update(peak_RSS_sampled_bytes=peak_rss,peak_GPU_sampled_bytes=peak_gpu,child_wall_seconds=time.monotonic()-launched);write(output/'SUPERVISOR.json',report)
  try:child.wait(timeout=.5)
  except subprocess.TimeoutExpired:pass
 result=read(output/'native/RESULT.json');report['result']=result
 require(child.returncode==0 and result.get('status')=='passed' and result.get('TEST_accessed') is False,'Child failed/incomplete; preserve without retry')
 for row in report['source_receipts']:verify(row)
 require(pin(qpath)==report['collection_terminal'],'Collection terminal changed');report['status']='passed'
except BaseException as error:report.update(status='failed',failure={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc(limit=8)})
finally:
 signal.alarm(0)
 if child is not None:
  try:
   os.killpg(child.pid,signal.SIGTERM);deadline=time.monotonic()+10
   while True:
    child.poll();os.killpg(child.pid,0)
    if time.monotonic()>=deadline:os.killpg(child.pid,signal.SIGKILL);report['forced_KILL_after_TERM']=True;break
    time.sleep(.1)
  except ProcessLookupError:pass
  try:child.wait(timeout=10)
  except subprocess.TimeoutExpired:report.update(status='failed',cleanup_incomplete=True)
 rp=output/'native/RESULT.json'
 if rp.exists():
  try:report.update(result=read(rp),child_result_receipt=pin(rp))
  except Exception as error:report['result_read_error']=str(error)
 use=resource.getrusage(resource.RUSAGE_CHILDREN);disk=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
 cost={'wall_seconds':time.monotonic()-started,'child_wall_seconds':None if launched is None else time.monotonic()-launched,'peak_RSS_sampled_bytes':peak_rss,'peak_GPU_sampled_bytes':peak_gpu,'own_output_disk_bytes':disk,'children_peak_RSS_bytes':use.ru_maxrss*1024,'children_user_seconds':use.ru_utime,'children_system_seconds':use.ru_stime,'scope':'own process group samples; OS children include metadata/GPU probes'}
 if disk>disk_cap or use.ru_maxrss*1024>rss_cap:report.update(status='failed',terminal_resource_violation=True)
 report.update(cost=cost,exit_code=None if child is None else child.returncode,unrecorded_inflight_work_possible=bool(child is not None and (report.get('forced_KILL_after_TERM') or report.get('result',{}).get('status')!='passed')))
 write(output/'COST.json',cost);write(output/'SUPERVISOR.json',report)
 if lock is not None:lock.close()
print(json.dumps(report))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True, choices=('NCNC_C4', 'buddy_seed1'))
    args = parser.parse_args(); here = Path(__file__).resolve().parent
    terminal = here / (args.case + '_RUNTIME_CHECK_31.json')
    with terminal.open('x') as stream: json.dump({'status': 'transport_in_progress', 'case': args.case}, stream)
    code = CODE.replace('@ROOT@', repr(REMOTE_ROOT)).replace('@EXP@', repr(REMOTE_EXPANSION)).replace('@CASE@', repr(args.case)).replace('@LAUNCHER@', repr(hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    try: result = json.loads(call(code, timeout=910 if args.case == 'NCNC_C4' else 3910))
    except BaseException as error: result = {'status': 'transport_failed_remote_outcome_unknown', 'case': args.case, 'failure': str(error), 'automatic_retries': 0}
    terminal.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result.get(k) for k in ('case', 'status', 'failure', 'cost')}))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__': raise SystemExit(main())
