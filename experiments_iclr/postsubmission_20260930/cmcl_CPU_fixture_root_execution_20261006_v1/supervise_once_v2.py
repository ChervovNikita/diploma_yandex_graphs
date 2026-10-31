"""One CPU CMCL check with the existing reviewed 806 physical/wait4 helpers."""
import argparse,hashlib,importlib.util,json,os,signal,socket,subprocess,time,traceback,sys
from pathlib import Path
REPO=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE=REPO/'experiments_iclr/postsubmission_20260930'
HELPER='matched_first_order_private_gradient_utility_native_supervisor_preparation_20261006_v4/supervise.py'
HELPER_SHA='80611cae8ab2e557a11db72c99321bf5f5d550992dbf16f63374364f48f8f808'
SOURCE_RELEASED=False

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def bound(row):
 r=Path(row['path']);assert not r.is_absolute() and '..' not in r.parts
 p=PHASE/r;assert p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(PHASE) and not p.stat().st_mode&0o222
 assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'];return p
def main():
 if sys.flags.optimize != 0:raise RuntimeError('Unoptimized normal CPU supervisor required before path access')
 parser=argparse.ArgumentParser();parser.add_argument('--execute-authorized',action='store_true');parser.add_argument('--scope');parser.add_argument('--scope-sha256');a=parser.parse_args()
 if not a.execute_authorized:print(json.dumps({'status':'DISABLED_CPU_CMCL_OWNED_INVOCATION'}));return 0
 assert SOURCE_RELEASED is False and socket.gethostname()=='anogena-2-0' and Path.cwd()==REPO
 scope_path=Path(a.scope);assert scope_path.is_file() and scope_path.resolve().is_relative_to(PHASE) and not scope_path.is_symlink() and not scope_path.stat().st_mode&0o222 and sha(scope_path)==a.scope_sha256
 s=json.loads(scope_path.read_text());assert s['root_engineering_invocation_authorized'] and s['fixed_before_execution'] and s['CPU_only'] and s['CUDA_VISIBLE_DEVICES']==''
 assert s['external_owned_supervisor_sha256']==sha(Path(__file__)) and s['external_terminal_resource_closure_required']
 review=json.loads(bound(s['external_supervisor_review']).read_text());assert review['status'].startswith('PASS_SOURCE') and review['supervisor_sha256']==sha(Path(__file__)) and review['execution_or_fit_authorized'] is False
 worker=bound(s['executor']);bound(s['executor_review']);bound(s['helper']);bound(s['helper_review']);bound(s['fixture_plan'])
 helper_path=PHASE/HELPER;assert not helper_path.is_symlink() and not helper_path.stat().st_mode&0o222 and sha(helper_path)==HELPER_SHA
 spec=importlib.util.spec_from_file_location('_cmcl_CPU_owned806',helper_path);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h);assert h.SOURCE_RELEASED is False
 binary=Path(s['python_executable']);assert binary.resolve().is_relative_to(REPO) and sha(binary)==s['python_binary_sha256']
 outer=h.deliberate(PHASE,PHASE/s['external_output_relative_path']);child_output=h.deliberate(PHASE,PHASE/s['output_relative_path'])
 assert outer!=child_output and outer.parent==child_output.parent and outer.parent.is_dir() and not outer.exists() and not child_output.exists()
 assert s['resource_limits']=={'max_elapsed_seconds':90,'max_process_rss_bytes':2147483648} and s['external_watchdog_seconds']==150 and s['terminate_grace_seconds']==10 and s['kill_reap_seconds']==30
 outer.mkdir();st=outer.stat();owner=(st.st_dev,st.st_ino)
 def write(name,v):
  assert not outer.is_symlink() and (outer.stat().st_dev,outer.stat().st_ino)==owner
  f=outer/name;tmp=outer/(name+'.tmp');assert not f.exists() and not f.is_symlink()
  with tmp.open('x') as stream:
   json.dump(v,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fchmod(stream.fileno(),0o444);os.fsync(stream.fileno())
  assert not outer.is_symlink() and (outer.stat().st_dev,outer.stat().st_ino)==owner and not f.exists()
  os.replace(tmp,f)
  directory_fd=os.open(outer,os.O_RDONLY)
  try:os.fsync(directory_fd)
  finally:os.close(directory_fd)
 argv=[str(binary),'-B',str(worker),'--execute-authorized','--source-root',str(PHASE),'--scope',str(scope_path),'--scope-sha256',a.scope_sha256,'--output',str(child_output)]
 observation={'reaped':False,'watchdog_fired':False,'signals':[]};terminal={'schema':'CMCL_CPU_806_owned_terminal_v1','status':'FAIL','scope_sha256':a.scope_sha256,'supervisor_sha256':sha(Path(__file__)),'owned_child':observation,'cleanup_errors':[],'held_scoring':False,'engineering_only':True,'resource_limits':s['resource_limits'],'launch_attempts':0}
 child=None;handlers={n:signal.getsignal(n) for n in (signal.SIGINT,signal.SIGTERM)}
 def interrupted(n,f):raise InterruptedError('Owned CPU check interrupted: '+str(n))
 try:
  write('LAUNCH_CLAIM.json',{'argv':argv,'cwd':str(REPO),'assigned_child_output':str(child_output),'scope_sha256':a.scope_sha256,'retry':False})
  for n in handlers:signal.signal(n,interrupted)
  with (outer/'stdout.log').open('xb') as out,(outer/'stderr.log').open('xb') as err:
   observation['monotonic_before_Popen']=time.monotonic();terminal['launch_attempts']=1
   child=subprocess.Popen(argv,cwd=REPO,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
   observation['PID']=child.pid;observation['physical_identity']=h.physical(child,argv,REPO);write('CHILD_LAUNCH.json',observation);h.owned_wait(child,observation,s)
  assert observation['exit_code']==0 and observation['reaped'] and not observation['watchdog_fired'] and not observation['signals'] and observation['physical_identity'].get('verified')
  assert child_output.is_dir() and not child_output.is_symlink() and child_output.resolve().is_relative_to(PHASE)
  result_path=child_output/'RESULT.json';publication_path=child_output/'PUBLICATION.json'
  for f in (result_path,publication_path):assert f.is_file() and not f.is_symlink() and not f.stat().st_mode&0o222
  result=json.loads(result_path.read_text());publication=json.loads(publication_path.read_text())
  assert publication['schema']=='CMCL_CPU_fixture_worker_publication_v1' and publication['result']=={'path':'RESULT.json','bytes':result_path.stat().st_size,'sha256':sha(result_path)} and publication['worker_exit_code_planned']==0
  assert result['status']==publication['status']=='PASS_CPU_FLOAT64_CMCL_ANALYTIC_FIXTURES_ENGINEERING_ONLY' and result['executor']==s['executor'] and publication['executor']==s['executor'] and result['scope_sha256']==a.scope_sha256
  assert len(result['checks'])==result['completed_frozen_checks']==19 and all(c['status']=='PASS' for c in result['checks']) and not result['restoration_errors']
  assert result['attempts']=={'positive_role_objective_calls':6,'positive_serving_calls':1,'ordinary_grad_API_calls':6,'rejection_calls':13}
  assert all(result[k]==0 for k in ('model_fits','native_callbacks','CUDA_API_calls','optimizer_updates')) and result['actual_dataset_labels_read'] is False and result['A_VALID_TEST_access'] is False
  peak=result['whole_process_resources'];assert observation['whole_child_wall_seconds']<=90 and observation['wait4_peak_RSS_bytes']<=2147483648 and peak['elapsed_seconds_including_imports_checks_and_restore']<=90 and peak['process_peak_rss_bytes']<=2147483648
  terminal.update(status='PASS_CPU_CMCL_OWNED_PUBLICATION_EXIT_RESOURCE_CLOSURE_ENGINEERING_ONLY',worker_result=h.descriptor(result_path,str(result_path.relative_to(PHASE))),publication=h.descriptor(publication_path,str(publication_path.relative_to(PHASE))),checks=19,worker_restoration_errors=[],wrapper_resource_closure_PASS=True)
 except BaseException as e:terminal.update(error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),wrapper_resource_closure_PASS=False)
 finally:
  for n in handlers:signal.signal(n,signal.SIG_IGN)
  if child is not None and not observation['reaped']:
   try:h.owned_wait(child,observation,s,force_reason='ROOT_CPU_CHECK_FAILURE_CLEANUP')
   except BaseException as e:terminal['cleanup_errors'].append({'type':type(e).__name__,'error':str(e)})
  for n,handler in handlers.items():signal.signal(n,handler)
  if terminal['cleanup_errors'] or not observation['reaped']:terminal.update(status='FAIL',wrapper_resource_closure_PASS=False)
  terminal['parent_preflight_and_terminal_tail_outside_child_caps']=True;write('TERMINAL.json',terminal)
 print(json.dumps({'status':terminal['status'],'owned_child':observation,'checks':terminal.get('checks'),'held_scoring':False}))
 return 0 if terminal.get('wrapper_resource_closure_PASS') else 1
if __name__=='__main__':raise SystemExit(main())
