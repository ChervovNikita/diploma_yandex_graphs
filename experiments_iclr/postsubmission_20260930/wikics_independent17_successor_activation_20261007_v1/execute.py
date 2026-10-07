"""Wait for the exact seed17 prefix terminal, then run its two fixed successors."""
import hashlib,importlib.util,json,os,socket,subprocess,time
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parent
REPO=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE=REPO/'experiments_iclr/postsubmission_20260930';GPU='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n')
def confined(rel):
 p=(PHASE/rel).resolve(strict=True)
 if not p.is_relative_to(PHASE) or not p.is_file():raise ValueError('Phase file required')
 return p
def physical():
 if Path.cwd()!=REPO or socket.gethostname()!='anogena-2-0':raise ValueError('Wrong host')
 if subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).split()!=[GPU]:raise ValueError('Wrong allocation')
def identity(pid):
 p=Path('/proc')/str(pid)
 try:
  s=(p/'stat').read_text();f=s[s.rfind(')')+2:].split();return {'PID':pid,'start_ticks':int(f[19]),'state':f[0]}
 except FileNotFoundError:return None
def load(name,b):
 p=confined(b['path'])
 if sha(p)!=b['sha256']:raise ValueError('Bound owner source changed')
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def main():
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--config-sha256',required=True);args=parser.parse_args()
 physical();cfg_path=ROOT/'CONFIG.json'
 if sha(cfg_path)!=args.config_sha256:raise ValueError('Exact prospective continuation release required')
 cfg=read(cfg_path)
 if sha(__file__)!=cfg['adapter_sha256'] or cfg['root_execution_authorized'] is not True:raise ValueError('Exact adapter authority required')
 for b in cfg['bindings']:
  if sha(confined(b['path']))!=b['sha256']:raise ValueError('Exact prospective binding changed')
 execution=PHASE/cfg['execution_root'];execution.mkdir();(execution/'logs').mkdir();(execution/'jobs').mkdir();(execution/'runs').mkdir()
 write(execution/'START.json',{'identity':identity(os.getpid()),'config_sha256':args.config_sha256,'dependent_science_started':False,'partial_quality_read':False})
 started=time.monotonic();dependency=PHASE/cfg['producer_root'];bank=dependency/'runs'/'DONOR_INDEPENDENT_17';terminal=dependency/'owner'/'logs'/'DONOR_INDEPENDENT_17.EXIT.json';producer_job=confined(cfg['producer_job']['path'])
 if sha(producer_job)!=cfg['producer_job']['sha256']:raise ValueError('Prefix producer job changed')
 while not terminal.exists():
  physical();who=identity(cfg['producer_identity']['PID'])
  if who is not None and who['start_ticks']!=cfg['producer_identity']['start_ticks']:raise ValueError('Prefix owner identity replaced')
  if who is None:raise ValueError('Prefix owner absent without complete native-bank terminal; no successor fit')
  if time.monotonic()-started>cfg['dependency_wait_seconds']:raise TimeoutError('Dependency wait cap; preserve prior work and no restart')
  write(execution/'WAIT.json',{'elapsed_seconds':time.monotonic()-started,'producer_owner_live':who,'dependent_science_started':False,'scores_read':False});time.sleep(30)
 receipt=read(terminal)
 if receipt['exit_code']!=0 or receipt['reason'] is not None or receipt['signals_sent'] or not receipt['terminal_wait_observed'] or receipt['job_sha256']!=cfg['producer_job']['sha256'] or (bank/'FAILURE.json').exists():raise ValueError('Prefix bank did not finish with clean exact owned terminal')
 child=receipt['raw_identity_observation'];who=identity(child['PID'])
 if who is not None and who['start_ticks']==child['start_ticks']:raise ValueError('Completed native prefix child still exists')
 frozen_path=bank/'DONOR_FREEZE.json';frozen=read(frozen_path)
 if frozen.get('complete') is not True or frozen['arm']!='DONOR_INDEPENDENT' or frozen['seed']!=17 or len(frozen['members'])!=4:raise ValueError('Four complete native prefixes required')
 for m,b in enumerate(frozen['members']):
  p=confined(b['path']);d=read(p)
  if sha(p)!=b['sha256'] or d['epochs']!=1100 or d['seed']!=17+1009*m or d['source_manifest_sha256']!=cfg['source_sha256'] or d['program_sha256']!=cfg['program_sha256']:raise ValueError('Prefix source/seed/horizon changed')
  confined(d['selected']);confined(d['end'])
 write(execution/'DEPENDENCY_ADMITTED.json',{'producer_terminal':{'path':str(terminal.relative_to(PHASE)),'sha256':sha(terminal)},'bank_freeze':{'path':str(frozen_path.relative_to(PHASE)),'sha256':sha(frozen_path)},'prefix_acquisition_count':1,'used_for':['I_native','U_stage'],'scores_read':False})
 supervisor=load('existing_successor_supervisor',cfg['existing_owner']);helper=load('existing_successor_helper',cfg['existing_helper']);helper.GPU=GPU
 source=PHASE/cfg['source_relative'];context=SimpleNamespace(REPO=REPO,SOURCE=source,SOURCE_SHA=cfg['source_sha256'],GPU_UUID=GPU,GPU_UUIDS=(GPU,),phase_file=confined,physical_host=physical,sha=sha,write=write)
 env=dict(os.environ,**cfg['environment']);env.pop('PYTHONHOME',None)
 entries=[]
 for arm in ('I_native','U_stage'):
  template=confined(cfg['fit_template']['path']);job=read(template);job.update(arm=arm,donor_freezes=frozen['members'],frozen_cache=None,cache_origin_E_stage_job=None,output_directory=str(execution/'runs'/arm))
  path=execution/'jobs'/(arm+'.json');write(path,job)
  entries.append({'cell_id':arm+'_17','job_relative':str(path.relative_to(PHASE)),'job_sha256':sha(path),'hard_seconds':job['hard_seconds'],'argv':[str(PHASE/'native_ncn_runtime_20261005_v1/.venv/bin/python'),'-B',str(source/'run.py'),'--job',str(path),'--output',job['output_directory']],'output_directory':job['output_directory']})
 write(execution/'JOBS_FROZEN.json',{'entries':entries,'prescribed_arms':['I_native','U_stage'],'prescribed_seed':17,'exact_plan':read(confined(cfg['fit_template']['path']))['plan'],'no_recipe_change':True,'scores_read':False})
 def fit(e):
  t=supervisor.run_fit(helper,execution,e,{'resource_limits':cfg['resource_limits']},env,Path(e['output_directory']),context)
  ok=t['exit_code']==0 and t['reason'] is None and not t['signals_sent'] and t['terminal_wait_observed'] and not (Path(e['output_directory'])/'FAILURE.json').exists();r={'cell_id':e['cell_id'],'complete':ok,'terminal':t}
  if ok:
   p=Path(e['output_directory'])/'FREEZE.json';f=read(p)
   if f.get('complete') is not True or f['seed']!=17 or f['arm']!=e['cell_id'].rsplit('_',1)[0] or f['source_manifest_sha256']!=cfg['source_sha256'] or f['job_sha256']!=e['job_sha256']:raise ValueError('Complete successor custody differs')
   r['freeze']={'path':str(p.relative_to(PHASE)),'sha256':sha(p)}
  return r
 results=[]
 with ThreadPoolExecutor(max_workers=2) as pool:
  fs={pool.submit(fit,e):e for e in entries}
  for f in as_completed(fs):
   try:results.append(f.result())
   except Exception as error:results.append({'cell_id':fs[f]['cell_id'],'complete':False,'error':type(error).__name__+': '+str(error)})
   write(execution/'PROGRESS.json',{'results':results,'scores_read':False})
 write(execution/('COMPLETE.json' if all(r['complete'] for r in results) else 'FAILURE.json'),{'complete':all(r['complete'] for r in results),'results':results,'scores_read':False,'TEST_access':False,'retry':False})
if __name__=='__main__':
 try:main()
 except BaseException as error:
  cfg=read(ROOT/'CONFIG.json');target=PHASE/cfg['execution_root']
  if target.is_dir() and not (target/'FAILURE.json').exists():write(target/'FAILURE.json',{'error':type(error).__name__+': '+str(error),'dependent_science_may_be_incomplete':True,'retry':False})
  raise
