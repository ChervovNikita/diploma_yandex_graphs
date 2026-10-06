"""Directly supervise three fixed native scientific fits in parallel."""
import argparse,hashlib,importlib.util,json,os,socket,subprocess
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor,as_completed
REPO=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE=REPO/'experiments_iclr/postsubmission_20260930'
GPU='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n')
def confined(rel):
 p=(PHASE/rel).resolve(strict=True)
 if not p.is_relative_to(PHASE) or not p.is_file():raise ValueError('Project file required')
 return p
def physical():
 if Path.cwd()!=REPO or socket.gethostname()!='anogena-2-0':raise ValueError('Wrong scientific host')
 if subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).split()!=[GPU]:raise ValueError('Wrong scientific allocation')
def load(name,row):
 path=confined(row['path'])
 if sha(path)!=row['sha256']:raise ValueError('Bound owner changed')
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--config-sha256',required=True);args=parser.parse_args();physical()
 if sha(args.config)!=args.config_sha256:raise ValueError('Configuration changed')
 cfg=read(args.config)
 if cfg['execution_enabled'] is not True or sha(__file__)!=cfg['adapter_sha256']:raise ValueError('Exact prospective root release required')
 for row in cfg['bindings']:
  if sha(confined(row['path']))!=row['sha256']:raise ValueError('Bound source/evidence changed')
 source=PHASE/cfg['source_relative'];root=PHASE/cfg['execution_root_relative'];work=root/'owner';work.mkdir();(work/'logs').mkdir()
 old=load('existing_native_fit_supervisor',cfg['existing_owner']);helper=load('existing_native_fit_helper',cfg['existing_helper']);helper.GPU=GPU
 context=SimpleNamespace(REPO=REPO,SOURCE=source,SOURCE_SHA=cfg['source_sha256'],GPU_UUID=GPU,GPU_UUIDS=(GPU,),phase_file=confined,physical_host=physical,sha=sha,write=write)
 environment=dict(os.environ,**cfg['environment']);environment.pop('PYTHONHOME',None)
 write(work/'START.json',{'identity':helper.identity(os.getpid()),'config_sha256':args.config_sha256,'full1100_fits':3,'parallel_workers':3,'reference_parity_gate':False,'partial_quality_read_by_owner':False})
 def fit(entry):
  physical();job_path=confined(entry['job_relative']);job=read(job_path)
  if sha(job_path)!=entry['job_sha256'] or job['TEST_access'] is not False or job['kind']!='native_fit' or job['seed'] not in (17,29,43):raise ValueError('Exact fixed native fit required')
  out=Path(entry['output_directory'])
  if out.exists():raise ValueError('No fit overwrite or retry')
  terminal=old.run_fit(helper,work,entry,{'resource_limits':cfg['resource_limits']},environment,out,context)
  if terminal['exit_code']!=0 or terminal['reason'] is not None or terminal['signals_sent'] or not terminal['terminal_wait_observed'] or (out/'FAILURE.json').exists():return {'cell_id':entry['cell_id'],'complete':False,'terminal':terminal}
  freeze=read(out/'NATIVE_FREEZE.json')
  if freeze['epochs']!=1100 or freeze['ordinary_Adam_updates_total']!=1100 or freeze['source_manifest_sha256']!=cfg['source_sha256'] or freeze['job_sha256']!=entry['job_sha256'] or freeze['TEST_access'] is not False:raise ValueError('Complete fixed native custody differs')
  for row in (freeze['selected'],freeze['end'],freeze['selected_replay']):
   if sha(confined(row['path']))!=row['sha256']:raise ValueError('Terminal native artifact changed')
  return {'cell_id':entry['cell_id'],'complete':True,'freeze':{'path':str((out/'NATIVE_FREEZE.json').relative_to(PHASE)),'sha256':sha(out/'NATIVE_FREEZE.json')},'terminal':terminal}
 completed=[]
 with ThreadPoolExecutor(max_workers=3) as pool:
  pending={pool.submit(fit,e):e['cell_id'] for e in cfg['entries']}
  for future in as_completed(pending):
   try:result=future.result()
   except BaseException as error:result={'cell_id':pending[future],'complete':False,'error':type(error).__name__+': '+str(error)}
   completed.append(result);write(work/'PROGRESS.json',{'completed':completed,'total':3,'partial_quality_read_by_owner':False})
 write(work/('COMPLETE.json' if all(r['complete'] for r in completed) else 'FAILURE.json'),{'completed':completed,'all_complete':all(r['complete'] for r in completed),'fits':3,'TEST_access':False,'retry':False})
if __name__=='__main__':main()
