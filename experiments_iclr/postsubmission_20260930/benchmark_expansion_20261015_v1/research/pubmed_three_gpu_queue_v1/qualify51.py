"""51-only two-update/three-VALID/fresh-restore engineering; no scientific fit."""
import argparse,copy,hashlib,importlib.util,json,os,resource,signal,socket,subprocess,sys,time,traceback
from pathlib import Path
def main():
 require=lambda ok,msg:None if ok else (_ for _ in()).throw(ValueError(msg))
 uuids=['GPU-9c6c20fa-a620-479d-f72f-163ac74deef2','GPU-30283973-140d-9e7b-e780-0fcc487dec54']
 require(socket.gethostname()=='olig'and subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()==uuids,'Literal verified51 host required before project reads')
 p=argparse.ArgumentParser();p.add_argument('--gpu-index',type=int,choices=(0,1),required=True);p.add_argument('--kind',choices=('native','structured'),required=True);p.add_argument('--output',required=True);p.add_argument('--source-sha256',required=True);p.add_argument('--binding-sha256',required=True);p.add_argument('--root-resource-admitted',action='store_true');args=p.parse_args()
 require(args.root_resource_admitted,'Root must admit bounded engineering resources first')
 here=Path(__file__).resolve().parent;root=Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git');phase=root/'experiments_iclr/postsubmission_20260930';e=phase/'benchmark_expansion_20261015_v1'
 require(here==e/'research/pubmed_three_gpu_queue_v1'and hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==args.source_sha256 and hashlib.sha256((here/'binding.py').read_bytes()).hexdigest()==args.binding_sha256,'Exact reviewed qualifier/binding source required')
 require(Path(sys.prefix).resolve()==(root/'.venv').resolve()and os.environ.get('CUDA_VISIBLE_DEVICES')==uuids[args.gpu_index],'Original51 virtualenv and one explicit physical GPU required')
 from binding import load_inventory,shard,relocated,install_residual_descriptor_view,restore_descriptor_view,pin
 inventory=load_inventory();worker='server51/gpu'+str(args.gpu_index);rows=shard(inventory,worker);row=next(r for r in rows if r['kind']==args.kind and(r['job'].get('body')in(None,0)))
 host={'project_root':str(root),'phase_relative':'experiments_iclr/postsubmission_20260930'};job,binding=relocated(row,host,worker,args.source_sha256,args.binding_sha256)
 for record in json.loads((here/'SOURCE_BINDINGS.json').read_text())['files']:
  current=pin(phase/record['path_relative_to_phase']);require(all(current[k]==record[k]for k in('bytes','sha256')),'Frozen source dependency changed')
 for record in json.loads((here/'PARENT_INPUTS.json').read_text())['data_pins']:
  path=phase/Path(record['path']).relative_to(Path(inventory['canonical_root'])/inventory['canonical_phase_relative']);current=pin(path);require(all(current[k]==record[k]for k in('bytes','sha256')),'Frozen data/basis/role bytes changed')
 output=Path(args.output).resolve();require(output.is_relative_to(root)and not output.exists(),'Fresh repo output required');output.mkdir(parents=True)
 report={'schema':'pubmed51-portable-lifecycle-v1','status':'running','worker':worker,'kind':args.kind,'canonical_job_sha256':row['job_hash'],'execution_binding':binding,'TRAIN_completed':0,'VALID_completed':0,'constructors':0,'optimizer_calls':None,'scientific_fit':False,'TEST_accessed':False,'source':pin(__file__),'binding_source':pin(here/'binding.py')};started=time.perf_counter();adapters=[];observer=None;view=None
 def save():
  report.update(wall_seconds=time.perf_counter()-started,own_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
  (output/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
 def stop(*_):raise TimeoutError('Bounded engineering alarm/interruption')
 signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.alarm(900)
 try:
  sys.path.insert(0,str(e/'runtime'));import torch
  from adapters import create_adapter
  from run_fit import validate_job,OptimizerObserver
  from common import seed_all,global_rng,restore_rng,cpu_clone
  torch.set_num_threads(2);torch.cuda.reset_peak_memory_stats();view=install_residual_descriptor_view(job,row,root)
  seed_all(torch,job['seed']);adapter=create_adapter(torch,copy.deepcopy(job),validate_job(job),root);adapters.append(adapter);report['constructors']+=1;observer=OptimizerObserver(torch,adapter.optimizers)
  first=adapter.train_epoch();report['TRAIN_completed']+=1;valid=adapter.validate();report['VALID_completed']+=1
  require(first['complete_native_TRAIN']is True and first['TRAIN_records_traversed']==11829 and len(valid['ids'])==3943 and valid['TEST_accessed']is False,'Full actual TRAIN/VALID required')
  torch.save({'adapter':cpu_clone(torch,adapter.state()),'rng':cpu_clone(torch,global_rng(torch)),'calls':observer.state(),'engineering_only':True},output/'ENGINEERING.pt');report['checkpoint']=pin(output/'ENGINEERING.pt');observer.close();observer=None;adapter.close();adapters.remove(adapter)
  seed_all(torch,job['seed']);fresh=create_adapter(torch,copy.deepcopy(job),validate_job(job),root);adapters.append(fresh);report['constructors']+=1;saved=torch.load(output/'ENGINEERING.pt',map_location='cpu',weights_only=False);fresh.load_state(saved['adapter']);replay=fresh.validate();report['VALID_completed']+=1
  require(torch.equal(valid['ids'],replay['ids'])and torch.equal(valid['outputs'],replay['outputs']),'Fresh restore changed whole VALID');report['fresh_restore_VALID_bit_exact']=True
  fresh.load_state(saved['adapter']);restore_rng(torch,saved['rng']);observer=OptimizerObserver(torch,fresh.optimizers,previous=saved['calls']);second=fresh.train_epoch();report['TRAIN_completed']+=1;final=fresh.validate();report['VALID_completed']+=1
  require(second['complete_native_TRAIN']is True and second['TRAIN_records_traversed']==11829 and len(final['ids'])==3943 and final['TEST_accessed']is False and observer.state()=={'attempts':[2],'completed':[2]},'Second full update/VALID or Adam accounting failed')
  torch.cuda.synchronize();report.update(status='passed',optimizer_calls=observer.state(),GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved(),Torch=str(torch.__version__))
 except BaseException as error:report.update(status='failed',failure={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc(limit=10)})
 finally:
  if observer is not None:report['optimizer_calls']=observer.state();observer.close()
  for adapter in adapters:
   try:adapter.close()
   except Exception as error:report.update(status='failed',close_error=str(error))
  restore_descriptor_view(view);signal.alarm(0);save()
 print(json.dumps({k:report.get(k)for k in('status','kind','worker','TRAIN_completed','VALID_completed','fresh_restore_VALID_bit_exact','failure')}));return 0 if report['status']=='passed'else 1
if __name__=='__main__':raise SystemExit(main())
