"""Time fixed full PubMed epochs; no selector, checkpoint, TEST or metric output."""
import argparse,copy,fcntl,importlib.util,json,os,resource,signal,sys,time,traceback
from pathlib import Path
from qualifier_contracts import (HERE,E,PILOT,ROOT,WARMUP_EPOCHS,TIMED_EPOCHS,PHASES,preflight,pin,read,write,identity,require,safe)
def main():
 p=argparse.ArgumentParser()
 for field in ('admission','admission-sha256','phase','arm'):p.add_argument('--'+field,required=True)
 args=p.parse_args();a,host,output,slot,ap=preflight(args.admission,args.admission_sha256)
 require(args.phase in PHASES and args.arm in PHASES[args.phase],'Fixed phase/arm required')
 require(os.environ.get('CUDA_VISIBLE_DEVICES')==a['physical_gpu_uuid'],'Exact one-GPU UUID mask required before Torch')
 claim=read(output/'OWNER_CLAIM.json');require(identity(claim['owner']['pid'])==claim['owner'] and os.getppid()==claim['owner']['pid'] and claim['admission']==pin(ap) and claim['source_data_verified'] is True,'Only the live verifying owner may start an engineering child')
 with slot.open('a') as handle:
  try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:pass
  else:fcntl.flock(handle,fcntl.LOCK_UN);raise ValueError('Owner must hold the prior exact PubMed slot lock')
 folder=output/'phases'/args.phase/args.arm;require(folder.is_dir() and not (folder/'RESULT.json').exists(),'Fresh own child folder required')
 started=time.monotonic();report={'schema':'server51-fixed-pubmed-epoch-timing-v1','status':'running','phase':args.phase,'arm':args.arm,
  'owner':claim['owner'],'child':identity(os.getpid()),'admission':pin(ap),'source_manifest':pin(HERE/'MANIFEST.json'),
  'scientific_fit':False,'accuracy_claim':False,'checkpoint_selection':False,'checkpoints_written':0,'TEST_accessed':False,
  'warmup_epochs_requested':WARMUP_EPOCHS,'timed_epochs_requested':TIMED_EPOCHS,'TRAIN_epochs_completed':0,'VALID_calls_completed':0,
  'optimizer_calls':{'attempts':[0],'completed':[0]},'epochs':[],'automatic_retries':0,
  'timed_progress_write_stride_epochs':25,'per_epoch_duration_excludes_progress_IO':True,'whole_window_includes_fixed_progress_IO':True}
 adapter=observer=view=None;binding_module=None;torch=None
 def save():
  if observer is not None:report['optimizer_calls']=observer.state()
  report['invocation_wall_seconds']=time.monotonic()-started
  report['own_peak_RSS_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
  write(folder/'RESULT.json',report)
 def stop(*_):raise InterruptedError('Owned bounded timing child interrupted; no retry')
 for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP,signal.SIGALRM):signal.signal(sig,stop)
 signal.alarm(a['maximum_child_wall_seconds']);save()
 try:
  # Frozen original pilot helpers; no training/controller/forward body is replaced.
  bs=importlib.util.spec_from_file_location('_pubmed_throughput_original_binding',PILOT/'binding.py')
  binding_module=importlib.util.module_from_spec(bs);bs.loader.exec_module(binding_module)
  inventory=binding_module.load_inventory();rows=binding_module.shard(inventory,a['worker'])
  row=next(r for r in rows if r['kind']==args.arm and r['job'].get('body') in (None,0))
  job,binding=binding_module.relocated(row,host,a['worker'],a['pilot_source_manifest']['sha256'],a['actual_target_map']['sha256'])
  report.update(canonical_reference_id=row['id'],canonical_reference_job_hash=row['job_hash'],execution_binding=binding,
   native_settings_unchanged=True,controller='fixed engineering epoch count; no selection; not a paper result')
  sys.path.insert(0,str(E/'runtime'));import torch
  from adapters import create_adapter
  from run_fit import validate_job,OptimizerObserver
  from common import seed_all
  require(str(torch.__version__)==a['expected_torch_version'],'Reviewed Torch build differs');torch.set_num_threads(2)
  torch.cuda.reset_peak_memory_stats();view=binding_module.install_residual_descriptor_view(job,row,ROOT)
  seed_all(torch,job['seed']);adapter=create_adapter(torch,copy.deepcopy(job),validate_job(job),ROOT)
  observer=OptimizerObserver(torch,adapter.optimizers);torch.cuda.synchronize()
  report['startup_seconds']=time.monotonic()-started;report['startup_GPU_peak_allocated_bytes']=torch.cuda.max_memory_allocated();report['startup_GPU_peak_reserved_bytes']=torch.cuda.max_memory_reserved()
  def epoch(stage,index):
   torch.cuda.synchronize();begin=time.monotonic_ns();train=adapter.train_epoch();report['TRAIN_epochs_completed']+=1
   require(train['complete_native_TRAIN'] is True and train['TRAIN_records_traversed']==11829,'One complete native PubMed TRAIN epoch required')
   valid=adapter.validate();report['VALID_calls_completed']+=1
   require(len(valid['ids'])==3943 and valid.get('TEST_accessed') is False,'One complete native PubMed VALID population required')
   torch.cuda.synchronize();end=time.monotonic_ns()
   record={'stage':stage,'index':index,'start_monotonic_ns':begin,'end_monotonic_ns':end,'seconds':(end-begin)/1e9,'TRAIN_records':11829,'VALID_records':3943}
   # Scores and tensors from the native VALID callback are deliberately not persisted or used for selection.
   report['epochs'].append(record)
   if stage=='warmup' and index+1==WARMUP_EPOCHS or stage=='timed' and(index+1)%25==0:save()
  for index in range(WARMUP_EPOCHS):epoch('warmup',index)
  report['warmup_wall_seconds']=sum(r['seconds'] for r in report['epochs']);report['warmup_ready_monotonic_ns']=time.monotonic_ns()
  write(folder/'WARMUP_READY.json',{'schema':'throughput-warmup-ready-v1','phase':args.phase,'arm':args.arm,'child':report['child'],'admission':pin(ap),'epochs':WARMUP_EPOCHS})
  barrier=folder.parent/'START_MEASUREMENT.json';waiting=time.monotonic()
  while not barrier.exists():
   require(identity(claim['owner']['pid'])==claim['owner'],'Owner disappeared before measurement barrier')
   require(time.monotonic()-waiting<300,'Finite warmup barrier wait');time.sleep(.1)
  start=read(barrier);require(start['owner']==claim['owner'] and start['admission']==pin(ap) and start['phase']==args.phase,'Exact owner-released barrier required')
  report['barrier']=pin(barrier);report['barrier_wait_seconds']=time.monotonic()-waiting
  torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();report['measurement_start_monotonic_ns']=time.monotonic_ns()
  for index in range(TIMED_EPOCHS):epoch('timed',index)
  torch.cuda.synchronize();report['measurement_end_monotonic_ns']=time.monotonic_ns()
  require(observer.state()=={'attempts':[WARMUP_EPOCHS+TIMED_EPOCHS],'completed':[WARMUP_EPOCHS+TIMED_EPOCHS]},'Exact fixed full-batch optimizer cost required')
  report.update(status='PASS',GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved(),Torch=str(torch.__version__))
 except BaseException as error:report.update(status='failed',failure={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc(limit=12)})
 finally:
  if observer is not None:report['optimizer_calls']=observer.state();observer.close()
  if adapter is not None:
   try:adapter.close()
   except Exception as error:report.update(status='failed',close_error=str(error))
  if binding_module is not None:binding_module.restore_descriptor_view(view)
  signal.alarm(0);save()
 print(json.dumps({k:report.get(k) for k in ('status','phase','arm','TRAIN_epochs_completed','VALID_calls_completed','optimizer_calls','failure')},sort_keys=True))
 return 0 if report['status']=='PASS' else 1
if __name__=='__main__':raise SystemExit(main())
