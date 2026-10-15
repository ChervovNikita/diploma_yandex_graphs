"""One detached command per GPU; eight fixed fits then four selected collectors."""
import argparse,fcntl,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
from binding import HERE,E,require,pin
from contracts import request,read,write,process,resource_check,safe
def main():
 p=argparse.ArgumentParser();p.add_argument('--worker',required=True);p.add_argument('--target-map',required=True);p.add_argument('--target-map-sha256',required=True);p.add_argument('--owner',action='store_true');args=p.parse_args()
 a,host,root,rows,ap=request(args.target_map,args.target_map_sha256,args.worker,full=args.owner)
 output=E/'results/pubmed_three_gpu_v1'/args.worker.replace('/','_')
 command=[host['python_executable'],'-B','-u',str(HERE/'run_queue.py'),'--worker',args.worker,'--target-map',str(ap),'--target-map-sha256',args.target_map_sha256,'--owner']
 if not args.owner:
  output.mkdir(parents=True,exist_ok=False)
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=host['physical_gpu_uuid'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
  with(output/'owner.log').open('xb')as log:child=subprocess.Popen(command,cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True,env=env)
  receipt={'schema':'pubmed-three-GPU-detached-launch-v1','worker':args.worker,'owner_pid':child.pid,'identity':process(child.pid),'command':command,'target_map':pin(ap),'canonical_ids':[r['id']for r in rows],'TEST_accessed':False}
  write(output/'LAUNCH.json',receipt);print(json.dumps(receipt));return 0
 require(os.environ.get('CUDA_VISIBLE_DEVICES')==host['physical_gpu_uuid'],'Exact assigned physical CUDA mask required for owner')
 require(output.is_dir(),'Detached starter must create output');claim=output/'OWNER_CLAIM.json'
 with claim.open('x')as stream:json.dump({'pid':os.getpid(),'identity':process(os.getpid()),'worker':args.worker,'target_map':pin(ap),'source_data_verified':True,'source_manifest':pin(HERE/'MANIFEST.json')},stream)
 child=None;started=time.monotonic();lock=None;report={'schema':'pubmed-three-GPU-cohort-owner-v1','status':'preflight','worker':args.worker,'cohort_seed':rows[0]['cohort_seed'],'new_fits':8,'new_collections':4,'outcomes':[],'collections':[],'automatic_retries':0,'TEST_accessed':False,'source_manifest':pin(HERE/'MANIFEST.json'),'target_map':pin(ap),'owner':process(os.getpid())}
 def save():report['wall_seconds']=time.monotonic()-started;write(output/'QUEUE.json',report)
 def stop(*_):raise InterruptedError('Cohort owner interrupted; no retry')
 def cleanup():
  if child is not None and child.poll()is None:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=20)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10);report['forced_KILL']=True
 for sig in(signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,stop)
 def invoke(mode,row=None):
  nonlocal child
  require(time.monotonic()-started<a['maximum_wall_seconds_per_owner'],'96h cohort owner bound')
  resource_check(host,root);cmd=[host['python_executable'],'-B','-u',str(HERE/'pipeline.py'),'--worker',args.worker,'--target-map',str(ap),'--target-map-sha256',args.target_map_sha256,'--mode',mode]
  if row is not None:cmd+=['--index',str(row['index'])]
  name=mode+('_'+row['token']if row is not None else'');entry={'mode':mode,'canonical_id':None if row is None else row['id'],'status':'starting','command':cmd};report['active']=entry;save();begin=time.monotonic()
  with(output/(name+'.log')).open('xb')as log:child=subprocess.Popen(cmd,cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
  entry['child']=process(child.pid);entry['status']='running';save()
  try:
   while child.poll()is None:
    require(time.monotonic()-begin<a['maximum_wall_seconds_per_fit']and time.monotonic()-started<a['maximum_wall_seconds_per_owner'],'Finite child/owner watchdog')
    current=process(child.pid);require(current is None or current['rss_bytes']<=host['reserved_host_bytes'],'Own child RSS exceeded approved envelope')
    memory=subprocess.check_output(['nvidia-smi','--id='+host['physical_gpu_uuid'],'--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=10)
    for line in memory.splitlines():
     if line.strip():
      pid,used=map(int,line.split(','))
      if pid==child.pid:require(used*1024**2<=host['reserved_GPU_bytes'],'Own child GPU envelope exceeded')
    require(sum(p.stat().st_size for p in output.rglob('*')if p.is_file())<=host['disk_reserved_bytes'],'Own shard disk envelope exceeded')
    try:child.wait(timeout=10)
    except subprocess.TimeoutExpired:pass
   entry.update(exit_code=child.returncode,wall_seconds=time.monotonic()-begin);require(child.returncode==0,'Owned child failed; preserve without retry')
   entry['status']='complete';return entry
  finally:cleanup();entry['wall_seconds']=time.monotonic()-begin;save()
 save()
 try:
  lockpath=safe(root/host['slot_lock_relative'],root);lockpath.parent.mkdir(parents=True,exist_ok=True);lock=lockpath.open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  resource_check(host,root);report['status']='running';save()
  for row in rows:
   entry=invoke('fit',row);result_path=output/'fits'/row['token']/'RESULT.json';result=read(result_path)
   require(result['status']=='complete'and result['job_id']==row['id']and result['TEST_accessed']is False and not result.get('close_error'),'Exact complete fit result required')
   report['outcomes'].append({'id':row['id'],'index':row['index'],'canonical_job_sha256':row['job_hash'],'RESULT':pin(result_path),'status':'complete','child':entry['child'],'exit_code':entry['exit_code'],'wall_seconds':entry['wall_seconds']});write(output/'FIT_OUTCOMES.json',report['outcomes']);save()
  write(output/'FIT_END.json',{'status':'complete','worker':args.worker,'cohort_seed':rows[0]['cohort_seed'],'outcomes':report['outcomes'],'count':8,'TEST_accessed':False,'target_map':pin(ap)})
  for row in rows:
   if row['kind']=='native':continue
   entry=invoke('collect',row);path=output/'collections'/row['token']/'RESULT.json';result=read(path);require(result['status']in('complete','complete_with_sanity_warning')and result['TEST_accessed']is False,'Complete own-selected collection required')
   report['collections'].append({'id':row['id'],'index':row['index'],'RESULT':pin(path),'status':result['status'],'wall_seconds':entry['wall_seconds']});write(output/'COLLECTION_OUTCOMES.json',report['collections']);save()
  invoke('export');report.update(status='complete',bundle=pin(output/'bundle/BUNDLE.json'))
 except BaseException as error:report.update(status='failed',failure={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc(limit=10)})
 finally:
  for sig in(signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,signal.SIG_IGN)
  try:cleanup()
  except BaseException as error:report.update(status='failed',cleanup_error=str(error),unrecorded_inflight_work_possible=True)
  if lock is not None:lock.close()
  save();write(output/'END.json',report)
 print(json.dumps({'status':report['status'],'worker':args.worker,'fits':len(report['outcomes']),'collections':len(report['collections'])}));return 0 if report['status']=='complete'else 1
if __name__=='__main__':raise SystemExit(main())
