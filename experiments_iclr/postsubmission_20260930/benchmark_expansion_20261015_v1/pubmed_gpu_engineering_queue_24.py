"""Seven bounded engineering invocations after the existing six-fit study.

Waits without a model/GPU context. Uses one GPU slot, stops on first failure,
never starts the24-fit pilot or6460-fit campaign, and never evaluates TEST.
"""
import argparse,fcntl,hashlib,importlib.util,json,os,signal,socket,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
GPU='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
E=ROOT/'experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1'
COMP='1c5d3669cec6681bbfbda045a9497805805c287ab3e03b4aca5aa65c0e3cf0dc'
PROBE='dab18f5eae00ad249bd3f9a1b855d51e53d7a0e3a8de55e70b3a3ab8ad9e36d2'
def write(p,x):
 temp=p.with_name(p.name+'.partial');temp.write_text(json.dumps(x,indent=2)+'\n');temp.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def require(v,msg):
 if not v:raise ValueError(msg)
def kill(p):
 if p is not None and p.poll() is None:
  os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
def main():
 require(socket.gethostname()=='anogena-2-0' and subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()==[GPU],'Wrong allocation')
 parser=argparse.ArgumentParser();parser.add_argument('--publication-commit',required=True);args=parser.parse_args()
 require(subprocess.check_output(['git','show',args.publication_commit+':'+str(Path(__file__).relative_to(ROOT))],cwd=ROOT)==Path(__file__).read_bytes(),'Published owner differs')
 q=E/'qualification/pubmed_cuda_queue_24';require(not q.exists(),'One queue only; preserve previous attempt');q.mkdir()
 report={'status':'waiting_for_existing_six_fits','owner_pid':os.getpid(),'owner_start_ticks':Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],'publication_commit':args.publication_commit,'scientific_fits':0,'TEST_accessed':False,'full_campaign_GO':False,'automatic_retries':0,'maximum_new_GPU_workers':1,'host_reservation_bytes':24*1024**3,'GPU_reservation_bytes':32*1024**3,'invocations':[]}
 child=None;started=time.monotonic()
 def save():report['wall_seconds']=time.monotonic()-started;write(q/'QUEUE.json',report)
 def stop(*args):raise InterruptedError('Owner interrupted')
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 save()
 try:
  # CPU-only bookkeeping imports; no torch or dataset construction in owner.
  sys.path.insert(0,str(E/'qualification/portablev4_link_queued_parent_authority_repair_v1'))
  from supervise import preflight,process,gpu_state
  spec=importlib.util.spec_from_file_location('_prior_scientific_gate',E/'pilot_20261015_structural_private_risk_v1/run.py');gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
  prior=E/'results/structural_private_risk_v1';deadline=time.monotonic()+24*3600
  while not (prior/'END.json').exists():
   require(time.monotonic()<deadline,'Wait deadline; no dependent execution')
   parent=process(848485);require(parent is not None and parent['start_ticks']=='6089052668','Existing scientific owner missing without terminal metadata')
   time.sleep(30)
  end=json.loads((prior/'END.json').read_text());outcomes=json.loads((prior/'OUTCOMES.json').read_text());pilot=json.loads((E/'pilot_20261015_structural_private_risk_v1/SPEC.json').read_text())
  require(sha(E/'pilot_20261015_structural_private_risk_v1/MANIFEST.json')=='d2bb359dcfd43f945749bf38c57fd6dc82fc14a16499dc1ae370fff4099fc67c','Existing study source changed')
  gate.terminal_gate(end,outcomes,pilot['jobs'])
  report['prerequisite']={'END_sha256':sha(prior/'END.json'),'OUTCOMES_sha256':sha(prior/'OUTCOMES.json'),'completed_fits':6};save()
  # Same slot used by the scientific owner; wait briefly for its final cleanup.
  lock=(E/'analysis_outputs/selected_VALID_collection_gpu_slot.lock').open('a');lock_deadline=time.monotonic()+60
  while True:
   try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
   except BlockingIOError:
    require(time.monotonic()<lock_deadline,'GPU slot occupied after predecessor completion');time.sleep(2)
  comp=E/'research/pubmed_cheb_residual_composition_v1';probe=E/'research/pubmed_cheb_production_cost_probe_v1'
  require(sha(comp/'MANIFEST.json')==COMP and sha(probe/'MANIFEST.json')==PROBE,'Reviewed source manifest differs')
  planned=[('lifecycle_'+v,comp/'qualify.py',COMP,v,False) for v in ('uniform','structured','permuted')]
  planned += [('production_'+v,probe/'probe.py',PROBE,v,True) for v in ('M1','G0','uniform','structured')]
  for name,script,manifest,variant,cost in planned:
   peers=[];old=process(640428)
   if old is not None:
    require(old['start_ticks']=='6055039165','Old peer PID reused')
    peers=[{'pid':640428,'start_ticks':6055039165,'peak_host_RSS_bytes':16*1024**3}]
   host=preflight({'device':'cpu','peers':peers},{'minimum_host_headroom_after_peer_growth_bytes':24*1024**3,'host_RSS_reservation_bytes':24*1024**3,'disk_reservation_bytes':8*1024**3})
   gpu=gpu_state();growth=0
   for p in gpu['processes']:
    require(p['pid']==640428 and p['start_ticks']=='6055039165','Unadmitted GPU peer')
    require(p['gpu_memory_bytes']<=32*1024**3,'Old peer exceeds reserved growth')
    growth+=32*1024**3-p['gpu_memory_bytes']
   require(len(gpu['processes'])<=1 and gpu['free_bytes']-growth>=32*1024**3,'Insufficient GPU headroom after peer growth')
   target=q/name;command=[str(ROOT/'.venv/bin/python'),'-B','-u',str(script),'--project-root',str(ROOT),'--manifest-sha256',manifest,'--variant',variant,'--device','cuda:0','--output',str(target)]
   if cost:command.append('--root-resource-admitted')
   row={'name':name,'status':'starting','command':command,'host_preflight':host,'GPU_preflight':gpu,'GPU_peer_growth_bytes':growth,'watchdog_seconds':900,'TERM_cleanup_seconds':10};report['invocations'].append(row)
   env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
   with (q/(name+'.log')).open('xb') as log:child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   row.update(status='running',child=process(child.pid));report['status']='running';save();begin=time.monotonic();limit=begin+900
   while child.poll() is None:
    if time.monotonic()>=limit:row['timed_out']=True;kill(child);break
    current=process(child.pid)
    if current and current['rss_bytes']>24*1024**3:row['resource_limit']='host_RSS';kill(child);break
    try:state=gpu_state()
    except ValueError as error:
     if str(error)=='GPU process could not be authenticated' and child.poll() is not None:break
     raise
    for p in state['processes']:
     require(p['pid'] in (child.pid,640428),'Third GPU model worker appeared')
     if p['pid']==640428:require(p['start_ticks']=='6055039165','GPU peer PID reused')
     if p['pid']==child.pid and p['gpu_memory_bytes']>32*1024**3:row['resource_limit']='GPU';kill(child);break
    try:child.wait(timeout=min(10,max(.01,limit-time.monotonic())))
    except subprocess.TimeoutExpired:pass
   row.update(exit_code=child.returncode,wall_seconds=time.monotonic()-begin)
   result=target/'RESULT.json';row['result']=json.loads(result.read_text()) if result.exists() else None
   row['status']='passed' if child.returncode==0 and row['result'] and row['result']['status']=='passed' and not row.get('timed_out') and not row.get('resource_limit') else 'failed';save()
   require(row['status']=='passed','Engineering invocation failed; no automatic retry or later invocation')
   child=None
  report['status']='passed';save();lock.close()
 except BaseException as error:
  kill(child);report.update(status='failed',failure={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc(limit=12)});save()
 return 0 if report['status']=='passed' else 1
if __name__=='__main__':raise SystemExit(main())
