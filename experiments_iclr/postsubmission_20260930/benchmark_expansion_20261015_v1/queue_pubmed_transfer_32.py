"""Published one-owner PubMed24 queue; source only until root invokes it."""
import argparse,fcntl,hashlib,json,os,signal,socket,subprocess,sys,time,traceback
from pathlib import Path
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');G='GPU-44039938-fd82-41d2-fefd-de71514e2fac';GiB=1024**3
E=R/'experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1';P=E/'research/pubmed_cheb_transfer_driver_v1'
def require(ok,message):
 if not ok:raise ValueError(message)
def pin(p):
 p=Path(p).resolve();require(p.is_relative_to(R),'Project-contained file required');b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def read(p):return json.loads(Path(p).read_text())
def fixed(p,sha):require(pin(p)['sha256']==sha,'Frozen source/evidence changed: '+str(p));return p
def write(p,value):
 t=p.with_name(p.name+'.partial');t.write_text(json.dumps(value,indent=2)+'\n');t.replace(p)
def main():
 require(socket.gethostname()=='anogena-2-0' and subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()==[G],'Wrong allocation before project reads')
 parser=argparse.ArgumentParser();parser.add_argument('--publication-commit',required=True);args=parser.parse_args()
 require(Path(sys.prefix).resolve()==(R/'.venv').resolve(),'Original repository virtualenv required')
 require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()==args.publication_commit,'Actual published checkout required')
 require(subprocess.check_output(['git','show',args.publication_commit+':'+str(Path(__file__).resolve().relative_to(R))],cwd=R)==Path(__file__).read_bytes(),'Published owner source differs')
 q=E/'qualification/pubmed_transfer_queue_32';q.mkdir(exist_ok=False);started=time.monotonic();driver=None;lock=None;workers={}
 study=E/'results/pubmed_cheb_structural_transfer_v1';jobs=study.with_name(study.name+'_jobs');bpath=E/'qualification/remaining_buddy_seed1_runtime_31/SUPERVISOR.json'
 report={'status':'waiting','owner_pid':os.getpid(),'owner_start_ticks':Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],'source':pin(__file__),'publication_commit':args.publication_commit,'new_fits':24,'new_GPU_workers_maximum':1,'automatic_retries':0,'TEST_accessed':False,'full_campaign_GO':False,'predecessor_wait_seconds':7200,'driver_watchdog_seconds':345600,'per_fit_watchdog_seconds':43200}
 def save():report['wall_seconds']=time.monotonic()-started;write(q/'QUEUE.json',report)
 def stop(*_):raise InterruptedError('Owner interrupted; preserve without retry')
 for s in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(s,stop)
 def own(p):
  try:args=(Path('/proc')/str(p['pid'])/'cmdline').read_bytes().split(b'\0');parent=int((Path('/proc')/str(p['pid'])/'stat').read_text().rsplit(')',1)[1].split()[1])
  except (FileNotFoundError,ProcessLookupError):return None
  return parent if str(E/'runtime/run_fit.py').encode() in args and any(v.startswith((str(jobs)+'/').encode())for v in args)else -1
 def remember():
  if (study/'OUTCOMES.json').exists():
   for row in read(study/'OUTCOMES.json'):
    if row.get('worker_pid') and row.get('worker_start_ticks'):
     p=process(row['worker_pid'])
     if p and p['start_ticks']==str(row['worker_start_ticks']) and own(p)not in (None,-1):workers[(p['pid'],p['start_ticks'])]=p
 def terminate():
  if driver is None:return
  remember()
  if driver.poll()is None:
   driver.terminate()
   try:driver.wait(timeout=30)
   except subprocess.TimeoutExpired:os.killpg(driver.pid,signal.SIGKILL);driver.wait(timeout=10);report['driver_forced_KILL']=True
  remember()
  for pid,ticks in workers:
   p=process(pid)
   if p and p['start_ticks']==ticks and own(p)not in (None,-1):
    try:os.killpg(pid,signal.SIGTERM)
    except ProcessLookupError:continue
    for _ in range(100):
     current=process(pid)
     if current is None or current['start_ticks']!=ticks:break
     time.sleep(.1)
    else:
     try:os.killpg(pid,signal.SIGKILL);report['worker_forced_KILL']=True
     except ProcessLookupError:pass
 save()
 try:
  helper=E/'qualification/portablev4_link_queued_parent_authority_repair_v1'
  fixed(helper/'supervise.py','a8056e0e29045031ae2302d94f67328273274c9084f98f6c163e6eeb3677a5d6');fixed(helper/'support.py','19ca77d9b69f0d49e932b5d0d14e5774328e9e4e0533d31ecc648d1e89bf6254')
  sys.path.insert(0,str(helper));from supervise import process,preflight,gpu_state
  require(not study.exists()and not jobs.exists(),'Preserve existing study; no retry or duplicate')
  template=fixed(E/'analysis/pubmed_cheb_fit_admission_candidate_v2/FITS_ADMISSION_CANDIDATE.json','c38eb1d24abe390d9bfef0c1d3df3d84c70eab1dd75585bcb59532ad1d5ce0ae');a=read(template)
  fixed(P/'MANIFEST.json',a['manifest_sha256']);fixed(P/'SPEC.json',a['SPEC_sha256']);deadline=started+7200
  while not bpath.exists()or read(bpath).get('status')in ('preflight','running'):
   require(time.monotonic()<deadline,'BUDDY predecessor wait exceeded2h');time.sleep(10)
  b=read(bpath);fixed(E/'run_remaining_runtime_check_31.py','120e5a8412c127ffb0e98e022f4d0724484551dba224b1678f2b173b25d86997')
  require(b.get('status')=='passed'and b.get('case')=='buddy_seed1'and b.get('exit_code')==0 and b.get('native_child_started')is True and b.get('automatic_retries')==0 and b.get('TEST_accessed')is False and b.get('full_campaign_GO')is False and b.get('launcher_sha256')=='120e5a8412c127ffb0e98e022f4d0724484551dba224b1678f2b173b25d86997','Exact BUDDY supervisor success required')
  require(not any(b.get(k)for k in ('failure','timed_out','terminal_resource_violation','cleanup_incomplete','forced_KILL_after_TERM')),'BUDDY supervision failed')
  bp=E/'qualification/remaining_buddy_seed1_runtime_31/native/RESULT.json';br=read(bp)
  require(b['child_result_receipt']==pin(bp)and b['result']==br and br.get('status')=='passed'and br.get('case')=='buddy_seed1'and br.get('native_qualification')is True and br.get('TEST_accessed')is False and br.get('source_manifest',{}).get('sha256')=='518784a9e83f4ba92c4f7bff55705999a787c7a67ff69c0cc9c09b5530550d38','Exact composed BUDDY child result required')
  qp=fixed(E/'qualification/pubmed_cuda_queue_24/QUEUE.json','f269d073b59856458b77c8d483a8618f3d8dba9a2bfecf6f3bb1c7d8137df92c');v=read(qp)
  names=['lifecycle_'+x for x in ('uniform','structured','permuted')]+['production_'+x for x in ('M1','G0','uniform','structured')]
  require(v['status']=='passed'and v['owner_pid']==849490 and str(v['owner_start_ticks'])=='6089402542'and v['scientific_fits']==0 and v['TEST_accessed']is False and [r['name']for r in v['invocations']]==names and all(r['status']=='passed'and r['exit_code']==0 and not r.get('timed_out')and not r.get('resource_limit')for r in v['invocations']),'All7 exact PubMed checks required')
  for receipt in a['lifecycle_evidence']+a['production_timing_evidence']:require(pin(receipt['path'])==receipt,'Actual lifecycle/timing evidence changed')
  report['prerequisites']={'BUDDY':pin(bpath),'BUDDY_child':pin(bp),'PubMed':pin(qp),'candidate':pin(template)}
  lock=Path(a['slot_lock_path']).open('a')
  while True:
   try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
   except BlockingIOError:require(time.monotonic()<deadline,'Existing owned slot did not release within2h');time.sleep(2)
  peers=[];old=process(640428)
  if old:require(old['start_ticks']=='6055039165','GRU PID reused');peers=[{'pid':640428,'start_ticks':6055039165,'peak_host_RSS_bytes':16*GiB}]
  resources=preflight({'device':'cpu','peers':peers},{'minimum_host_headroom_after_peer_growth_bytes':8*GiB,'host_RSS_reservation_bytes':8*GiB,'disk_reservation_bytes':8*GiB});state=gpu_state();growth=0
  require(state['uuid']==G and len(state['processes'])<=1,'Sole allocation and GRU peer required')
  for p in state['processes']:require(peers and p['pid']==640428 and p['start_ticks']=='6055039165'and p['gpu_memory_bytes']<=32*GiB,'Unadmitted GPU peer');growth+=32*GiB-p['gpu_memory_bytes']
  require(state['free_bytes']-growth>=8*GiB,'Own8GiB GPU envelope unavailable after peer growth');resources.update(GPU=state,GPU_peer_growth_bytes=growth);write(q/'RESOURCES.json',resources)
  a.update(approved=True,publication_approved=True,resource_admission_approved=True,all_numerical_lifecycles_approved=True,isolated_production_timing_approved=True,publication_commit=args.publication_commit,allowed_other_GPU_processes=[{'pid':p['pid'],'start_ticks':p['start_ticks']}for p in peers],evidence_review_note='Root-published queued invocation; exact prerequisites and fresh resource checks passed. See sibling QUEUE.json/RESOURCES.json.')
  ap=q/'ADMISSION.json';write(ap,a);report.update(admission=pin(ap),resources=pin(q/'RESOURCES.json'),status='starting_driver');save();lock.close();lock=None
  command=[str(R/'.venv/bin/python'),'-B','-u',str(P/'driver.py'),'--project-root',str(R),'--mode','fits','--admission',str(ap),'--admission-sha256',pin(ap)['sha256']]
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
  with (q/'driver.log').open('xb')as log:driver=subprocess.Popen(command,cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  report.update(status='running',driver=process(driver.pid),command=command);launched=time.monotonic();save()
  while driver.poll()is None:
   require(time.monotonic()-launched<345600,'Total driver watchdog96h');remember();old=process(640428)
   if old:require(peers and old['start_ticks']=='6055039165'and old['rss_bytes']<=16*GiB,'GRU host envelope/identity changed')
   try:state=gpu_state()
   except ValueError as error:
    if str(error)=='GPU process could not be authenticated'and driver.poll()is not None:break
    require(str(error)=='GPU process could not be authenticated','GPU telemetry failed');time.sleep(.2);state=gpu_state()
   own_gpu=0
   for p in state['processes']:
    if p['pid']==640428:require(peers and p['start_ticks']=='6055039165'and p['gpu_memory_bytes']<=32*GiB,'GRU GPU envelope changed')
    else:
     parent=own(p)
     if parent is None:continue
     require(parent==driver.pid,'Unadmitted GPU worker');workers[(p['pid'],p['start_ticks'])]=p;own_gpu+=p['gpu_memory_bytes']
   live=[p for pid,ticks in workers if (p:=process(pid))and p['start_ticks']==ticks];parent=process(driver.pid)
   require(state['uuid']==G and len(state['processes'])<=2 and len(live)<=1 and own_gpu<=8*GiB and sum(p['rss_bytes']for p in live)+(parent['rss_bytes']if parent else 0)<=8*GiB,'Own worker/resource envelope exceeded')
   require(sum(p.stat().st_size for folder in (q,study,jobs)if folder.exists()for p in folder.rglob('*')if p.is_file())<=8*GiB,'Own output disk envelope exceeded')
   report.update(driver_wall_seconds=time.monotonic()-launched,observed_own_GPU_bytes=own_gpu);save()
   try:driver.wait(timeout=10)
   except subprocess.TimeoutExpired:pass
  require(driver.returncode==0,'Scientific driver failed; preserve without retry');end=read(study/'END.json');outcomes=read(study/'OUTCOMES.json')
  require(end['status']=='complete'and end.get('failure')is None and end['TEST_accessed']is False and end['admission']==pin(ap)and len(outcomes)==24 and all(r['status']=='complete'and r['returncode']==0 for r in outcomes),'All24 scientific terminal outcomes required')
  report.update(status='complete',END=pin(study/'END.json'),OUTCOMES=pin(study/'OUTCOMES.json'))
 except BaseException as error:report.update(status='failed',failure={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc(limit=8)})
 finally:
  for s in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(s,signal.SIG_IGN)
  try:terminate()
  except BaseException as error:report.update(status='failed',cleanup_failure=str(error),unrecorded_inflight_work_possible=True)
  if lock is not None:lock.close()
  report['driver_exit_code']=None if driver is None else driver.returncode;save()
 print(json.dumps(report));return 0 if report['status']=='complete'else 1
if __name__=='__main__':raise SystemExit(main())
