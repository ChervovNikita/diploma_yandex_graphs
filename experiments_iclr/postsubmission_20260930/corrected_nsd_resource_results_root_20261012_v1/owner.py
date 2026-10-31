
import base64,hashlib,json,os,shutil,socket,subprocess,sys,time,zlib
from pathlib import Path
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
phase=repo/'experiments_iclr/postsubmission_20260930'

import signal
out=Path(__file__).resolve().parent
proposal=json.loads((phase/'corrected_nsd_resource_admission_root_20261012_v1/HANDOFF.json').read_text())
assert proposal['case_count']==3 and proposal['maximum_optimizer_updates']==0 and proposal['GO'] is False
def save(n,v):
 p=out/n;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def ident(pid):
 f=Path('/proc')/str(pid)/'stat'
 if not f.exists():return None
 a=f.read_text().rsplit(')',1)[1].split();return {'pid':pid,'start_ticks':int(a[19]),'pgid':int(a[2])}
def custody():
 p=phase/'private_GRU_acquisition_root_20261011_v1/owner_v1';start=json.loads((p/'OWNER_START.json').read_text());seen={}
 for k in ('owner','child'):
  expected=start[k];now=ident(expected['pid']);seen[k]=now
  if now is not None:assert all(now[x]==expected[x] for x in now)
  else:assert (p/'OWNER_END.json').exists()
 return seen
def checked(row):
 p=phase/row['path'];raw=p.read_bytes();assert p.resolve().is_relative_to(phase) and len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'];return p
save('ROOT_ENGINEERING_DECISION.json',{'exact_case_ids':[r['case']['case_id'] for r in proposal['cases']],'maximum_complete_TRAIN_callbacks':6,'maximum_actual_optimizer_updates':0,'scientific_fits':0,'GO':False,'concurrent_children':1,'retry':False,'scope':'The two corrected formerly nonfinite native NSD cases plus new cold geometry; no selectors, heldout targets or scores. Source helper40percent CUDA allocator ceiling and330second external child limit;64GiB RSS limit.'})
records=[]
for item in proposal['cases']:
 c=item['case'];helper=checked(item['helper']);admission=checked(item['admission']);before=custody()
 free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>34816
 target=out/('case_'+c['case_id']);command=[str(repo/'.venv/bin/python'),'-B','-u',str(helper),'--project-root',str(phase),'--admission',str(admission.relative_to(phase)),'--case-id',c['case_id'],'--output',str(target.relative_to(phase))]
 start=time.monotonic();terminated=None;RSS_peak=0
 with (out/(c['case_id']+'.log')).open('x') as log:
  child=subprocess.Popen(command,cwd=out,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='0'))
  owner=ident(child.pid);assert owner is not None and owner['pgid']==child.pid
  row={'case_id':c['case_id'],'child':owner,'Physics_before':before,'command':command};save(c['case_id']+'_OWNER.json',row)
  while child.poll() is None:
   now=ident(child.pid);assert now==owner
   status=Path('/proc')/str(child.pid)/'status'
   rss=sum(int(x.split()[1])*1024 for x in status.read_text().splitlines() if x.startswith('VmRSS:')) if status.exists() else 0
   RSS_peak=max(RSS_peak,rss);elapsed=time.monotonic()-start;current=custody()
   save('ACTIVE.json',{'case_id':c['case_id'],'elapsed_seconds':elapsed,'RSS_peak_bytes':RSS_peak,'child':now,'Physics':current,'scientific_fits':0})
   if elapsed>330:terminated='wall_limit'
   if rss>64*1024**3:terminated='RSS_limit'
   if terminated:
    assert ident(child.pid)==owner;os.killpg(owner['pgid'],signal.SIGTERM)
    try:child.wait(timeout=10)
    except subprocess.TimeoutExpired:
     assert ident(child.pid)==owner;os.killpg(owner['pgid'],signal.SIGKILL);child.wait(timeout=10)
    break
   time.sleep(3)
  code=child.wait()
 row.update(exit_code=code,child_absent=ident(child.pid) is None,Physics_after=custody(),elapsed_seconds=time.monotonic()-start,RSS_peak_bytes=RSS_peak,termination_reason=terminated)
 result=target/'RESULT.json'
 if result.exists():
  v=json.loads(result.read_text());assert v['actual_optimizer_updates']==v['scores_emitted']==v['heldout_target_files_opened']==0
  row['result']={'path':str(result.relative_to(phase)),'bytes':result.stat().st_size,'sha256':hashlib.sha256(result.read_bytes()).hexdigest()}
  row['resource']={k:v[k] for k in ['CUDA_peak_allocated_bytes','CUDA_peak_reserved_bytes','parameter_bytes','manual_moment_bytes','gradient_bytes','warm_median_seconds','random_complete_engine_restore','CPU_bank_export_proxy','full_graph_serving_seconds','source_complete_native_modules']}
 else:row['failure_log_tail']=(out/(c['case_id']+'.log')).read_text()[-6000:]
 records.append(row);save('PROGRESS.json',{'attempts':records,'scientific_fits':0,'GO':False})
closure={'attempts':records,'all_children_closed':all(r['child_absent'] for r in records),'scientific_fits':0,'actual_optimizer_updates':0,'GO':False,'automatic_remaining_queue':False};save('CLOSURE.json',closure);print(json.dumps(closure),flush=True)
