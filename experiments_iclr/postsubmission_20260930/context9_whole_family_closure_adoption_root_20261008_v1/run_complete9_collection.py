from pathlib import Path
import os,sys,socket,subprocess,json,hashlib,datetime,time,signal
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';S=P/'context9_whole_family_collection_entry_source_20261008_v1';A=P/'context_positive_stage1_scientific_activation_root_20261008_v1';D=P/'context9_whole_family_closure_adoption_root_20261008_v1'
assert Path.cwd()==R and socket.gethostname()=='anogena-2-0'
GPU='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()==[GPU]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def binding(p):return dict(path=str(p.relative_to(P)),sha256=sha(p),bytes=p.stat().st_size)
def write(p,x):
 with p.open('x') as h:json.dump(x,h,indent=2);h.write('\n')
def proc(pid):
 try:
  s=Path('/proc',str(pid),'stat').read_text();f=s[s.rfind(')')+2:].split();return dict(pid=pid,start_ticks=int(f[19]),state=f[0],group=int(f[2]),session=int(f[3]))
 except FileNotFoundError:return None
family=read(A/'FAMILY_CLOSURE.json');parent=read(A/'PARENT_OWNER.json')
assert family['complete'] and family['fixed_total']==9 and len(family['completed'])==9 and parent['pid']==526200 and parent['start_ticks']==6021896966 and proc(parent['pid']) is None
pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=10).splitlines();pids={x.strip() for x in pids}
assert str(parent['pid']) not in pids
D.mkdir(exist_ok=True);terminal=read(S/'TERMINAL_EVIDENCE_TEMPLATE_DISABLED.json');terminal.update(complete=True,root_observed=True,owner_and_children_terminal=True,parent_absent=True,parent_no_CUDA_rows=True,parent_identity=parent,observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),family_closure=binding(A/'FAMILY_CLOSURE.json'),parent_owner=binding(A/'PARENT_OWNER.json'),template_only=False)
for row,closed in zip(terminal['children'],family['completed']):
 assert row['cell_id']==closed['cell_id'];o=A/'logs'/(row['cell_id']+'.OWNER.json');e=A/'logs'/(row['cell_id']+'.EXIT.json');er=read(e);child=er['child'];assert er==closed['exit_receipt'] and er['exit_code']==0 and er['child_reaped'] and not er['timeout'] and not er['signals'];assert proc(child['pid']) is None and str(child['pid']) not in pids
 row.update(identity=child,owner_receipt=binding(o),exit_receipt=binding(e),child_absent=True,child_no_CUDA_rows=True,child_reaped=True,exit_authority=er['exit_authority'])
t=D/'TERMINAL_EVIDENCE.json';write(t,terminal)
cfg=read(S/'RELEASE_TEMPLATE_DISABLED.json')
for k in ('enabled','root_execution_authorized','source_review_approved','entire9_complete','owner_and_children_terminal','trusted_checkpoint_deserialization_authorized','runtime_resource_readiness_confirmed','collection_and_analysis_cost_charged','external_owned_bound_confirmed'):cfg[k]=True
cfg.update(template_only=False,collection_manifest_sha256=sha(S/'MANIFEST.json'),family_closure=binding(A/'FAMILY_CLOSURE.json'),parent_owner=binding(A/'PARENT_OWNER.json'),terminal_evidence=binding(t),output='context9_whole_family_collection_execution_root_20261008_v1',owned_GPU_memory_cap_bytes=20*1024**3,minimum_fresh_free_GPU_bytes=24*1024**3,minimum_fresh_disk_free_bytes=2*1024**3,external_active_seconds=300,external_cleanup_seconds=10,external_hard_seconds=310)
pins=read(S/'SOURCE_BINDINGS.json');assert sha(S/'MANIFEST.json')=='704365c25a9e8723eeec31bfc7c568b91a7ca3e30966e80af81da6b4dec69524';f=D/'RELEASE_ROOT.json';write(f,cfg)
env=dict(os.environ,PYTHONPATH=pins['runtime']['PYTHONPATH'],CUDA_VISIBLE_DEVICES=GPU,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2');env.pop('PYTHONHOME',None)
argv=[pins['runtime']['python'],'-B',str(S/'collect_closed9.py'),'--release',str(f),'--release-sha256',sha(f)];began=time.monotonic();actions=[];max_rss=max_gpu=0;reason=None;status=None
with (D/'collection.stdout.log').open('xb') as out,(D/'collection.stderr.log').open('xb') as err:
 child=subprocess.Popen(argv,cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True);saved=proc(child.pid);assert saved and saved['group']==saved['session']==child.pid;write(D/'COLLECTION_OWNER.json',dict(owner=saved,argv=argv,release_sha256=sha(f)))
 def stop(sig):
  actual=proc(child.pid)
  if actual and actual['start_ticks']==saved['start_ticks'] and actual['group']==saved['group']:
   os.killpg(saved['group'],sig);actions.append(int(sig))
 try:
  while child.poll() is None:
   if time.monotonic()-began>=300:reason='external active deadline';break
   text=Path('/proc',str(child.pid),'status').read_text();rss=next((int(x.split()[1])*1024 for x in text.splitlines() if x.startswith('VmRSS:')),0);max_rss=max(max_rss,rss)
   rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=5).splitlines();mem=sum(int(v[1].strip())*1024**2 for v in (x.split(',') for x in rows) if len(v)==2 and v[0].strip()==str(child.pid) and v[1].strip().isdigit());max_gpu=max(max_gpu,mem)
   if rss>8*1024**3 or mem>cfg['owned_GPU_memory_cap_bytes']:reason='owned resource cap';break
   try:status=child.wait(timeout=min(2,max(.01,300-(time.monotonic()-began))))
   except subprocess.TimeoutExpired:pass
  if reason:stop(signal.SIGTERM)
  try:status=child.wait(timeout=max(.01,310-(time.monotonic()-began)))
  except subprocess.TimeoutExpired:
   stop(signal.SIGKILL);status=child.wait(timeout=1)
 except BaseException:
  stop(signal.SIGTERM)
  try:child.wait(timeout=5)
  except subprocess.TimeoutExpired:stop(signal.SIGKILL);child.wait(timeout=1)
  raise
 finally:
  value=dict(exit_code=child.returncode,reaped=child.poll() is not None,owned_after=proc(child.pid),reason=reason,signals=actions,inclusive_seconds=time.monotonic()-began,max_sampled_RSS_bytes=max_rss,max_sampled_owned_GPU_bytes=max_gpu,hard_cap_exceeded=time.monotonic()-began>310,source_manifest_sha256=sha(S/'MANIFEST.json'),release_sha256=sha(f),training_updates=0,TEST_access=False)
  write(D/'COLLECTION_EXIT.json',value)
print(json.dumps(value))
if child.returncode or reason:print((D/'collection.stderr.log').read_text()[-6000:]);raise SystemExit(child.returncode or 1)
