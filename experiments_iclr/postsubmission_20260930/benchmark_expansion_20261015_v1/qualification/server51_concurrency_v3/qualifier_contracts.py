"""Finite server51 engineering admission, source/data custody and capacity."""
import fcntl,hashlib,importlib.util,json,os,socket,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
E=HERE.parent.parent
ROOT=Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
PHASE_RELATIVE='experiments_iclr/postsubmission_20260930'
PHASE=ROOT/PHASE_RELATIVE
PILOT=E/'research/pubmed_three_gpu_queue_v1'
UUIDS=['GPU-9c6c20fa-a620-479d-f72f-163ac74deef2','GPU-30283973-140d-9e7b-e780-0fcc487dec54']
WARMUP_EPOCHS=10
TIMED_EPOCHS=100
PHASES={'native_alone':['native'],'structured_alone':['structured'],'concurrent':['native','structured']}
def require(ok,message):
 if not ok:raise ValueError(message)
def pin(path):
 path=Path(path).resolve();raw=path.read_bytes();return {'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def safe(path,root=ROOT):
 path=Path(path).resolve();require(path.is_relative_to(root.resolve()),'Deliberate file access escapes project');return path
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
 path=Path(path);temporary=path.with_name(path.name+'.tmp-'+str(os.getpid()));temporary.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');temporary.replace(path)
def verify(receipt):
 path=safe(receipt['path']);require(pin(path)==receipt,'Exact receipt changed: '+str(path));return read(path)
def identity(pid):
 try:
  p=Path('/proc')/str(pid);f=(p/'stat').read_text().rsplit(')',1)[1].split()
  if f[0]=='Z':return None
  return {'pid':int(pid),'UID':p.stat().st_uid,'start_ticks':int(f[19]),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
 except (FileNotFoundError,ProcessLookupError):return None
def rss(pid):
 try:return int(next(x.split()[1] for x in Path('/proc',str(pid),'status').read_text().splitlines() if x.startswith('VmRSS:')))*1024
 except (FileNotFoundError,StopIteration):return 0
def guard():
 require(socket.gethostname()=='olig','Only verified server51, before project reads')
 actual=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()
 require(actual==UUIDS,'Exact verified two-GPU inventory required')
 require(HERE==ROOT/PHASE_RELATIVE/'benchmark_expansion_20261015_v1/qualification/server51_concurrency_v3','Exact dedicated source layout required')
def cache_aware_available(host_available,groups):
 available=host_available;details=[]
 for group in groups:
  maximum,current,stats=group['maximum'],group['current'],group['stats'];reclaimable=max(int(stats.get('inactive_file',0))-int(stats.get('shmem',0)),0)
  value=host_available if maximum=='max' else min(int(maximum),max(int(maximum)-current,0)+reclaimable)
  available=min(available,value);details.append(dict(group,cache_aware_available_bytes=value))
 return available,details
def capacity(admission,arms):
 r=admission['resources'];gpu=admission['physical_gpu_uuid']
 free=int(subprocess.check_output(['nvidia-smi','--id='+gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=10).strip())*1024**2
 host=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
 groups=[]
 for value in r['cgroup_v2_memory_paths']:
  p=Path(value);require(p.is_absolute() and p.is_relative_to('/sys/fs/cgroup') and '..' not in p.parts,'Qualified cgroup path required')
  groups.append({'path':str(p),'maximum':(p/'memory.max').read_text().strip(),'current':int((p/'memory.current').read_text()),'stats':{k:int(v) for k,v in (line.split() for line in (p/'memory.stat').read_text().splitlines())}})
 available,details=cache_aware_available(host,groups);fs=os.statvfs(ROOT);disk=fs.f_bavail*fs.f_frsize
 need_gpu=sum(r[arm]['GPU_bytes'] for arm in arms)+r['GPU_headroom_bytes']+r['external_GPU_growth_bytes']
 need_ram=sum(r[arm]['RAM_bytes'] for arm in arms)+r['host_RAM_headroom_bytes']+r['external_RAM_growth_bytes']
 need_disk=r['disk_reserved_bytes']+r['disk_headroom_bytes']
 ok=free>=need_gpu and available>=need_ram and disk>=need_disk
 return ok,{'free_GPU_bytes':free,'needed_GPU_bytes':need_gpu,'available_RAM_bytes':available,'needed_RAM_bytes':need_ram,
  'free_disk_bytes':disk,'needed_disk_bytes':need_disk,'memory_accounting':details,'other_process_PID_whitelist':False,'host_settings_changed':False}
def preflight(path,sha,full=False):
 guard();ap=safe(path);require(pin(ap)['sha256']==sha,'Exact root-reviewed admission SHA required');a=read(ap)
 require(a['schema']=='server51-pubmed-throughput-pair-admission-v1' and a['approved'] is True and a['engineering_only'] is True and a['TEST_access'] is False,'Explicit engineering admission required')
 require(a['worker'] in ('server51/gpu0','server51/gpu1'),'Only authorized server51 workers')
 require(a['physical_gpu_uuid']==UUIDS[int(a['worker'][-1])],'Worker/UUID mismatch')
 require(Path(sys.prefix).resolve()==(ROOT/'.venv').resolve() and sys.prefix!=sys.base_prefix,'Repository venv required')
 require(sys.executable==a['python_executable'] and Path(a['python_executable'])==ROOT/'.venv/bin/python','Exact lexical interpreter required')
 require(a['warmup_epochs']==WARMUP_EPOCHS and a['timed_epochs']==TIMED_EPOCHS and a['threads_per_child']==2,'Frozen timing protocol required')
 require(a['maximum_child_wall_seconds']==1800 and a['maximum_owner_wall_seconds']==7200,'Finite source watchdogs required')
 require(pin(HERE/'MANIFEST.json')==a['source_manifest'],'Exact qualifier source manifest required')
 for row in read(HERE/'MANIFEST.json')['files']:
  actual=pin(HERE/row['path']);require(all(actual[k]==row[k] for k in ('bytes','sha256')),'Qualifier source changed')
 require(pin(PILOT/'MANIFEST.json')==a['pilot_source_manifest'],'Frozen pilot source changed')
 actual=verify(a['actual_target_map']);require(actual['schema']=='root-reviewed-pubmed-three-GPU-map-v1','Exact existing target map required')
 require(actual['queue_manifest_sha256']==a['pilot_source_manifest']['sha256'],'Prior map must bind the same immutable pilot source')
 host=actual['workers'][a['worker']];require(host['physical_gpu_uuid']==a['physical_gpu_uuid'] and host['project_root']==str(ROOT) and host['phase_relative']==PHASE_RELATIVE,'Prior host identity/layout changed')
 terminal=verify(a['prior_terminal']);claim=verify(a['prior_owner_claim'])
 expected=E/'results/pubmed_three_gpu_v1'/a['worker'].replace('/','_')
 require(Path(a['prior_terminal']['path'])==expected/'END.json' and Path(a['prior_owner_claim']['path'])==expected/'OWNER_CLAIM.json','Exact prior PubMed owner artifacts required')
 require(terminal['worker']==a['worker'] and terminal['status'] in ('complete','failed') and not terminal.get('cleanup_error') and not terminal.get('unrecorded_inflight_work_possible'),'Prior owner must be terminal with completed child cleanup')
 require(claim['worker']==a['worker'] and claim['target_map']==a['actual_target_map'],'Prior owner/map binding differs')
 current=identity(claim['pid'])
 require(current is None or current['start_ticks']!=claim['identity']['start_ticks'],'Prior PubMed owner remains live')
 r=a['resources'];require(r['cgroup_v2_memory_paths'] and r['all_effective_ancestor_limits_accounted'] is True and r['host_CPU_slots']>=4,'Reviewed aggregate host limits required')
 for name in ('native','structured'):
  require(all(type(r[name][k]) is int and r[name][k]>0 for k in ('GPU_bytes','RAM_bytes')),'Measured positive per-arm envelopes required')
 for name in ('GPU_headroom_bytes','external_GPU_growth_bytes','host_RAM_headroom_bytes','external_RAM_growth_bytes','disk_reserved_bytes','disk_headroom_bytes'):
  require(type(r[name]) is int and r[name]>=0,'Explicit conservative capacity reserve required')
 require(a['resource_evidence'],'Actual root-reviewed resource witnesses required')
 for receipt in a['resource_evidence']:
  safe(receipt['path']);require(pin(receipt['path'])==receipt,'Resource witness bytes changed')
 output=safe(a['output']);require(output.parent==HERE/'results','Fresh dedicated output under qualifier/results required')
 if full:
  for row in read(PILOT/'MANIFEST.json')['files']:
   actualpin=pin(PILOT/row['path']);require(all(actualpin[k]==row[k] for k in ('bytes','sha256')),'Frozen pilot source member changed')
  for row in read(PILOT/'SOURCE_BINDINGS.json')['files']:
   actualpin=pin(safe(PHASE/row['path_relative_to_phase']));require(all(actualpin[k]==row[k] for k in ('bytes','sha256')),'Frozen native dependency changed')
  inv=read(PILOT/'CANONICAL.json');origin=Path(inv['canonical_root'])/inv['canonical_phase_relative']
  for row in read(PILOT/'PARENT_INPUTS.json')['data_pins']:
   path=safe(PHASE/Path(row['path']).relative_to(origin));actualpin=pin(path)
   require(all(actualpin[k]==row[k] for k in ('bytes','sha256')),'Frozen PubMed provider bytes changed')
 return a,host,output,safe(ROOT/host['slot_lock_relative']),ap
