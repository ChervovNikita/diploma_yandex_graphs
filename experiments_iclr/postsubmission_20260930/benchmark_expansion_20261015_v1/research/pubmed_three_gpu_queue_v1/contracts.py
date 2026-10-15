"""Root-reviewed three-worker map and finite source/data custody."""
import fcntl,json,os,socket,subprocess,sys
from pathlib import Path
from binding import HERE,E,require,pin,load_inventory,shard
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
 p=Path(path);t=p.with_name(p.name+'.partial');t.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');t.replace(p)
def safe(path,root):
 p=Path(path).resolve();require(p.is_relative_to(Path(root).resolve()),'Project-contained operation required');return p
def verify(value,root):
 p=safe(value['path'],root);require(all(pin(p)[k]==value[k]for k in('bytes','sha256')),'Receipt changed: '+str(p));return p
def process(pid):
 try:
  f=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
  return None if f[0]=='Z'else {'pid':int(pid),'start_ticks':int(f[19]),'rss_bytes':int(f[21])*os.sysconf('SC_PAGE_SIZE')}
 except (FileNotFoundError,ProcessLookupError):return None
def guard(host):
 require(socket.gethostname()==host['expected_hostname'],'Unexpected host')
 uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()
 require(uuids==host['expected_gpu_uuids']and host['physical_gpu_uuid']in uuids,'Unexpected physical GPU inventory')
def request(path,sha,worker,full=False):
 p=Path(path).resolve();require(pin(p)['sha256']==sha,'Exact root target-map pin required');a=read(p)
 inv=load_inventory();require(worker in inv['assignments']and set(a['workers'])==set(inv['assignments']),'Only three frozen workers exist')
 require(len({(h['expected_hostname'],h['physical_gpu_uuid'])for h in a['workers'].values()})==3,'Workers must map to three distinct physical host/GPU pairs')
 host=a['workers'][worker];root=Path(host['project_root']).resolve()
 guard(host);safe(p,root);require(HERE==root/host['phase_relative']/'benchmark_expansion_20261015_v1/research/pubmed_three_gpu_queue_v1','Source layout differs')
 require(a['schema']=='root-reviewed-pubmed-three-GPU-map-v1'and a['approved']is True and a['new_fits']==24 and a['TEST_access']is False and a['full_campaign_GO']is False,'Explicit bounded scientific approval required')
 require(host['runtime_qualified']is True and host['resource_admission_approved']is True and host['evidence'],'Actual per-host runtime/resource review required')
 require(pin(HERE/'MANIFEST.json')['sha256']==a['queue_manifest_sha256']and pin(HERE/'CANONICAL.json')['sha256']==a['canonical_inventory_sha256'],'Frozen queue/inventory changed')
 for row in read(HERE/'MANIFEST.json')['files']:
  now=pin(HERE/row['path']);require(all(now[k]==row[k]for k in('bytes','sha256')),'Queue member changed')
 require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==a['publication_commit'],'Published host checkout differs')
 require(Path(sys.executable).resolve()==Path(host['python_executable']).resolve(),'Reviewed interpreter required')
 require(Path(sys.prefix).resolve()==(root/'.venv').resolve(),'Reviewed repository virtualenv prefix required')
 for receipt in host['evidence']:verify(receipt,root)
 if full:
  phase=root/host['phase_relative']
  for row in read(HERE/'SOURCE_BINDINGS.json')['files']:
   actual=pin(safe(phase/row['path_relative_to_phase'],root));require(all(actual[k]==row[k]for k in('bytes','sha256')),'Frozen dependency changed')
  for row in read(HERE/'PARENT_INPUTS.json')['data_pins']:
   rel=Path(row['path']).relative_to(Path(inv['canonical_root'])/inv['canonical_phase_relative']);actual=pin(safe(phase/rel,root))
   require(all(actual[k]==row[k]for k in('bytes','sha256')),'Exact data/basis/role bytes changed')
 return a,host,root,shard(inv,worker),p
def resource_check(host,root):
 for peer in host['allowed_GPU_peers']:
  p=process(peer['pid']);require(p is None or p['start_ticks']==peer['start_ticks'],'GPU peer PID reused')
 apps=subprocess.check_output(['nvidia-smi','--id='+host['physical_gpu_uuid'],'--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=10)
 allowed={p['pid']:p for p in host['allowed_GPU_peers']}
 for line in apps.splitlines():
  if line.strip():
   pid,memory=map(int,line.split(','));p=process(pid)
   require(pid in allowed and p is not None and p['start_ticks']==allowed[pid]['start_ticks']and memory*1024**2<=allowed[pid]['maximum_GPU_bytes'],'Unadmitted assigned-GPU peer')
 free=int(subprocess.check_output(['nvidia-smi','--id='+host['physical_gpu_uuid'],'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=10).strip())*1024**2
 require(type(host['GPU_peer_growth_reserve_bytes'])is int and free-host['GPU_peer_growth_reserve_bytes']>=host['reserved_GPU_bytes'],'GPU headroom below approved envelope')
 available=int(next(x.split()[1]for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:')))*1024
 c=Path('/sys/fs/cgroup')
 if (c/'memory.max').exists()and (limit:=(c/'memory.max').read_text().strip())!='max':
  stats=dict(x.split()for x in(c/'memory.stat').read_text().splitlines());clean=max(0,min(int(stats.get('file',0)),int(stats.get('active_file',0))+int(stats.get('inactive_file',0)))-sum(int(stats.get(k,0))for k in('shmem','file_dirty','file_writeback','unevictable')))
  available=min(available,max(0,int(limit)-int((c/'memory.current').read_text()))+3*clean//4)
 require(type(host['host_peer_growth_reserve_bytes'])is int and available-host['host_peer_growth_reserve_bytes']>=host['host_shared_reserved_bytes'],'Host headroom below approved shared envelope')
 fs=os.statvfs(root);require(fs.f_bavail*fs.f_frsize>=host['disk_reserved_bytes'],'Disk headroom below approved envelope')
 return {'available_host_bytes':available,'free_assigned_GPU_bytes':free,'no_host_settings_changed':True}
def child_context(host,root,output,worker,map_pin):
 require(os.environ.get('CUDA_VISIBLE_DEVICES')==host['physical_gpu_uuid'],'Exact assigned physical CUDA mask required before Torch')
 claim=read(output/'OWNER_CLAIM.json');current=process(claim['pid'])
 require(claim['worker']==worker and claim['target_map']==map_pin and claim.get('source_data_verified')is True and claim.get('source_manifest')==pin(HERE/'MANIFEST.json')and current is not None and current['start_ticks']==claim['identity']['start_ticks']and os.getppid()==claim['pid'],'Only the live claimed source/data-verifying owner may launch its child')
 with safe(root/host['slot_lock_relative'],root).open('a')as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return
  fcntl.flock(lock,fcntl.LOCK_UN);raise ValueError('Live owner must hold its GPU slot')
