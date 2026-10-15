"""Guarded one-attempt NCNC interaction CPU admission and compact observation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

E = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(E))
import remote_control as remote
Q = E / 'qualification/interaction_link_ncnc_v1_cpu_v1'
OUTPUT = remote.REMOTE_EXPANSION + '/results/interaction_link_ncnc_v1_cpu_v1_attempt01'
GATE_SHA = 'bf32fd6f6aa0cce6fc7da77255964d2ae25923618a1d0288b50abca5ed4b9cc4'
SOURCE_SHA = 'e7fb265ea3c3650d7ba4c24157adda04b20cdba88ad46ebc54a498b9b892b0c1'

def save(name, value):
    path = Q / name
    if path.exists():
        raise ValueError('Preserve earlier receipt: ' + str(path))
    path.write_text(json.dumps(value, indent=2) + '\n')

def launch():
    if (Q/'LAUNCH.json').exists():
        raise ValueError('Existing attempt cannot be relaunched')
    code = f'''import datetime,hashlib,json,os,subprocess
from pathlib import Path
root=Path({remote.REMOTE_ROOT!r});exp=Path({remote.REMOTE_EXPANSION!r});source=exp/'qualification/interaction_link_ncnc_v1_cpu_v1'
out=Path({OUTPUT!r});log=Path(str(out)+'.driver.log')
assert not out.exists() and not log.exists(),'Preserve existing attempt'
prior=json.loads((exp/'results/context_affinity_ncnc_native_cpu_v1_attempt01/SUPERVISOR.json').read_text())
assert prior['status'] in ('complete','failed'),'Previous C3 must be terminal'
for folder,expected in [(source,{GATE_SHA!r}),(exp/'interaction_link_ncnc_v1',{SOURCE_SHA!r})]:
 assert hashlib.sha256((folder/'MANIFEST.json').read_bytes()).hexdigest()==expected
 for row in json.loads((folder/'MANIFEST.json').read_text())['files']:
  p=(folder/row['path']).resolve();assert p.is_relative_to(folder)
  b=p.read_bytes();assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256']
c=Path('/sys/fs/cgroup');current=int((c/'memory.current').read_text());maximum=(c/'memory.max').read_text().strip()
stats=dict(line.split() for line in (c/'memory.stat').read_text().splitlines())
host_available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
hard=host_available if maximum=='max' else max(int(maximum)-current,0)
reclaimable=max(int(stats.get('inactive_file',0))-int(stats.get('shmem',0)),0)
available=min(host_available,hard+reclaimable)
quota=(c/'cpu.max').read_text().split();capacity=len(os.sched_getaffinity(0)) if quota[0]=='max' else int(quota[0])/int(quota[1])
processes=[]
for line in subprocess.check_output(['ps','-u',str(os.getuid()),'-o','pid=,ppid=,rss=,args='],text=True).splitlines():
 f=line.strip().split(None,3)
 if len(f)==4:processes.append(dict(pid=int(f[0]),ppid=int(f[1]),rss_bytes=int(f[2])*1024,args=f[3]))
assert not any('context_affinity_ncnc_native_cpu_v1/qualify.py' in p['args'] and '-c ' not in p['args'] for p in processes),'C3 child remains active'
buddy=[p for p in processes if 'buddy' in p['args'].lower() and ('qualify' in p['args'].lower() or 'supervis' in p['args'].lower()) and '-c ' not in p['args']]
selected={{p['pid'] for p in buddy}}
while True:
 more={{p['pid'] for p in processes if p['ppid'] in selected}}
 if more.issubset(selected):break
 selected.update(more)
buddy=[p for p in processes if p['pid'] in selected]
rss=sum(p['rss_bytes'] for p in buddy)
reserve=max(8*1024**3-rss,0)
remaining=available-reserve
vfs=os.statvfs(root);disk=vfs.f_bavail*vfs.f_frsize
admitted=remaining>=20*1024**3 and capacity>=4 and disk>=1024**3
preflight=dict(admitted=admitted,cgroup_memory_current_bytes=current,cgroup_memory_max=maximum,host_available_bytes=host_available,hard_headroom_bytes=hard,inactive_file_reclaimable_bytes=reclaimable,cache_accounted_available_bytes=available,BUDDY_concurrent_peak_envelope_bytes=8*1024**3,BUDDY_current_RSS_bytes=rss,BUDDY_growth_reserved_bytes=reserve,BUDDY_reservation_applies_even_before_launch=True,BUDDY_processes=buddy,root_metadata_worker_reserve_bytes=0,available_after_growth_reserves_bytes=remaining,minimum_required_bytes=20*1024**3,own_peak_reservation_bytes=20*1024**3,CPU_capacity_cores=capacity,own_CPU_threads=2,peer_CPU_threads_reserved=2,free_disk_bytes=disk,memory_events=(c/'memory.events').read_text(),prior_C3_status=prior['status'])
if not admitted:print(json.dumps(dict(status='deferred',preflight=preflight,native_jobs_started=0)))
else:
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
 command=[str(root/'.venv/bin/python'),'-B','-u',str(source/'supervise.py'),'--root',str(root),'--output',str(out)]
 with log.open('xb') as stream:child=subprocess.Popen(command,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
 ticks=(Path('/proc')/str(child.pid)/'stat').read_text().split()[21]
 print(json.dumps(dict(status='launched',pid=child.pid,process_start_ticks=ticks,command=command,output=str(out),log=str(log),launched_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),preflight=preflight,source_manifest_sha256={SOURCE_SHA!r},gate_manifest_sha256={GATE_SHA!r},maximum_optimizer_updates=8,maximum_full_VALID_calls=12,maximum_native_constructions=8,maximum_extra_reverse_collections=0,CPU_threads=2,perchild_AS_GiB=64,watchdog_seconds_per_case=1200,automatic_retries=0,scientific_fits=0,TEST_accessed=False,GPU_updates=0)))
'''
    value=json.loads(remote.call(code))
    value['local_control_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    name='LAUNCH.json' if value['status']=='launched' else 'RESOURCE_DEFERRAL_'+str(len(list(Q.glob('RESOURCE_DEFERRAL_*.json')))+1).zfill(2)+'.json'
    save(name,value)
    print(json.dumps(value,indent=2))

def observe():
    launch=json.loads((Q/'LAUNCH.json').read_text())
    code=f'''import json
from pathlib import Path
p=Path({OUTPUT!r});proc=Path('/proc')/str({launch['pid']})/'stat';fields=proc.read_text().split() if proc.exists() else None
alive=fields is not None and fields[2]!='Z' and fields[21]=={launch['process_start_ticks']!r}
sup=json.loads((p/'SUPERVISOR.json').read_text()) if (p/'SUPERVISOR.json').exists() else None
cases=[]
for i in range(4):
 f=p/str(i)/'RESULT.json'
 if f.exists():
  v=json.loads(f.read_text());cases.append({{k:v.get(k) for k in ('method','status','stage','elapsed_seconds','actual_optimizer_transitions','optimizer_attempts','VALID_calls','VALID_attempts','native_constructions','native_constructions_attempted','actual_reverse_collections_attempted','actual_reverse_collections_completed','process_peak_RSS_KiB','failure')}})
print(json.dumps(dict(alive=alive,supervisor=sup,cases=cases)))
'''
    value=json.loads(remote.call(code));save('OBSERVATION_'+str(len(list(Q.glob('OBSERVATION_*.json')))+1).zfill(2)+'.json',value)
    print(json.dumps(value,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=('launch','observe'));args=parser.parse_args()
    globals()[args.operation]()
