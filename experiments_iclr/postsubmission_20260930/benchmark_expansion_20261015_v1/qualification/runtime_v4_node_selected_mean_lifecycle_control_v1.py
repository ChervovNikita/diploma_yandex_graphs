"""One guarded admitted native v4 CPU lifecycle attempt; no automatic retry."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

EXPANSION = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXPANSION))
import remote_control as remote

HERE = EXPANSION / "qualification/runtime_v4_node_selected_mean_lifecycle_v1"
OUTPUT = remote.REMOTE_EXPANSION + "/qualification/runtime_v4_node_selected_mean_lifecycle_attempt01"
RUNTIME_SHA = "6ea4735aac075314e66d0f18a46873d430f98b1803f8452ad24ba4e39ce91d98"
QUALIFICATION_SHA = "acecde92546013e9317d0946ae48a8ad2414e7e119c08d6dd4e67e68b460e10a"


def save(name, value):
    path = HERE / name
    if path.exists():
        raise ValueError("Preserve earlier receipt: " + str(path))
    path.write_text(json.dumps(value, indent=2) + "\n")


def launch():
    if QUALIFICATION_SHA is None or (HERE / "LAUNCH.json").exists():
        raise ValueError("Frozen one-launch admission only")
    code = f'''import os,json,time,hashlib,subprocess
from pathlib import Path
root=Path({remote.REMOTE_ROOT!r});exp=Path({remote.REMOTE_EXPANSION!r})
runtime=exp/'runtime_v4';source=exp/'qualification/runtime_v4_node_selected_mean_lifecycle_v1'
output=Path({OUTPUT!r});log=Path(str(output)+'.log')
assert not output.exists() and not log.exists(),'Preserve existing attempt'
for folder,expected in [(runtime,{RUNTIME_SHA!r}),(source,{QUALIFICATION_SHA!r})]:
 assert hashlib.sha256((folder/'MANIFEST.json').read_bytes()).hexdigest()==expected
 for row in json.loads((folder/'MANIFEST.json').read_text())['files']:
  p=folder/row['path'];assert p.resolve().is_relative_to(folder) and p.stat().st_size==row['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
c=Path('/sys/fs/cgroup');quota=(c/'cpu.max').read_text().split()
capacity=len(os.sched_getaffinity(0)) if quota[0]=='max' else int(quota[0])/int(quota[1])
current=int((c/'memory.current').read_text());maximum=(c/'memory.max').read_text().strip()
stats=dict(line.split() for line in (c/'memory.stat').read_text().splitlines())
host_available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
hard_headroom=host_available if maximum=='max' else max(int(maximum)-current,0)
reclaimable=max(int(stats.get('inactive_file',0))-int(stats.get('shmem',0)),0)
available=min(host_available,hard_headroom+reclaimable)
processes=[]
for line in subprocess.check_output(['ps','-u',str(os.getuid()),'-o','pid=,ppid=,rss=,args='],text=True).splitlines():
 fields=line.strip().split(None,3)
 if len(fields)==4:processes.append(dict(pid=int(fields[0]),ppid=int(fields[1]),rss_bytes=int(fields[2])*1024,args=fields[3]))
groups=[]
for name,pid,envelope,marker in [('NCN',834717,14*1024**3,'interaction_link_ncn_v2_cpu_v2'),('BUDDY',834307,8*1024**3,'buddy')]:
 selected={{pid}}
 selected.update(p['pid'] for p in processes if marker.lower() in p['args'].lower() and 'qualify' in p['args'].lower() and '-c ' not in p['args'])
 while True:
  more={{p['pid'] for p in processes if p['ppid'] in selected}}
  if more.issubset(selected):break
  selected.update(more)
 members=[p for p in processes if p['pid'] in selected]
 rss=sum(p['rss_bytes'] for p in members)
 growth=max(envelope-rss,0) if members else 0
 groups.append(dict(name=name,owner_pid=pid,peak_envelope_bytes=envelope,current_RSS_bytes=rss,reserved_growth_bytes=growth,processes=members))
reserved_growth=sum(g['reserved_growth_bytes'] for g in groups)
remaining=available-reserved_growth
disk=os.statvfs(root);free_disk=disk.f_bavail*disk.f_frsize
admitted=capacity>=2 and remaining>=6*1024**3 and free_disk>=512*1024**2
preflight=dict(admitted=admitted,affinity_CPU_count=len(os.sched_getaffinity(0)),cgroup_CPU_capacity_cores=capacity,cgroup_memory_current_bytes=current,cgroup_memory_max=maximum,hard_headroom_bytes=hard_headroom,inactive_file_reclaimable_bytes=reclaimable,conservative_available_bytes=available,concurrent_owned_groups=groups,reserved_concurrent_growth_bytes=reserved_growth,available_after_growth_reserves_bytes=remaining,minimum_remaining_bytes=6*1024**3,own_reserved_peak_RSS_bytes=4*1024**3,own_reservation_already_in_NCN_metadata_allowance=True,free_disk_bytes=free_disk,own_threads=2,host_settings_changed=False,memory_events=(c/'memory.events').read_text())
if not admitted:
 print(json.dumps(dict(admitted=False,preflight=preflight,native_jobs_started=0,reason='Immediate conservative resource threshold not met')))
else:
 command=['/usr/bin/timeout','--signal=TERM','--kill-after=5s','180s','/usr/bin/time','-v',str(root/'.venv/bin/python'),'-B','-u',str(source/'qualify.py'),str(root),str(output)]
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
 started=time.time()
 with log.open('xb') as stream:
  child=subprocess.Popen(command,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
 proc=Path('/proc')/str(child.pid)/'stat';ticks=proc.read_text().split()[21] if proc.exists() else None
 print(json.dumps(dict(admitted=True,pid=child.pid,process_start_ticks=ticks,command=command,output=str(output),log=str(log),started_unix=started,preflight=preflight,runtime_manifest_sha256={RUNTIME_SHA!r},qualification_manifest_sha256={QUALIFICATION_SHA!r},maximum_TRAIN_epochs=2,maximum_optimizer_steps=2,maximum_native_constructions=3,maximum_complete_VALID_calls=4,maximum_selected_mean_exports=2,CUDA_VISIBLE_DEVICES='',threads=2,wall_watchdog_seconds=180,automatic_retries=0,scientific_fits=0,TEST_accessed=False,GPU_worker_changes=0)))
'''
    value = json.loads(remote.call(code, timeout=60))
    value["local_control"] = {"path": str(Path(__file__).resolve()), "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if value["admitted"]:
        save("LAUNCH.json", value)
    else:
        save("RESOURCE_DEFERRAL_" + str(len(list(HERE.glob("RESOURCE_DEFERRAL_*.json"))) + 1).zfill(2) + ".json", value)
    print(json.dumps(value, indent=2))


def observe():
    launch = json.loads((HERE / "LAUNCH.json").read_text())
    code = f'''import json
from pathlib import Path
p=Path({OUTPUT!r});proc=Path('/proc')/str({launch['pid']})/'stat'
fields=proc.read_text().split() if proc.exists() else None
alive=fields is not None and fields[2]!='Z' and fields[21]=={launch['process_start_ticks']!r}
report=json.loads((p/'RESULT.json').read_text()) if (p/'RESULT.json').exists() else None
summary=None if report is None else {{k:report.get(k) for k in ['status','elapsed_seconds','native_constructions_attempted','native_constructions_completed','TRAIN_epochs_started','complete_native_TRAIN_epochs','source_optimizer_updates_observed','complete_native_VALID_calls','same_call_mean_diagnostics','selected_mean_exports','failure']}}
log=Path(str(p)+'.log')
print(json.dumps(dict(alive=alive,process_state=None if fields is None else fields[2],result=summary,log_tail=log.read_text()[-4000:] if log.exists() else None)))
'''
    value = json.loads(remote.call(code, timeout=60))
    save("OBSERVATION_" + str(len(list(HERE.glob("OBSERVATION_*.json"))) + 1).zfill(2) + ".json", value)
    print(json.dumps(value, indent=2))


def fetch():
    code = f'''import json
from pathlib import Path
p=Path({OUTPUT!r});log=Path(str(p)+'.log')
print(json.dumps(dict(result=json.loads((p/'RESULT.json').read_text()),log_mtime_unix=log.stat().st_mtime,log=log.read_text())))
'''
    value = json.loads(remote.call(code, timeout=60))
    result = value["result"]
    if result["status"] not in ("passed", "failed"):
        raise ValueError("Wait for terminal qualification")
    save("RESULT.json", result)
    save("FINAL_LOG_AND_TIMING.json", {k: v for k, v in value.items() if k != "result"})
    launch = json.loads((HERE / "LAUNCH.json").read_text())
    cost = {key: result.get(key) for key in ("status", "elapsed_seconds", "process_user_seconds", "process_system_seconds",
        "process_peak_RSS_KiB", "TRAIN_epochs_started", "complete_native_TRAIN_epochs", "native_constructions_attempted",
        "native_constructions_completed", "complete_native_VALID_calls", "same_call_mean_diagnostics", "selected_mean_exports",
        "optimizer_attempts", "optimizer_completed", "source_optimizer_updates_observed")}
    cost.update(schema="runtime-v4-native-node-selected-mean-lifecycle-cost-v1", attempts=1, scientific_fits=0,
        TEST_accessed=False, GPU_worker_changes=0, full_launch_to_log_end_seconds=value["log_mtime_unix"]-launch["started_unix"],
        concurrent_CPU_qualification=launch["preflight"]["concurrent_owned_groups"], expected_controlled_fault_preserved=True,
        automatic_retries=0, wall_watchdog_seconds=180)
    save("COST.json", cost)
    print(json.dumps(cost, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("launch", "observe", "fetch"))
    globals()[parser.parse_args().mode]()
