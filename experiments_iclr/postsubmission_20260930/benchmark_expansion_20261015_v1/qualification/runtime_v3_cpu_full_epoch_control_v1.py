"""Local guarded launch/observe/fetch for the frozen v3 CPU qualification."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

EXPANSION = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXPANSION))
import remote_control as remote

HERE = EXPANSION / "qualification/runtime_v3_cpu_full_epoch_v1"
OUTPUT = remote.REMOTE_EXPANSION + "/qualification/runtime_v3_cpu_full_epoch_attempt01"
RUNTIME_SHA = "d04193717372bb08eb85b780822338ad25b0205e1fd5a1ef72680d293506fd93"
QUALIFICATION_SHA = "e1a7869fe9a4c0de75651a586525ac94108e5b930e218d650ba97d6016c5e440"


def save(name, value):
    path = HERE / name
    if path.exists():
        raise ValueError("Preserve prior receipt: " + str(path))
    path.write_text(json.dumps(value, indent=2) + "\n")


def launch():
    if (HERE / "LAUNCH.json").exists():
        raise ValueError("One frozen launch only")
    code = f'''import os,json,time,hashlib,subprocess
from pathlib import Path
root=Path({remote.REMOTE_ROOT!r});exp=Path({remote.REMOTE_EXPANSION!r})
runtime=exp/'runtime_v3';source=exp/'qualification/runtime_v3_cpu_full_epoch_v1'
output=Path({OUTPUT!r});log=Path(str(output)+'.log')
assert not output.exists() and not log.exists()
for folder,expected in [(runtime,{RUNTIME_SHA!r}),(source,{QUALIFICATION_SHA!r})]:
 assert hashlib.sha256((folder/'MANIFEST.json').read_bytes()).hexdigest()==expected
 for row in json.loads((folder/'MANIFEST.json').read_text())['files']:
  p=folder/row['path'];assert p.resolve().is_relative_to(folder) and p.stat().st_size==row['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
c=Path('/sys/fs/cgroup');quota=(c/'cpu.max').read_text().split()
capacity=len(os.sched_getaffinity(0)) if quota[0]=='max' else int(quota[0])/int(quota[1]);assert capacity>=8
current=int((c/'memory.current').read_text());maximum=(c/'memory.max').read_text().strip()
stats=dict(line.split() for line in (c/'memory.stat').read_text().splitlines())
host_available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
hard_headroom=host_available if maximum=='max' else max(int(maximum)-current,0)
reclaimable=max(int(stats.get('inactive_file',0))-int(stats.get('shmem',0)),0)+int(stats.get('slab_reclaimable',0))
available=min(host_available,hard_headroom+reclaimable);assert available>=8*1024**3
processes=subprocess.check_output(['ps','-u',str(os.getuid()),'-o','pid=,args='],text=True).splitlines()
other=[p for p in processes if str(exp) in p and 'qualify' in p and '.venv/bin/python' in p and '-c ' not in p and 'runtime_v3_cpu_full_epoch' not in p]
preflight=dict(affinity_cpu_count=len(os.sched_getaffinity(0)),cgroup_cpu_capacity_cores=capacity,cgroup_memory_current_bytes=current,cgroup_memory_max=maximum,hard_headroom_bytes=hard_headroom,anon_bytes=int(stats['anon']),inactive_file_bytes=int(stats.get('inactive_file',0)),slab_reclaimable_bytes=int(stats.get('slab_reclaimable',0)),conservative_available_bytes=available,other_qualifiers=other,concurrent_CPU_work_disclosed=True,own_threads=2,host_settings_changed=False)
command=[str(root/'.venv/bin/python'),'-B','-u',str(source/'qualify.py'),str(root),str(output)]
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
started=time.time()
with log.open('xb') as stream:
 child=subprocess.Popen(command,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
print(json.dumps(dict(pid=child.pid,command=command,output=str(output),log=str(log),started_unix=started,preflight=preflight,runtime_manifest_sha256={RUNTIME_SHA!r},qualification_manifest_sha256={QUALIFICATION_SHA!r},maximum_full_TRAIN_epochs=5,maximum_native_constructions=6,maximum_complete_VALID_calls=10,expected_optimizer_steps_if_no_source_skip=190,CUDA_VISIBLE_DEVICES='',threads=2,scientific_fits=0,TEST_accessed=False,GPU_worker_changes=0)))
'''
    value = json.loads(remote.call(code, timeout=60))
    value["local_control"] = {"path": str(Path(__file__).resolve()),
        "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    save("LAUNCH.json", value)
    print(json.dumps(value))


def observe():
    launch = json.loads((HERE / "LAUNCH.json").read_text())
    code = f'''import json
from pathlib import Path
p=Path({OUTPUT!r});proc=Path('/proc')/str({launch['pid']})/'stat';state=proc.read_text().split()[2] if proc.exists() else None
report=json.loads((p/'RESULT.json').read_text()) if (p/'RESULT.json').exists() else None
summary=None if report is None else {{k:report.get(k) for k in ['status','elapsed_seconds','complete_native_TRAIN_epochs','native_constructions','complete_VALID_calls','source_optimizer_updates_observed','checks','failure']}}
curves=[]
if p.exists():
 for f in p.glob('*/PROGRESS.json'):
  v=json.loads(f.read_text());curves.append(dict(folder=f.parent.name,epoch=v['train']['epoch'],optimizer_calls=v['optimizer_calls']))
log=Path(str(p)+'.log')
print(json.dumps(dict(alive=state is not None and state!='Z',process_state=state,result=summary,progress=curves,log_tail=log.read_text()[-4000:] if log.exists() else None)))
'''
    value = json.loads(remote.call(code, timeout=60))
    save("OBSERVATION_" + str(len(list(HERE.glob("OBSERVATION_*.json"))) + 1).zfill(2) + ".json", value)
    print(json.dumps(value))


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
        "process_peak_RSS_KiB", "TRAIN_epochs_started", "complete_native_TRAIN_epochs", "native_constructions",
        "complete_VALID_calls", "source_optimizer_updates_observed", "optimizer_attempts", "optimizer_completed")}
    cost.update(schema="runtime-v3-CPU-full-epoch-cost-v1", attempts=1, scientific_fits=0,
        TEST_accessed=False, GPU_worker_changes=0, full_launch_to_log_end_seconds=value["log_mtime_unix"]-launch["started_unix"],
        concurrent_CPU_qualification=launch["preflight"]["other_qualifiers"], expected_controlled_fault_preserved=True,
        zero_work_launch_deferrals=2)
    save("COST.json", cost)
    print(json.dumps(cost))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("launch", "observe", "fetch"))
    globals()[parser.parse_args().mode]()
