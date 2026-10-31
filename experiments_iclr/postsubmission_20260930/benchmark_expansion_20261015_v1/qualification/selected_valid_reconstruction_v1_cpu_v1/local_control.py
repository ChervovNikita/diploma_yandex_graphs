"""One guarded zero-step reconstruction qualification attempt."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
EXPANSION = HERE.parent.parent
sys.path.insert(0, str(EXPANSION))
import remote_control as remote

SCRIPTS = remote.REMOTE_EXPANSION + "/qualification/selected_valid_reconstruction_v1_cpu_v1"
OUTPUT = remote.REMOTE_EXPANSION + "/qualification/selected_valid_reconstruction_v1_cpu_attempt01"
PACKAGE = remote.REMOTE_EXPANSION + "/selected_valid_reconstruction_v1"
SOURCE_MANIFEST = "e397d3f1738e763daf8ecfe614833ce64464b974f978f23a67c4eb874c7af27c"


def save(name, value):
    path = HERE / name
    if path.exists():
        raise ValueError("Preserve previous receipt: " + name)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def stage():
    paths = ["selected_valid_reconstruction_v1", "qualification/selected_valid_reconstruction_v1_cpu_v1"]
    result = subprocess.run([sys.executable, "-B", str(EXPANSION / "stage_selected.py"), *paths],
        capture_output=True, text=True, check=True, timeout=110)
    value = json.loads(result.stdout)
    save("TRANSFER_RECEIPT.json", value)
    print(json.dumps(value))


def launch():
    manifest = hashlib.sha256((HERE / "MANIFEST.json").read_bytes()).hexdigest()
    code = f'''import hashlib,importlib.util,json,os,subprocess,sys,time
from pathlib import Path
root=Path({remote.REMOTE_ROOT!r});scripts=Path({SCRIPTS!r});output=Path({OUTPUT!r});package=Path({PACKAGE!r})
log=Path(str(output)+'.log')
assert all(p.resolve().is_relative_to(root) for p in [scripts,output,package,log])
assert not output.exists() and not log.exists()
assert hashlib.sha256((scripts/'MANIFEST.json').read_bytes()).hexdigest()=={manifest!r}
for row in json.loads((scripts/'MANIFEST.json').read_text())['files']:
 p=scripts/row['path'];assert p.resolve().is_relative_to(scripts)
 assert p.stat().st_size==row['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
spec=importlib.util.spec_from_file_location('_selected_replay_preflight',package/'replay.py')
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
module.admit_cpu();module.verify_package(package,{SOURCE_MANIFEST!r},root)
command=[str(root/'.venv/bin/python'),'-B','-u',str(scripts/'qualify.py'),'--project-root',str(root),'--output',str(output)]
environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
with log.open('xb') as stream:
 process=subprocess.Popen(command,cwd=root,env=environment,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
print(json.dumps(dict(pid=process.pid,command=command,output=str(output),log=str(log),device='CPU',threads=2,
 CUDA_VISIBLE_DEVICES='',maximum_optimizer_steps=0,maximum_native_constructions=32,maximum_complete_VALID_calls=32,
 production_selected_cases=4,existing_engineering_cases=14,TEST_accessed=False,scientific_fits_started=0,GPU_worker_changes=0,
 source_manifest_sha256={SOURCE_MANIFEST!r},qualification_manifest_sha256={manifest!r},
 started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))))
'''
    value = json.loads(remote.call(code, timeout=60))
    save("LAUNCH.json", value)
    print(json.dumps(value))


def observe():
    launched = json.loads((HERE / "LAUNCH.json").read_text())
    code = f'''import json
from pathlib import Path
output=Path({OUTPUT!r});log=Path(str(output)+'.log');proc=Path('/proc')/str({launched['pid']!r})/'stat'
state=proc.read_text().split()[2] if proc.exists() else None
r=json.loads((output/'RESULT.json').read_text()) if (output/'RESULT.json').exists() else None
summary=None if r is None else dict(status=r['status'],elapsed_seconds=r['elapsed_seconds'],native_constructions=r['native_constructions'],
 complete_VALID_calls=r['complete_VALID_calls'],optimizer_steps=r['optimizer_steps'],optimizer_attempts=r['optimizer_attempts'],
 production=[dict(id=c.get('id'),status=c['status'],method=c.get('procedure'),failure=c.get('failure',{{}}).get('message')) for c in r['production_selected_cases']],
 engineering=[dict(method=c['method'],family=c['family'],status=c['status'],failure=c.get('failure',{{}}).get('message')) for c in r['engineering_existing_state_cases']])
print(json.dumps(dict(pid={launched['pid']!r},alive=state is not None and state!='Z',process_state=state,result=summary,
 log_tail=log.read_text()[-6000:] if log.exists() else None)))
'''
    value = json.loads(remote.call(code, timeout=60))
    save("OBSERVATION_" + str(len(list(HERE.glob("OBSERVATION_*.json"))) + 1).zfill(2) + ".json", value)
    print(json.dumps(value))


def fetch():
    code = f'''from pathlib import Path
import json
p=Path({OUTPUT!r});r=p/'RESULT.json';log=Path(str(p)+'.log')
value=json.loads(r.read_text())
print(json.dumps(dict(result=value,timing=dict(result_mtime_unix=r.stat().st_mtime,log_mtime_unix=log.stat().st_mtime))))
'''
    value = json.loads(remote.call(code, timeout=60))
    result = value["result"]
    if result["status"] not in ("passed", "failed"):
        raise ValueError("Wait for completed qualification")
    save("RESULT.json", result)
    save("TIMING_RECEIPT.json", value["timing"])
    cost = {key: result[key] for key in ("status", "elapsed_seconds", "process_user_seconds", "process_system_seconds",
        "process_peak_RSS_KiB", "native_constructions", "complete_VALID_calls", "optimizer_steps", "optimizer_attempts")}
    cost.update(schema="selected-VALID-zero-step-qualification-cost-v1", attempts=1, scientific_fits=0,
                TEST_accessed=False, GPU_worker_changes=0, result_sha256=hashlib.sha256((HERE / "RESULT.json").read_bytes()).hexdigest(),
                fetched_utc=datetime.now(timezone.utc).isoformat())
    save("COST.json", cost)
    print(json.dumps(cost))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("stage", "launch", "observe", "fetch"))
    globals()[parser.parse_args().mode]()
