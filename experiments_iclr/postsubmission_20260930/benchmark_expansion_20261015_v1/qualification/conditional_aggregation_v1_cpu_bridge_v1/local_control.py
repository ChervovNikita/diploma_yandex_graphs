"""Guarded targeted stage/launch/observe/fetch for one fixed CPU attempt."""
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

SOURCE_MANIFEST = "edef4b1f552a495e09e64855afbd96c7eb14ce5ef02d26cc13ec530d80e9805f"
REMOTE_PACKAGE = remote.REMOTE_EXPANSION + "/conditional_aggregation_v1"
REMOTE_SCRIPTS = remote.REMOTE_EXPANSION + "/qualification/conditional_aggregation_v1_cpu_bridge_v1"
REMOTE_OUTPUT = remote.REMOTE_EXPANSION + "/qualification/conditional_aggregation_v1_cpu_attempt01"
REMOTE_JOB = remote.REMOTE_EXPANSION + "/results/molbace_pilot_v1/fits/004_69902ee9d5d7427d/JOB.json"


def save(name, value):
    path = HERE / name
    if path.exists():
        raise ValueError("Preserve existing receipt; do not overwrite " + str(path))
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def stage():
    selected = ["conditional_aggregation_v1"] + ["qualification/conditional_aggregation_v1_cpu_bridge_v1/" + name
        for name in ("qualify_molecular_cpu.py", "local_control.py", "PLAN.md", "MANIFEST.json")]
    result = subprocess.run([sys.executable, "-B", str(EXPANSION / "stage_selected.py"), *selected],
                            capture_output=True, text=True, check=True, timeout=110)
    value = json.loads(result.stdout)
    save("TRANSFER_RECEIPT.json", value)
    print(json.dumps(value))


def launch():
    expected_qualification = hashlib.sha256((HERE / "MANIFEST.json").read_bytes()).hexdigest()
    code = f'''import hashlib,importlib.util,json,os,subprocess,sys,time
from pathlib import Path
root=Path({remote.REMOTE_ROOT!r})
scripts=Path({REMOTE_SCRIPTS!r});package=Path({REMOTE_PACKAGE!r})
job_path=Path({REMOTE_JOB!r});output=Path({REMOTE_OUTPUT!r});log=Path(str(output)+'.log')
assert all(p.resolve().is_relative_to(root) for p in (scripts,package,job_path,output,log))
assert not output.exists() and not log.exists(),'Preserve previous attempt; no overwrite'
assert hashlib.sha256((scripts/'MANIFEST.json').read_bytes()).hexdigest()=={expected_qualification!r}
for row in json.loads((scripts/'MANIFEST.json').read_text())['files']:
 p=scripts/row['path'];assert p.resolve().is_relative_to(scripts)
 assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'] and p.stat().st_size==row['bytes']
spec=importlib.util.spec_from_file_location('_conditional_preflight',package/'molecular_transform.py')
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
receipts=module._bindings({SOURCE_MANIFEST!r})
job=json.loads(job_path.read_text())
assert job['procedure']=='gnnm_boundary_native' and job['family']=='ogb' and job['TEST_access'] is False
assert job['recipe']['dataset']=='ogbg-molbace'
command=[str(root/'.venv/bin/python'),'-B','-u',str(scripts/'qualify_molecular_cpu.py'),
 '--project-root',str(root),'--job',str(job_path),'--method-package',str(package),'--output',str(output)]
environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
with log.open('xb') as stream:
 child=subprocess.Popen(command,cwd=root,env=environment,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
print(json.dumps(dict(pid=child.pid,command=command,output=str(output),log=str(log),device='cpu',threads=2,
 CUDA_VISIBLE_DEVICES='',maximum_optimizer_transitions=6,GPU_jobs_started=0,scientific_fits_started=0,TEST_accessed=False,
 source_manifest_sha256={SOURCE_MANIFEST!r},qualification_manifest_sha256={expected_qualification!r},
 source_preflight_receipts=len(receipts),source_job_data_root=job['data_root'],source_job_dataset=job['recipe']['dataset'],
 started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))))
'''
    value = json.loads(remote.call(code, timeout=60))
    save("LAUNCH.json", value)
    print(json.dumps(value))


def observe():
    launch_value = json.loads((HERE / "LAUNCH.json").read_text())
    code = f'''import json
from pathlib import Path
pid={launch_value['pid']!r};output=Path({REMOTE_OUTPUT!r});log=Path(str(output)+'.log')
proc=Path('/proc')/str(pid)/'stat'
state=proc.read_text().split()[2] if proc.exists() else None
result=json.loads((output/'RESULT.json').read_text()) if (output/'RESULT.json').exists() else None
summary=None if result is None else {{'status':result['status'],'elapsed_seconds':result['elapsed_seconds'],
 'actual_optimizer_transitions':result['actual_optimizer_transitions'],'native_constructions':result['native_constructions'],
 'cases':[{{'method':c['method'],'status':c['status'],'steps':c['actual_optimizer_transitions'],
 'VALID_calls':c['VALID_calls'],'failure':c.get('failure',{{}}).get('message')}} for c in result['cases']]}}
print(json.dumps(dict(pid=pid,process_state=state,alive=state is not None and state!='Z',result=summary,
 log_tail=log.read_text()[-6000:] if log.exists() else None)))
'''
    value = json.loads(remote.call(code, timeout=60))
    number = len(list(HERE.glob("OBSERVATION_*.json"))) + 1
    save(f"OBSERVATION_{number:02d}.json", value)
    print(json.dumps(value))


def fetch():
    code = f'''from pathlib import Path
path=Path({REMOTE_OUTPUT!r})/'RESULT.json'
print(path.read_text())
'''
    value = json.loads(remote.call(code, timeout=60))
    if value["status"] not in ("passed", "failed"):
        raise ValueError("Wait for a completed engineering attempt")
    save("RESULT.json", value)
    cost = {key: value.get(key) for key in ("status", "actual_optimizer_transitions", "optimizer_attempts",
        "native_constructions", "elapsed_seconds", "process_user_seconds", "process_system_seconds", "process_peak_RSS_KiB")}
    cost.update(schema="conditional-aggregation-CPU-engineering-cost-v1", attempts=1, device="cpu", threads=2,
                scientific_fits=0, GPU_models_or_updates=0, TEST_accessed=False,
                result_sha256=hashlib.sha256((HERE / "RESULT.json").read_bytes()).hexdigest(),
                fetched_utc=datetime.now(timezone.utc).isoformat())
    save("COST.json", cost)
    print(json.dumps(cost))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("stage", "launch", "observe", "fetch"))
    args = parser.parse_args()
    globals()[args.mode]()
