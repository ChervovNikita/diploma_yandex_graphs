"""One authorized CPU component check, not a predictive experiment."""
from pathlib import Path
from datetime import datetime, timezone
import base64
import hashlib
import importlib.util
import json
import sys

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
SOURCE = PHASE / "native_endpoint_private_factor_mechanism_assessment_20261005_v1"
PARENT = PHASE / "graph_ncNC_member_completion_qualification_preparation_20261003_v2"
TRANSPORT = PHASE / "ncnc_heldout_wrapper_qualification_execution_root_20261004_v1/stage_and_launch_qa.py"
spec = importlib.util.spec_from_file_location("endpoint_component_transport", TRANSPORT)
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
transport.HERE = HERE


def save(name, value):
    with (HERE / name).open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def main():
    rows = []
    for path in (SOURCE / "adapter.py", SOURCE / "check_cpu.py", PARENT / "prototype.py", PARENT / "graph_ops.py"):
        raw = path.read_bytes()
        rows.append(dict(path=str(path.relative_to(PHASE)), sha256=hashlib.sha256(raw).hexdigest(),
                         bytes=len(raw), data=base64.b64encode(raw).decode()))
    save("PROTOCOL.json", dict(UTC=datetime.now(timezone.utc).isoformat(),
        scope="One fabricated CPU construction/RNG/gradient fixture. No fitting or task-quality inference.",
        files=[{k:v for k,v in r.items() if k != "data"} for r in rows],
        expected_host="peptide", CUDA_VISIBLE_DEVICES="", threads=2,
        fit_count=0, optimizer_steps=0, data_checkpoint_outcome_access=False,
        automatic_retry=False))
    code = '''from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,os,subprocess,time
repo=Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
phase=repo/'experiments_iclr/postsubmission_20260930'
assert Path.cwd()==repo and os.uname().nodename=='peptide'
for row in ROWS:
 p=phase/row['path'];assert p.resolve().is_relative_to(phase)
 raw=base64.b64decode(row['data'],validate=True)
 assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 if p.exists():assert p.is_file() and not p.is_symlink() and p.read_bytes()==raw
 else:
  p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as stream:stream.write(raw)
root=phase/'endpoint_frame_component_execution_20261005_v1'
root.mkdir(parents=True,exist_ok=True)
assert not (root/'ATTEMPT.json').exists()
with (root/'ATTEMPT.json').open('x') as stream:json.dump(dict(UTC=datetime.now(timezone.utc).isoformat(),sources=[{k:v for k,v in r.items() if k!='data'} for r in ROWS]),stream)
command=['/disk/10tb/home/shmelev/miniconda3/envs/rapids-25.06/bin/python3.12','-B',str(phase/'native_endpoint_private_factor_mechanism_assessment_20261005_v1/check_cpu.py')]
environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(repo/'.gnnm_runtime/buddy_extra_v1/site'))
start=time.monotonic()
try:
 child=subprocess.run(command,cwd=repo,env=environment,capture_output=True,text=True,timeout=60)
 result=dict(UTC=datetime.now(timezone.utc).isoformat(),status='PASS' if child.returncode==0 else 'FAIL',exit_code=child.returncode,stdout=child.stdout,stderr=child.stderr,elapsed_seconds=time.monotonic()-start,command=command,CUDA_VISIBLE_DEVICES='',fit_count=0,optimizer_steps=0,data_checkpoint_outcome_access=False)
except subprocess.TimeoutExpired as error:
 result=dict(UTC=datetime.now(timezone.utc).isoformat(),status='TIMEOUT',elapsed_seconds=time.monotonic()-start,stdout=str(error.stdout),stderr=str(error.stderr),fit_count=0,optimizer_steps=0,data_checkpoint_outcome_access=False)
with (root/'RECEIPT.json').open('x') as stream:json.dump(result,stream,indent=2);stream.write(chr(10))
print(json.dumps(result))
'''
    code = "ROWS=" + repr(rows) + "\n" + code
    result = transport.run("endpoint_frame_one_cpu_component_20261005_v1", code)
    save("RECEIPT.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
