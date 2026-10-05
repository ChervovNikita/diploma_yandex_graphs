"""Metadata-only physical closure of the one released qualifier."""
from pathlib import Path
import importlib.util,json
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('closure77_transport',HERE.parent/'shared_private_transfer_gpu77_environment_execution_20261005_v1/remote_transport.py')
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t);t.HERE=HERE
code="""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,socket,subprocess
repo=Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
phase=repo/'experiments_iclr/postsubmission_20260930'
root=phase/'shared_private_transfer_gpu77_fp32_execution_root_20261005_v2'
release=phase/'shared_private_transfer_gpu77_qualification_root_release_20261005_v2'
assert Path.cwd()==repo and socket.gethostname()=='peptide'
launch=json.loads((release/'DETACHED_LAUNCH.json').read_text())
receipt=json.loads((root/'EXECUTION_RECEIPT.json').read_text())
child=receipt['child_identity'];parent=launch['supervisor_PID']
rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=20).strip().splitlines()
pids={int(r.split(',')[1].strip()) for r in rows if len(r.split(','))==3 and r.split(',')[1].strip().isdigit()}
result=dict(UTC=datetime.now(timezone.utc).isoformat(),owned_child_PID=child['PID'],owned_child_start_ticks=child['start_ticks'],owned_supervisor_PID=parent,owned_supervisor_start_ticks=launch['supervisor_start_ticks'],owned_child_PID_absent=not Path('/proc',str(child['PID'])).exists(),owned_supervisor_PID_absent=not Path('/proc',str(parent)).exists(),owned_child_no_CUDA_rows=child['PID'] not in pids,owned_supervisor_no_CUDA_rows=parent not in pids,compute_rows=rows,exit_code=receipt['exit_code'],terminal_wait_observed=receipt['terminal_wait_observed'],exit_code_authority=receipt['exit_code_authority'],attempts=1,numerical_launches=1,retry=False,fits=0,VALID_TEST_values_access=False)
assert all(result[k] for k in ('owned_child_PID_absent','owned_supervisor_PID_absent','owned_child_no_CUDA_rows','owned_supervisor_no_CUDA_rows','terminal_wait_observed'))
result['status']='PASS'
path=root/'FINAL_PHYSICAL_TERMINAL.json'
with path.open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True);stream.write('\\n')
raw=path.read_bytes();print(json.dumps(dict(result=result,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),raw=raw.decode())))
"""
r=t.run('private_transfer77_qualifier_final_physical_v3_20261005_v1',code)
raw=r.pop('raw').encode()
assert len(raw)==r['bytes']
with (HERE/'FINAL_PHYSICAL_TERMINAL.json').open('xb') as stream:stream.write(raw)
with (HERE/'FINAL_PHYSICAL_CUSTODY.json').open('x') as stream:json.dump(r,stream,indent=2,sort_keys=True);stream.write('\n')
print(json.dumps(r))
