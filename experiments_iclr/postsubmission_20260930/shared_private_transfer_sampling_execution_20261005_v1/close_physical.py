"""Metadata-only closure of the exact successful finite sampler PIDs on both hosts."""
from pathlib import Path
import hashlib
import json
from stage_and_launch import call, HOSTS, HERE

for host in ('singleton','gpu77'):
    h=HOSTS[host];local=HERE/host
    execution=json.loads((local/'EXECUTION_RECEIPT.json').read_text())
    launch=json.loads((local/'DETACHED_LAUNCH.json').read_text())
    assert execution['exit_code']==0 and execution['terminal_wait_observed'] is True and execution['signals_sent']==[]
    child=execution['child_identity'];repo=h['repo'];root=repo+'/experiments_iclr/postsubmission_20260930/'+h['execution']
    code=f'''
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,socket,subprocess
repo=Path({repo!r});root=Path({root!r})
assert Path.cwd()==repo and socket.gethostname()=={h['hostname']!r}
execution=json.loads((root/'EXECUTION_RECEIPT.json').read_text());assert execution['exit_code']==0 and execution['terminal_wait_observed'] is True and execution['signals_sent']==[]
def owned_absent(pid,ticks):
 p=Path('/proc',str(pid),'stat')
 try:
  fields=p.read_text().split(') ',1)[1].split()
  return int(fields[19])!=ticks
 except FileNotFoundError:return True
assert owned_absent({child['PID']},{child['start_ticks']}) and owned_absent({launch['supervisor_PID']},{launch['supervisor_start_ticks']})
query=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=20)
assert not any(len(parts)>=2 and parts[1].strip() in {{{str(child['PID'])!r},{str(launch['supervisor_PID'])!r}}} for parts in (line.split(',') for line in query.stdout.splitlines()))
receipt=dict(UTC=datetime.now(timezone.utc).isoformat(),status='PASS',hostname=socket.gethostname(),owned_child_PID_absent=True,owned_supervisor_PID_absent=True,owned_PIDs_no_CUDA_rows=True,child_PID={child['PID']},child_start_ticks={child['start_ticks']},supervisor_PID={launch['supervisor_PID']},supervisor_start_ticks={launch['supervisor_start_ticks']},execution_receipt_sha256=hashlib.sha256((root/'EXECUTION_RECEIPT.json').read_bytes()).hexdigest(),raw_result_sha256=hashlib.sha256((root/'result/RESULT.json').read_bytes()).hexdigest(),exit_code=0,terminal_wait_observed=True,signals_sent=[],fits=0)
(root/'PHYSICAL_TERMINAL.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\\n')
print(json.dumps(receipt))
'''
    result=call(host,f'private_transfer_{host}_sampling_closure_20261005_v1',code)
    with (local/'PHYSICAL_TERMINAL.json').open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True);stream.write('\n')
    assert result['raw_result_sha256']==hashlib.sha256((local/'result/RESULT.json').read_bytes()).hexdigest()
    print(json.dumps(result),flush=True)
