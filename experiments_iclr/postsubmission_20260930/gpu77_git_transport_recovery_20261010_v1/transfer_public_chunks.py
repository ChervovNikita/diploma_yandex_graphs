from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import subprocess
import time

here=Path(__file__).resolve().parent
workspace=here.parents[1]
wrapper=workspace/'postsubmission_research_20260930/gpu77_connection_recovery_v1/run_gpu77_v3.py'
plan=json.loads((here/'CHUNK_PLAN.json').read_text())
completed=[item["chunk"] for item in plan["chunks"] if (here/f'chunk_{item["chunk"]:03d}_transport_receipt.json').is_file() and json.loads((here/f'chunk_{item["chunk"]:03d}_transport_receipt.json').read_text())["exit_code"]==0]
failed=[]
start=time.monotonic()

def transfer(item):
    index=item['chunk']
    argv=['python3',str(wrapper),'--id',f'git77_public_bundle_chunk_{index:03d}_20261010_v1','--command-file',str(workspace/item['command_file'])]
    try:
        r=subprocess.run(argv,capture_output=True,text=True,timeout=160,cwd=workspace)
        record={'chunk':index,'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'bytes':item['bytes'],'sha256':item['sha256']}
    except subprocess.TimeoutExpired:
        record={'chunk':index,'exit_code':124,'bytes':item['bytes'],'sha256':item['sha256'],'timeout_seconds':160}
    (here/f'chunk_{index:03d}_transport_receipt.json').write_text(json.dumps(record,indent=2)+'\n')
    return record

with ThreadPoolExecutor(max_workers=6) as pool:
    pending=[pool.submit(transfer,item) for item in plan['chunks'] if item['chunk'] not in completed]
    for future in as_completed(pending):
        record=future.result()
        (completed if record['exit_code']==0 else failed).append(record['chunk'])
        progress={'completed_chunks':sorted(completed),'failed_chunks':sorted(failed),'expected_chunks':len(plan['chunks']),
                  'elapsed_seconds':time.monotonic()-start,'checkout_modified':False,'private_credentials_transferred':False}
        temporary=here/'TRANSFER_PROGRESS.json.tmp'
        temporary.write_text(json.dumps(progress,indent=2)+'\n')
        temporary.replace(here/'TRANSFER_PROGRESS.json')
        if len(completed)%12==0 or failed or len(completed)+len(failed)==len(plan['chunks']):
            print(json.dumps({'completed_chunks':len(completed),'failed_chunks':failed,'expected_chunks':len(plan['chunks']),'elapsed_seconds':round(time.monotonic()-start,2)}),flush=True)
raise SystemExit(1 if failed else 0)
