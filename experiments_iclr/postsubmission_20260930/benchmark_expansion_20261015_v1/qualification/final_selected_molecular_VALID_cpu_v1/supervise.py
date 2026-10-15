"""Three fixed serial CPU workers; own-process finite resource supervision."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from support import HERE, admission, receipt, write_json


def rss_bytes(pid):
    try:
        lines=Path('/proc')/str(pid)/'status'
        for row in lines.read_text().splitlines():
            if row.startswith('VmRSS:'): return int(row.split()[1])*1024
    except (FileNotFoundError,ProcessLookupError): pass
    return 0


def stop(proc):
    if proc.poll() is not None: return
    proc.terminate()
    try: proc.wait(timeout=5)
    except subprocess.TimeoutExpired: proc.kill(); proc.wait(timeout=5)


def run(root,admission_path):
    request,inputs,output,admission_pin=admission(admission_path,root)
    if output.exists(): raise ValueError('One fresh complete attempt; preserve previous output/no automatic retry')
    output.mkdir(parents=True)
    record={'schema':'genuine-selected-MolBACE-CPU-VALID-supervision-v1','status':'running','admission':admission_pin,
        'harness_manifest':receipt(HERE/'MANIFEST.json'),'TEST_accessed':False,'automatic_retries':0,
        'case_attempts':[],'case_ids':[row['id'] for row in inputs['cases']],'new_scientific_fits':0}
    write_json(output/'START.json',record); started=time.perf_counter(); proc=None
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    try:
        for case in inputs['cases']:
            began=time.perf_counter(); peak=0; reason=None
            command=[sys.executable,str(HERE/'qualify.py'),'--root',str(root),'--admission',str(admission_path),'--case',case['id']]
            with (output/(case['id']+'.log')).open('x') as log:
                proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
                while proc.poll() is None:
                    peak=max(peak,rss_bytes(proc.pid))
                    if peak>request['host_RSS_reservation_bytes']: reason='own_worker_RSS_bound'; break
                    if time.perf_counter()-began>request['wall_seconds_per_case']+10: reason='own_worker_wall_bound'; break
                    time.sleep(.2)
                if reason is not None: stop(proc)
                code=proc.wait(); proc=None
            path=output/case['id']/'RESULT.json'
            result=json.loads(path.read_text()) if path.exists() else None
            row={'case_id':case['id'],'worker_exit_code':code,'supervisor_reason':reason,'own_worker_peak_RSS_sampled_bytes':peak,
                'attempt_wall_seconds':time.perf_counter()-began,'RESULT':receipt(path) if path.exists() else None,
                'status':None if result is None else result['status']}
            record['case_attempts'].append(row)
            write_json(output/(case['id']+'_SUPERVISION.json'),row)
            # Continue the fixed independent cases after a numerical mismatch;
            # source/identity/resource failures stop the finite bundle for root.
            if reason is not None or result is None or result['status'] not in ('passed','failed_numerical_reconstruction'):
                record['stopped_before_remaining_cases']=True; break
    finally:
        if proc is not None: stop(proc)
        rows=record['case_attempts']
        record.update(status='passed' if len(rows)==3 and all(row['status']=='passed' and row['worker_exit_code']==0 and row['supervisor_reason'] is None for row in rows) else 'failed',
            wall_seconds=time.perf_counter()-started,case_attempts_started=len(rows),
            native_execution_source_scope='CPU engineering under authentic saved producers, not exact original CUDA-device or portable runtime qualification')
        write_json(output/'RESULT.json',record)
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--root',required=True); parser.add_argument('--admission',required=True)
    args=parser.parse_args(); result=run(Path(args.root).resolve(),Path(args.admission).resolve())
    print(json.dumps({'status':result['status'],'case_attempts_started':result['case_attempts_started']}))
    raise SystemExit(0 if result['status']=='passed' else 2)
