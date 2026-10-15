"""One reviewed finite private-BN engineering child; preserve first failure."""
import argparse
import json
import hashlib
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


def main():
    assert socket.gethostname()=='anogena-2-0'
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--context-manifest',required=True);p.add_argument('--output',required=True);p.add_argument('--admitted-plan-sha256',required=True);a=p.parse_args()
    root=Path(a.root).resolve();output=Path(a.output).resolve();source=Path(__file__).resolve().parent
    assert all(path.is_relative_to(root) for path in (output,source,Path(a.context_manifest).resolve())) and not output.exists()
    assert hashlib.sha256((source/'SMOKE_PLAN.json').read_bytes()).hexdigest()==a.admitted_plan_sha256
    for row in json.loads((source/'MANIFEST.json').read_text())['files']:
        raw=(source/row['path']).read_bytes()
        assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
    output.mkdir(parents=True);started=time.perf_counter();child_output=output/'gnnm_private_bn_state_v3'
    summary={'schema':'finite-BUDDY-private-BN-CPU-supervision-v1','status':'running','maximum_children':1,'threads':2,'address_space_limit_bytes':64*1024**3,
        'maximum_native_constructions':3,'maximum_actual_optimizer_transitions':2,'maximum_full_VALID_calls':3,'maximum_native_model_forwards':36,'maximum_reverse_collections':3,
        'scientific_fits':0,'GPU_updates':0,'TEST_accessed':False,'automatic_retries':0,'output':str(child_output)}
    def write():
        summary['elapsed_seconds']=time.perf_counter()-started;temporary=output/'SUMMARY.json.part';temporary.write_text(json.dumps(summary,sort_keys=True,indent=2)+'\n');temporary.replace(output/'SUMMARY.json')
    write();env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    command=[str(root/'.venv/bin/python'),'-B','-u',str(source/'qualify_cpu.py'),'--root',str(root),'--context-manifest',str(Path(a.context_manifest).resolve()),'--output',str(child_output)]
    with (output/'child.log').open('xb') as stream:child=subprocess.Popen(command,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    summary.update(child_pid=child.pid,command=command);write()
    try:returncode=child.wait(timeout=1800)
    except subprocess.TimeoutExpired:
        summary['timeout_seconds']=1800;os.killpg(child.pid,signal.SIGTERM)
        try:returncode=child.wait(timeout=15)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);returncode=child.wait(timeout=15)
    result_path=child_output/'RESULT.json';result=json.loads(result_path.read_text()) if result_path.exists() else None
    summary.update(status='passed' if returncode==0 and result and result['status']=='passed' else 'failed',returncode=returncode,
        actual_result=None if result is None else {k:result.get(k) for k in ('status','stage','native_constructions','actual_optimizer_transitions','VALID_calls','native_model_forwards','reverse_collections','elapsed_seconds','process_peak_RSS_KiB','failure')})
    write();print(json.dumps(summary));return 0 if summary['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
