"""Exact reviewed worker supervisor; disabled jobs fail before any worker.

Own child process group only. Hard expiry includes startup and writes terminal
cost/failure custody even if the child cannot write a Python exception receipt.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from runtime import allocation,verify_manifest,sha,ROOT,PHASE,PYTHON,GPU,proc_start,cell_identity


def main():
    started=time.monotonic()
    parser=argparse.ArgumentParser();parser.add_argument('--job',required=True)
    args=parser.parse_args();job_path=Path(args.job).resolve();job=json.loads(job_path.read_text())
    if job.get('root_execution_authorized') is not True or job.get('source_review_approved') is not True or job.get('fixed_protocol_adopted') is not True:
        raise ValueError('Disabled source/fit authority')
    allocation();assert job['source_manifest_sha256']==verify_manifest()
    receipt_path=(PHASE/job['supervisor_receipt_path']).resolve()
    assert receipt_path.is_relative_to(PHASE) and not receipt_path.exists() and receipt_path.parent.is_dir()
    live=dict(schema='internal-be-live-supervisor-v2',cell_identity=cell_identity(job),job_sha256=sha(job_path),
        supervisor_pid=os.getpid(),supervisor_start_ticks=proc_start(os.getpid()),
        supervisor_source_sha256=sha(ROOT/'supervise.py'),hard_seconds=job['hard_seconds'])
    receipt_path.write_text(json.dumps(live,indent=2,sort_keys=True)+'\n')
    # Avoid a self-referential job/receipt hash. Worker binds the exact path and
    # hash through environment; job binds path, source, identity and deadline.
    env=dict(os.environ,INTERNAL_BE_SUPERVISOR_RECEIPT=str(receipt_path),
             INTERNAL_BE_SUPERVISOR_RECEIPT_SHA256=sha(receipt_path))
    child=None;status='startup_failed';exit_code=None
    try:
        child=subprocess.Popen([str(PYTHON),'-B',str(ROOT/'run.py'),'--job',str(job_path)],
            cwd=str(PHASE.parents[1]),env=env,start_new_session=True)
        child_start=proc_start(child.pid)
        while child.poll() is None:
            if time.monotonic()-started>=job['hard_seconds']:
                # Popen ownership + unchanged start time + own session prove
                # this signal targets this supervisor's child, not a queue peer.
                if proc_start(child.pid)!=child_start or os.getpgid(child.pid)!=child.pid:
                    raise ValueError('Owned child identity changed')
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                status='hard_timeout';break
            time.sleep(.25)
        exit_code=child.wait()
        if status!='hard_timeout':status='complete' if exit_code==0 else 'worker_failed'
    finally:
        terminal=dict(schema='internal-be-supervisor-terminal-v2',live_receipt_sha256=sha(receipt_path),
            cell_identity=live['cell_identity'],status=status,exit_code=exit_code,
            inclusive_seconds=time.monotonic()-started,automatic_retry=False,predictive_scores_closed=True)
        path=receipt_path.with_name(receipt_path.stem+'_TERMINAL.json')
        if path.exists():raise ValueError('No terminal overwrite')
        path.write_text(json.dumps(terminal,indent=2,sort_keys=True)+'\n')
    if exit_code!=0:raise SystemExit(1)


if __name__=='__main__':main()
