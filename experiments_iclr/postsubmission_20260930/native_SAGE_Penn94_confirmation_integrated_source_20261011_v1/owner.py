"""Existing finite native-owner pattern, scoped to one admitted Penn94 stage."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import subprocess
import time
from contracts import gate, scoped, read, write, describe, require, UUID, REPO


def identity(pid):
    p = Path('/proc',str(pid),'stat')
    if not p.exists(): return None
    fields=p.read_text().rsplit(')',1)[1].split()
    return {'PID':pid,'start_ticks':int(fields[19]),'state':fields[0]}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--admission',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=('qualify','acquire','calibrate','confirm'),required=True);args=parser.parse_args()
    cfg,admission,custody=gate(args.config,args.admission,args.stage)
    output=scoped(args.output);require(not output.exists(),'Fresh stage output only')
    sidecar=scoped(output.parent/(output.name+'_owner'));sidecar.mkdir(exist_ok=False)
    require(identity(os.getpid()) is not None,'Owner proc identity required')
    worker='runtime.py' if args.stage in ('qualify','acquire') else 'readout.py'
    command=[str(REPO/'.venv/bin/python'),'-B',str(Path(__file__).with_name(worker)), '--config',str(args.config),
        '--admission',str(args.admission),'--output',str(output),'--stage',args.stage]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=UUID,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
    start=time.monotonic()
    with (sidecar/'stdout.log').open('x') as stdout,(sidecar/'stderr.log').open('x') as stderr:
        child=subprocess.Popen(command,cwd=REPO,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
        child_id=identity(child.pid)
        write(sidecar/'OWNER_START.json',{'UTC':datetime.now(timezone.utc).isoformat(),'owner':identity(os.getpid()),
            'child':child_id,'stage':args.stage,'finite_seconds':cfg['owner_seconds'][args.stage],
            'custody':custody,'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()})
        timed_out=False
        try:code=child.wait(timeout=cfg['owner_seconds'][args.stage])
        except subprocess.TimeoutExpired:
            timed_out=True;os.killpg(child.pid,signal.SIGTERM)
            try:code=child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait(timeout=15)
    cuda_pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines()
    stage_path=output/'STAGE_END.json';stage=read(stage_path) if stage_path.exists() else None
    stage_terminal=stage is not None and stage.get('terminal') is True
    value={'UTC':datetime.now(timezone.utc).isoformat(),'exit_code':code,'timed_out':timed_out,'direct_child_wait':True,
        'child':child_id,'child_pid_absent':identity(child.pid) is None,'owned_cuda_pid_absent':str(child.pid) not in [p.strip() for p in cuda_pids],
        'stage_terminal':stage_terminal,'stage_result':stage,'stage':args.stage,'seconds':time.monotonic()-start,
        'custody':custody,'worker_command':command,'stage_end_receipt':describe(stage_path) if stage_path.exists() else None,
        'scientific_success':code==0 and stage_terminal and stage.get('success') is True,
        'CONFIRM_truth_access':args.stage=='confirm','failure_or_timeout_preserved':timed_out or code!=0,
        'cost_scope':'Observed normal-host sequential stage costs; no isolated benchmark or ETA.'}
    write(sidecar/'OWNER_END.json',value)
    return 0 if code==0 and stage_terminal and value['child_pid_absent'] and value['owned_cuda_pid_absent'] else 1


if __name__=='__main__':raise SystemExit(main())
