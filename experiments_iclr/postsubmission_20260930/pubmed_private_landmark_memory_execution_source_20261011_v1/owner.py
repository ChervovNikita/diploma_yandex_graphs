"""Original direct-child timer/terminate/kill/reap ownership, parameterized once."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from runtime import HERE,REPO,UUID,checked,receipt,route,write

def identity(pid):
    path=Path('/proc')/str(pid)/'stat'
    if not path.exists():return None
    fields=path.read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,start_ticks=int(fields[19]),pgid=int(fields[2]))

def main():
    route();parser=argparse.ArgumentParser();parser.add_argument('config',type=Path);args=parser.parse_args()
    config=json.loads(args.config.read_text());cap=config['limits']['wall_seconds']
    if type(cap) is not int or cap<=0:raise ValueError('Positive finite wall required')
    output=Path(config['owner_output']);output.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=UUID,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',PILOT_OWNER_PID=str(os.getpid()))
    env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
    binding=json.loads((HERE/'SOURCE_BINDINGS.json').read_text())['server_data_and_runtime']['runtime']
    interpreter=checked(binding['python']);env['PYTHONPATH']=binding['PYTHONPATH']
    argv=[str(interpreter),'-B','-u',str(Path(__file__).with_name('worker.py')),str(args.config.resolve())]
    start=time.monotonic();end=start+cap+60
    result=dict(stage=config['stage'],config=receipt(args.config),source_manifest=config['source_manifest'],root_decision=config['root_decision'],output=config['output'],child_wall_seconds=cap,external_wall_seconds=cap+60,direct_child_wait=True,retry=False)
    with (output/'child.log').open('xb') as log:
        child=subprocess.Popen(argv,cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
        birth=identity(child.pid);assert birth and birth['pgid']==child.pid
        result.update(child=birth,owner=identity(os.getpid()),command=argv);write(output/'OWNER_START.json',result)
        print(json.dumps(dict(status='OWNER_STARTED',stage=config['stage'],child_pid=child.pid,child_wall_seconds=cap)),flush=True)
        timeout=False
        try:code=child.wait(timeout=cap)
        except subprocess.TimeoutExpired:
            timeout=True;assert identity(child.pid)==birth;os.killpg(child.pid,signal.SIGTERM)
            try:code=child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                assert identity(child.pid)==birth;os.killpg(child.pid,signal.SIGKILL);code=child.wait(timeout=max(.1,end-time.monotonic()-15))
        cuda_pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=15).splitlines()
        result.update(status='COMPLETE_EXIT_0' if code==0 and not timeout else 'FAILED_RETAINED',exit_code=code,timed_out=timeout,seconds=time.monotonic()-start,child_pid_absent=identity(child.pid) is None,owned_cuda_pid_absent=str(child.pid) not in [s.strip() for s in cuda_pids])
        artifact=Path(config['output'])/dict(prepare='CUSTODY.json',qualify='QUALIFICATION.json',acquire='PANEL.json')[config['stage']]
        if artifact.is_file():result['produced_artifact']=receipt(artifact)
        write(output/'OWNER_END.json',result)
    print(json.dumps(result),flush=True)
    raise SystemExit(0 if result['status']=='COMPLETE_EXIT_0' and result['child_pid_absent'] and result['owned_cuda_pid_absent'] else 1)
if __name__=='__main__':main()
