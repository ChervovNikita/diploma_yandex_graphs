"""One root-released CPU sampler, bounded to 2100s; no fit or GPU work."""
from datetime import datetime,timezone
from pathlib import Path
import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import time

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def identity(pid):
    try:
        p=Path('/proc')/str(pid);raw=(p/'stat').read_text();fields=raw[raw.rfind(')')+2:].split()
        return {'PID':pid,'start_ticks':int(fields[19]),'pgid':int(fields[2]),'sid':int(fields[3]),
                'state':fields[0],'argv':[x.decode() for x in (p/'cmdline').read_bytes().split(bytes([0])) if x]}
    except FileNotFoundError:
        return None

def save(path,value):
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True);stream.write('\n')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release',type=Path,required=True)
    p.add_argument('--release-sha256',required=True)
    a=p.parse_args()
    assert sha(a.release)==a.release_sha256
    release=json.loads(a.release.read_text())
    assert release['sampling_execution_enabled'] is True and release['retry'] is False
    assert release['supervisor_sha256']==sha(__file__) and release['hard_seconds']==2100 and release['soft_seconds']==1800
    assert release['models_authorized'] is False and release['fits_authorized'] is False
    assert release['VALID_values_access'] is False and release['TEST_access'] is False
    repo=Path(release['repository']);phase=repo/'experiments_iclr/postsubmission_20260930'
    assert Path.cwd().resolve()==repo and socket.gethostname()==release['expected_hostname']
    root=a.release.resolve().parent;assert root.is_relative_to(phase) and not (root/'CHILD_STARTED.json').exists()
    assert not (root/'EXECUTION_RECEIPT.json').exists() and not Path(release['output_directory']).exists()
    program=Path(release['program_path']);assert program.is_relative_to(phase) and sha(program)==release['program_sha256']
    env=dict(os.environ);env.pop('PYTHONHOME',None)
    env.update(CUDA_VISIBLE_DEVICES='',PYTHONPATH=release['declared_PYTHONPATH'],PYTHONNOUSERSITE='1',
               PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2')
    argv=[release['python_executable'],'-B',str(program),'--release',str(a.release),'--release-sha256',a.release_sha256]
    started=time.monotonic();signals=[];reason=None
    with (root/'child.stdout.log').open('xb') as out,(root/'child.stderr.log').open('xb') as err:
        child=subprocess.Popen(argv,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
        owner=identity(child.pid)
        assert owner is not None and owner['pgid']==owner['sid']==child.pid
        save(root/'CHILD_STARTED.json',{'UTC':datetime.now(timezone.utc).isoformat(),'child_identity':owner,
             'declared_argv':argv,'CUDA_VISIBLE_DEVICES':'','declared_PYTHONPATH':release['declared_PYTHONPATH'],
             'hard_seconds':2100,'soft_seconds':1800,'attempts':1,'retry':False,'fits':0})
        try:
            child.wait(timeout=2100)
        except subprocess.TimeoutExpired:
            reason='hard_wall_bound';live=identity(child.pid)
            if live is not None and live['state']!='Z':
                assert live['start_ticks']==owner['start_ticks'] and live['pgid']==live['sid']==child.pid
                os.killpg(child.pid,signal.SIGKILL)
                signals.append({'PID':child.pid,'start_ticks':owner['start_ticks'],'signal':'SIGKILL','reason':reason})
            child.wait(timeout=20)
    assert identity(child.pid) is None
    result=Path(release['output_directory'])/'RESULT.json'
    receipt={'UTC':datetime.now(timezone.utc).isoformat(),'status':'PASS' if child.returncode==0 and result.exists() and reason is None else 'FAIL',
             'release_sha256':a.release_sha256,'program_sha256':release['program_sha256'],'supervisor_sha256':sha(__file__),
             'exit_code':child.returncode,'exit_code_authority':'subprocess.Popen.wait','terminal_wait_observed':True,
             'child_identity':owner,'observed_cwd':str(repo),'reason':reason,'signals_sent':signals,
             'owned_PID_absent':True,'attempts':1,'retry':False,'models_executed':False,'fits':0,
             'VALID_TEST_values_access':False,'wall_seconds':time.monotonic()-started,'result_sha256':sha(result) if result.exists() else None,
             'logs':{name:{'bytes':(root/name).stat().st_size,'sha256':sha(root/name)} for name in ('child.stdout.log','child.stderr.log')}}
    save(root/'EXECUTION_RECEIPT.json',receipt)
    print(json.dumps(receipt),flush=True)

if __name__=='__main__':
    main()
