"""Run the prospectively fixed remaining fits sequentially; preserve every failure."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import socket
import subprocess
import time

REPO=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE=REPO/'experiments_iclr/postsubmission_20260930'
ROOT=PHASE/'citeseer_endpoint_frame_paired_development_20261005_v1'
SOURCE=PHASE/'citeseer_heart_ncn_trainval_runner_source_20261005_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name,value):
    (ROOT/name).write_text(json.dumps(value,indent=2)+'\n')


def main():
    assert Path.cwd()==REPO and socket.gethostname()=='anogena-2-0'
    assert subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
    assert not (ROOT/'QUEUE_START.json').exists()
    admission=json.loads((ROOT/'DISPATCH_ADMISSION.json').read_text())
    assert admission['source_manifest_sha256']==sha(SOURCE/'SOURCE_MANIFEST.json')
    assert admission['plan_sha256']==sha(ROOT/'PLAN.json')
    qualification=PHASE/'citeseer_frame_native_runtime_qualification_20261005_v1/RESULT.json'
    assert admission['arm_qualification_sha256']==sha(qualification)
    assert json.loads(qualification.read_text())['status']=='PASS'
    plan=json.loads((ROOT/'PLAN.json').read_text())
    rows=plan['physical_fits'];assert len(rows)==36
    baseline=ROOT/'fits/b0_native_single_seed0/FREEZE.json'
    assert sha(baseline)==admission['baseline_freeze_sha256']
    assert json.loads(baseline.read_text())['TEST_access'] is False
    template=json.loads((ROOT/'b0_native_single_seed0.job.json').read_text())
    interpreter=PHASE/'native_ncn_runtime_20261005_v1/.venv/bin/python'
    pythonpath=str(PHASE/'native_ncn_dependency_overlay_20261005_v1')+os.pathsep+str(REPO/'.venv/lib/python3.11/site-packages')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
             PYTHONPATH=pythonpath,PYTHONDONTWRITEBYTECODE='1')
    jobs=ROOT/'jobs';jobs.mkdir()
    write('QUEUE_START.json',dict(UTC=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),plan_sha256=sha(ROOT/'PLAN.json'),physical_fits=36,remaining_fits=35,TEST_access=False))
    started=time.monotonic();completed=[dict(id=rows[0]['id'],freeze_sha256=sha(baseline),baseline_reused=True)]
    try:
        for row in rows[1:]:
            output=ROOT/'fits'/row['id'];assert not output.exists()
            job={k:v for k,v in template.items() if k not in {'id','seed','arm','member_count','paired_seed_block','ensemble_member_index','axis_index','factor_seed'}}
            job.update(row)
            job['exact_arm_runtime_qualification']=dict(path=str(qualification.relative_to(PHASE)),sha256=sha(qualification))
            jobpath=jobs/(row['id']+'.json');jobpath.write_text(json.dumps(job,indent=2)+'\n')
            command=[str(interpreter),'-B',str(SOURCE/'run.py'),'--job',str(jobpath),'--output',str(output)]
            with (jobs/(row['id']+'.stdout.log')).open('xb') as out,(jobs/(row['id']+'.stderr.log')).open('xb') as err:
                child=subprocess.Popen(command,cwd=REPO,env=env,stdout=out,stderr=err)
                write('QUEUE_PROGRESS.json',dict(UTC=datetime.now(timezone.utc).isoformat(),completed=len(completed),total=36,current=row['id'],child_pid=child.pid,command=command))
                exit_code=child.wait()
            if exit_code!=0:raise RuntimeError(row['id']+' failed with exit '+str(exit_code)+'; artifacts/logs preserved, no retry.')
            freeze=output/'FREEZE.json';value=json.loads(freeze.read_text())
            assert value['TEST_access'] is False and value['cohort_plan_sha256']==sha(ROOT/'PLAN.json')
            completed.append(dict(id=row['id'],freeze_sha256=sha(freeze),baseline_reused=False))
            write('QUEUE_PROGRESS.json',dict(UTC=datetime.now(timezone.utc).isoformat(),completed=len(completed),total=36,current=None))
        write('COHORT_FREEZE.json',dict(UTC=datetime.now(timezone.utc).isoformat(),plan_sha256=sha(ROOT/'PLAN.json'),source_manifest_sha256=sha(SOURCE/'SOURCE_MANIFEST.json'),completed_physical_fits=completed,inclusive_queue_seconds=time.monotonic()-started,TEST_access=False,comparative_scoring_performed=False,complete=True))
        print(json.dumps(dict(status='complete',physical_fits=36,TEST_access=False,comparative_scoring_performed=False)))
    except Exception as error:
        write('QUEUE_FAILURE.json',dict(UTC=datetime.now(timezone.utc).isoformat(),error=type(error).__name__+': '+str(error),completed=len(completed),total=36,inclusive_queue_seconds=time.monotonic()-started,TEST_access=False,retry=False,all_partial_artifacts_preserved=True))
        raise


if __name__=='__main__':
    main()
