"""Bounded scheduling observation of two exact owned jobs, with automatic resume."""
from pathlib import Path
import datetime
import json
import os
import signal
import socket
import subprocess
import time

R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
P=R/'experiments_iclr/postsubmission_20260930'
D=Path(__file__).resolve().parent
A=P/'context_positive_stage1_scientific_activation_root_20261008_v1'
C=P/'context_positive_stage1_scientific_execution_root_20261008_v1/8101_shared_route_permuted/PROGRESS.json'
M=P/'internal_BE_molhiv18_family_execution_root_20261007_v1/fits/P_7101/PROGRESS.json'
CHILD=(527200,6022574923)
CONTEXT_PARENT=(526200,6021896966)
MOL_PARENT=(523400,6019318952)
MOL_CHILD=(526095,6021762957)


def identity(pid):
    try:
        raw=(Path('/proc')/str(pid)/'stat').read_text()
        f=raw[raw.rfind(')')+2:].split()
        return dict(pid=pid,start_ticks=int(f[19]),state=f[0],
                    CPU_ticks=int(f[11])+int(f[12]))
    except FileNotFoundError:
        return None


def exact(pair):
    obj=identity(pair[0])
    return obj if obj and obj['start_ticks']==pair[1] else None


def progress(path):
    if not path.is_file():
        return None
    obj=json.loads(path.read_text())
    return {k:v for k,v in obj.items() if k in ('epoch','steps','complete_epochs','complete')}


def snapshot():
    return dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        context_parent=exact(CONTEXT_PARENT),context_child=exact(CHILD),
        mol_parent=exact(MOL_PARENT),mol_child=exact(MOL_CHILD),
        context_progress=progress(C),mol_progress=progress(M),
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free,utilization.gpu',
            '--format=csv,noheader,nounits'],text=True,timeout=10).strip())


def write(name,obj):
    tmp=D/(name+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    os.replace(tmp,D/name)


def main():
    assert Path.cwd()==R and socket.gethostname()=='anogena-2-0'
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],
        text=True,timeout=10).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
    assert not (D/'RESULT.json').exists() and not (D/'START.json').exists()
    for pair in (CHILD,CONTEXT_PARENT,MOL_PARENT,MOL_CHILD):
        assert exact(pair),('Original live process required',pair)
    current=json.loads((A/'CURRENT.json').read_text())
    assert current['cell_id']=='8101_shared_route_permuted'
    assert current['child']['pid']==CHILD[0] and current['child']['start_ticks']==CHILD[1]
    argv=(Path('/proc')/str(CHILD[0])/'cmdline').read_bytes().split(b'\0')
    assert any(b'context_positive' in item and str(P).encode() in item for item in argv)
    record=dict(operation='Bounded scheduling probe, not a fit or quality comparison',
        duration_seconds=180,source_or_recipe_or_horizon_or_original_bounds_changed=False,
        quality_or_dataset_or_checkpoint_access=False,scientific_retry=False,
        owner=identity(os.getpid()),before=snapshot(),samples=[],signals=[],resume=None)
    write('START.json',record)
    def interrupted(signum,frame):
        raise RuntimeError('Probe interrupted by signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    stopped=False
    began=time.monotonic()
    try:
        assert exact(CHILD)
        os.kill(CHILD[0],signal.SIGSTOP)
        stopped=True
        record['signals'].append('SIGSTOP_EXACT_OWNED_CONTEXT_WORKER')
        stop_deadline=time.monotonic()+5
        while True:
            actual=exact(CHILD)
            assert actual,'Context worker identity vanished'
            if actual['state'] in ('T','t'):
                break
            if time.monotonic()>=stop_deadline:
                raise RuntimeError('Owned worker did not enter stopped state within5seconds')
            time.sleep(.1)
        record['stopped_confirmed']=actual
        write('PROGRESS.json',record)
        deadline=began+180
        while time.monotonic()<deadline:
            time.sleep(min(30,deadline-time.monotonic()))
            actual=exact(CHILD)
            assert actual and actual['state'] in ('T','t'),'Paused original worker must remain live'
            record['samples'].append(snapshot())
            write('PROGRESS.json',record)
    except BaseException as error:
        record['failure']=dict(type=type(error).__name__,message=str(error))
        raise
    finally:
        # Interrupts cannot prevent restoring the exact worker we stopped.
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        signal.signal(signal.SIGINT,signal.SIG_IGN)
        if stopped:
            actual=exact(CHILD)
            if actual:
                os.kill(CHILD[0],signal.SIGCONT)
                record['signals'].append('SIGCONT_EXACT_OWNED_CONTEXT_WORKER')
                resumed_deadline=time.monotonic()+5
                while time.monotonic()<resumed_deadline:
                    actual=exact(CHILD)
                    if actual is None or actual['state'] not in ('T','t'):
                        break
                    time.sleep(.1)
                record['resume']=dict(signal_sent=True,actual=actual,
                    resumed_or_terminal=actual is None or actual['state'] not in ('T','t'))
            else:
                record['resume']=dict(signal_sent=False,original_worker_missing=True)
        record['elapsed_seconds']=time.monotonic()-began
        record['after']=snapshot()
        write('RESULT.json',record)


if __name__=='__main__':
    main()
