"""Fixed12 pilot cells with owned-session caps; never reads predictive output."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True
from execution_common import ROOT, PHASE, pin, save, sha, source_inventory, update, utc
from owned_rss import members_of_session

HELD_CHILDREN = []


def gpu_window(uuid):
    result = subprocess.run(['nvidia-smi','-i',uuid,'--query-gpu=uuid,memory.total,memory.free',
                             '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=15)
    fields = [value.strip() for value in result.stdout.strip().split(',')]
    assert len(fields)==3 and fields[0]==uuid
    return dict(UTC=utc(),GPU_UUID=uuid,memory_total_MiB=int(fields[1]),memory_free_MiB=int(fields[2]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--release-sha256',required=True)
    args = parser.parse_args()
    release_path = ROOT/'ROOT_RELEASE.json'
    assert sha(release_path)==args.release_sha256
    release = json.loads(release_path.read_text())
    plan = json.loads((ROOT/'PLAN.json').read_text())
    assert release['execution_authorized'] is True and release['plan_sha256']==sha(ROOT/'PLAN.json')
    assert release['root_source_manifest_sha256']==sha(ROOT/'ROOT_SOURCE_MANIFEST.json')
    assert release['source_inventory']==source_inventory()
    assert os.uname().nodename=='peptide' and str(Path.cwd())==plan['repository']
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==plan['GPU_UUID']
    assert signal.getsignal(signal.SIGCHLD)==signal.SIG_DFL
    assert sys.version_info[:2]==(3,12) and sha(plan['interpreter'])==plan['interpreter_sha256']
    assert sha(ROOT/'SCIENTIFIC_RELEASE.json')==release['scientific_release_sha256']
    assert plan['allocator_fraction']==0.30 and plan['caps']['host_RSS_bytes']==32*1024**3
    assert plan['caps']['wall_seconds_per_cell']==8*3600
    assert plan['minimum_free_MiB_before_each_cell']==34*1024 and plan['minimum_global_headroom_MiB']==10*1024
    expected = [dict(seed=seed,arm=arm,cell_id=f'{arm}_seed{seed}') for seed in (0,1,2)
                for arm in ('native_m1','target_only','joint','separate')]
    assert plan['queue']==expected and plan['epochs_per_cell']==100
    source = PHASE/plan['owned_supervisor_source']
    assert sha(source)==plan['owned_supervisor_sha256']
    sys.path.insert(0,str(source.parent))
    spec = importlib.util.spec_from_file_location('reviewed_owned_identity_primitives',source)
    owned = importlib.util.module_from_spec(spec); spec.loader.exec_module(owned)
    received = []
    for signum in (signal.SIGINT,signal.SIGTERM):
        signal.signal(signum,lambda number,frame:received.append(number))
    save(ROOT/'QUEUE_STARTED.json',dict(UTC=utc(),root_release_sha256=args.release_sha256,
         queue=expected,predictive_values_read=False,automatic_retry=False))
    completed, terminals, failure = [], [], None
    for cell in expected:
        if received:
            failure = dict(cell=cell,reason='SUPERVISOR_SIGNAL_BEFORE_CELL'); break
        window = gpu_window(plan['GPU_UUID'])
        if window['memory_free_MiB']<plan['minimum_free_MiB_before_each_cell']:
            failure = dict(cell=cell,reason='INSUFFICIENT_CAPACITY_BEFORE_CELL',window=window); break
        slot = ROOT/'supervision'/cell['cell_id']; slot.mkdir(parents=True,exist_ok=False)
        fit = ROOT/'fits'/cell['arm']/f'seed_{cell["seed"]}'
        assert not fit.exists()
        command = [plan['interpreter'],'-B',str(PHASE/plan['candidate_packet']/'train_cell.py'),
                   '--config',str(ROOT/'SCIENTIFIC_RELEASE.json'),'--arm',cell['arm'],
                   '--seed',str(cell['seed']),'--device','cuda:0','--output-dir',str(fit)]
        child = identity = None
        started = time.monotonic(); peak_rss = 0
        stop, errors, actions = None, [], []
        session_closed = reaped = False
        last_window, next_window, next_progress = window, started, started
        # These private logs contain scientific values and are excluded from the monitor.
        with (slot/'PRIVATE_STDOUT.txt').open('xb') as out,(slot/'PRIVATE_STDERR.txt').open('xb') as err:
            try:
                child = subprocess.Popen(command,cwd=plan['repository'],stdin=subprocess.DEVNULL,
                                         stdout=out,stderr=err,start_new_session=True)
                HELD_CHILDREN.append(child)
                identity = owned.process_identity(child.pid)
                assert identity is not None and identity['session']==identity['group']==child.pid
                save(slot/'OWNED_CHILD.json',dict(identity=identity,command=command,cell=cell,
                     root_release_sha256=args.release_sha256,window_before_launch=window))
                while True:
                    owned.held_identity(child,identity)
                    members = members_of_session(child.pid)
                    assert any(row['pid']==child.pid and row['start_ticks']==identity['start_ticks'] for row in members)
                    peak_rss = max(peak_rss,sum(row['RSS_bytes'] for row in members))
                    live = [row for row in members if row['state']!='Z']
                    ended = os.waitid(os.P_PID,child.pid,os.WEXITED|os.WNOHANG|os.WNOWAIT)
                    if any(row['group']!=child.pid for row in members): stop='UNEXPECTED_OWNED_PROCESS_GROUP'
                    elif received: stop='SUPERVISOR_SIGNAL'
                    elif time.monotonic()-started>=plan['caps']['wall_seconds_per_cell']: stop='OWNED_WALL_CAP'
                    elif peak_rss>plan['caps']['host_RSS_bytes']: stop='OWNED_RSS_CAP'
                    if stop is None and live and time.monotonic()>=next_window:
                        last_window = gpu_window(plan['GPU_UUID']); next_window=time.monotonic()+5
                        if last_window['memory_free_MiB']<plan['minimum_global_headroom_MiB']:
                            stop='PROTECT_GLOBAL_GPU_HEADROOM'
                    if stop:
                        owned.kill_owned(child,identity,actions,stop); break
                    if ended is not None and not live:
                        assert ended.si_pid==child.pid
                        child.wait(timeout=0); reaped=session_closed=True; break
                    if time.monotonic()>=next_progress:
                        update(ROOT/'QUEUE_PROGRESS.json',dict(UTC=utc(),status='RUNNING',current_cell=cell,
                            completed_cells=completed,elapsed_cell_seconds=time.monotonic()-started,
                            sampled_peak_session_RSS_bytes=peak_rss,owned_session_members=members,
                            last_GPU_window=last_window,predictive_values_read=False))
                        next_progress=time.monotonic()+5
                    time.sleep(0.25)
            except BaseException as error:
                stop=stop or 'SUPERVISION_EXCEPTION'
                errors.append(dict(type=type(error).__name__,message=str(error)))
            finally:
                if child is not None and not reaped:
                    try:
                        owned.kill_owned(child,identity,actions,stop or 'FINAL_OWNED_CLEANUP')
                        deadline=time.monotonic()+30
                        while time.monotonic()<deadline:
                            owned.held_identity(child,identity)
                            members=members_of_session(child.pid)
                            assert any(row['pid']==child.pid and row['start_ticks']==identity['start_ticks'] for row in members)
                            peak_rss=max(peak_rss,sum(row['RSS_bytes'] for row in members))
                            ended=os.waitid(os.P_PID,child.pid,os.WEXITED|os.WNOHANG|os.WNOWAIT)
                            if ended is not None and all(row['state']=='Z' and row['group']==child.pid for row in members):
                                assert ended.si_pid==child.pid
                                child.wait(timeout=0); reaped=session_closed=True; break
                            time.sleep(0.25)
                    except BaseException as error:
                        errors.append(dict(type=type(error).__name__,message=str(error)))
        terminal = dict(UTC=utc(),cell=cell,identity=identity,
            physical_exit_code=None if child is None else child.returncode,direct_child_reaped=reaped,
            physical_session_closed=session_closed,wall_seconds=time.monotonic()-started,
            sampled_peak_session_RSS_bytes=peak_rss,stop=stop,errors=errors,termination_actions=actions,
            last_GPU_window=last_window,root_release_sha256=args.release_sha256,
            predictive_values_read=False,scientific_completion_adopted=False,automatic_retry=False,
            RSS_method='same_proc_stat_snapshot_identity_and_RSS_pages',RSS_sample_interval_seconds=0.25,
            RSS_scope='maximum sampled sum over owned session tasks; not true or instantaneous peak RSS',
            instantaneous_RSS_guarantee=False,process_escape_sandbox=False)
        save(slot/'PHYSICAL_TERMINAL.json',terminal); terminals.append(terminal)
        if not(child is not None and child.returncode==0 and reaped and session_closed and stop is None and not errors):
            failure=dict(cell=cell,reason='FAILED_OR_INCOMPLETE_PHYSICAL_CELL',terminal=terminal); break
        completed.append(cell)
    complete = failure is None and len(completed)==12
    final=dict(UTC=utc(),status='COMPLETE_PHYSICAL_12CELL_QUEUE' if complete else 'FAILED_OR_INCOMPLETE',
        completed_cells=completed,terminals=terminals,failure=failure,
        unattempted_cells=expected[len(terminals):],root_release_sha256=args.release_sha256,
        predictive_values_read=False,scientific_completion_adopted=False,full500_qualification=False,automatic_retry=False)
    save(ROOT/'QUEUE_TERMINAL.json',final); update(ROOT/'QUEUE_PROGRESS.json',final)
    # Preserve held handles until exact child reaping/session closure; no destructor poll.
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0 if complete else 1)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        if not isinstance(error,SystemExit):
            path=ROOT/'QUEUE_EXCEPTION.json'
            if not path.exists(): save(path,dict(UTC=utc(),type=type(error).__name__,message=str(error),predictive_values_read=False,automatic_retry=False))
            sys.stdout.flush(); sys.stderr.flush(); os._exit(1)
        raise
