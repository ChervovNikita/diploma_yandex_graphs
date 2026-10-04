"""One fresh PENCIL resource child; inherited bounded ownership/physical terminal."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from common import (DATA_FILES, HERE, PHASE, final_file_custody, gate, pin, require, sha, write)

# Retain the Popen object until os._exit. Its destructor otherwise polls and
# can reap an unresolved zombie leader before its owned session is closed.
HELD_CHILDREN = []


def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def durable_write(path, value):
    write(path, value)
    fsync_directory(path.parent)


def process_identity(pid):
    try:
        raw = (Path('/proc')/str(pid)/'stat').read_text()
        fields = raw[raw.rfind(')')+2:].split()
        return dict(pid=pid,start_ticks=int(fields[19]),session=int(fields[3]),group=int(fields[2]),state=fields[0])
    except FileNotFoundError:
        return None


def members_of_session(pid):
    members = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            item = process_identity(int(path.name))
            if item is None or item['session'] != pid:
                continue
            text = (path/'status').read_text()
            lines = [line for line in text.splitlines() if line.startswith('VmRSS:')]
            require(len(lines) == 1 or item['state'] == 'Z', 'Owned live RSS observation missing')
            rss = int(lines[0].split()[1])*1024 if lines else 0
            members.append(dict(item,RSS_bytes=rss))
        except FileNotFoundError:
            continue
    return members


def held_identity(process, identity):
    current = process_identity(process.pid)
    require(current is not None and current['start_ticks'] == identity['start_ticks']
            and current['session'] == current['group'] == process.pid, 'Held direct-child identity changed')
    return current


def kill_owned(process, identity, actions, reason):
    if process is None:
        return
    # No poll/wait has reaped this exact direct child. Default SIGCHLD is
    # required before spawn. Only its original dedicated group is signalled.
    current = process_identity(process.pid)
    if current is None:
        actions.append(dict(reason=reason,action='IDENTITY_UNRESOLVED_NO_SIGNAL'))
        return
    if identity is not None:
        require(current['start_ticks'] == identity['start_ticks'], 'Birth identity differs; no signal')
    require(current['session'] == current['group'] == process.pid, 'Owned group differs; no broad signal')
    os.killpg(process.pid,signal.SIGKILL)
    actions.append(dict(UTC=datetime.now(timezone.utc).isoformat(),reason=reason,pid=process.pid,
                        session=current['session'],group=current['group'],signal='SIGKILL'))


def read_json(path):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 64*1024**2, 'Invalid/big receipt file')
    return json.loads(path.read_text())


def collect(execution, plan, runtime, release_sha):
    child = execution/'run01'
    required = {'RESOURCE.json','PROGRESS.json','FILE_CUSTODY.json','FINAL_CUSTODY.json'}
    require(child.is_dir() and not child.is_symlink(), 'Resource output absent')
    require({path.name for path in child.iterdir()} == required
            and all(path.is_file() and not path.is_symlink() for path in child.iterdir()), 'Partial/extra child output')
    result, final, files = (read_json(child/name) for name in ('RESOURCE.json','FINAL_CUSTODY.json','FILE_CUSTODY.json'))
    require(result.get('status') == 'COMPLETE_RESOURCE_ONLY' and result.get('schema') == 'pencil-collab-resource-v1'
            and result.get('source_manifest_sha256') == sha(HERE/'MANIFEST.json')
            and result.get('release_sha256') == release_sha and result.get('workload') == plan['workload']
            and result.get('data_files_opened') == list(DATA_FILES)
            and result.get('TEST_reads') is False and result.get('predictive_metrics_computed') is False
            and result.get('scores_saved') is False and result.get('checkpoint_saved') is False
            and result.get('state_donor') is False and result.get('completed_native_epochs') == 1,
            'Resource scope differs')
    expected_batches = (result['TRAIN_queries'] + 1023)//1024
    require(result['TRAIN_batches'] == expected_batches and result['optimizer_updates'] == (expected_batches+7)//8
            and result['VALID_queries'] == 160084 and result['VALID_batches'] == 157
            and result['VALID_sequential_indices'] is True and result['all_observed_outputs_finite'] is True,
            'Native epoch/VALID coverage incomplete')
    require(result['shapes']['structural_width'] == 306 and result['shapes']['raw_feature_width'] == 128
            and 2 <= result['shapes']['max_sequence_positions'] <= 154,
            'Observed actual PENCIL shape differs')
    require(result['cuda_peak_allocated_bytes'] <= plan['caps']['cuda_allocated_bytes']
            and result['cuda_peak_reserved_bytes'] <= plan['caps']['cuda_reserved_bytes']
            and result['inclusive_child_wall_seconds'] <= plan['caps']['wall_seconds'], 'Observed child caps differ')
    require(final.get('completed') is True and final.get('file_custody_sha256') == sha(child/'FILE_CUSTODY.json')
            and final['inclusive_child_wall_seconds_through_custody'] <= plan['caps']['wall_seconds'], 'Final custody incomplete')
    rows = final['files']
    require(len(rows) == 3 and {row['path'] for row in rows} == required-{'FINAL_CUSTODY.json'}, 'Final output set differs')
    for row in rows:
        require(Path(row['path']).name == row['path'], 'Output custody path differs'); pin(child/row['path'], row)
    expected = final_file_custody(plan,runtime,execution/'ROOT_RELEASE.json',release_sha,
                                 Path(json.loads((HERE/'metadata/DATA_AUTHORITY.json').read_text())['dataset_root']).resolve())
    require(files == expected and files['status'] == 'MATCH', 'Final source/runtime/data custody differs')
    gate(execution/'ROOT_RELEASE.json',release_sha)
    fsync_directory(child)
    return dict(status='COLLECTED_COMPLETE_RESOURCE_ONLY',result_path=str(child/'RESOURCE.json'),
                result_sha256=sha(child/'RESOURCE.json'),final_custody_path=str(child/'FINAL_CUSTODY.json'),
                final_custody_sha256=sha(child/'FINAL_CUSTODY.json'),file_custody_sha256=sha(child/'FILE_CUSTODY.json'))



def inventory(path):
    if not path.exists():
        return []
    rows = []
    for item in sorted(path.rglob('*')):
        require(not item.is_symlink() and item.resolve().is_relative_to(path), 'Invalid owned output path')
        if item.is_file():
            rows.append(dict(path=str(item.relative_to(path)),bytes=item.stat().st_size,sha256=sha(item)))
    return rows


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument('--release',type=Path,required=True)
    parser.add_argument('--release-sha256',required=True)
    args = parser.parse_args()
    plan,runtime,execution = gate(args.release,args.release_sha256)
    source_manifest_sha = sha(HERE/'MANIFEST.json')
    require(sys.platform=='linux' and signal.getsignal(signal.SIGCHLD)==signal.SIG_DFL,
            'Linux/default SIGCHLD required for held waitid/WNOWAIT identity')
    require(not (execution/'run01').exists() and not (execution/'supervision/run01').exists()
            and execution.resolve()==execution, 'Fresh canonical output required')
    execution.mkdir(parents=True,exist_ok=True)
    lock = execution/'RESOURCE_ATTEMPT_SPENT.json'
    with lock.open('x') as stream:
        json.dump(dict(UTC=datetime.now(timezone.utc).isoformat(),release_sha256=args.release_sha256,
            source_manifest_sha256=source_manifest_sha,parent_pid=os.getpid(),automatic_retry=False),stream)
        stream.flush();os.fsync(stream.fileno())
    fsync_directory(execution)
    output = execution/'supervision/run01';output.mkdir(parents=True,exist_ok=False)
    fsync_directory(output.parent)
    process = identity = None;stop = None;peak = 0
    physical_complete = False;reaped = False;received = [];errors = [];actions = [];last_members = [];exit_observed = None
    command = [sys.executable,'-B',str(HERE/'worker.py'),'--release',str(args.release),'--release-sha256',args.release_sha256]
    rank_environment = dict(RANK='0',LOCAL_RANK='0',WORLD_SIZE='1')
    child_environment = dict(os.environ,**rank_environment)
    durable_write(output/'STARTED.json',dict(UTC=datetime.now(timezone.utc).isoformat(),command=command,
        source_manifest_sha256=source_manifest_sha,release_sha256=args.release_sha256,caps=plan['caps'],automatic_retry=False,
        rank_environment=rank_environment,rendezvous_file=str(execution/'run01/RANK0_RENDEZVOUS')))
    def interrupted(number,frame):
        nonlocal stop
        received.append(number)
        stop = stop or "SUPERVISOR_SIGNAL"
    handlers = {number:signal.signal(number,interrupted) for number in (signal.SIGTERM,signal.SIGINT)}
    try:
        with (output/'STDOUT.txt').open('xb') as out,(output/'STDERR.txt').open('xb') as err:
            process = subprocess.Popen(command,cwd=plan['repository'],stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,env=child_environment)
            HELD_CHILDREN.append(process)
            identity = process_identity(process.pid)
            require(identity is not None and identity['session']==identity['group']==process.pid, 'Owned child session missing')
            durable_write(output/'LAUNCH.json',dict(identity=identity,command=command,release_sha256=args.release_sha256,rank_environment=rank_environment,
                rendezvous_file=str(execution/'run01/RANK0_RENDEZVOUS')))
            while True:
                held_identity(process,identity)
                ended = os.waitid(os.P_PID,process.pid,os.WEXITED|os.WNOHANG|os.WNOWAIT)
                last_members = members_of_session(process.pid)
                require(any(item['pid']==process.pid for item in last_members), 'Held session leader absent')
                peak = max(peak,sum(item['RSS_bytes'] for item in last_members))
                if any(item['group']!=process.pid for item in last_members):stop='UNEXPECTED_OWNED_PROCESS_GROUP'
                if received:stop='SUPERVISOR_SIGNAL'
                if time.monotonic()-started>plan['caps']['wall_seconds']:stop='WALL'
                if peak>plan['caps']['host_RSS_bytes']:stop='SAMPLED_SESSION_RSS'
                if sum(path.stat().st_size for path in execution.rglob('*') if path.is_file())>plan['caps']['output_bytes']:stop='OUTPUT_BYTES'
                if ended is not None:
                    require(ended.si_pid==process.pid,'Exact child waitid differs')
                    exit_observed=dict(pid=ended.si_pid,code=ended.si_code,status=ended.si_status)
                if stop:
                    kill_owned(process,identity,actions,stop);break
                if ended is not None:
                    require(all(item['state']=='Z' for item in last_members),'Live owned session remains after exit')
                    process.wait(timeout=0);reaped=True;physical_complete=True;break
                durable_write(output/'STATUS.json',dict(elapsed_seconds=time.monotonic()-started,members=last_members,peak_observed_session_RSS_bytes=peak))
                time.sleep(.25)
    except BaseException as error:
        stop=stop or 'SUPERVISOR_EXCEPTION'
        errors.append(type(error).__name__+': '+str(error))
    finally:
        if process is not None and not reaped:
            try:kill_owned(process,identity,actions,stop or 'FINAL_CLEANUP')
            except BaseException as error:errors.append(type(error).__name__+': '+str(error))
            deadline=time.monotonic()+plan['parent_cleanup_seconds']
            while time.monotonic()<deadline:
                try:
                    if identity is not None:held_identity(process,identity)
                    ended=os.waitid(os.P_PID,process.pid,os.WEXITED|os.WNOHANG|os.WNOWAIT)
                    last_members=members_of_session(process.pid)
                    peak=max(peak,sum(item['RSS_bytes'] for item in last_members))
                    if ended is not None:
                        require(ended.si_pid==process.pid,'Cleanup exact waitid differs')
                        exit_observed=dict(pid=ended.si_pid,code=ended.si_code,status=ended.si_status)
                        require(any(item['pid']==process.pid for item in last_members),'Cleanup held leader absent')
                        if all(item['state']=='Z' and item['group']==process.pid for item in last_members):
                            process.wait(timeout=0);reaped=True;physical_complete=True;break
                except BaseException as error:errors.append(type(error).__name__+': '+str(error))
                time.sleep(.05)
            if not reaped:stop=stop or 'CLEANUP_UNRESOLVED'
        for number,handler in handlers.items():signal.signal(number,handler)
        if received:
            stop = stop or "SUPERVISOR_SIGNAL"
        physical=dict(UTC=datetime.now(timezone.utc).isoformat(),identity=identity,physical_session_closed=physical_complete,
            direct_child_reaped=reaped,physical_exit_code=None if process is None else process.returncode,
            exit_observed_unreaped=exit_observed,stop=stop,received_supervisor_signals=list(received),errors=errors,termination_actions=actions,
            last_owned_session_members=last_members,wall_seconds=time.monotonic()-started,
            peak_observed_session_RSS_bytes=peak,source_manifest_sha256=source_manifest_sha,
            release_sha256=args.release_sha256,sampled_RSS_interval_seconds=.25,instantaneous_OS_RSS_limit=False,
            process_escape_sandbox=False,completed_scientific_fits=0,TEST_reads=False,VALID_resource_reads=True,automatic_retry=False,
            unresolved_cleanup=(process is not None and not reaped),
            ownership_relinquished_on_supervisor_exit=(process is not None and not reaped),
            implicit_Popen_destructor_reap_disabled=True)
        # Physical evidence is committed before any child JSON/hash collection.
        physical_sha = None;physical_commit_error = None
        try:
            durable_write(output/'PHYSICAL_TERMINAL.json',physical)
            physical_sha = sha(output/'PHYSICAL_TERMINAL.json')
        except BaseException as error:
            physical_commit_error = type(error).__name__+': '+str(error)
        linked=dict(status='MISSING_OR_UNQUALIFIED')
        final_budget = dict(status='NOT_CHECKED',last_completed_check=None)
        def check_final_parent_budget(phase):
            # Check after the finite output scan, so the recorded elapsed value
            # includes that scan. These are observations at named checkpoints,
            # not an instantaneous/infinite guarantee for future publication.
            observed_output_bytes = sum(path.stat().st_size for path in execution.rglob('*') if path.is_file())
            observed_elapsed = time.monotonic()-started
            passed = observed_elapsed <= plan['caps']['wall_seconds'] and observed_output_bytes <= plan['caps']['output_bytes']
            final_budget.update(status='PASS' if passed else 'FAIL',last_completed_check=phase,
                inclusive_parent_wall_seconds_at_last_completed_check=observed_elapsed,
                observed_output_bytes_at_last_completed_check=observed_output_bytes,
                wall_seconds_cap=plan['caps']['wall_seconds'],output_bytes_cap=plan['caps']['output_bytes'],
                sampled_RSS_only=True,post_child_parent_RSS_unsampled=True)
            require(passed,'Final parent elapsed/output cap exceeded at '+phase)
        try:
            require(physical_commit_error is None,'Physical receipt commit/hash failed: '+str(physical_commit_error))
            require(physical_complete and reaped and stop is None and process.returncode==0,'Physical/stop outcome cannot qualify')
            require(physical['wall_seconds']<=plan['caps']['wall_seconds'] and peak<=plan['caps']['host_RSS_bytes'],'Observed parent caps differ')
            linked=collect(execution,plan,runtime,args.release_sha256)
            check_final_parent_budget('after_collection_and_reauthentication')
        except BaseException as error:
            linked=dict(status='FAILED_OR_INCOMPLETE_COLLECTION',type=type(error).__name__,condition=str(error))
        success=linked.get('status')=='COLLECTED_COMPLETE_RESOURCE_ONLY'
        terminal=dict(physical,status='COMPLETE_RESOURCE_ONLY' if success else 'FAILED_NO_RESOURCE_ADOPTION',linked=linked,
            physical_terminal_path=str(output/'PHYSICAL_TERMINAL.json'),physical_terminal_sha256=physical_sha,
            physical_terminal_commit_error=physical_commit_error,final_parent_budget_check=dict(final_budget),
            final_tail_limitations={
                'post_check_status_telemetry_terminal_custody_hash_write_and_fsync_tail_unmeasured': True,
                'exceptional_failure_republication_and_traceback_tail_unmeasured': True,
                'final_stdio_print_flush_and_os_exit_tail_unmeasured': True,
                'instantaneous_parent_RSS_or_end_to_infinite_publication_bound_claimed': False})
        durable_write(output/'TERMINAL.json',terminal)
        try:
            custody=dict(status=terminal['status'],terminal_path=str(output/'TERMINAL.json'),terminal_sha256=sha(output/'TERMINAL.json'),
                child_output_files=inventory(execution/'run01'),supervisor_output_files=inventory(output),root_lock_sha256=sha(lock),automatic_retry=False)
        except BaseException as error:
            success=False;terminal['status']='CUSTODY_FAILED_NO_RESOURCE_ADOPTION'
            terminal['custody_collection_error']=type(error).__name__+': '+str(error)
            terminal_sha = None;terminal_commit_error = None
            try:
                durable_write(output/'TERMINAL.json',terminal)
                terminal_sha = sha(output/'TERMINAL.json')
            except BaseException as failure:
                terminal_commit_error=type(failure).__name__+': '+str(failure)
            custody=dict(status=terminal['status'],terminal_path=str(output/'TERMINAL.json'),terminal_sha256=terminal_sha,
                error=terminal['custody_collection_error'],terminal_commit_error=terminal_commit_error,automatic_retry=False)
        durable_write(output/'SUPERVISOR_CUSTODY.json',custody)
        # The initial terminal, inventories, hashes and custody have now been
        # published. A late overrun must revoke success before return/stdio.
        try:
            check_final_parent_budget('after_receipt_and_custody_publication')
        except BaseException as error:
            success=False
            terminal['status']='FINAL_PARENT_CAP_FAILED_NO_RESOURCE_ADOPTION'
            terminal['final_parent_budget_error']=type(error).__name__+': '+str(error)
            if final_budget['status'] == 'NOT_CHECKED' or final_budget['last_completed_check'] != 'after_receipt_and_custody_publication':
                final_budget.update(status='ERROR',failed_check='after_receipt_and_custody_publication',
                                    observed_elapsed_seconds_on_error=time.monotonic()-started)
        terminal['final_parent_budget_check']=dict(final_budget)
        # Finite telemetry/status republication is the explicitly disclosed
        # tail after the last completed check; no recursive check/write loop.
        durable_write(output/'TERMINAL.json',terminal)
        final_terminal_sha = sha(output/'TERMINAL.json')
        final_terminal_bytes = (output/'TERMINAL.json').stat().st_size
        for row in custody.get('supervisor_output_files', []):
            if row['path'] == 'TERMINAL.json':
                row.update(bytes=final_terminal_bytes,sha256=final_terminal_sha)
        custody.update(status=terminal['status'],terminal_sha256=final_terminal_sha,
                       final_parent_budget_check=dict(final_budget),
                       final_tail_limitations=terminal['final_tail_limitations'])
        durable_write(output/'SUPERVISOR_CUSTODY.json',custody)
    print(json.dumps(terminal),flush=True)
    return 0 if success else 1


if __name__=='__main__':
    status = 1
    try:
        status = main()
    except BaseException:
        traceback.print_exc()
    finally:
        # Never let Python's Popen teardown reap an unresolved held child.
        try:
            sys.stdout.flush();sys.stderr.flush()
        finally:
            os._exit(status)
