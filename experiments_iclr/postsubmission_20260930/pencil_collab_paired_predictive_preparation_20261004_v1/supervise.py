"""One fresh PENCIL scientific fit child; inherited bounded ownership/physical terminal."""
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


def collect(execution,plan,runtime,release_sha,release_path,seed):
    child=execution/'run01'
    required={'FIT.json','PROGRESS.json','EPOCH_HISTORY.json','SELECTION.json','SELECTED_FULL_STATE.pt',
              'VALID_POSITIVE_SCORES.npy','VALID_NEGATIVE_SCORES.npy','FILE_CUSTODY.json','FINAL_CUSTODY.json'}
    require(child.is_dir() and not child.is_symlink() and {p.name for p in child.iterdir()}==required
            and all(p.is_file() and not p.is_symlink() for p in child.iterdir()),'Exact complete scientific output required')
    result=read_json(child/'FIT.json');final=read_json(child/'FINAL_CUSTODY.json');files=read_json(child/'FILE_CUSTODY.json')
    history=read_json(child/'EPOCH_HISTORY.json')['epochs'];selected=read_json(child/'SELECTION.json')
    require(result['status']=='COMPLETE_PREDICTIVE_FIT' and result['seed']==seed and result['completed_native_epochs']==20
            and result['source_manifest_sha256']==sha(HERE/'MANIFEST.json') and result['release_sha256']==release_sha
            and result['workload']==plan['workload'] and result['data_files_opened']==list(DATA_FILES),'Scientific identity/scope differs')
    require(len(history)==20 and [r['epoch_index'] for r in history]==list(range(20)),'Complete native epoch history required')
    for row in history:
        require(row['TRAIN_batches']==(row['TRAIN_queries']+1023)//1024
                and row['optimizer_updates']==(row['TRAIN_batches']+7)//8
                and row['VALID_queries']==160084 and row['VALID_batches']==157,'Actual native epoch/update/VALID coverage differs')
    require(result['optimizer_updates']==sum(r['optimizer_updates'] for r in history)
            and result['TRAIN_batches']==sum(r['TRAIN_batches'] for r in history)
            and result['TRAIN_queries']==sum(r['TRAIN_queries'] for r in history)
            and result['VALID_queries']==20*160084 and result['VALID_batches']==20*157,'Scientific counters differ')
    index=max(range(20),key=lambda i:history[i]['VALID_hits50'])
    require(selected['seed']==seed and selected['epoch_index']==result['selected_epoch_index']==index
            and selected['hits50']==result['selected_VALID_hits50']==history[index]['VALID_hits50']
            and selected['first_tie'] is True,'Strict VALID selection/first tie differs')
    require(selected['metric']=='VALID_hits@50' and selected['source_manifest_sha256']==sha(HERE/'MANIFEST.json')
            and selected['release_sha256']==release_sha and selected['positive_queries']==60084
            and selected['negative_queries']==100000 and selected['TEST_reads'] is False
            and selected['heldout_release'] is False and selected['resource_donor_loaded'] is False
            and selected['other_fit_donor'] is False,'Selected scientific identity differs')
    best=-float('inf')
    for row in history:
        score=row['VALID_hits50']
        require(isinstance(score,(int,float)) and 0<=score<=1
                and row['selected_on_strict_improvement']==(score>best),'Strict improvement history differs')
        if score>best:best=score
    require(selected['native_score_dtype'] in ('torch.bfloat16','torch.float32')
            and selected['score_dtype']=='float32' and selected['native_values_preserved'] is True
            and result['native_score_dtype']==selected['native_score_dtype']
            and result['native_score_values_preserved'] is True,'Native score value export differs')
    require(result['TEST_reads'] is False and result['heldout_release'] is False and result['state_donor'] is False
            and result['resource_fit_state_loaded'] is False and result['fresh_model_optimizer'] is True
            and result['scores_saved'] is True and result['checkpoint_saved'] is True and result['automatic_retry'] is False,
            'Scientific state/TEST boundary differs')
    require(result['result_class']=='adapted_native_PENCIL_scientific_baseline'
            and result['exact_author_reproduction'] is False and result['selected_state_server_side'] is True
            and result['historical_Collab_TEST_consumed'] is True
            and result['protocol_differences']==plan['protocol_differences'],'Adaptation/history disclosure differs')
    require(result['cuda_peak_allocated_bytes']<=plan['caps']['cuda_allocated_bytes']
            and result['cuda_peak_reserved_bytes']<=plan['caps']['cuda_reserved_bytes']
            and result['inclusive_child_wall_seconds']<=plan['caps']['wall_seconds'],'Scientific child caps differ')
    require(final['completed'] is True and final['seed']==seed and final['file_custody_sha256']==sha(child/'FILE_CUSTODY.json')
            and final['inclusive_child_wall_seconds_through_custody']<=plan['caps']['wall_seconds'],'Final scientific custody differs')
    require(len(final['files'])==8 and {r['path'] for r in final['files']}==required-{'FINAL_CUSTODY.json'},'Final exact output set differs')
    for row in final['files']:
        require(Path(row['path']).name==row['path'],'Custody path escaped');pin(child/row['path'],row)
    require({r['path'] for r in selected['files']}=={'SELECTED_FULL_STATE.pt','VALID_POSITIVE_SCORES.npy','VALID_NEGATIVE_SCORES.npy'},'Selected artifact set differs')
    for row in selected['files']:pin(child/row['path'],row)
    require(selected['files']==result['selected_artifacts'],'Selected result/custody differs')
    expected=final_file_custody(plan,runtime,release_path,release_sha,
        Path(json.loads((HERE/'metadata/DATA_AUTHORITY.json').read_text())['dataset_root']).resolve())
    require(files==expected and files['status']=='MATCH','Final scientific source/runtime/TRAIN-VALID custody differs')
    gate(release_path,release_sha,seed);fsync_directory(child)
    return dict(status='COLLECTED_COMPLETE_PREDICTIVE_FIT',seed=seed,result_path=str(child/'FIT.json'),
                result_sha256=sha(child/'FIT.json'),final_custody_path=str(child/'FINAL_CUSTODY.json'),
                final_custody_sha256=sha(child/'FINAL_CUSTODY.json'),file_custody_sha256=sha(child/'FILE_CUSTODY.json'),
                selected_epoch_index=index,selected_VALID_hits50=selected['hits50'],selected_artifacts=selected['files'])



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
    parser.add_argument('--seed',type=int,choices=(0,1,2),required=True)
    args = parser.parse_args()
    plan,runtime,execution = gate(args.release,args.release_sha256,args.seed)
    source_manifest_sha = sha(HERE/'MANIFEST.json')
    require(sys.platform=='linux' and signal.getsignal(signal.SIGCHLD)==signal.SIG_DFL,
            'Linux/default SIGCHLD required for held waitid/WNOWAIT identity')
    require(not (execution/'run01').exists() and not (execution/'supervision/run01').exists()
            and execution.resolve()==execution, 'Fresh canonical output required')
    execution.mkdir(parents=True,exist_ok=True)
    lock = execution/'FIT_ATTEMPT_SPENT.json'
    with lock.open('x') as stream:
        json.dump(dict(UTC=datetime.now(timezone.utc).isoformat(),release_sha256=args.release_sha256,
            source_manifest_sha256=source_manifest_sha,parent_pid=os.getpid(),seed=args.seed,automatic_retry=False),stream)
        stream.flush();os.fsync(stream.fileno())
    fsync_directory(execution)
    active=execution.parent/'ACTIVE_FIT.json'
    with active.open('x') as stream:
        json.dump(dict(seed=args.seed,supervisor_pid=os.getpid(),release_sha256=args.release_sha256),stream)
        stream.flush();os.fsync(stream.fileno())
    fsync_directory(execution.parent)
    output = execution/'supervision/run01';output.mkdir(parents=True,exist_ok=False)
    fsync_directory(output.parent)
    process = identity = None;stop = None;peak = 0
    physical_complete = False;reaped = False;received = [];errors = [];actions = [];last_members = [];exit_observed = None
    command = [sys.executable,'-B',str(HERE/'worker.py'),'--release',str(args.release),'--release-sha256',args.release_sha256,'--seed',str(args.seed)]
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
            process_escape_sandbox=False,seed=args.seed,completed_scientific_fits=0,TEST_reads=False,VALID_scientific_reads=True,automatic_retry=False,
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
            linked=collect(execution,plan,runtime,args.release_sha256,args.release,args.seed)
            check_final_parent_budget('after_collection_and_reauthentication')
        except BaseException as error:
            linked=dict(status='FAILED_OR_INCOMPLETE_COLLECTION',type=type(error).__name__,condition=str(error))
        success=linked.get('status')=='COLLECTED_COMPLETE_PREDICTIVE_FIT'
        terminal=dict(physical,status='COMPLETE_PREDICTIVE_FIT' if success else 'FAILED_NO_PREDICTIVE_ADOPTION',linked=linked,
            physical_terminal_path=str(output/'PHYSICAL_TERMINAL.json'),physical_terminal_sha256=physical_sha,
            physical_terminal_commit_error=physical_commit_error,final_parent_budget_check=dict(final_budget),
            final_tail_limitations={
                'post_check_status_telemetry_terminal_custody_hash_write_and_fsync_tail_unmeasured': True,
                'exceptional_failure_republication_and_traceback_tail_unmeasured': True,
                'final_stdio_print_flush_and_os_exit_tail_unmeasured': True,
                'instantaneous_parent_RSS_or_end_to_infinite_publication_bound_claimed': False})
        terminal['completed_scientific_fits']=1 if success else 0
        durable_write(output/'TERMINAL.json',terminal)
        try:
            custody=dict(status=terminal['status'],terminal_path=str(output/'TERMINAL.json'),terminal_sha256=sha(output/'TERMINAL.json'),
                child_output_files=inventory(execution/'run01'),supervisor_output_files=inventory(output),root_lock_sha256=sha(lock),automatic_retry=False)
        except BaseException as error:
            success=False;terminal['status']='CUSTODY_FAILED_NO_PREDICTIVE_ADOPTION'
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
            terminal['status']='FINAL_PARENT_CAP_FAILED_NO_PREDICTIVE_ADOPTION'
            terminal['final_parent_budget_error']=type(error).__name__+': '+str(error)
            if final_budget['status'] == 'NOT_CHECKED' or final_budget['last_completed_check'] != 'after_receipt_and_custody_publication':
                final_budget.update(status='ERROR',failed_check='after_receipt_and_custody_publication',
                                    observed_elapsed_seconds_on_error=time.monotonic()-started)
        terminal['final_parent_budget_check']=dict(final_budget)
        terminal['completed_scientific_fits']=1 if success else 0
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
    if physical_complete and reaped:
        require(read_json(active)['supervisor_pid']==os.getpid(),'Active fit ownership differs')
        active.unlink();fsync_directory(execution.parent)
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
