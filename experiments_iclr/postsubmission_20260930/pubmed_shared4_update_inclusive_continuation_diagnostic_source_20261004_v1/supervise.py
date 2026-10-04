"""One fresh native Pubmed control child; bounded cleanup and exact receipts."""
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
from common import HERE, PHASE, EXECUTION, REPO, gate, require, sha, write

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
    # At most 17*2*2*65536 histogram rows occur in a seed receipt. 1 GiB
    # accommodates this finite JSON envelope without accepting arbitrary size.
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 1024**3, 'Invalid/big receipt file')
    return json.loads(path.read_text())


def digest_field(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def integer(value):
    return type(value) is int and value >= 0


def collect(execution, plan, release, release_sha):
    child = execution / 'update_inclusive/run01'
    required = {'DIAGNOSTIC.json', 'TRAIN_INSPECTION.json', 'RESTORE_ALIAS_PROGRESS.json',
                'STEP_DIVERGENCE_PROGRESS.json', 'PROGRESS.json', 'CUDA_PEAKS.json', 'FINAL_CUSTODY.json'}
    require(child.is_dir() and not child.is_symlink() and {p.name for p in child.iterdir()} == required
            and all(p.is_file() and not p.is_symlink() for p in child.iterdir()), 'Exact update-inclusive child files required')
    result, custody = read_json(child / 'DIAGNOSTIC.json'), read_json(child / 'FINAL_CUSTODY.json')
    require(result.get('schema') == 'pubmed-shared4-update-inclusive-continuation-diagnostic-v1'
            and result.get('status') == 'COMPLETE' and result['source_manifest_sha256'] == release['source_manifest_sha256']
            and result['root_release_sha256'] == release_sha
            and result['owned_serialized_identity_sha256'] == release['owned_serialized_identity_sha256']
            and all(result.get(k) is False for k in ('engineering_qualification_PASS', 'scientific_fit_admitted',
                'state_donor_allowed', 'VALID_files_opened', 'TEST_files_opened', 'score_files_opened',
                'numeric_rule_changed', 'restore_helper_changed', 'kernel_causality_established')), 'Update-inclusive scope differs')
    progress = result['progress']
    expected = dict(Adam_started=72, Adam_completed=72, native_epochs_started=2, native_epochs_completed=2,
                    factory_calls=2, weights_only_loads=2, state_file_loads=1, feature_weights_only_loads=1,
                    VALID_started=0, VALID_completed=0, serializations=0,
                    diagnostic_optimizer_pre_observations=72, diagnostic_optimizer_post_observations=72)
    require(all(type(progress.get(k)) is int and progress[k] == v for k, v in expected.items()), 'Complete native work differs')
    require(read_json(child / 'PROGRESS.json') == progress, 'Final progress copy differs')
    evidence = result['diagnostic_evidence']
    channels = evidence['selected_update_channels']
    require(channels['steps'] == [1, 2] and channels['backward_calls'] == {'reference': 36, 'candidate': 36}
            and channels['no_extra_replay'] is True and channels['retained_tensor_payload_bytes'] == 0
            and len(channels['comparisons']) == 2, 'Selected update coverage differs')
    declared = {'encoder_output', 'positive_raw_logits', 'negative_raw_logits', 'loss', 'named_preAdam_gradients'}
    for role in ('reference', 'candidate'):
        require([r['step'] for r in channels['roles'][role]] == [1, 2], 'Selected role steps differ')
        for row in channels['roles'][role]:
            require(set(row['channel_identities']) == declared and all(digest_field(v) for v in row['channel_identities'].values()),
                    'Selected channel identities differ')
    for row, step in zip(channels['comparisons'], (1, 2)):
        require(row['step'] == step and {v['channel'] for v in row['original_fixed_comparator_verdicts']} == declared
                and len(row['original_fixed_comparator_verdicts']) == 5, 'Original selected comparator coverage differs')
    steps = evidence['per_step_observer']
    require(steps['hook_counts'] == {'reference_pre': 36, 'reference_post': 36, 'candidate_pre': 36, 'candidate_post': 36}
            and steps['hook_RNG_neutral_checks'] == 144 and steps['retained_reference_steps'] == 0
            and steps['retained_reference_tensor_payload_bytes'] == 0, 'Complete inherited optimizer observer differs')
    for role in ('reference', 'candidate'):
        require([(r['step'], r['phase']) for r in steps[role]] == [(i, p) for i in range(1, 37) for p in ('pre', 'post')],
                'Original36-step order differs')
        require([r['step'] for r in steps[role] if 'moment_step_transitions' in r] == [1, 2], 'First-two transition coverage differs')
    require(evidence['decision']['preserved_old_qualification_status'] == 'FAILED'
            and evidence['decision']['automatic_retry'] is False and evidence['decision']['scientific_fit_admitted'] is False
            and evidence['decision']['tolerance_changed'] is False, 'Preserved failure/decision scope differs')
    require(len(evidence['unchanged_original_comparator_verdicts']) == 10, 'Original final comparator coverage differs')
    peaks = read_json(child / 'CUDA_PEAKS.json')
    require(peaks.get('CUDA_observed') is True and all(integer(peaks[k]) and peaks[k] <= plan['caps'][k]
            for k in ('cuda_peak_allocated_bytes', 'cuda_peak_reserved_bytes')), 'CUDA cap metadata differs')
    require(custody.get('stage') == 'update_inclusive' and custody.get('completed') is True, 'Child custody incomplete')
    rows = custody['files']
    require({r['path'] for r in rows} == required - {'FINAL_CUSTODY.json'} and len(rows) == 6, 'Exact child custody rows differ')
    for row in rows:
        path = child / row['path']
        require(path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Child custody differs')
    gate(EXECUTION / 'ROOT_RELEASE_update_inclusive.json', release_sha, 'update_inclusive')
    return dict(status='COLLECTED_COMPLETE_UPDATE_INCLUSIVE_DIAGNOSTIC_ONLY', result_path=str(child / 'DIAGNOSTIC.json'),
                result_sha256=sha(child / 'DIAGNOSTIC.json'), final_custody_path=str(child / 'FINAL_CUSTODY.json'),
                final_custody_sha256=sha(child / 'FINAL_CUSTODY.json'), decision=evidence['decision'],
                engineering_qualification_PASS=False, scientific_fit_admitted=False)


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
    started=time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-release',dest='release',type=Path,required=True)
    parser.add_argument('--release-sha256',required=True)
    args = parser.parse_args()
    release,plan = gate(args.release,args.release_sha256,'update_inclusive')
    execution=EXECUTION
    source_manifest_sha = sha(HERE/'MANIFEST.json')
    require(sys.platform=='linux' and signal.getsignal(signal.SIGCHLD)==signal.SIG_DFL,
            'Linux/default SIGCHLD required for held waitid/WNOWAIT identity')
    require(not (execution/'update_inclusive/run01').exists() and not (execution/'supervision/update_inclusive/run01').exists()
            and execution.resolve()==execution, 'Fresh canonical output required')
    execution.mkdir(parents=True,exist_ok=True)
    lock = execution/'UPDATE_INCLUSIVE_ATTEMPT_SPENT.json'
    with lock.open('x') as stream:
        json.dump(dict(UTC=datetime.now(timezone.utc).isoformat(),release_sha256=args.release_sha256,
            source_manifest_sha256=source_manifest_sha,parent_pid=os.getpid(),automatic_retry=False),stream)
        stream.flush();os.fsync(stream.fileno())
    fsync_directory(execution)
    output = execution/'supervision/update_inclusive/run01';output.mkdir(parents=True,exist_ok=False)
    fsync_directory(output.parent)
    process = identity = None;stop = None;peak = 0
    require(time.monotonic()-started<=plan['caps']['wall_seconds'],'Control cap spent in metadata gate')
    physical_complete = False;reaped = False;received = [];errors = [];actions = [];last_members = [];exit_observed = None
    command = [sys.executable,'-B',str(HERE/'continuation_diagnostic.py'),'--root-release',str(args.release),'--release-sha256',args.release_sha256]
    durable_write(output/'STARTED.json',dict(UTC=datetime.now(timezone.utc).isoformat(),command=command,
        source_manifest_sha256=source_manifest_sha,release_sha256=args.release_sha256,caps=plan['caps'],automatic_retry=False))
    def interrupted(number,frame):
        nonlocal stop
        received.append(number)
        stop = stop or "SUPERVISOR_SIGNAL"
    handlers = {number:signal.signal(number,interrupted) for number in (signal.SIGTERM,signal.SIGINT)}
    try:
        with (output/'STDOUT.txt').open('xb') as out,(output/'STDERR.txt').open('xb') as err:
            process = subprocess.Popen(command,cwd=REPO,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
            HELD_CHILDREN.append(process)
            identity = process_identity(process.pid)
            require(identity is not None and identity['session']==identity['group']==process.pid, 'Owned child session missing')
            durable_write(output/'LAUNCH.json',dict(identity=identity,command=command,release_sha256=args.release_sha256))
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
            process_escape_sandbox=False,fits=0,VALID_TEST_reads=False,automatic_retry=False,
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
        try:
            require(physical_commit_error is None,'Physical receipt commit/hash failed: '+str(physical_commit_error))
            require(physical_complete and reaped and stop is None and process.returncode==0,'Physical/stop outcome cannot qualify')
            require(physical['wall_seconds']<=plan['caps']['wall_seconds'] and peak<=plan['caps']['host_RSS_bytes'],'Observed parent caps differ')
            linked=collect(execution,plan,release,args.release_sha256)
            require(time.monotonic()-started<=plan['caps']['wall_seconds'],'Inclusive control collection wall cap exceeded')
        except BaseException as error:
            linked=dict(status='FAILED_OR_INCOMPLETE_COLLECTION',type=type(error).__name__,condition=str(error))
        success=linked.get('status')=='COLLECTED_COMPLETE_UPDATE_INCLUSIVE_DIAGNOSTIC_ONLY'
        terminal=dict(physical,inclusive_supervisor_wall_seconds=time.monotonic()-started,status='COMPLETE_UPDATE_INCLUSIVE_DIAGNOSTIC_ONLY' if success else 'FAILED_NO_DIAGNOSTIC_ADOPTION',linked=linked,
            physical_terminal_path=str(output/'PHYSICAL_TERMINAL.json'),physical_terminal_sha256=physical_sha,
            physical_terminal_commit_error=physical_commit_error)
        durable_write(output/'TERMINAL.json',terminal)
        try:
            custody=dict(status=terminal['status'],terminal_path=str(output/'TERMINAL.json'),terminal_sha256=sha(output/'TERMINAL.json'),
                child_output_files=inventory(execution/'update_inclusive/run01'),supervisor_output_files=inventory(output),root_lock_sha256=sha(lock),automatic_retry=False)
            require(time.monotonic()-started<=plan['caps']['wall_seconds'],'Inclusive final custody wall cap exceeded')
        except BaseException as error:
            success=False;terminal['status']='CUSTODY_FAILED_NO_DIAGNOSTIC_ADOPTION'
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
