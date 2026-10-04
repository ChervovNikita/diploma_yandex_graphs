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


def validate_stream(stream):
    require(stream.get('full_batches')==36 and stream.get('trained_positive_rows')==36864
            and stream.get('dropped_positive_tail')==812 and stream.get('original_body_reused') is True
            and stream.get('native_sampler_defaults_unchanged') is True
            and len(stream['negative_calls'])==len(stream['iterator_calls'])==1 and len(stream['batches'])==36,
            'Exact native stream coverage differs')
    require(all(digest_field(stream.get(name)) for name in ('start_RNG_sha256','end_RNG_sha256')), 'Native RNG digest absent')
    for row in stream['negative_calls']+stream['batches']:
        require(digest_field(row['sha256']), 'Native draw/batch digest absent')
    iterator=stream['iterator_calls'][0]
    require(iterator['native_length']==36 and iterator['training'] is True and iterator['batch_size']==1024
            and iterator['full_order']['shape']==[37676] and iterator['dropped_tail']['shape']==[812]
            and digest_field(iterator['full_order']['sha256']) and digest_field(iterator['dropped_tail']['sha256']),
            'Native iterator/tail differs')


def collect(execution, plan, release, release_sha):
    child=execution/'native_continuation_control/run01'
    required={'DIAGNOSTIC.json','TRAIN_INSPECTION.json','WARMUP.json','PROGRESS.json','CUDA_PEAKS.json',
              'RESTORE_ALIAS_PROGRESS.json','STEP_DIVERGENCE_PROGRESS.json','FINAL_CUSTODY.json'}
    require(child.is_dir() and not child.is_symlink()
            and {path.name for path in child.iterdir()}==required
            and all(path.is_file() and not path.is_symlink() for path in child.iterdir()), 'Exact complete child files absent')
    result=read_json(child/'DIAGNOSTIC.json');custody=read_json(child/'FINAL_CUSTODY.json')
    require(result.get('schema')=='pubmed-native-only-continuation-control-v1' and result.get('status')=='COMPLETE'
            and result.get('source_manifest_sha256')==release['source_manifest_sha256']
            and result.get('root_release_sha256')==release_sha and result.get('maximum_native_updates')==252
            and result.get('state_file_loads')==0 and result.get('engineering_qualification_PASS') is False
            and result.get('scientific_fit_admitted') is False and result.get('state_donor_allowed') is False
            and all(result.get(name) is False for name in ('VALID_files_opened','TEST_files_opened','score_files_opened','numeric_rule_changed','restore_helper_changed')), 'Result identity/scope differs')
    progress=result['progress']
    for name,expected in dict(Adam_started=252,Adam_completed=252,native_epochs_started=7,native_epochs_completed=7,
            factory_calls=3,native_reference_factory_calls=3,weights_only_loads=1,VALID_started=0,VALID_completed=0,
            serializations=0,diagnostic_optimizer_pre_observations=72,diagnostic_optimizer_post_observations=72).items():
        require(type(progress.get(name)) is int and progress[name]==expected, 'Exact native work differs: '+name)
    require(result['warmup']==dict(epochs=5,updates=180,native_RNG_neutral_checks=190)
            and result['continuation']==dict(epochs=2,updates=72,native_RNG_neutral_checks=76), 'Warm/continuation coverage differs')
    warm=read_json(child/'WARMUP.json')
    require(len(warm['rows'])==5 and [row['epoch'] for row in warm['rows']]==[1,2,3,4,5]
            and warm.get('state_file_loads')==0 and warm.get('engineering_only') is True
            and warm.get('scientific_donor_allowed') is False, 'Fresh warm evidence absent')
    for row in warm['rows']:validate_stream(row['stream'])
    evidence=result['diagnostic_evidence'];steps=evidence['per_step_observer']
    require(type(evidence.get('repeat_control_preconditions_exact')) is bool
            and evidence.get('RNG_neutral_observations')==266
            and set(evidence['native_streams'])=={'first','second'}, 'Complete native stream evidence absent')
    for stream in evidence['native_streams'].values():validate_stream(stream)
    require(steps['hook_counts']==dict(reference_pre=36,reference_post=36,candidate_pre=36,candidate_post=36)
            and steps['hook_RNG_neutral_checks']==144 and steps['retained_reference_tensor_payload_bytes']==0
            and steps['retained_reference_steps']==0 and steps['reference_snapshots_written_to_disk'] is False
            and steps['observer_is_causal_kernel_probe'] is False
            and steps['numerical_mismatch_truncates_native_epoch'] is False
            and 0<=steps['peak_reference_tensor_payload_bytes']<=plan['diagnostic_observer_limits']['reference_tensor_payload_bytes'],
            'Complete read-only hook evidence absent')
    for role in ('reference','candidate'):
        require(len(steps[role])==72, 'Per-step receipt incomplete')
        for index,row in enumerate(steps[role]):
            require(row['step']==index//2+1 and row['phase']==('pre' if index%2==0 else 'post')
                    and row['native_epoch']==6 and digest_field(row['complete_state_sha256']), 'Per-step order/identity differs')
    loss=evidence['loss'];tolerance=plan['engineering_tolerance']
    require(loss['atol']==tolerance['atol'] and loss['rtol']==tolerance['rtol']
            and loss['reference_on_right']=='original_loss' and loss['finiteness_telemetry_changes_original_comparator'] is False,
            'Fixed finite loss rule differs')
    require(len(evidence['unchanged_original_comparator_verdicts'])==10
            and all(row.get('comparison_collected') is True and type(row.get('passed_original_fixed_predicate')) is bool
                    for row in evidence['unchanged_original_comparator_verdicts']), 'Fixed original comparator coverage absent')
    require(custody.get('completed') is True and custody.get('stage')=='native_continuation_control'
            and custody.get('source_manifest_sha256')==release['source_manifest_sha256']
            and custody.get('root_release_sha256')==release_sha
            and custody.get('final_source_runtime_input_gate_rechecked') is True
            and custody.get('final_gate_error') is None and custody.get('array_loads_in_final_gate') is False,
            'Final exact source/runtime/input gate failed')
    rows=custody['files']
    require(len(rows)==7 and {row['path'] for row in rows}==required-{'FINAL_CUSTODY.json'}, 'Exact result file custody absent')
    for row in rows:
        path=child/row['path']
        require(Path(row['path']).name==row['path'] and path.stat().st_size==row['bytes'] and sha(path)==row['sha256'], 'Actual output custody differs')
    inspection=read_json(child/'TRAIN_INSPECTION.json')
    require(set(inspection['files'])=={'train_pos.txt','gnn_feature'}
            and all(inspection.get(name) is False for name in ('VALID_files_opened','TEST_files_opened','score_files_opened','state_files_opened')),
            'TRAIN-only input projection differs')
    peaks=read_json(child/'CUDA_PEAKS.json')
    require(peaks.get('CUDA_observed') is True and all(0<=peaks[name]<=plan['caps'][name]
            for name in ('cuda_peak_allocated_bytes','cuda_peak_reserved_bytes')), 'Observed CUDA caps differ')
    gate(execution/'ROOT_RELEASE_native_continuation_control.json',release_sha,'native_continuation_control')
    fsync_directory(child)
    return dict(status='COLLECTED_COMPLETE_NATIVE_CONTROL_ONLY',result_path=str(child/'DIAGNOSTIC.json'),
        result_sha256=sha(child/'DIAGNOSTIC.json'),final_custody_path=str(child/'FINAL_CUSTODY.json'),
        final_custody_sha256=sha(child/'FINAL_CUSTODY.json'),observed_numerical_parity=evidence['observed_final_continuation_parity_within_fixed_rule'],
        repeat_control_preconditions_exact=evidence['repeat_control_preconditions_exact'],
        engineering_qualification_PASS=False,scientific_fit_admitted=False)


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
    release,plan = gate(args.release,args.release_sha256,'native_continuation_control')
    execution=EXECUTION
    source_manifest_sha = sha(HERE/'MANIFEST.json')
    require(sys.platform=='linux' and signal.getsignal(signal.SIGCHLD)==signal.SIG_DFL,
            'Linux/default SIGCHLD required for held waitid/WNOWAIT identity')
    require(not (execution/'native_continuation_control/run01').exists() and not (execution/'supervision/native_continuation_control/run01').exists()
            and execution.resolve()==execution, 'Fresh canonical output required')
    execution.mkdir(parents=True,exist_ok=True)
    lock = execution/'NATIVE_CONTROL_ATTEMPT_SPENT.json'
    with lock.open('x') as stream:
        json.dump(dict(UTC=datetime.now(timezone.utc).isoformat(),release_sha256=args.release_sha256,
            source_manifest_sha256=source_manifest_sha,parent_pid=os.getpid(),automatic_retry=False),stream)
        stream.flush();os.fsync(stream.fileno())
    fsync_directory(execution)
    output = execution/'supervision/native_continuation_control/run01';output.mkdir(parents=True,exist_ok=False)
    fsync_directory(output.parent)
    process = identity = None;stop = None;peak = 0
    require(time.monotonic()-started<=plan['caps']['wall_seconds'],'Control cap spent in metadata gate')
    physical_complete = False;reaped = False;received = [];errors = [];actions = [];last_members = [];exit_observed = None
    command = [sys.executable,'-B',str(HERE/'native_continuation_control.py'),'--root-release',str(args.release),'--release-sha256',args.release_sha256]
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
        success=linked.get('status')=='COLLECTED_COMPLETE_NATIVE_CONTROL_ONLY'
        terminal=dict(physical,inclusive_supervisor_wall_seconds=time.monotonic()-started,status='COMPLETE_NATIVE_CONTROL_ONLY' if success else 'FAILED_NO_CONTROL_ADOPTION',linked=linked,
            physical_terminal_path=str(output/'PHYSICAL_TERMINAL.json'),physical_terminal_sha256=physical_sha,
            physical_terminal_commit_error=physical_commit_error)
        durable_write(output/'TERMINAL.json',terminal)
        try:
            custody=dict(status=terminal['status'],terminal_path=str(output/'TERMINAL.json'),terminal_sha256=sha(output/'TERMINAL.json'),
                child_output_files=inventory(execution/'native_continuation_control/run01'),supervisor_output_files=inventory(output),root_lock_sha256=sha(lock),automatic_retry=False)
            require(time.monotonic()-started<=plan['caps']['wall_seconds'],'Inclusive final custody wall cap exceeded')
        except BaseException as error:
            success=False;terminal['status']='CUSTODY_FAILED_NO_CONTROL_ADOPTION'
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
