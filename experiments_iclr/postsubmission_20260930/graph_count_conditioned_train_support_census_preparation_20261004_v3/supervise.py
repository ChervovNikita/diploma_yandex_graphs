"""One fresh Linux census child; bounded owned cleanup and guarded collection."""
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
from census import HERE, PHASE, gate, require, sha, write

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


def validate_population(population, plan):
    queries = plan['batch_size']
    require(population.get('queries') == queries and len(population['sides']) == 2, 'Population size/sides differ')
    require(digest_field(population.get('query_digest')) and digest_field(population.get('support_count_digest')), 'Population digests absent')
    categories = []
    for side in population['sides']:
        table = side['n_k_frequency']
        require(isinstance(table,list) and table, 'Side histogram absent')
        seen, frequency = set(), 0
        total = dict(empty=0,zero_count=0,full_count=0,categorical_or_complement=0,genuine_subset=0,
                     total_slots=0,total_observed_bits=0,n_times_r_sum=0)
        maximum = dict(max_n=0,max_k=0,max_r=0)
        groups = [0,0,0]
        for row in table:
            require(isinstance(row,list) and len(row)==3 and all(integer(x) for x in row), 'Invalid histogram row')
            n,k,f = row
            require(0 <= k <= n <= plan['nodes'] and f>0 and (n,k) not in seen, 'Invalid/duplicate histogram support')
            seen.add((n,k));frequency += f;r = min(k,n-k);groups[0 if r==0 else 1 if r==1 else 2] += f
            for name, condition in (('empty',n==0),('zero_count',k==0),('full_count',n>0 and k==n),
                                    ('categorical_or_complement',r==1),('genuine_subset',r>1)):
                total[name] += f if condition else 0
            total['total_slots'] += n*f;total['total_observed_bits'] += k*f;total['n_times_r_sum'] += n*r*f
            maximum['max_n'] = max(maximum['max_n'],n);maximum['max_k'] = max(maximum['max_k'],k);maximum['max_r'] = max(maximum['max_r'],r)
        require(frequency==queries and all(side.get(name)==value for name,value in {**total,**maximum}.items()), 'Side histogram accounting differs')
        categories.append(groups)
    joint = population['joint_group_counts']
    require(isinstance(joint,list) and len(joint)==3 and all(isinstance(row,list) and len(row)==3 for row in joint)
            and all(integer(value) for row in joint for value in row), 'Invalid joint strata')
    require([sum(row) for row in joint]==categories[0] and [sum(joint[i][j] for i in range(3)) for j in range(3)]==categories[1],
            'Joint side marginals differ')
    require(population.get('both_sides_nonconstant')==sum(joint[i][j] for i in (1,2) for j in (1,2))
            and population.get('both_sides_genuine_subset')==joint[2][2]
            and population.get('either_side_genuine_subset')==sum(joint[i][j] for i in range(3) for j in range(3) if i==2 or j==2),
            'Joint derived strata differ')


def collect(execution, plan, runtime, release_sha):
    child = execution/'run01'
    required = {'CENSUS.json','seed0_COUNTS.json','seed1_COUNTS.json','seed2_COUNTS.json','PROGRESS.json','FILE_CUSTODY.json','FINAL_CUSTODY.json'}
    require(child.is_dir() and not child.is_symlink(), 'Actual child output absent')
    actual = {path.name for path in child.iterdir()}
    require(actual==required and all(path.is_file() and not path.is_symlink() for path in child.iterdir()), 'Partial/extra/nested child output cannot be adopted')
    result, final, file_custody = (read_json(child/name) for name in ('CENSUS.json','FINAL_CUSTODY.json','FILE_CUSTODY.json'))
    require(result.get('status')=='COMPLETE_TRAIN_CENSUS_ONLY' and result.get('source_manifest_sha256')==sha(HERE/'MANIFEST.json')
            and result.get('release_sha256')==release_sha and result.get('streams')==3
            and result.get('full_batches_per_stream')==17 and result.get('positive_records_per_stream')==17*65536
            and result.get('negative_records_per_stream')==17*65536, 'Actual result identity/coverage differs')
    require(result.get('data_files_opened')==['split/time/train.pt','raw/edge.csv.gz'] and result.get('optimizer_updates')==0
            and all(result.get(name) is False for name in ('features_read','VALID_TEST_read','learned_models')), 'Result scientific roles differ')
    require(type(result.get('wall_seconds')) in (int,float) and 0<=result['wall_seconds']<=plan['caps']['wall_seconds']
            and integer(result.get('cuda_peak_allocated_bytes')) and result['cuda_peak_allocated_bytes']<=plan['caps']['cuda_allocated_bytes']
            and integer(result.get('cuda_peak_reserved_bytes')) and result['cuda_peak_reserved_bytes']<=plan['caps']['cuda_reserved_bytes'], 'Child observed cap receipt differs')
    for seed in (0,1,2):
        stream = read_json(child/('seed%d_COUNTS.json'%seed))
        require(type(stream.get('seed')) is int and stream['seed']==seed and isinstance(stream.get('rows'),list) and len(stream['rows'])==17
                and all(digest_field(stream.get(name)) for name in ('negative_draw_sha256','permutation_sha256','dropped_tail_sha256')), 'Actual seed stream incomplete')
        for batch,row in enumerate(stream['rows'],1):
            require(type(row.get('batch')) is int and row['batch']==batch and digest_field(row.get('record_ids_sha256'))
                    and set(row['populations'])=={'positive','negative'}, 'Actual batch/order/populations differ')
            for population in row['populations'].values():
                validate_population(population,plan)
    require(final.get('completed') is True and final.get('file_custody_sha256')==sha(child/'FILE_CUSTODY.json'), 'Final child custody absent/failed')
    rows = final.get('files')
    require(isinstance(rows,list) and len(rows)==len(required)-1 and {row['path'] for row in rows}==required-{'FINAL_CUSTODY.json'},
            'Child custody lacks exact result/seed/file set')
    for row in rows:
        require(Path(row['path']).name==row['path'], 'Custody path invalid')
        path = child/row['path']
        require(path.stat().st_size==row['bytes'] and sha(path)==row['sha256'], 'Actual child custody hash differs')
    require(file_custody.get('schema')=='TRAIN-census-final-file-custody-v2' and file_custody.get('status')=='MATCH'
            and file_custody.get('source_manifest_sha256')==sha(HERE/'MANIFEST.json')
            and file_custody.get('plan_sha256')==sha(HERE/'PLAN.json') and file_custody.get('release_sha256')==release_sha
            and file_custody.get('interpreter_sha256')==runtime['interpreter_sha256']
            and file_custody.get('distribution_versions')==runtime['distribution_versions']
            and file_custody.get('actual_profile_fields_beyond_original_guards_verified') is False
            and all(file_custody.get(name) is False for name in ('GPU_operations','array_loads','additional_scientific_input_roles')), 'Final file/runtime custody differs')
    expected_files = []
    for scope,path,pin in (('source_manifest',HERE/'MANIFEST.json',{'sha256':sha(HERE/'MANIFEST.json')}),
                           ('root_release',execution/'ROOT_RELEASE.json',{'sha256':release_sha}),
                           ('interpreter',Path(runtime['interpreter_path']),{'sha256':runtime['interpreter_sha256']})):
        expected_files.append(dict(scope=scope,path=str(path),**pin))
    for base,pins,scope in ((HERE,read_json(HERE/'MANIFEST.json')['files'],'source'),
                            (PHASE,plan['source_pins'],'producer_or_authority')):
        expected_files.extend(dict(scope=scope,path=str(base/pin['path']),bytes=pin['bytes'],sha256=pin['sha256']) for pin in pins)
    expected_files.extend(dict(scope='runtime_file',path=pin['path'],sha256=pin['sha256'],
                               **({'bytes':pin['bytes']} if 'bytes' in pin else {}))
                          for pin in runtime['runtime_source_pins']+runtime['runtime_binary_files']+[runtime['negative_sampler']])
    expected = {(pin['scope'],pin['path']):pin for pin in expected_files}
    files = file_custody.get('files')
    require(len(expected)==len(expected_files) and isinstance(files,list) and len(files)==len(expected)
            and {(row['scope'],row['path']) for row in files}==set(expected), 'Final source/runtime/control file set differs')
    for row in files:
        pin=expected[(row['scope'],row['path'])];path=Path(row['path'])
        require(integer(row.get('bytes')) and row['sha256']==pin['sha256']
                and ('bytes' not in pin or row['bytes']==pin['bytes'])
                and path.is_file() and (row['scope']=='interpreter' or not path.is_symlink())
                and path.stat().st_size==row['bytes'] and sha(path)==pin['sha256'],
                'Parent final source/runtime/control custody differs')
    authority = read_json(PHASE/plan['data_authority'])
    expected_train = {str(Path(authority['dataset_root'])/name):authority['files'][name] for name in ('split/time/train.pt','raw/edge.csv.gz')}
    train = file_custody.get('TRAIN_files')
    require(isinstance(train,list) and len(train)==2 and {row['path'] for row in train}==set(expected_train), 'Final TRAIN file set differs')
    for row in train:
        pin = expected_train[row['path']]
        path=Path(row['path'])
        require(row['relative_path'] in ('split/time/train.pt','raw/edge.csv.gz')
                and path==Path(authority['dataset_root'])/row['relative_path']
                and row['bytes']==pin['bytes'] and row['sha256']==pin['sha256']
                and path.is_file() and not path.is_symlink() and path.stat().st_size==pin['bytes']
                and sha(path)==pin['sha256'], 'Parent final TRAIN pin differs')
    progress=read_json(child/'PROGRESS.json')
    require(progress.get('completed_streams')==2 and progress.get('seed')==2 and progress.get('batches')==17,
            'Final progress seed/batch differs')
    # Parent rechecks source/runtime metadata gate without loading any arrays.
    gate(execution/'ROOT_RELEASE.json',release_sha)
    fsync_directory(child)
    return dict(status='COLLECTED_COMPLETE_TRAIN_METADATA_ONLY',result_path=str(child/'CENSUS.json'),
        result_sha256=sha(child/'CENSUS.json'),final_custody_path=str(child/'FINAL_CUSTODY.json'),
        final_custody_sha256=sha(child/'FINAL_CUSTODY.json'),file_custody_path=str(child/'FILE_CUSTODY.json'),
        file_custody_sha256=sha(child/'FILE_CUSTODY.json'),seed_files=[dict(path=str(child/('seed%d_COUNTS.json'%seed)),
        sha256=sha(child/('seed%d_COUNTS.json'%seed))) for seed in (0,1,2)])


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
    lock = execution/'CENSUS_ATTEMPT_SPENT.json'
    with lock.open('x') as stream:
        json.dump(dict(UTC=datetime.now(timezone.utc).isoformat(),release_sha256=args.release_sha256,
            source_manifest_sha256=source_manifest_sha,parent_pid=os.getpid(),automatic_retry=False),stream)
        stream.flush();os.fsync(stream.fileno())
    fsync_directory(execution)
    output = execution/'supervision/run01';output.mkdir(parents=True,exist_ok=False)
    fsync_directory(output.parent)
    started = time.monotonic();process = identity = None;stop = None;peak = 0
    physical_complete = False;reaped = False;received = [];errors = [];actions = [];last_members = [];exit_observed = None
    command = [sys.executable,'-B',str(HERE/'census.py'),'--release',str(args.release),'--release-sha256',args.release_sha256]
    durable_write(output/'STARTED.json',dict(UTC=datetime.now(timezone.utc).isoformat(),command=command,
        source_manifest_sha256=source_manifest_sha,release_sha256=args.release_sha256,caps=plan['caps'],automatic_retry=False))
    def interrupted(number,frame):
        nonlocal stop
        received.append(number)
        stop = stop or "SUPERVISOR_SIGNAL"
    handlers = {number:signal.signal(number,interrupted) for number in (signal.SIGTERM,signal.SIGINT)}
    try:
        with (output/'STDOUT.txt').open('xb') as out,(output/'STDERR.txt').open('xb') as err:
            process = subprocess.Popen(command,cwd=plan['repository'],stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
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
            linked=collect(execution,plan,runtime,args.release_sha256)
        except BaseException as error:
            linked=dict(status='FAILED_OR_INCOMPLETE_COLLECTION',type=type(error).__name__,condition=str(error))
        success=linked.get('status')=='COLLECTED_COMPLETE_TRAIN_METADATA_ONLY'
        terminal=dict(physical,status='COMPLETE_TRAIN_METADATA_ONLY' if success else 'FAILED_NO_CENSUS_ADOPTION',linked=linked,
            physical_terminal_path=str(output/'PHYSICAL_TERMINAL.json'),physical_terminal_sha256=physical_sha,
            physical_terminal_commit_error=physical_commit_error)
        durable_write(output/'TERMINAL.json',terminal)
        try:
            custody=dict(status=terminal['status'],terminal_path=str(output/'TERMINAL.json'),terminal_sha256=sha(output/'TERMINAL.json'),
                child_output_files=inventory(execution/'run01'),supervisor_output_files=inventory(output),root_lock_sha256=sha(lock),automatic_retry=False)
        except BaseException as error:
            success=False;terminal['status']='CUSTODY_FAILED_NO_CENSUS_ADOPTION'
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
