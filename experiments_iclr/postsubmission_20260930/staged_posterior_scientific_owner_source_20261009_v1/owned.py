"""Disabled normal-host owner for the sealed staged-posterior full12 driver."""
import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
SEEDS = (6101, 6203, 6307)
ARMS = ('C4', 'S_joint4head', 'U4_sharedB', 'S_one_path')
ACTIVE, CLEANUP, STARTUP, ADMISSION, TERMINAL = 7200, 15, 120, 60, 30
FAMILY = STARTUP + 3 * (ADMISSION + ACTIVE + CLEANUP + TERMINAL)
CUSTODY = ('pid', 'start_ticks', 'group', 'session', 'boot_id')


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value, exclusive=False):
    path = Path(path)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if exclusive:
        with path.open('x') as stream:
            stream.write(text)
    else:
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(text)
        os.replace(temporary, path)


def identity(pid):
    try:
        value = Path('/proc', str(pid), 'stat').read_text()
        parts = value[value.rfind(')') + 2:].split()
        return dict(pid=int(pid), start_ticks=int(parts[19]), state=parts[0],
                    group=int(parts[2]), session=int(parts[3]),
                    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    except (FileNotFoundError, ProcessLookupError):
        return None


def same(current, saved):
    return current is not None and saved is not None and all(current[k] == saved[k] for k in CUSTODY)


def members(saved):
    result = []
    for entry in Path('/proc').iterdir():
        if entry.name.isdigit():
            item = identity(int(entry.name))
            if item and item['group'] == saved['group'] and item['session'] == saved['session']:
                result.append(item)
    return result


def bound(phase, row):
    path = (phase / row['path']).resolve(strict=True)
    require(path.is_relative_to(phase) and path.stat().st_size == row['bytes']
            and sha(path) == row['sha256'], 'Changed exact binding: ' + row['path'])
    return path


def seal(root, digest):
    require(sha(root / 'MANIFEST.json') == digest, 'Changed sealed manifest')
    require(read(root / 'SEAL.json')['manifest_sha256'] == digest, 'Changed sealed manifest receipt')
    for row in read(root / 'MANIFEST.json')['files']:
        bound(root, row)


def release_config(release_path, release_sha256, authorized):
    require(authorized is True and sha(release_path) == release_sha256,
            'Separate exact enabled root release required')
    cfg = read(release_path)
    fixed = read(HERE / 'ROOT_RELEASE_TEMPLATE_DISABLED.json')
    variable = {'enabled', 'later_execution_authorized', 'scientific_execution_authorized',
                'source_review_approved', 'owner_manifest_sha256', 'execution_source_commit'}
    require(set(cfg) == set(fixed) and all(cfg[k] == fixed[k] for k in fixed if k not in variable)
            and all(cfg[k] is True for k in ('enabled', 'later_execution_authorized',
                'scientific_execution_authorized', 'source_review_approved')),
            'Disabled until exact frozen root scientific release')
    require(isinstance(cfg['execution_source_commit'], str) and len(cfg['execution_source_commit']) == 40
            and all(c in '0123456789abcdef' for c in cfg['execution_source_commit']),
            'Actual committed execution source required')
    seal(HERE, cfg['owner_manifest_sha256'])
    pins = read(HERE / 'SOURCE_BINDINGS.json')
    phase = Path(pins['runtime']['phase']).resolve(strict=True)
    require(HERE.parent == phase and socket.gethostname() == pins['runtime']['hostname'],
            'Declared normal allocation host and phase only')
    require(Path(release_path).resolve(strict=True).parent == phase / cfg['activation_relative'],
            'Release in the separate root activation directory')
    return cfg, pins, phase


def gpu_rows():
    result = []
    for row in subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
            '--format=csv,noheader,nounits'], text=True, timeout=5).splitlines():
        parts = [part.strip() for part in row.split(',')]
        require(len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit(),
                'Readable GPU process accounting required')
        result.append((int(parts[0]), int(parts[1]) * 1024 ** 2))
    return result


def validate(release_path, release_sha256, authorized):
    cfg, pins, phase = release_config(release_path, release_sha256, authorized)
    runtime = pins['runtime']
    require(subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
            text=True, timeout=5).splitlines() == [runtime['GPU_uuid']], 'Sole frozen physical GPU UUID')
    root = phase / pins['stage_directory']
    seal(root, pins['stage_manifest_sha256'])
    # The sealed stage owns its dependency, role, native-state and scientific custody checks.
    stage_module(pins, phase).verify_bindings()
    for key in ('native_metadata', 'runtime_metadata', 'original_protocol', 'projection_manifest',
                'official_manifest', 'qualification_local_result', 'qualification_adoption',
                'qualification_result'):
        bound(phase, pins[key])
    require(read(phase / pins['runtime_metadata']['path']) == runtime, 'Original native runtime metadata')
    projection = read(phase / pins['projection_manifest']['path'])
    roles = {k: read(root / 'INPUT_FILES.json')[k] for k in ('train', 'valid')}
    require(projection['schema'] == 'internal-be-official-role-projection-v2'
            and projection['official_split_preserved'] is True and projection['split_index'] == 0
            and projection['train_count'] == 580 and projection['valid_count'] == 5274
            and projection['TEST_values_in_payload'] is False and projection['payloads'] == roles
            and projection['source_custody']['safe_payload'] == pins['safe_payload']
            and projection['source_custody']['official_manifest'] == pins['official_manifest'],
            'Original split0 TRAIN and development roles; no TEST')
    metadata = read(phase / pins['native_metadata']['path'])
    require([row['seed'] for row in metadata['native_states']] == list(SEEDS)
            and metadata['weights_or_outcome_values_exported'] is False,
            'Original three native metadata identities')
    for row in metadata['native_states']:
        origin = row['run_identity']
        require(origin['source']['protocol_sha256'] == pins['original_protocol']['sha256']
                and origin['data']['train_npz_sha256'] == roles['train']['sha256']
                and origin['data']['valid_npz_sha256'] == roles['valid']['sha256'],
                'Native acquisition protocol and original numeric role custody')
    result = read(phase / pins['qualification_result']['path'])
    adoption = read(phase / pins['qualification_adoption']['path'])
    require(result['complete'] is True and result['source_manifest_sha256'] == pins['stage_manifest_sha256']
            and result['scientific_fit'] is False and result['development_scores_computed'] is False
            and result['TEST_access'] is False and result['native_training_updates'] == 0
            and result['discarded_updates'] == 4 and result['new_native_captures'] == 1
            and result['peak_GPU_reserved_bytes'] <= cfg['max_owned_GPU_bytes']
            and result['peak_RSS_bytes'] <= cfg['max_owned_RSS_bytes'],
            'Successful unscored discarded complete-input qualification only')
    require(adoption['qualified'] is True and adoption['runtime_qualification'] is True
            and adoption['source_manifest_sha256'] == pins['stage_manifest_sha256']
            and adoption['qualification_path'] == pins['qualification_result']['path']
            and adoption['qualification_sha256'] == pins['qualification_result']['sha256']
            and adoption['selected_serving_validation_later_after_all12'] is True,
            'Exact root qualification adoption')
    commit = cfg['execution_source_commit']
    subprocess.run(['git', 'merge-base', '--is-ancestor', pins['qualified_source_commit'], commit],
                   cwd=runtime['repository'], check=True, timeout=10)
    for directory, digest in ((HERE.name, cfg['owner_manifest_sha256']),
                              (pins['stage_directory'], pins['stage_manifest_sha256'])):
        data = subprocess.check_output(['git', 'show', commit + ':' + pins['repository_phase_relative']
                + '/' + directory + '/MANIFEST.json'], cwd=runtime['repository'], timeout=10)
        require(hashlib.sha256(data).hexdigest() == digest, 'Exact manifest in actual execution commit')
    require(Path(runtime['python']).is_file() and all(Path(p).is_dir() for p in runtime['PYTHONPATH']),
            'Existing pinned normal runtime only')
    return cfg, pins, phase


def stage_module(pins, phase):
    path = bound(phase, pins['stage_program'])
    spec = importlib.util.spec_from_file_location('_owned_sealed_staged_posterior', path)
    stage = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = stage
    spec.loader.exec_module(stage)
    return stage


def stop(child, saved, known, actions, deadline):
    deadline = min(deadline, time.monotonic() + CLEANUP)

    def send(number):
        witness = next((item for item in known if same(identity(item['pid']), item)), None)
        require(witness is not None and saved['group'] == saved['session'] == saved['pid']
                and witness['group'] == saved['group'] and witness['session'] == saved['session'],
                'Signal only own witnessed PID/birth process group')
        try:
            os.killpg(saved['group'], number)
            actions.append(dict(signal=int(number),witness=witness))
        except ProcessLookupError:
            pass

    if members(saved):
        send(signal.SIGTERM)
    try:
        child.wait(timeout=max(.01, min(5, deadline - time.monotonic())))
    except subprocess.TimeoutExpired:
        send(signal.SIGKILL)
        child.wait(timeout=max(.01, deadline - time.monotonic()))
    if members(saved):
        send(signal.SIGKILL)
    while members(saved) and time.monotonic() < deadline:
        time.sleep(min(.05, max(0., deadline-time.monotonic())))
    require(time.monotonic() <= deadline and not members(saved), 'Owned group cleanup within 15 seconds')


def run_owned(*, release_path, release_sha256, authorized=False, launch_started=None):
    began = time.monotonic() if launch_started is None else launch_started
    cfg, pins, phase = release_config(release_path, release_sha256, authorized)
    activation = phase / cfg['activation_relative']
    output = phase / cfg['output_relative']
    require(not output.exists(), 'Fresh full12 only; no retry/resume/overwrite')
    output.mkdir(exist_ok=False)
    (output / 'handles').mkdir()
    (output / 'logs').mkdir()
    parent = identity(os.getpid())
    completed = []
    child = saved = None
    known = []
    active_seed = None
    closed = False
    error = None

    def interrupted(number, frame):
        raise RuntimeError('Owned family stop/deadline ' + str(number))

    def timer(deadline):
        require(time.monotonic() < deadline, 'Finite owner phase reserve')
        signal.setitimer(signal.ITIMER_REAL, deadline - time.monotonic())

    for number in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM):
        signal.signal(number, interrupted)
    try:
        timer(began + STARTUP)
        require(parent and parent['group'] == parent['session'] == parent['pid'], 'Detached scientific parent')
        deadline = min(began + STARTUP, time.monotonic() + 5)
        while not same(read(activation / 'LAUNCH.json').get('parent'), parent):
            require(time.monotonic() < deadline, 'Finite detached-parent launch registration')
            time.sleep(.01)
        write(output / 'PARENT_OWNER.json', dict(identity=parent, release_sha256=release_sha256,
              execution_source_commit=cfg['execution_source_commit'], qualified_source_commit=pins['qualified_source_commit'],
              family_seconds=FAMILY, serial_one_active_seed=True, coexecution='Mol18', scores_read=False), True)
        cfg, pins, phase = validate(release_path, release_sha256, True)
        for seed in SEEDS:
            active_seed = seed
            admitted = time.monotonic()
            timer(min(began + FAMILY - CLEANUP, admitted + ADMISSION))
            free = int(subprocess.check_output(['nvidia-smi', '--id=' + pins['runtime']['GPU_uuid'],
                '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True, timeout=5).strip()) * 1024 ** 2
            require(free >= cfg['minimum_fresh_GPU_bytes'], 'Fresh 12GiB GPU headroom; no arbitrary waiting')
            cell = output / ('seed' + str(seed))
            require(not cell.exists(), 'Fresh named seed block only')
            handle = output / 'handles' / ('seed' + str(seed) + '.json')
            value = dict(parent=parent, seed=seed, release_path=str(Path(release_path).resolve()),
                         release_sha256=release_sha256, output=str(cell), child=None)
            write(handle, value, True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=pins['runtime']['GPU_uuid'],
                       PYTHONPATH=os.pathsep.join(pins['runtime']['PYTHONPATH']), PYTHONDONTWRITEBYTECODE='1',
                       OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2')
            env.pop('PYTHONHOME', None)
            start = time.monotonic()
            reason = None
            actions = []
            max_gpu = max_rss = 0
            timer(min(began + FAMILY - CLEANUP, start + ACTIVE))
            with (output / 'logs' / ('seed' + str(seed) + '.log')).open('xb') as log:
                child = subprocess.Popen([pins['runtime']['python'], '-B', str(HERE / 'owned.py'), '--worker', str(handle)],
                    cwd=pins['runtime']['repository'], env=env, stdin=subprocess.DEVNULL, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
                saved = identity(child.pid)
                require(saved and saved['group'] == saved['session'] == child.pid, 'Exact newly owned seed process group')
                known = [saved]
                value['child'] = saved
                write(handle, value)
                # The monitor samples only process metadata and owned resource usage.
                signal.setitimer(signal.ITIMER_REAL, min(began + FAMILY - CLEANUP, start + ACTIVE + CLEANUP) - time.monotonic())
                while child.poll() is None:
                    if time.monotonic() - start >= ACTIVE:
                        reason = 'fixed7200second_active_safety_bound'
                        break
                    require(same(identity(child.pid), saved), 'Live owned PID/birth custody')
                    current = members(saved)
                    known = list({(item['pid'], item['start_ticks']): item for item in known + current}.values())
                    rss = 0
                    for item in current:
                        try:
                            status = Path('/proc', str(item['pid']), 'status').read_text()
                            require(same(identity(item['pid']), item), 'Resource-sample PID/birth custody')
                            rss += next((int(row.split()[1]) * 1024 for row in status.splitlines() if row.startswith('VmRSS:')), 0)
                        except FileNotFoundError:
                            pass
                    pids = {item['pid'] for item in current}
                    memory = sum(used for pid, used in gpu_rows() if pid in pids)
                    max_gpu, max_rss = max(max_gpu, memory), max(max_rss, rss)
                    if memory > cfg['max_owned_GPU_bytes'] or rss > cfg['max_owned_RSS_bytes']:
                        reason = 'owned8GiB_resource_safety_cap'
                        break
                    try:
                        child.wait(timeout=min(2, max(.01, ACTIVE - (time.monotonic() - start))))
                    except subprocess.TimeoutExpired:
                        pass
                if reason:
                    stop(child, saved, known, actions, min(start + ACTIVE + CLEANUP, began + FAMILY))
                else:
                    child.wait(timeout=.01)
            elapsed = time.monotonic() - start
            terminal_start = time.monotonic()
            timer(min(began + FAMILY, terminal_start + TERMINAL))
            absent = not members(saved)
            no_cuda = not ({pid for pid, _ in gpu_rows()} & {item['pid'] for item in known})
            receipt = dict(seed=seed, child=saved, witnessed_owned_members=known, exit_code=child.returncode,
                reaped=child.poll() is not None, group_absent=absent, no_owned_CUDA=no_cuda, reason=reason, signals=actions,
                active_and_cleanup_seconds=elapsed, resource_admission_seconds=start-admitted,
                max_sampled_owned_GPU_bytes=max_gpu, max_sampled_owned_RSS_bytes=max_rss,
                partial_work_and_costs_retained=True, scores_read=False, automatic_retry=False)
            write(output / 'handles' / ('seed' + str(seed) + '.EXIT.json'), receipt, True)
            require(child.returncode == 0 and reason is None and absent and no_cuda and elapsed <= ACTIVE + CLEANUP,
                    'Successful exact-owned terminal seed required; failed work retained without retry')
            done = read(cell / 'COMPLETE.json')
            require(done['complete'] is True and done['seed'] == seed and done['label_updates'] == 1100
                    and done['arms'] == list(ARMS) and done['source']['manifest_sha256'] == pins['stage_manifest_sha256']
                    and done['work']['all_bank_update_attempts'] == done['work']['all_bank_update_completions']
                    == done['work']['common_Q_checks'] == done['work']['complete_VALID_events'] == 1100
                    and done['work']['native_capture_calls'] == 1
                    and done['work']['native_updates'] == done['work']['learned_restores'] == done['new_native_training'] == 0,
                    'Full1100 staged work; one fixed native capture and no native training')
            cost = read(cell / 'WORKER_COST.json')
            require(cost['success'] is True and cost['peak_RSS_bytes'] <= cfg['max_owned_RSS_bytes']
                    and max(cost['peak_cuda_allocated_bytes'], cost['peak_cuda_reserved_bytes']) <= cfg['max_owned_GPU_bytes'],
                    'Successful worker costs within frozen caps')
            completed.append(dict(seed=seed, completion_sha256=sha(cell / 'COMPLETE.json'),
                worker_cost_sha256=sha(cell / 'WORKER_COST.json'), required_arm_count=4, exit_receipt=receipt))
            child = saved = None
            known = []
            active_seed = None
            write(output / 'PROGRESS.json', dict(completed=completed, fixed_seeds=list(SEEDS), scores_read=False))
            if seed == SEEDS[-1]:
                stage = stage_module(pins, phase)
                records = stage.verify_closed_family(output)
                require(len(records) == 12 and set(records) == {(s, a) for s in SEEDS for a in ARMS},
                        'Sealed mechanical all3/all12 exact-state/full1100 barrier')
                write(output / 'FAMILY_CLOSURE.json', dict(complete=True, completed=completed, fixed_seeds=list(SEEDS),
                    required_bank_records=12, label_updates_per_bank=1100, exact_all12_barrier='stage.verify_closed_family',
                    stage_source=stage.source_identity(), source_manifest_sha256=cfg['owner_manifest_sha256'],
                    execution_source_commit=cfg['execution_source_commit'], qualified_source_commit=pins['qualified_source_commit'],
                    qualification_result=pins['qualification_result'], native_metadata=pins['native_metadata'],
                    original_protocol=pins['original_protocol'], stage_protocol=pins['stage_protocol'],
                    scores_read=False, comparative_opening_authorized=False, selected_serving_validation_deferred=True,
                    automatic_retry=False), True)
                closed = True
            require(time.monotonic() - terminal_start < TERMINAL, 'Finite per-seed terminal/closure reserve')
    except BaseException as caught:
        error = dict(type=type(caught).__name__, message=str(caught))
        for number in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM):
            signal.signal(number, signal.SIG_IGN)
        signal.setitimer(signal.ITIMER_REAL, 0)
        cleanup = None
        if child is not None:
            actions = []
            try:
                stop(child, saved, known, actions, time.monotonic() + CLEANUP)
                cleanup = dict(reaped=True, exit_code=child.returncode, signals=actions, group_absent=True)
            except BaseException as failed:
                cleanup = dict(reaped=child.poll() is not None, error=type(failed).__name__ + ': ' + str(failed), signals=actions)
        write(output / 'FAMILY_FAILURE.json', dict(error=error, completed=completed, active_seed=active_seed,
              owned_cleanup=cleanup, all_partial_files_and_costs_retained=True, scores_read=False, automatic_retry=False), True)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        write(output / 'PARENT_TERMINAL.json', dict(family_complete=closed, error=error, completed_seed_count=len(completed),
              seconds=time.monotonic()-began, owner=parent, scores_read=False, automatic_retry=False), True)
    return dict(family_complete=closed, seeds=list(SEEDS), output=str(output), scores_read=False)


def worker(handle_path):
    started = time.monotonic()
    torch = None
    cfg = None
    success = False
    error = None
    handle = read(handle_path)
    output = Path(handle['output'])

    def interrupted(number, frame):
        raise RuntimeError('Owned seed stop/deadline ' + str(number))

    for number in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM):
        signal.signal(number, interrupted)
    signal.setitimer(signal.ITIMER_REAL, ACTIVE)
    try:
        deadline = started + 5
        while not handle.get('child'):
            require(time.monotonic() < deadline, 'Finite owned-child registration')
            time.sleep(.01)
            handle = read(handle_path)
        actual = identity(os.getpid())
        parent = identity(os.getppid())
        require(same(actual, handle['child']) and same(parent, handle['parent']), 'Exact child and detached-parent PID/birth custody')
        require(ctypes.CDLL(None, use_errno=True).prctl(1, int(signal.SIGTERM), 0, 0, 0) == 0, 'Owned parent-death stop signal')
        require(os.getppid() == parent['pid'] and same(identity(parent['pid']), parent), 'Parent live through registration')
        cfg, pins, phase = validate(handle['release_path'], handle['release_sha256'], True)
        require(handle['seed'] in SEEDS and Path(sys.executable).resolve() == Path(pins['runtime']['python']).resolve(),
                'Original pinned child interpreter and seed')
        import torch
        require(str(torch.__version__) == pins['runtime']['torch'] and torch.cuda.device_count() == 1,
                'Qualified one-GPU provider')
        torch.cuda.set_per_process_memory_fraction(cfg['max_owned_GPU_bytes'] / torch.cuda.get_device_properties(0).total_memory, 0)
        stage = stage_module(pins, phase)
        stage.run_complete(seed=handle['seed'], output=output,
                           later_execution_authorized=True, scientific_execution_authorized=True)
        success = True
    except BaseException as caught:
        error = dict(type=type(caught).__name__, message=str(caught))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        cost_path = output / 'WORKER_COST.json' if output.is_dir() else Path(handle_path).parent / ('seed' + str(handle['seed']) + '.WORKER_COST.json')
        allocated = reserved = None
        try:
            if torch is not None and torch.cuda.is_initialized():
                allocated, reserved = torch.cuda.max_memory_allocated(0), torch.cuda.max_memory_reserved(0)
        except Exception:
            pass
        write(cost_path, dict(success=success, error=error, seed=handle['seed'], seconds=time.monotonic()-started,
              CPU_user_seconds=usage.ru_utime, CPU_system_seconds=usage.ru_stime, peak_RSS_bytes=usage.ru_maxrss*1024,
              peak_cuda_allocated_bytes=allocated, peak_cuda_reserved_bytes=reserved,
              execution_source_commit=cfg['execution_source_commit'] if cfg else None,
              release_sha256=handle['release_sha256'], partial_driver_outputs_preserved=True, automatic_retry=False), True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--release', type=Path)
    parser.add_argument('--release-sha256')
    parser.add_argument('--authorized', action='store_true')
    parser.add_argument('--launch-started', type=float)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    else:
        print(json.dumps(run_owned(release_path=args.release, release_sha256=args.release_sha256,
              authorized=args.authorized, launch_started=args.launch_started)))


if __name__ == '__main__':
    main()
