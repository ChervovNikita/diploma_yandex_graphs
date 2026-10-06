"""Disabled host-only once-only bridge engineering launch, supervision and poll."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback

SOURCE_RELEASED = False
REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
PYTHON = PHASE / 'native_ncn_runtime_20261005_v1/.venv/bin/python'
HOST = 'anogena-2-0'
GPU = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
OUTPUT = 'public_path_responsibility_two_hop_paired_qualification_engineering_execution_root_20261006_v1'
QUALIFIER = {'path': 'public_path_responsibility_two_hop_paired_qualification_preparation_20261006_v3/qualify_bridge.py',
             'bytes': 47895, 'sha256': 'b8eaf5774cc3b53f43408079a3c1dbb23f6e324c713018513522e61e6c10cd3b'}
QUALIFIER_MANIFEST = {'path': 'public_path_responsibility_two_hop_paired_qualification_preparation_20261006_v3/MANIFEST.json',
                      'bytes': 1439, 'sha256': '3a0f629344028b3bd83c0e9937946aa247c30c932f2cb04cc0c6845827a0857d'}
HELPER = {'path': 'amazon_ordinary_shared_bank_two_gpu_scheduling_preparation_20261006_v1/process_supervisor.py',
          'bytes': 15840,
          'sha256': '6106ee06d4764293cf58e77870f96a157bcc23f442936258a1d982c63dd740cf'}
CAPS = {'max_elapsed_seconds': 1800, 'max_process_rss_bytes': 8589934592,
        'max_cuda_allocated_bytes': 79456894976, 'max_cuda_reserved_bytes': 83751862272}
WATCHDOG, GRACE, REAP = 1850, 5, 10
PRIOR = [('native_reference', 483967, 6006990021), ('final_transfer', 484269, 6007178957)]


def require(value, message):
    if not value: raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def bound(root, row):
    relative = Path(row['path'])
    require(relative.parts and not relative.is_absolute() and '..' not in relative.parts, 'Exact relative source/metadata required')
    path = root / relative
    require(path.is_file() and path.resolve().is_relative_to(root.resolve()), 'Source/metadata path escapes root')
    for item in (path, *path.parents):
        if item == root.parent: break
        require(not item.is_symlink(), 'Symlink source/metadata refused')
    require(path.stat().st_mode & 0o222 == 0 and sha(path) == row['sha256']
            and ('bytes' not in row or path.stat().st_size == row['bytes']), 'Immutable exact source/metadata differs')
    return path


class ObservationTransition(RuntimeError):
    def __init__(self, row, pgid, sid):
        self.observation = {'pid': row['pid'], 'starttime_ticks': row['starttime_ticks'],
                            'pgid_before': row['pgid'], 'pgid_after': pgid,
                            'sid_before': row['sid'], 'sid_after': sid}
        super().__init__(json.dumps(self.observation, sort_keys=True))


def identity(pid):
    proc = Path('/proc') / str(pid)
    try:
        fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
        row = {'pid': pid, 'starttime_ticks': int(fields[19]), 'parent_pid': int(fields[1]),
               'state': fields[0], 'pgid': int(fields[2]), 'sid': int(fields[3])}
        row['argv'] = [x.decode() for x in (proc / 'cmdline').read_bytes().split(b'\0') if x]
        if row['state'] != 'Z':
            row.update(cwd=str((proc / 'cwd').resolve(strict=True)), exe=str((proc / 'exe').resolve(strict=True)))
        again = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
        require(int(again[19]) == row['starttime_ticks'], 'Process start-time changed during observation')
        if int(again[2]) != row['pgid'] or int(again[3]) != row['sid']:
            raise ObservationTransition(row, int(again[2]), int(again[3]))
        return row
    except FileNotFoundError:
        return None


def physical(pid, argv, parent=None, start=None):
    deadline = time.monotonic() + 3
    transitions = []
    while True:
        try: row = identity(pid)
        except ObservationTransition as error:
            transitions.append(error.observation)
            require((start is None or error.observation['starttime_ticks'] == start)
                    and error.observation['starttime_ticks'] == transitions[0]['starttime_ticks'],
                    'Owned process start-time differs; observation retry refused: ' + json.dumps(transitions, sort_keys=True))
            require(time.monotonic() < deadline,
                    'Process group/session observation did not stabilize within 3s: ' + json.dumps(transitions, sort_keys=True))
            time.sleep(0.02)
            continue
        except RuntimeError as error:
            if transitions: raise RuntimeError(str(error) + ': ' + json.dumps(transitions, sort_keys=True)) from error
            raise
        observation_suffix = ': ' + json.dumps(transitions, sort_keys=True) if transitions else ''
        require(row is not None and row['state'] != 'Z', 'No live complete process identity; no signal authorized' + observation_suffix)
        require(not transitions or row['starttime_ticks'] == transitions[0]['starttime_ticks'],
                'Process start-time changed after observation transition; no signal authorized' + observation_suffix)
        if row['argv'] == argv:
            require(row['cwd'] == str(REPO) and row['exe'] == str(Path(argv[0]).resolve())
                    and row['pgid'] == row['sid'] == pid
                    and (parent is None or row['parent_pid'] == parent)
                    and (start is None or row['starttime_ticks'] == start), 'Owned process argv/cwd/exe/start/group differs' + observation_suffix)
            if transitions: row['stabilized_observation_transitions'] = transitions
            return row
        require(time.monotonic() < deadline, 'Incomplete/different argv; no signal authorized' + observation_suffix)
        time.sleep(0.02)


class Owner:
    def __init__(self, path, record):
        self.path, self.record = path, record

    def verify(self, token):
        require(token == self.record['token'] and not self.path.is_symlink() and self.path.is_dir()
                and self.path.resolve().is_relative_to(PHASE)
                and (self.path.stat().st_dev, self.path.stat().st_ino) == tuple(self.record['device_inode']), 'Created output ownership changed')
        return self.path


def write(owner, name, value, immutable=False):
    base = owner.verify(owner.record['token']); path = base / name; temporary = base / (name + '.tmp')
    require(not path.is_symlink() and not temporary.exists(), 'Receipt target ownership differs')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())
    owner.verify(owner.record['token']); os.replace(temporary, path)
    if immutable: path.chmod(0o444)


def source_request(args):
    require(SOURCE_RELEASED is False and socket.gethostname() == HOST and sys.platform.startswith('linux')
            and Path.cwd().resolve() == REPO and REPO.resolve() == REPO and PHASE.resolve() == PHASE
            and Path(sys.executable).absolute() == PYTHON and sys.dont_write_bytecode
            and not any(x == 'torch' or x.startswith('torch.') for x in sys.modules), 'Exact fresh normal host/runtime/-B required')
    path = Path(args.request).absolute()
    require(path.is_relative_to(PHASE), 'Root request outside phase')
    path = bound(PHASE, {'path': str(path.relative_to(PHASE)), 'sha256': args.request_sha256})
    request = read(path)
    require(request['schema'] == 'root_once_only_paired_bridge_engineering_request_v1'
            and all(request[k] is True for k in ('root_engineering_approved', 'supervision_source_review_approved',
                'qualifier_source_review_approved', 'exclusive_window_root_granted'))
            and all(request[k] is False for k in ('fits_authorized', 'held_scoring', 'persistent_updates_authorized',
                'automatic_retry', 'automatic_launch_after_WAIT', 'cap_expansion', 'priority_override')),
            'Exact engineering-only root authority required')
    require(request['qualifier'] == QUALIFIER and request['qualifier_manifest'] == QUALIFIER_MANIFEST
            and request['process_helper'] == HELPER and request['resource_limits'] == CAPS
            and request['external_watchdog_seconds'] == WATCHDOG
            and request['termination_grace_seconds'] == GRACE and request['termination_reap_seconds'] == REAP
            and request['host'] == HOST and request['GPU_UUID'] == GPU and request['output_relative'] == OUTPUT
            and request['python'] == str(PYTHON) and request['minimum_free_GPU_bytes'] == CAPS['max_cuda_reserved_bytes'], 'Frozen target/source/caps/output differ')
    require(bound(PHASE, request['supervisor_source']).resolve() == Path(__file__).resolve(), 'Exact reviewed supervisor source required')
    for row in read(bound(PHASE, request['packet_manifest']))['files']: bound(PHASE, row)
    require(request['source_reviews'], 'Actual qualifier/supervision review bindings required')
    for row in request['source_reviews']: bound(PHASE, row)
    qualifier = bound(PHASE, QUALIFIER)
    for row in read(bound(PHASE, QUALIFIER_MANIFEST))['files']: bound(qualifier.parent, row)
    bound(PHASE, HELPER)
    return request, path


def availability(request):
    require(len(request['prior_job_closures']) == len(PRIOR), 'Both actual predecessor closures required')
    for row, (role, pid, start) in zip(request['prior_job_closures'], PRIOR):
        closure = read(bound(PHASE, row))
        require(closure['schema'] == 'root_actual_owned_supervisor_terminal_cleanup_closure_v1'
                and (closure['role'], closure['supervisor_pid'], closure['supervisor_start_ticks']) == (role, pid, start)
                and all(closure[k] is True for k in ('actual_terminal_observed', 'owned_children_reaped', 'cleanup_complete', 'root_evidence_reviewed'))
                and closure['cleanup_errors'] == [] and closure['actual_terminal_and_cleanup_evidence'], 'Actual prior terminal/cleanup closure missing')
        for evidence in closure['actual_terminal_and_cleanup_evidence']: read(bound(PHASE, evidence))
        live = identity(pid)
        require(live is None or live['starttime_ticks'] != start, 'WAIT: original supervisor still present')
        for child in closure['owned_children']:
            live = identity(child['pid'])
            require(live is None or live['starttime_ticks'] != child['start_ticks'], 'WAIT: prior owned child still present')
    sample = subprocess.run(['nvidia-smi', '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, timeout=15)
    require(sample.returncode == 0 and len(sample.stdout.strip().splitlines()) == 1, 'WAIT: one physical GPU identity unavailable')
    uuid, free = [x.strip() for x in sample.stdout.strip().split(',')]
    require(uuid == GPU and int(free) * 1024**2 >= request['minimum_free_GPU_bytes'], 'WAIT: exact GPU/free-memory requirement failed')
    apps = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'],
                          capture_output=True, text=True, timeout=15)
    require(apps.returncode == 0 and not any(x.split(',')[0].strip() == GPU for x in apps.stdout.strip().splitlines()), 'WAIT: selected GPU has another compute process')
    return {'GPU_UUID': uuid, 'free_MiB': int(free), 'minimum_free_bytes': request['minimum_free_GPU_bytes'], 'selected_GPU_compute_apps_empty': True}


def helper():
    path = bound(PHASE, HELPER); name = '_bridge_exact_owned_process_helpers'
    require(name not in sys.modules, 'Fresh process helper required')
    spec = importlib.util.spec_from_file_location(name, path); module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module; spec.loader.exec_module(module)
    require(module.SOURCE_RELEASED is False and not any(x == 'torch' or x.startswith('torch.') for x in sys.modules), 'Stdlib-only exact disabled helper required')
    original_signal = module.signal_owned
    def verified_signal(child, signum):
        if child['wait4_closed'] or module.poll_child(child): return
        require(type(child['starttime_ticks']) is int and child['starttime_ticks'] > 0,
                'Incomplete owned-child start identity; no signal authorized')
        argv = [child['python'], '-B', child['worker'], *child['argv']]
        try: physical(child['pid'], argv, os.getpid(), child['starttime_ticks'])
        except BaseException as error:
            child.setdefault('signal_identity_observation_errors', []).append(str(error))
            if module.poll_child(child): return
            raise
        original_signal(child, signum)
    module.signal_owned = verified_signal
    return module


def launch(args, request):
    try: sample = availability(request)
    except BaseException as error:
        print(json.dumps({'status': 'WAIT_NO_LAUNCH', 'error': str(error), 'automatic_launch_after_WAIT': False})); return 2
    base = PHASE / OUTPUT
    require(not base.exists() and not base.is_symlink() and base.parent.is_dir(), 'Once-only output/launch sentinel already claimed')
    base.mkdir(mode=0o700); record = {'token': os.urandom(16).hex(), 'device_inode': [base.stat().st_dev, base.stat().st_ino]}
    owner = Owner(base, record)
    argv = [str(PYTHON), '-B', str(Path(__file__).resolve()), '--execute-authorized', '--mode', 'run',
            '--request', str(Path(args.request).absolute()), '--request-sha256', args.request_sha256]
    claim = {'schema': 'once_only_bridge_launch_claim_v1', 'request_sha256': args.request_sha256, 'owner': record,
             'supervisor_argv': argv, 'child_output': str(base / 'qualifier'), 'pre_framework_availability': sample,
             'no_retry': True, 'UTC': datetime.now(timezone.utc).isoformat()}
    write(owner, 'LAUNCH_SENTINEL.json', claim, True)
    with (base / 'supervisor.stdout.log').open('xb') as out, (base / 'supervisor.stderr.log').open('xb') as err:
        supervisor = subprocess.Popen(argv, cwd=REPO, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                      env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), start_new_session=True)
    write(owner, 'SUPERVISOR_START_ATTEMPT.json', {'pid': supervisor.pid, 'argv': argv, 'request_sha256': args.request_sha256}, True)
    observed = physical(supervisor.pid, argv, parent=os.getpid())
    handle = {'schema': 'detached_owned_bridge_supervisor_handle_v1', 'request_sha256': args.request_sha256,
              'supervisor': observed, 'output_relative': OUTPUT, 'owner': record, 'automatic_retry': False}
    write(owner, 'LAUNCH_HANDLE.json', handle, True)
    print(json.dumps({'status': 'DETACHED_SUPERVISOR_HANDLE_RETURNED', 'handle': str(base / 'LAUNCH_HANDLE.json'),
                      'supervisor_pid': supervisor.pid, 'supervisor_start_ticks': observed['starttime_ticks'], 'wait_in_tool_connection': False})); return 0


def run(args, request):
    base = PHASE / OUTPUT; claim = read(base / 'LAUNCH_SENTINEL.json'); owner = Owner(base, claim['owner'])
    owner.verify(claim['owner']['token']); require(claim['request_sha256'] == args.request_sha256, 'Launch claim request differs')
    terminal = {'schema': 'once_only_paired_bridge_owned_engineering_terminal_v1', 'status': 'FAIL_ENGINEERING_SUPERVISION',
                'request_sha256': args.request_sha256, 'qualifier': QUALIFIER, 'launch_attempts': 0, 'children': [],
                'model_fits': 0, 'held_scoring': False, 'persistent_updates': 0, 'numeric_imports_in_supervisor': False,
                'cleanup_errors': [], 'automatic_retry': False, 'resource_limits': CAPS}
    module = None; child = None; code = 1
    def save(): write(owner, 'SUPERVISOR_STATUS.json', terminal)
    old = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    def interrupted(number, frame): raise InterruptedError('Owned engineering supervisor interrupted: ' + str(number))
    try:
        deadline = time.monotonic() + 5
        while not (base / 'LAUNCH_HANDLE.json').exists():
            require(time.monotonic() < deadline, 'No verified launch handle; no child launch'); time.sleep(0.05)
        handle = read(base / 'LAUNCH_HANDLE.json')
        require(handle['supervisor']['pid'] == os.getpid() and handle['request_sha256'] == args.request_sha256, 'Supervisor handle mismatch')
        physical(os.getpid(), claim['supervisor_argv'], start=handle['supervisor']['starttime_ticks'])
        terminal['pre_child_availability'] = availability(request)
        module = helper(); module.validate_caps(CAPS, WATCHDOG, GRACE, REAP)
        for s in old: signal.signal(s, interrupted)
        arguments = ['--execute-authorized', '--source-root', str(PHASE), '--output', str(base / 'qualifier')]
        terminal['launch_attempts'] = 1; save()
        child = module.launch(REPO, owner, claim['owner']['token'], 'qualifier', str(PYTHON), PHASE / QUALIFIER['path'], arguments,
                              dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, PYTHONDONTWRITEBYTECODE='1'), terminal['children'])
        child['verified_physical_identity'] = physical(child['pid'], [str(PYTHON), '-B', str(PHASE / QUALIFIER['path']), *arguments], os.getpid(), child['starttime_ticks'])
        save()
        while not module.watch(child, WATCHDOG, GRACE, REAP): time.sleep(0.25)
        save()
        row = module.immutable_descriptor(PHASE, base / 'qualifier/RESULT.json'); result = read(PHASE / row['path'])
        terminal.update(original_worker_result=row, worker_status=result.get('status'), original_worker_errors={
            k: result.get(k) for k in ('error_type', 'error', 'traceback', 'native_body_error', 'native_cleanup_errors',
                'qualification_body_error', 'restoration_errors', 'terminal_resource_errors', 'terminal_resource_cap_breaches', 'cleanup_receipt_errors')},
            original_worker_resources=result.get('whole_child_resources'))
        module.resource_closure(child, {'whole_process_resources': result['whole_child_resources']}, CAPS)
        require(result['status'] == 'PASS_PAIRED_BRIDGE_NATIVE_PARITY_RECOMMIT_RESOURCE_ONLY'
                and result['worker_sha256'] == QUALIFIER['sha256'] and result['model_fits'] == result['persistent_updates'] == 0
                and result['A_scoring'] is False and result['VALID_TEST_access'] is False
                and result['actual_complete_native_callback_total'] == 284
                and not any(result.get(k) for k in ('native_cleanup_errors', 'restoration_errors', 'terminal_resource_errors', 'terminal_resource_cap_breaches', 'cleanup_receipt_errors')),
                'Exact engineering-only qualifier/resource/restoration PASS required')
        bound(PHASE, QUALIFIER); terminal['status'] = 'PASS_ENGINEERING_ONLY_OWNED_WHOLE_CHILD_CLOSED'; code = 0
    except BaseException as error:
        terminal.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        for s in old:
            try: signal.signal(s, signal.SIG_IGN)
            except BaseException as error: terminal['cleanup_errors'].append({'stage': 'ignore_repeated_signal', 'error': str(error)})
        if module is not None:
            try: terminal['cleanup_errors'].extend(module.cleanup(terminal['children'], GRACE, REAP))
            except BaseException as error: terminal['cleanup_errors'].append({'stage': 'owned_child_cleanup', 'error': str(error), 'traceback': traceback.format_exc()})
        for s, handler in old.items():
            try: signal.signal(s, handler)
            except BaseException as error: terminal['cleanup_errors'].append({'stage': 'restore_signal_handler', 'error': str(error)})
        if terminal['cleanup_errors'] or any(not c['wait4_closed'] for c in terminal['children']):
            terminal['status'] = 'FAIL_ENGINEERING_SUPERVISION_UNCLOSED_CHILD'; code = 1
        path = base / 'qualifier/RESULT.json'
        if module is not None and path.is_file():
            try:
                terminal['original_worker_result'] = module.immutable_descriptor(PHASE, path)
                result = read(path)
                terminal['worker_status'] = result.get('status')
                terminal['original_worker_resources'] = result.get('whole_child_resources')
                terminal['original_worker_errors'] = {k: result.get(k) for k in (
                    'error_type', 'error', 'traceback', 'native_body_error', 'native_cleanup_errors',
                    'qualification_body_error', 'restoration_errors', 'terminal_resource_errors',
                    'terminal_resource_cap_breaches', 'cleanup_receipt_errors')}
            except BaseException as error: terminal.setdefault('receipt_errors', []).append(str(error)); code = 1
        if code and terminal['status'] == 'PASS_ENGINEERING_ONLY_OWNED_WHOLE_CHILD_CLOSED':
            terminal['status'] = 'FAIL_ENGINEERING_SUPERVISION'
        terminal['engineering_only_resource_closure_PASS'] = code == 0
        terminal['UTC_terminal'] = datetime.now(timezone.utc).isoformat()
        write(owner, 'TERMINAL.json', terminal, True)
        for name in ('qualifier.stdout.log', 'qualifier.stderr.log'):
            if (base / name).is_file(): (base / name).chmod(0o444)
    return code


def poll(args, request):
    base = PHASE / OUTPUT; handle = read(base / 'LAUNCH_HANDLE.json'); owner = Owner(base, handle['owner'])
    owner.verify(handle['owner']['token']); require(handle['request_sha256'] == args.request_sha256, 'Poll request differs')
    if (base / 'TERMINAL.json').exists():
        require((base / 'TERMINAL.json').stat().st_mode & 0o222 == 0, 'Immutable terminal required')
        print(json.dumps({'status': 'TERMINAL', 'terminal': str(base / 'TERMINAL.json'), 'sha256': sha(base / 'TERMINAL.json')})); return 0
    live = identity(handle['supervisor']['pid'])
    if live is None or live['starttime_ticks'] != handle['supervisor']['starttime_ticks']:
        print(json.dumps({'status': 'OBSERVATION_FAILURE_NO_TERMINAL', 'automatic_retry': False, 'relaunch_authorized': False})); return 1
    try: physical(live['pid'], handle['supervisor']['argv'], start=live['starttime_ticks'])
    except BaseException as error:
        print(json.dumps({'status': 'OBSERVATION_FAILURE_NO_VERIFIED_SUPERVISOR', 'error': str(error),
                          'automatic_retry': False, 'relaunch_authorized': False})); return 1
    print(json.dumps({'status': 'RUNNING', 'handle': str(base / 'LAUNCH_HANDLE.json'), 'automatic_retry': False})); return 0


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--execute-authorized', action='store_true')
    parser.add_argument('--mode', choices=('launch', 'run', 'poll')); parser.add_argument('--request'); parser.add_argument('--request-sha256')
    args = parser.parse_args()
    if not args.execute_authorized:
        print(json.dumps({'status': 'DISABLED_ENGINEERING_EXECUTION_PREPARATION', 'SOURCE_RELEASED': False, 'numeric_imports': False})); return 0
    require(args.mode and args.request and args.request_sha256, 'Explicit mode and exact immutable root request required')
    request, _ = source_request(args)
    return {'launch': launch, 'run': run, 'poll': poll}[args.mode](args, request)


if __name__ == '__main__': raise SystemExit(main())
