"""Disabled once-only owned caller for the exact CMCL native parity qualifier."""
import argparse
import hashlib
import importlib.util
import json
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
GPU = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
OUTPUT = 'cmcl_graph_common400_native_first_order_parity_owned_execution_root_20261006_v2'
QUALIFIER = {'path': 'cmcl_graph_common400_native_first_order_parity_preparation_20261006_v1/qualify_native.py', 'bytes': 31563,
             'sha256': '2a2127e63bf58da8a3f48be70e9eaa339a6d4313c960e6cc158b9d35568d6c58'}
OWNED = {'path': 'public_path_responsibility_two_hop_paired_qualification_owned_execution_preparation_20261006_v2/control.py',
         'bytes': 23700, 'sha256': '5f8d214660dceaa55bb2393bc3b7a2f7c840ce7508da0e630ee30a694dc6a458'}
CAPS = {'max_elapsed_seconds': 900, 'max_process_rss_bytes': 8589934592,
        'max_cuda_allocated_bytes': 77309411328, 'max_cuda_reserved_bytes': 79456894976}
WATCHDOG, FREE, GRACE, REAP = 1020, 83751862272, 5, 10


def require(value, message):
    if not value: raise RuntimeError(message)


def authenticate(args):
    require(SOURCE_RELEASED is False and sys.platform.startswith('linux') and socket.gethostname() == 'anogena-2-0'
            and Path.cwd().resolve() == REPO and PHASE.resolve() == PHASE and Path(sys.executable).absolute() == PYTHON
            and sys.dont_write_bytecode and not any(n == 'torch' or n.startswith('torch.') for n in sys.modules), 'Exact fresh normal allocation runtime/-B required')
    source = PHASE / OWNED['path']
    require(source.is_file() and not source.is_symlink() and source.stat().st_mode & 0o222 == 0
            and source.stat().st_size == OWNED['bytes'] and hashlib.sha256(source.read_bytes()).hexdigest() == OWNED['sha256'], 'Exact reviewed stdlib ownership helper required')
    name = '_CMCL_reviewed_owned_helpers'; require(name not in sys.modules, 'Fresh ownership helper required')
    spec = importlib.util.spec_from_file_location(name, source); owned = importlib.util.module_from_spec(spec)
    sys.modules[name] = owned; spec.loader.exec_module(owned)
    request_path = Path(args.request).absolute(); require(request_path.is_relative_to(PHASE), 'Immutable root request must stay in phase')
    request = owned.read(owned.bound(PHASE, {'path': str(request_path.relative_to(PHASE)), 'sha256': args.request_sha256}))
    require(request['schema'] == 'root_CMCL_once_owned_caller_request_v1' and request['root_owned_caller_authorized'] is True
            and request['caller_source_review_approved'] is True and request['automatic_retry'] is False
            and request['automatic_launch_after_WAIT'] is False and request['output_relative'] == OUTPUT, 'Exact once-only reviewed root caller authority required')
    require(owned.bound(PHASE, request['caller_source']).resolve() == Path(__file__).resolve(), 'Reviewed caller source differs')
    for row in owned.read(owned.bound(PHASE, request['caller_manifest']))['files']: owned.bound(PHASE, row)
    owned.bound(PHASE, request['caller_review'])
    scope_path = owned.bound(PHASE, request['native_scope']); scope = owned.read(scope_path)
    require(scope['schema'] == 'root_CMCL_native_common400_first_order_parity_scope_v1' and scope['executor'] == QUALIFIER
            and scope['root_engineering_invocation_authorized'] is True and scope['fixed_before_execution'] is True
            and scope['one_owned_invocation_no_retry'] is True and scope['no_fit_or_scoring'] is True
            and scope['A_VALID_TEST_access'] is False and scope['exclusive_native_parity_window'] is True
            and scope['root_scheduling_clearance_verified'] is True and scope['resource_limits'] == CAPS
            and scope['external_watchdog_seconds'] == WATCHDOG and scope['minimum_initial_cuda_free_bytes'] == FREE
            and scope['cuda_headroom_bytes'] == 4294967296 and scope['GPU_UUID'] == GPU
            and scope['python_executable'] == str(PYTHON), 'Existing fixed CMCL scope/window/resource declarations required')
    owned.bound(PHASE, QUALIFIER)
    for row in scope['deployed_sources'].values(): owned.bound(PHASE, row)
    for key in ('root_scheduling_clearance', 'fresh_GPU_memory_review', 'caller_source_review', 'parity_source_review', 'CPU_worker_PASS', 'CPU_owned_terminal_PASS'):
        owned.bound(PHASE, scope[key])
    return owned, request, scope


def available(owned, request, scope):
    clearance = owned.read(PHASE / scope['root_scheduling_clearance']['path'])
    memory = owned.read(PHASE / scope['fresh_GPU_memory_review']['path'])
    require(clearance['schema'] == 'root_CMCL_native_parity_scheduling_clearance_v1' and clearance['fixed_before_execution'] is True
            and clearance['exclusive_native_parity_window'] is True and clearance['GPU_UUID'] == GPU and clearance['no_fit_or_scoring'] is True
            and memory['schema'] == 'root_CMCL_native_parity_fresh_memory_review_v1' and memory['root_reviewed'] is True
            and memory['GPU_UUID'] == GPU and memory['resource_limits'] == CAPS and memory['cuda_headroom_bytes'] == 4294967296
            and memory['minimum_initial_cuda_free_bytes'] == FREE and memory['actual_observed_free_bytes'] >= FREE,
            'Actual existing CMCL scheduling/memory receipts required')
    return owned.availability({'prior_job_closures': request['prior_job_closures'], 'minimum_free_GPU_bytes': FREE})


def launch(args, owned, request, scope):
    try: sample = available(owned, request, scope)
    except BaseException as error:
        print(json.dumps({'status': 'WAIT_NO_LAUNCH', 'error': str(error), 'automatic_launch_after_WAIT': False})); return 2
    base = PHASE / OUTPUT; require(not base.exists() and not base.is_symlink(), 'Once-only CMCL output already claimed')
    base.mkdir(mode=0o700); record = {'token': os.urandom(16).hex(), 'device_inode': [base.stat().st_dev, base.stat().st_ino]}
    owner = owned.Owner(base, record)
    argv = [str(PYTHON), '-B', str(Path(__file__).resolve()), '--execute-authorized', '--mode', 'run', '--request',
            str(Path(args.request).absolute()), '--request-sha256', args.request_sha256]
    owned.write(owner, 'LAUNCH_SENTINEL.json', {'owner': record, 'request_sha256': args.request_sha256,
                'supervisor_argv': argv, 'pre_framework_availability': sample, 'native_scope': request['native_scope'], 'no_retry': True}, True)
    with (base / 'supervisor.stdout.log').open('xb') as out, (base / 'supervisor.stderr.log').open('xb') as err:
        supervisor = subprocess.Popen(argv, cwd=REPO, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                      env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), start_new_session=True)
    owned.write(owner, 'SUPERVISOR_START_ATTEMPT.json', {'pid': supervisor.pid, 'argv': argv}, True)
    observed = owned.physical(supervisor.pid, argv, parent=os.getpid())
    owned.write(owner, 'LAUNCH_HANDLE.json', {'owner': record, 'request_sha256': args.request_sha256, 'supervisor': observed}, True)
    print(json.dumps({'status': 'DETACHED_CMCL_HANDLE_RETURNED', 'handle': str(base / 'LAUNCH_HANDLE.json'),
                      'supervisor_pid': supervisor.pid, 'supervisor_start_ticks': observed['starttime_ticks'], 'wait_in_tool_connection': False})); return 0


def run(args, owned, request, scope):
    base = PHASE / OUTPUT; claim = owned.read(base / 'LAUNCH_SENTINEL.json'); owner = owned.Owner(base, claim['owner'])
    owner.verify(claim['owner']['token']); require(claim['request_sha256'] == args.request_sha256, 'CMCL launch claim request differs')
    receipt = {'schema': 'owned_CMCL_native_parity_engineering_terminal_v1', 'status': 'FAIL_OWNED_CMCL_ENGINEERING',
               'native_scope': request['native_scope'], 'qualifier': QUALIFIER, 'launch_attempts': 0, 'children': [],
               'cleanup_errors': [], 'model_fits': 0, 'held_scoring': False, 'persistent_updates': 0, 'automatic_retry': False, 'resource_limits': CAPS}
    process = None; code = 1; old = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    def save(): owned.write(owner, 'SUPERVISOR_STATUS.json', receipt)
    def interrupted(number, frame): raise InterruptedError('Owned CMCL supervisor interrupted: ' + str(number))
    try:
        deadline = time.monotonic() + 5
        while not (base / 'LAUNCH_HANDLE.json').exists():
            require(time.monotonic() < deadline, 'No verified handle; no CMCL child launch'); time.sleep(0.05)
        handle = owned.read(base / 'LAUNCH_HANDLE.json')
        require(handle['supervisor']['pid'] == os.getpid() and handle['request_sha256'] == args.request_sha256, 'CMCL supervisor handle differs')
        owned.physical(os.getpid(), claim['supervisor_argv'], start=handle['supervisor']['starttime_ticks'])
        receipt['pre_child_availability'] = available(owned, request, scope)
        process = owned.helper(); process.validate_caps(CAPS, WATCHDOG, GRACE, REAP)
        for s in old: signal.signal(s, interrupted)
        argv = ['--execute-authorized', '--source-root', str(PHASE), '--scope', str(PHASE / request['native_scope']['path']),
                '--scope-sha256', request['native_scope']['sha256'], '--output', str(base / 'qualifier')]
        receipt['launch_attempts'] = 1; save()
        child = process.launch(REPO, owner, claim['owner']['token'], 'qualifier', str(PYTHON), PHASE / QUALIFIER['path'], argv,
                               dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, PYTHONDONTWRITEBYTECODE='1'), receipt['children'])
        child['verified_physical_identity'] = owned.physical(child['pid'], [str(PYTHON), '-B', str(PHASE / QUALIFIER['path']), *argv], os.getpid(), child['starttime_ticks']); save()
        while not process.watch(child, WATCHDOG, GRACE, REAP): time.sleep(0.25)
        result = owned.read(base / 'qualifier/RESULT.json'); process.resource_closure(child, result, CAPS)
        require(result['status'] == 'PASS_CMCL_NATIVE_COMMON400_FIRST_ORDER_PARITY_ENGINEERING_ONLY'
                and result['executor'] == QUALIFIER and result['root_scope_sha256'] == request['native_scope']['sha256']
                and result['model_fits'] == result['persistent_updates'] == 0 and result['A_VALID_TEST_access'] is False
                and not result['restoration_errors'], 'Existing CMCL engineering PASS and owned closure required')
        owned.bound(PHASE, QUALIFIER); receipt['status'] = 'PASS_OWNED_CMCL_ENGINEERING_CLOSED'; code = 0
    except BaseException as error:
        receipt.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        for s in old:
            try: signal.signal(s, signal.SIG_IGN)
            except BaseException as error: receipt['cleanup_errors'].append(str(error))
        if process is not None:
            try: receipt['cleanup_errors'].extend(process.cleanup(receipt['children'], GRACE, REAP))
            except BaseException as error: receipt['cleanup_errors'].append(str(error))
        for s, handler in old.items():
            try: signal.signal(s, handler)
            except BaseException as error: receipt['cleanup_errors'].append(str(error))
        if receipt['cleanup_errors'] or any(not c['wait4_closed'] for c in receipt['children']): code = 1
        path = base / 'qualifier/RESULT.json'
        if process is not None and path.is_file():
            try:
                receipt['original_worker_result'] = process.immutable_descriptor(PHASE, path)
                result = owned.read(path)
                receipt['original_worker_status'] = result.get('status'); receipt['original_worker_resources'] = result.get('whole_process_resources')
                receipt['original_worker_error_receipts'] = {k: result.get(k) for k in ('error_type', 'error', 'traceback', 'restoration_errors', 'final_resource_error', 'output_finalization_error', 'failure_publication_error')}
                receipt['original_bill'] = result.get('bill')
            except BaseException as error: receipt.setdefault('receipt_errors', []).append(str(error)); code = 1
        if code: receipt['status'] = 'FAIL_OWNED_CMCL_ENGINEERING'
        receipt['owned_resource_closure_PASS'] = code == 0; owned.write(owner, 'TERMINAL.json', receipt, True)
        for name in ('qualifier.stdout.log', 'qualifier.stderr.log'):
            if (base / name).is_file(): (base / name).chmod(0o444)
    return code


def poll(args, owned, request, scope):
    base = PHASE / OUTPUT; handle = owned.read(base / 'LAUNCH_HANDLE.json'); owner = owned.Owner(base, handle['owner'])
    owner.verify(handle['owner']['token']); require(handle['request_sha256'] == args.request_sha256, 'CMCL poll request differs')
    if (base / 'TERMINAL.json').exists():
        row = owned.bound(PHASE, {'path': OUTPUT + '/TERMINAL.json', 'sha256': owned.sha(base / 'TERMINAL.json')})
        print(json.dumps({'status': 'TERMINAL', 'path': str(row), 'sha256': owned.sha(row)})); return 0
    try: owned.physical(handle['supervisor']['pid'], handle['supervisor']['argv'], start=handle['supervisor']['starttime_ticks'])
    except BaseException as error:
        print(json.dumps({'status': 'OBSERVATION_FAILURE_NO_RELAUNCH', 'error': str(error), 'automatic_retry': False})); return 1
    print(json.dumps({'status': 'RUNNING', 'handle': str(base / 'LAUNCH_HANDLE.json'), 'automatic_retry': False})); return 0


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--execute-authorized', action='store_true')
    parser.add_argument('--mode', choices=('launch', 'run', 'poll')); parser.add_argument('--request'); parser.add_argument('--request-sha256')
    args = parser.parse_args()
    if not args.execute_authorized:
        print(json.dumps({'status': 'DISABLED_CMCL_OWNED_ENGINEERING_CALLER', 'SOURCE_RELEASED': False})); return 0
    require(args.mode and args.request and args.request_sha256, 'Explicit mode and immutable exact root request required')
    owned, request, scope = authenticate(args)
    return {'launch': launch, 'run': run, 'poll': poll}[args.mode](args, owned, request, scope)


if __name__ == '__main__': raise SystemExit(main())
