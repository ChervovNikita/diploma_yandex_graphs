"""Disabled once-only owner for the exact existing Amazon comparison child."""
import time
STARTED = time.monotonic()
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import traceback

SOURCE_RELEASED = False
OWNED = {'path': 'public_path_responsibility_two_hop_paired_qualification_owned_execution_preparation_20261006_v2/control.py',
         'bytes': 23700, 'sha256': '5f8d214660dceaa55bb2393bc3b7a2f7c840ce7508da0e630ee30a694dc6a458'}
EVALUATOR = {'path': 'amazon_G0_complete_pool_comparison_preparation_20261006_v4/compare_complete.py', 'bytes': 51066, 'sha256': 'd3c84d7bb333f7874427acc267c8a8842ade7ddc953dcbb83f744af49a1f2c58'}
EVALUATOR_REVIEW = {'path': 'amazon_G0_complete_comparison_serialization_successor_root_20261006_v1/EVALUATOR_DELTA_REVIEW.json', 'bytes': 972, 'sha256': '4160a53207ccdcced0de6455c5a0a74f12d3d7c9658b2f310d92414b5af72aa6'}
CAPS = {'max_elapsed_seconds': 900, 'max_process_rss_bytes': 8589934592,
        'max_cuda_allocated_bytes': 77309411328, 'max_cuda_reserved_bytes': 79456894976}
WATCHDOG, FREE, GRACE, REAP = 1020, 83751862272, 5, 10
OUTPUT = 'amazon_G0_complete_pool_comparison_owned_execution_root_20261006_v3'
EVALUATION_OUTPUT = 'amazon_G0_complete_pool_comparison_execution_root_20261006_v4'


def require(value, message):
    if not value: raise RuntimeError(message)


def ownership(root):
    path = root / OWNED['path']
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root)
        and path.stat().st_mode & 0o222 == 0 and path.stat().st_size == OWNED['bytes']
        and hashlib.sha256(path.read_bytes()).hexdigest() == OWNED['sha256'], 'Exact existing ownership helper required')
    name = '_G0_comparison_existing_ownership'
    require(name not in sys.modules, 'Fresh ownership namespace required')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)
    require(module.SOURCE_RELEASED is False, 'Preserved disabled ownership source required')
    module.bound(root, OWNED)
    return module


def authenticate(args):
    root = Path(args.source_root).absolute()
    require(root.resolve() == root and sys.dont_write_bytecode
        and not any(n == 'torch' or n.startswith('torch.') for n in sys.modules), 'Fresh resolved phase/-B/pre-Torch required')
    owned = ownership(root)
    scope_path = Path(args.scope).absolute()
    scope_path = owned.bound(root, {'path': str(scope_path.relative_to(root)),
        'bytes': scope_path.stat().st_size, 'sha256': args.scope_sha256})
    scope = owned.read(scope_path)
    require(scope['schema'] == 'root_owned_G0_complete_pool_comparison_scope_v1'
        and all(scope[k] is True for k in ('root_owned_evaluation_authorized', 'owner_source_review_approved',
            'fixed_before_evaluation', 'exclusive_GPU_window'))
        and all(scope[k] is False for k in ('fits_authorized', 'selection_authorized', 'automatic_retry', 'VALID_TEST_access')),
        'Separate reviewed root scoring-only owner scope required')
    require(socket.gethostname() == owned.HOST and Path.cwd().resolve() == owned.REPO and root == owned.PHASE
        and Path(sys.executable).absolute() == owned.PYTHON
        and str(Path(sys.executable).resolve()) == scope['python_resolved']
        and scope['GPU_UUID'] == owned.GPU and scope['resource_limits'] == CAPS
        and scope['external_watchdog_seconds'] == WATCHDOG and scope['minimum_initial_cuda_free_bytes'] == FREE
        and scope['supervisor_output_relative'] == OUTPUT and scope['evaluation_output_relative'] == EVALUATION_OUTPUT,
        'Exact allocation runtime, proposed fixed caps and unique outputs required')
    require(owned.bound(root, scope['executor']).resolve() == Path(__file__).resolve(), 'Exact owner source required')
    owned.bound(root, scope['owner_review'])
    require(scope['ownership_helper'] == OWNED and scope['process_helper'] == owned.HELPER
        and scope['evaluator'] == EVALUATOR and scope['evaluator_review'] == EVALUATOR_REVIEW,
        'Exact existing helpers/evaluator/review required')
    for key in ('ownership_helper', 'process_helper', 'evaluator', 'evaluator_review'): owned.bound(root, scope[key])
    comparison_path = owned.bound(root, scope['comparison_scope']); comparison = owned.read(comparison_path)
    require(comparison['schema'] == 'root_existing_G0_complete_pool_comparison_scope_v4'
        and all(comparison[k] is True for k in ('root_scoring_authorized', 'caller_source_review_approved',
            'all_training_terminal_closed', 'exclusive_evaluation_window', 'external_owned_supervision_required'))
        and all(comparison[k] is False for k in ('fits_authorized', 'selection_authorized', 'automatic_retry', 'VALID_TEST_access')),
        'Exact fully admitted comparison scope required')
    require(comparison['executor'] == EVALUATOR and comparison['caller_review'] == EVALUATOR_REVIEW
        and comparison['host'] == owned.HOST and comparison['repository'] == str(owned.REPO)
        and comparison['python_executable'] == str(owned.PYTHON)
        and comparison['python_resolved'] == scope['python_resolved'] and comparison['GPU_UUID'] == owned.GPU
        and comparison['resource_limits'] == CAPS and comparison['external_watchdog_seconds'] == WATCHDOG
        and comparison['output_relative'] == EVALUATION_OUTPUT,
        'Bound child invocation/runtime/output/caps differ')
    require(scope['planned_prediction_artifacts'] == 12 and scope['planned_served_member_forwards'] == 44,
        'Original fixed twelve-artifact/44-forward evaluation plan required')
    return root, owned, scope, comparison_path


def available(owned, scope):
    return owned.availability({'prior_job_closures': scope['prior_job_closures'], 'minimum_free_GPU_bytes': FREE})


def launch(args, root, owned, scope, comparison_path):
    try: sample = available(owned, scope)
    except BaseException as error:
        print(json.dumps({'status': 'WAIT_NO_LAUNCH', 'error': str(error), 'automatic_launch_after_WAIT': False})); return 2
    base = root / OUTPUT; evaluation = root / EVALUATION_OUTPUT
    require(not base.exists() and not base.is_symlink() and not evaluation.exists() and not evaluation.is_symlink(),
        'One fixed supervisor/evaluation output already claimed')
    base.mkdir(mode=0o700)
    record = {'token': os.urandom(16).hex(), 'device_inode': [base.stat().st_dev, base.stat().st_ino]}
    owner = owned.Owner(base, record)
    argv = [str(owned.PYTHON), '-B', str(Path(__file__).resolve()), '--execute-authorized', '--mode', 'supervise',
        '--source-root', str(root), '--scope', str(Path(args.scope).absolute()), '--scope-sha256', args.scope_sha256]
    owned.write(owner, 'LAUNCH_SENTINEL.json', {'owner': record, 'scope_sha256': args.scope_sha256,
        'comparison_scope': scope['comparison_scope'], 'supervisor_argv': argv, 'availability': sample,
        'one_evaluator_no_retry': True}, True)
    with (base / 'supervisor.stdout.log').open('xb') as out, (base / 'supervisor.stderr.log').open('xb') as err:
        child = subprocess.Popen(argv, cwd=owned.REPO, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), start_new_session=True)
    owned.write(owner, 'SUPERVISOR_START_ATTEMPT.json', {'pid': child.pid, 'argv': argv,
        'scope_sha256': args.scope_sha256}, True)
    observed = owned.physical(child.pid, argv, parent=os.getpid())
    owned.write(owner, 'LAUNCH_HANDLE.json', {'owner': record, 'scope_sha256': args.scope_sha256,
        'supervisor': observed}, True)
    print(json.dumps({'status': 'DETACHED_G0_COMPARISON_OWNER_HANDLE', 'pid': child.pid,
        'start_ticks': observed['starttime_ticks'], 'wait_in_tool_connection': False})); return 0


def capture(root, owned, process, receipt):
    path = root / EVALUATION_OUTPUT / 'RESULT.json'
    if not path.is_file(): return None
    require(len(receipt['children']) == 1 and receipt['children'][0]['wait4_closed'], 'Capture only after owned child reaped')
    require(not path.is_symlink() and not path.parent.is_symlink() and path.resolve().is_relative_to(root),
        'Exact evaluator result path required')
    readonly = path.stat().st_mode & 0o222 == 0
    row = {'path': str(path.relative_to(root)), 'bytes': path.stat().st_size, 'sha256': process.sha(path)}
    result = owned.read(path)
    receipt.update(original_worker_result=row, worker_status=result.get('status'),
        original_worker_result_readonly=readonly,
        original_worker_resources=result.get('whole_process_resources'),
        original_worker_errors={k: result.get(k) for k in ('error_type', 'error', 'traceback', 'body_error', 'restoration_errors')},
        original_worker_costs={k: result.get(k) for k in ('completed_served_member_forwards', 'model_fits', 'persistent_updates',
            'A_labels_opened', 'evaluation_outputs_complete', 'original_G0_metric_gate_pass')})
    return result


def supervise(args, root, owned, scope, comparison_path):
    base = root / OUTPUT; claim = owned.read(base / 'LAUNCH_SENTINEL.json')
    owner = owned.Owner(base, claim['owner']); owner.verify(claim['owner']['token'])
    receipt = {'schema': 'once_only_G0_comparison_owned_terminal_v1', 'status': 'FAIL_OWNED_G0_COMPARISON',
        'scope_sha256': args.scope_sha256, 'comparison_scope': scope['comparison_scope'], 'evaluator': EVALUATOR,
        'launch_attempts': 0, 'children': [], 'cleanup_errors': [], 'publication_errors': [],
        'numeric_imports_in_supervisor': False, 'fits_authorized': False, 'selection_authorized': False,
        'VALID_TEST_access': False, 'automatic_retry': False, 'resource_limits': CAPS,
        'planned_prediction_artifacts': 12, 'planned_served_member_forwards': 44}
    process = None; code = 1
    old = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    def save():
        receipt['supervisor_elapsed_seconds'] = time.monotonic() - STARTED
        owned.write(owner, 'SUPERVISOR_STATUS.json', receipt)
    def interrupted(number, frame): raise InterruptedError('Owned comparison supervisor interrupted: ' + str(number))
    try:
        require(claim['scope_sha256'] == args.scope_sha256 and claim['comparison_scope'] == scope['comparison_scope'],
            'Once-only owner claim differs')
        deadline = time.monotonic() + 5
        while not (base / 'LAUNCH_HANDLE.json').exists():
            require(time.monotonic() < deadline, 'No verified owner handle; no evaluator launch'); time.sleep(0.05)
        handle = owned.read(base / 'LAUNCH_HANDLE.json')
        require(handle['supervisor']['pid'] == os.getpid() and handle['scope_sha256'] == args.scope_sha256, 'Owner handle differs')
        owned.physical(os.getpid(), claim['supervisor_argv'], start=handle['supervisor']['starttime_ticks'])
        receipt['pre_child_availability'] = available(owned, scope)
        require(not (root / EVALUATION_OUTPUT).exists(), 'Once-only evaluator output already exists')
        process = owned.helper(); process.validate_caps(CAPS, WATCHDOG, GRACE, REAP)
        for s in old: signal.signal(s, interrupted)
        argv = ['--execute-authorized', '--source-root', str(root), '--scope', str(comparison_path),
            '--scope-sha256', scope['comparison_scope']['sha256'], '--output', str(root / EVALUATION_OUTPUT)]
        receipt['launch_attempts'] = 1; receipt['evaluator_argv'] = argv; save()
        child = process.launch(owned.REPO, owner, claim['owner']['token'], 'evaluator', str(owned.PYTHON),
            root / EVALUATOR['path'], argv, dict(os.environ, CUDA_VISIBLE_DEVICES=scope['GPU_UUID'],
                PYTHONDONTWRITEBYTECODE='1'), receipt['children'])
        child['verified_physical_identity'] = owned.physical(child['pid'],
            [str(owned.PYTHON), '-B', str(root / EVALUATOR['path']), *argv], os.getpid(), child['starttime_ticks']); save()
        while not process.watch(child, WATCHDOG, GRACE, REAP): time.sleep(0.25)
        result = capture(root, owned, process, receipt); process.resource_closure(child, result, CAPS)
        require(result['status'] == 'COMPLETE_POOL_COMPARISON_SCORED_RESOURCE_ONLY'
            and receipt['original_worker_result_readonly'] is True
            and result['worker_sha256'] == EVALUATOR['sha256'] and result['scope_sha256'] == scope['comparison_scope']['sha256']
            and result['evaluation_outputs_complete'] is True and result['A_labels_opened'] is True
            and result['model_fits'] == result['persistent_updates'] == 0 and result['automatic_retry'] is False
            and result['completed_served_member_forwards'] == 44 and not result['restoration_errors'],
            'Exact complete scoring-only evaluator result required')
        artifacts = {}
        for name in ('PREDICTIONS_COMPLETE.json', 'SCORES.json', 'ERROR_FLOW.json'):
            artifacts[name] = process.immutable_descriptor(root, root / EVALUATION_OUTPUT / name)
        predictions = owned.read(owned.bound(root, artifacts['PREDICTIONS_COMPLETE.json']))
        require(predictions['A_labels_opened'] is False and predictions['served_member_forwards'] == 44
            and predictions['prediction_artifacts'] == 12 and len(predictions['predictions']) == 12,
            'Original complete frozen prediction manifest required')
        for name, row in predictions['predictions'].items():
            relative = str((Path(EVALUATION_OUTPUT) / row['path']))
            owned.bound(root, dict(row, path=relative))
        receipt['evaluation_artifacts'] = artifacts; receipt['prediction_artifacts'] = predictions['predictions']
        owned.bound(root, EVALUATOR); owned.bound(root, scope['comparison_scope'])
        receipt['status'] = 'G0_COMPARISON_SCORED_OWNED_WHOLE_CHILD_CLOSED'; code = 0
    except BaseException as error:
        receipt.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        for s in old:
            try: signal.signal(s, signal.SIG_IGN)
            except BaseException: receipt['cleanup_errors'].append({'stage': 'ignore_repeated_signal', 'traceback': traceback.format_exc()})
        if process is not None:
            try: receipt['cleanup_errors'].extend(process.cleanup(receipt['children'], GRACE, REAP))
            except BaseException: receipt['cleanup_errors'].append({'stage': 'owned_cleanup', 'traceback': traceback.format_exc()})
            try: capture(root, owned, process, receipt)
            except BaseException: receipt.setdefault('receipt_errors', []).append(traceback.format_exc()); code = 1
        for s, handler in old.items():
            try: signal.signal(s, handler)
            except BaseException: receipt['cleanup_errors'].append({'stage': 'restore_signal', 'traceback': traceback.format_exc()})
        receipt['owned_child_closure_complete'] = (len(receipt['children']) == 1
            and all(c['wait4_closed'] for c in receipt['children']) and not receipt['cleanup_errors'])
        if not receipt['owned_child_closure_complete']: code = 1
        if code: receipt['status'] = 'FAIL_OWNED_G0_COMPARISON'
        receipt['owned_scoring_resource_closure_PASS'] = code == 0
        receipt['supervisor_elapsed_seconds'] = time.monotonic() - STARTED
        receipt['UTC_terminal'] = datetime.now(timezone.utc).isoformat()
        try: owned.write(owner, 'TERMINAL.json', receipt, True)
        except BaseException:
            receipt['publication_errors'].append(traceback.format_exc()); code = 1
            receipt['status'] = 'FAIL_OWNED_G0_COMPARISON'; receipt['owned_scoring_resource_closure_PASS'] = False
        print(json.dumps(receipt, sort_keys=True, allow_nan=False))
    return code


def poll(args, root, owned, scope, comparison_path):
    base = root / OUTPUT
    if (base / 'TERMINAL.json').exists():
        print(json.dumps({'status': 'TERMINAL', 'path': str(base / 'TERMINAL.json'), 'automatic_retry': False})); return 0
    handle = owned.read(base / 'LAUNCH_HANDLE.json')
    owner = owned.Owner(base, handle['owner']); owner.verify(handle['owner']['token'])
    require(handle['scope_sha256'] == args.scope_sha256, 'Poll owner scope differs')
    try: owned.physical(handle['supervisor']['pid'], handle['supervisor']['argv'], start=handle['supervisor']['starttime_ticks'])
    except BaseException as error:
        print(json.dumps({'status': 'OBSERVATION_FAILURE_NO_RELAUNCH', 'error': str(error), 'automatic_retry': False})); return 1
    print(json.dumps({'status': 'RUNNING', 'automatic_retry': False})); return 0


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--execute-authorized', action='store_true')
    parser.add_argument('--mode', choices=('launch', 'supervise', 'poll'))
    for name in ('source-root', 'scope', 'scope-sha256'): parser.add_argument('--' + name)
    args = parser.parse_args()
    if not args.execute_authorized:
        print(json.dumps({'status': 'DISABLED_OWNED_G0_COMPARISON', 'SOURCE_RELEASED': False})); return 0
    require(SOURCE_RELEASED is False and all((args.mode, args.source_root, args.scope, args.scope_sha256)),
        'Explicit reviewed root owner scope required')
    context = authenticate(args)
    return {'launch': launch, 'supervise': supervise, 'poll': poll}[args.mode](args, *context)


if __name__ == '__main__': raise SystemExit(main())
