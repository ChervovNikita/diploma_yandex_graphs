"""Fresh fixed100-epoch fit child; preserve live and final driver CUDA peaks."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import threading

PHASE = Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git/experiments_iclr/postsubmission_20260930')
EXECUTION = PHASE / 'exact_cb_support_bucket_paired_predictive_execution_root_20261004_v1'
DRIVER = Path(__file__).resolve().parent / 'paired_run.py'
DRIVER_SHA = None
PEAK_NAMES = ('cuda_peak_allocated_bytes', 'cuda_peak_reserved_bytes')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def peaks(value):
    require(type(value) is dict and all(name in value for name in PEAK_NAMES), 'Driver peak fields missing')
    require(all(type(value[name]) is int and value[name] >= 0 for name in PEAK_NAMES), 'Invalid driver peak byte count')
    return {name: value[name] for name in PEAK_NAMES}


def merge(state, category, value):
    observed = peaks(value)
    prior = state.setdefault(category, {name: 0 for name in PEAK_NAMES})
    for name in PEAK_NAMES:
        prior[name] = max(prior[name], observed[name])
        state[name] = max(state[name], observed[name])


def driver_captures(output, state, release_sha, arm, seed):
    """Read only the owned driver's atomic accounting/complete-fit metadata."""
    def read(name):
        path = output / name
        if not path.exists():
            return None
        raw = path.read_bytes()
        value = json.loads(raw)
        require(value.get('identity', {}).get('driver_manifest_sha256') == DRIVER_SHA, 'Driver capture source identity differs')
        state.setdefault('driver_capture_receipts', {})[name] = {
            'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
        return value

    accounting = read('ATTEMPTS.json')
    if accounting is not None:
        require(accounting.get('schema') == 'ncnc-pilot-inclusive-attempts-v1', 'Wrong driver accounting schema')
        require(len(accounting['attempts']) == 1, 'Fit child requires exactly one fresh attempt')
        row = accounting['attempts'][0]
        require(row['stage'] == 'fit' and row['unit'] == arm and row['base_seed'] == seed,
                'Wrong captured fit invocation')
        require(row['root_release_sha256'] == release_sha, 'Capture belongs to another root release')
        captured = row.get('observed_CUDA_peak_before_arm_resets')
        if captured is not None:
            merge(state, 'captured_pre_reset_peak_bytes', captured)
            state['driver_pre_reset_capture_observed'] = True
        if 'failure_peak_allocated_bytes' in row and 'failure_peak_reserved_bytes' in row:
            merge(state, 'captured_failure_peak_bytes', {
                'cuda_peak_allocated_bytes': row['failure_peak_allocated_bytes'],
                'cuda_peak_reserved_bytes': row['failure_peak_reserved_bytes']})

    result = read('COMPLETE.json')
    if result is not None:
        common = sys.modules.get('pilot_common')
        require(common is not None and Path(common.__file__).resolve() == DRIVER.parent / 'pilot_common.py',
                'V5 fit profile helper shadowed')
        common.require_profile_receipt(result)
        require(result.get('schema') == 'ncnc-pattern-complete-fit-v1' and result.get('arm') == arm
                and result.get('seed') == seed and result.get('epochs') == 100 and result.get('optimizer_steps') == 1700
                and result.get('selection_candidates') == 100 and result.get('full_VALID_evaluations') == 102
                and result.get('extra_complete_VALID_replay_evaluations') == 2
                and result.get('selected_roundtrip_and_full_served_replay') is True
                and result.get('resource_state_donor') is False and result.get('test_file_opened') is False
                and result.get('predictive_values_exposed') is False,
                'Incomplete fixed fit capture')
        require(result['identity']['runtime_profile_id'] == common.RUNTIME_PROFILE_ID
                and result['identity']['runtime_profile_transition'] == common.RUNTIME_PROFILE_TRANSITION,
                'Wrong captured deterministic profile identity')
        merge(state, 'captured_final_driver_peak_bytes', result)
        state['driver_complete_capture_observed'] = True


def main():
    global DRIVER_SHA
    DRIVER_SHA = hashlib.sha256((DRIVER.parent/'MANIFEST.json').read_bytes()).hexdigest()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-release', required=True, type=Path)
    parser.add_argument('--supervision-output', required=True, type=Path)
    parser.add_argument('--arm', required=True, choices=('target_only', 'joint', 'separate'))
    parser.add_argument('--seed', required=True, type=int, choices=(0,1,2))
    args = parser.parse_args()
    require('torch' not in sys.modules and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8',
            'Fit workspace must be configured before first Torch import')
    release_raw = args.root_release.read_bytes()
    release = json.loads(release_raw)
    release_sha = hashlib.sha256(release_raw).hexdigest()
    expected = {'stage': 'fit', 'unit': args.arm, 'base_seed': args.seed,
                'output_directory': str(EXECUTION / ('fit_' + args.arm + '_seed' + str(args.seed) + '/run01'))}
    require(release['authorized_stages'] == ['fit'] and release['authorized_invocations'] == [expected],
            'This child supports only one fresh fixed fit')
    require(release.get('execution_enabled') is True and release.get('root_authorization_reference') and release['driver_manifest_sha256'] == DRIVER_SHA,
            'Exact V5 root release required')
    require(release.get('runtime_profile_transition') == {'deterministic_algorithms_before': False,
            'deterministic_algorithms_after': True, 'warn_only_after': False, 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'},
            'Exact V5 deterministic profile transition required')
    require(Path(release['fit_driver_path']).resolve() == DRIVER
            and args.supervision_output.resolve() == EXECUTION / ('supervision/fit_' + args.arm + '_seed' + str(args.seed) + '/run01')
            and args.root_release.resolve().is_relative_to(EXECUTION), 'Wrong per-arm fit paths')
    caps = release['fit_caps']
    output = Path(expected['output_directory'])
    stopped = threading.Event()
    state = {'schema': 'ncnc-pattern-fit-child-CUDA-peaks-v1', 'PID': os.getpid(),
             'CUDA_observed': False, 'cuda_peak_allocated_bytes': 0, 'cuda_peak_reserved_bytes': 0,
             'caps': caps, 'source_reset_or_model_hooks': False,
             'driver_pre_reset_capture_observed': False, 'driver_complete_capture_observed': False,
             'arm': args.arm, 'seed': args.seed, 'resume_requested': False}
    path = args.supervision_output / 'CHILD_CUDA_PEAKS.json'

    def sample():
        module = sys.modules.get('torch')
        cuda = getattr(module, 'cuda', None)
        if cuda is not None and hasattr(cuda, 'is_initialized') and cuda.is_initialized():
            state['CUDA_observed'] = True
            merge(state, 'live_persistent_counter_maxima', {
                'cuda_peak_allocated_bytes': int(cuda.max_memory_allocated(0)),
                'cuda_peak_reserved_bytes': int(cuda.max_memory_reserved(0))})
        driver_captures(output, state, release_sha, args.arm, args.seed)
        state['UTC'] = datetime.now(timezone.utc).isoformat()
        exceeded = [name for name in PEAK_NAMES if state[name] > caps[name]]
        if exceeded:
            state['cap_violation'] = {'kind': 'CUDA_peak', 'fields': exceeded, 'exit_code': 88}
        write(path, state)
        if exceeded:
            os._exit(88)

    def observe():
        while not stopped.wait(.25):
            try:
                sample()
            except Exception as error:
                state['monitor_failure'] = {'type': type(error).__name__, 'condition': str(error)}
                write(path, state)
                os._exit(89)

    argv = [str(DRIVER), '--root-release', str(args.root_release.resolve()), '--stage', 'fit',
            '--unit', args.arm, '--base-seed', str(args.seed), '--output', str(output)]
    state['driver_argv'] = argv
    write(path, state)
    observer = threading.Thread(target=observe, name='own-fit-CUDA-peak-observer', daemon=True)
    observer.start()
    try:
        sys.path.insert(0, str(DRIVER.parent))
        sys.argv = argv
        runpy.run_path(str(DRIVER), run_name='__main__')
    finally:
        stopped.set()
        observer.join()
        sample()
        write(path, state)
    require(state['CUDA_observed'] and state['driver_complete_capture_observed'],
            'Successful fresh fit child lacks final driver peak custody')


if __name__ == '__main__':
    main()
