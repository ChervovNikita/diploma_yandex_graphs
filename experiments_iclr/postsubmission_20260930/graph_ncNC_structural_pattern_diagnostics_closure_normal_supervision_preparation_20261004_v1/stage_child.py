"""Fixed V5 diagnostic/stdlib closure child; own-stage peak and receipt accounting."""
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
EXECUTION = PHASE / 'graph_ncNC_structural_pattern_execution_root_20261004_v5'
DRIVER = PHASE / 'graph_ncNC_structural_pattern_pilot_preparation_20261004_v5/pattern_run.py'
DRIVER_SHA = '9fc539b8f92d7d4e883224c3b0aa85ae583b64c70f2a701a4a648fd818aa32a1'
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


def driver_captures(output, state, release_sha, stage):
    """Read owned atomic accounting/completion metadata; no private outcome JSON."""
    def read(name):
        path = output / name
        if not path.exists():
            return None
        raw = path.read_bytes(); value = json.loads(raw)
        require(value.get('identity', {}).get('driver_manifest_sha256') == DRIVER_SHA, 'Driver capture source identity differs')
        state.setdefault('driver_capture_receipts', {})[name] = {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
        return value
    accounting = read('ATTEMPTS.json')
    if accounting is not None:
        require(accounting.get('schema') == 'ncnc-pilot-inclusive-attempts-v1' and len(accounting['attempts']) == 1, 'One fresh owned stage attempt required')
        row = accounting['attempts'][0]
        require(row['stage'] == stage and row['unit'] == 'pair' and row['base_seed'] == 0 and row['root_release_sha256'] == release_sha, 'Captured stage invocation differs')
        captured = row.get('observed_CUDA_peak_before_arm_resets')
        if captured is not None:
            merge(state, 'captured_pre_reset_peak_bytes', captured)
            state['driver_pre_reset_capture_observed'] = True
        if 'failure_peak_allocated_bytes' in row and 'failure_peak_reserved_bytes' in row:
            merge(state, 'captured_failure_peak_bytes', {'cuda_peak_allocated_bytes': row['failure_peak_allocated_bytes'], 'cuda_peak_reserved_bytes': row['failure_peak_reserved_bytes']})
    result = read('COMPLETE.json' if stage == 'diagnostics' else 'CLOSURE.json')
    if result is None:
        return
    if stage == 'diagnostics':
        common = sys.modules.get('pilot_common')
        require(common is not None and Path(common.__file__).resolve() == DRIVER.parent / 'pilot_common.py', 'V5 profile helper shadowed')
        common.require_profile_receipt(result)
        require(result.get('schema') == 'ncnc-pattern-diagnostics-complete-v1' and result.get('status') == 'COMPLETE'
                and result.get('arms') == ['J', 'F'] and result.get('full_selected_VALID_evaluations') == 2
                and result.get('full_TRAIN_mask_epochs') == 2 and result.get('matched_mask_supports') is True
                and result.get('test_file_opened') is False and result.get('predictive_values_exposed') is False,
                'Fixed diagnostic coverage differs')
        merge(state, 'captured_final_driver_peak_bytes', result)
    else:
        require(result.get('schema') == 'ncnc-pattern-pair-closure-v1' and result.get('status') == 'CLOSED'
                and result.get('unique_scientific_fits') == 2 and result.get('scientific_optimizer_steps') == 3400
                and result.get('scientific_selector_candidates') == 200
                and result.get('complete_scientific_VALID_evaluations_including_replay_and_diagnostics') == 206
                and result.get('matched_pair_streams_RNG_and_supports') is True and result.get('TEST_supported') is False,
                'Fixed stdlib closure coverage differs')
    state['driver_complete_capture_observed'] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-release', required=True, type=Path)
    parser.add_argument('--supervision-output', required=True, type=Path)
    parser.add_argument('--stage', required=True, choices=('diagnostics', 'close'))
    args = parser.parse_args()
    require('torch' not in sys.modules and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'Workspace configured before first Torch import required')
    release_raw = args.root_release.read_bytes(); release = json.loads(release_raw)
    release_sha = hashlib.sha256(release_raw).hexdigest()
    expected = {'stage': args.stage, 'unit': 'pair', 'base_seed': 0, 'output_directory': str(EXECUTION / (args.stage + '/run01'))}
    require(release['authorized_stages'] == [args.stage] and release['authorized_invocations'] == [expected], 'One fresh fixed pair stage required')
    require(release.get('execution_enabled') is True and release.get('root_authorization_reference') and release['driver_manifest_sha256'] == DRIVER_SHA, 'Exact V5 root release required')
    require(release.get('runtime_profile_transition') == {'deterministic_algorithms_before': False, 'deterministic_algorithms_after': True, 'warn_only_after': False, 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}, 'Exact V5 profile transition required')
    require(Path(release['stage_driver_path']).resolve() == DRIVER and args.supervision_output.resolve() == EXECUTION / ('supervision/' + args.stage + '/run01') and args.root_release.resolve().is_relative_to(EXECUTION), 'Wrong fixed stage paths')
    caps = release['stage_caps']; output = Path(expected['output_directory'])
    stopped = threading.Event()
    state = {'schema': 'ncnc-pattern-diagnostic-closure-child-capture-v1', 'PID': os.getpid(), 'stage': args.stage,
             'CUDA_observed': False, 'cuda_peak_allocated_bytes': 0, 'cuda_peak_reserved_bytes': 0, 'caps': caps,
             'source_reset_or_model_hooks': False, 'driver_pre_reset_capture_observed': False,
             'driver_complete_capture_observed': False, 'new_optimization_or_selection': False, 'resume_requested': False}
    path = args.supervision_output / 'CHILD_STAGE_CAPTURE.json'

    def sample():
        module = sys.modules.get('torch')
        cuda = getattr(module, 'cuda', None)
        if args.stage == 'close':
            require(not any(name in sys.modules for name in ('torch', 'numpy', 'scipy', 'torch_sparse', 'torch_geometric')), 'Stdlib closure imported a scientific library')
            state['stdlib_closure_scientific_imports_absent'] = True
        if cuda is not None and hasattr(cuda, 'is_initialized') and cuda.is_initialized():
            state['CUDA_observed'] = True
            merge(state, 'live_persistent_counter_maxima', {'cuda_peak_allocated_bytes': int(cuda.max_memory_allocated(0)), 'cuda_peak_reserved_bytes': int(cuda.max_memory_reserved(0))})
        driver_captures(output, state, release_sha, args.stage)
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

    argv = [str(DRIVER), '--root-release', str(args.root_release.resolve()), '--stage', args.stage,
            '--unit', 'pair', '--base-seed', '0', '--output', str(output)]
    state['driver_argv'] = argv; write(path, state)
    observer = threading.Thread(target=observe, name='own-fixed-stage-receipt-observer', daemon=True)
    observer.start()
    try:
        sys.path.insert(0, str(DRIVER.parent)); sys.argv = argv
        runpy.run_path(str(DRIVER), run_name='__main__')
    finally:
        stopped.set(); observer.join(); sample(); write(path, state)
    require(state['driver_complete_capture_observed'], 'Complete fixed stage capture missing')
    require(state['CUDA_observed'] if args.stage == 'diagnostics' else state['stdlib_closure_scientific_imports_absent'], 'Stage runtime capture differs')


if __name__ == '__main__':
    main()
