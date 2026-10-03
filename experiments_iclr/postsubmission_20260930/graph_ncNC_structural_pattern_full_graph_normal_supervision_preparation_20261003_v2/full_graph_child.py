"""Full-real-graph-only child; preserve live and driver-captured CUDA peaks."""
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
EXECUTION = PHASE / 'graph_ncNC_structural_pattern_full_graph_execution_root_20261003_v2'
DRIVER = PHASE / 'graph_ncNC_structural_pattern_pilot_preparation_20261003_v3/pattern_run.py'
DRIVER_SHA = 'fa7b2a7a2c6ec83362f3c820fb4f7ad5288e5cc9fb0ee5139614d6690f3f7f89'
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


def driver_captures(output, state, release_sha):
    """Read only the owned driver's atomic accounting/qualification metadata."""
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
        require(len(accounting['attempts']) == 1, 'Full-graph child requires exactly one fresh attempt')
        row = accounting['attempts'][0]
        require(row['stage'] == 'full_graph' and row['unit'] == 'pair' and row['base_seed'] == 0,
                'Wrong captured full-graph invocation')
        require(row['root_release_sha256'] == release_sha, 'Capture belongs to another root release')
        captured = row.get('observed_CUDA_peak_before_arm_resets')
        if captured is not None:
            merge(state, 'captured_pre_reset_peak_bytes', captured)
            state['driver_pre_reset_capture_observed'] = True
        if 'failure_peak_allocated_bytes' in row and 'failure_peak_reserved_bytes' in row:
            merge(state, 'captured_failure_peak_bytes', {
                'cuda_peak_allocated_bytes': row['failure_peak_allocated_bytes'],
                'cuda_peak_reserved_bytes': row['failure_peak_reserved_bytes']})

    qualification = read('QUALIFICATION.json')
    if qualification is not None:
        require(qualification.get('schema') == 'ncnc-pattern-qualification-v1'
                and qualification.get('stage') == 'full_graph' and qualification.get('status') == 'PASS',
                'Wrong full-graph qualification capture')
        require(qualification['full_TRAIN_epochs'] == 2 and qualification['full_VALID_evaluations'] == 2
                and qualification['project_metric_computed'] is False and qualification['state_donor'] is False
                and qualification['test_file_opened'] is False
                and qualification['full_graph_numerical_and_serialized_replay'] is True,
                'Incomplete full-graph qualification coverage')
        probe = peaks(qualification['numerical_probe_peaks'])
        state['captured_probe_peaks'] = probe
        merge(state, 'captured_pre_reset_peak_bytes', probe)
        require(len(qualification['records']) == 2
                and {record['arm'] for record in qualification['records']} == {'J', 'F'},
                'Both complete arm captures required')
        for record in qualification['records']:
            require(record['TRAIN']['full_batches'] == 17 and record['TRAIN']['optimizer_steps'] == 17,
                    'Full TRAIN batches/steps omitted from capture')
            require(record['VALID']['positive_queries'] == 60084
                    and record['VALID']['negative_queries'] == 100000
                    and record['VALID']['route_count'] == 5,
                    'Full five-route VALID capture required')
            observed = peaks(record)
            state.setdefault('captured_arm_peaks', {})[record['arm']] = observed
            merge(state, 'captured_arm_peak_bytes', observed)
        merge(state, 'captured_final_driver_peak_bytes', qualification)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-release', required=True, type=Path)
    parser.add_argument('--supervision-output', required=True, type=Path)
    args = parser.parse_args()
    release_raw = args.root_release.read_bytes()
    release = json.loads(release_raw)
    release_sha = hashlib.sha256(release_raw).hexdigest()
    expected = {'stage': 'full_graph', 'unit': 'pair', 'base_seed': 0,
                'output_directory': str(EXECUTION / 'full_graph/run01')}
    require(release['authorized_stages'] == ['full_graph'] and release['authorized_invocations'] == [expected],
            'This child supports only one complete full-real-graph pair')
    require(release.get('root_authorization_reference') and release['driver_manifest_sha256'] == DRIVER_SHA,
            'Exact V3 root release required')
    require(Path(release['full_graph_driver_path']).resolve() == DRIVER
            and args.supervision_output.resolve() == EXECUTION / 'supervision/run01'
            and args.root_release.resolve().is_relative_to(EXECUTION), 'Wrong full-graph paths')
    caps = release['full_graph_caps']
    output = Path(expected['output_directory'])
    stopped = threading.Event()
    state = {'schema': 'ncnc-pattern-full-graph-child-CUDA-peaks-v1', 'PID': os.getpid(),
             'CUDA_observed': False, 'cuda_peak_allocated_bytes': 0, 'cuda_peak_reserved_bytes': 0,
             'caps': caps, 'source_reset_or_model_hooks': False,
             'driver_pre_reset_capture_observed': False, 'captured_arm_peaks': {}}
    path = args.supervision_output / 'CHILD_CUDA_PEAKS.json'

    def sample():
        module = sys.modules.get('torch')
        cuda = getattr(module, 'cuda', None)
        if cuda is not None and hasattr(cuda, 'is_initialized') and cuda.is_initialized():
            state['CUDA_observed'] = True
            merge(state, 'live_persistent_counter_maxima', {
                'cuda_peak_allocated_bytes': int(cuda.max_memory_allocated(0)),
                'cuda_peak_reserved_bytes': int(cuda.max_memory_reserved(0))})
        driver_captures(output, state, release_sha)
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

    argv = [str(DRIVER), '--root-release', str(args.root_release.resolve()), '--stage', 'full_graph',
            '--unit', 'pair', '--base-seed', '0', '--output', str(output)]
    state['driver_argv'] = argv
    write(path, state)
    observer = threading.Thread(target=observe, name='own-full-graph-CUDA-peak-observer', daemon=True)
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
    require(state['CUDA_observed'] and state['driver_pre_reset_capture_observed']
            and set(state['captured_arm_peaks']) == {'J', 'F'}, 'Successful child lacks complete peak custody')


if __name__ == '__main__':
    main()
