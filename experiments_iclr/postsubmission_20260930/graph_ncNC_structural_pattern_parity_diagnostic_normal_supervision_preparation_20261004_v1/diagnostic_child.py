"""Normal-host diagnostic child with persistent CUDA peak caps; no resets/hooks."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import threading

HERE = Path(__file__).resolve().parent
PHASE = Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git/experiments_iclr/postsubmission_20260930')
EXECUTION = PHASE / 'graph_ncNC_structural_pattern_parity_diagnostic_execution_root_20261004_v1'
DIAGNOSTIC_SHA = 'b665affd187593875a2be81e30dbe7a3c8e3ce231f7e344d791d67a800331ebe'
CAPS = {'wall_seconds': 1800, 'host_RSS_bytes': 32 * 1024 ** 3,
        'cuda_peak_allocated_bytes': 70 * 1024 ** 3, 'cuda_peak_reserved_bytes': 75 * 1024 ** 3}
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


def merge(state, category, observed):
    require(all(type(observed[name]) is int and observed[name] >= 0 for name in PEAK_NAMES), 'Invalid CUDA peak bytes')
    prior = state.setdefault(category, {name: 0 for name in PEAK_NAMES})
    for name in PEAK_NAMES:
        prior[name] = max(prior[name], observed[name])
        state[name] = max(state[name], observed[name])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-admission', required=True, type=Path)
    parser.add_argument('--admission-sha256', required=True)
    parser.add_argument('--supervision-output', required=True, type=Path)
    args = parser.parse_args()
    admission_path = args.root_admission.resolve()
    raw = admission_path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == args.admission_sha256, 'Child admission bytes differ')
    admission = json.loads(raw)
    require(admission_path == EXECUTION / 'ROOT_DIAGNOSTIC_ADMISSION.json'
            and args.supervision_output.resolve() == EXECUTION / 'supervision/run01'
            and admission.get('schema') == 'ncnc-pattern-parity-diagnostic-root-admission-v1'
            and admission.get('status') == 'APPROVED' and admission.get('root_authorization_reference')
            and admission['diagnostic_manifest_sha256'] == DIAGNOSTIC_SHA
            and admission['caps'] == CAPS and admission['authorized_optimizer_updates'] == 12
            and admission['authorized_member_trajectory_updates'] == 48
            and admission['runner_path'] == str(HERE / 'diagnostic_run.py'), 'Wrong separately admitted diagnostic child')
    output = EXECUTION / 'diagnostic/run01'
    stopped = threading.Event()
    state = {'schema': 'ncnc-pattern-parity-diagnostic-child-CUDA-peaks-v1', 'PID': os.getpid(),
             'admission_sha256': args.admission_sha256, 'diagnostic_manifest_sha256': DIAGNOSTIC_SHA,
             'CUDA_observed': False, 'cuda_peak_allocated_bytes': 0, 'cuda_peak_reserved_bytes': 0,
             'caps': CAPS, 'source_reset_or_model_hooks': False, 'driver_peak_capture_observed': False}
    path = args.supervision_output / 'CHILD_CUDA_PEAKS.json'
    def sample():
        module = sys.modules.get('torch')
        cuda = getattr(module, 'cuda', None)
        if cuda is not None and hasattr(cuda, 'is_initialized') and cuda.is_initialized():
            state['CUDA_observed'] = True
            merge(state, 'live_persistent_counter_maxima', {
                'cuda_peak_allocated_bytes': int(cuda.max_memory_allocated(0)),
                'cuda_peak_reserved_bytes': int(cuda.max_memory_reserved(0))})
        for name in ('ATTEMPT.json', 'FINAL.json', 'FAILED.json'):
            capture = output / name
            if not capture.exists():
                continue
            raw_capture = capture.read_bytes()
            value = json.loads(raw_capture)
            require(value.get('identity', {}).get('admission_sha256') == args.admission_sha256
                    and value.get('identity', {}).get('diagnostic_manifest_sha256') == DIAGNOSTIC_SHA,
                    'Diagnostic peak capture identity differs')
            state.setdefault('driver_capture_receipts', {})[name] = {
                'path': str(capture), 'bytes': len(raw_capture), 'sha256': hashlib.sha256(raw_capture).hexdigest()}
            if all(field in value for field in PEAK_NAMES):
                merge(state, 'captured_driver_peak_maxima', value)
                state['driver_peak_capture_observed'] = True
        state['UTC'] = datetime.now(timezone.utc).isoformat()
        exceeded = [name for name in PEAK_NAMES if state[name] > CAPS[name]]
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
    argv = [str(HERE / 'diagnostic_run.py'), '--root-admission', str(admission_path),
            '--admission-sha256', args.admission_sha256]
    state['driver_argv'] = argv
    write(path, state)
    observer = threading.Thread(target=observe, name='own-diagnostic-CUDA-peak-observer', daemon=True)
    observer.start()
    try:
        sys.path.insert(0, str(HERE))
        sys.argv = argv
        runpy.run_path(str(HERE / 'diagnostic_run.py'), run_name='__main__')
    finally:
        stopped.set()
        observer.join()
        sample()
        write(path, state)
    require(state['CUDA_observed'] and state['driver_peak_capture_observed'], 'Successful child lacks complete CUDA custody')


if __name__ == '__main__':
    main()
