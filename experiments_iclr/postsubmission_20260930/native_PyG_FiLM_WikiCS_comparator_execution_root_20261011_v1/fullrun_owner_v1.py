"""Existing finite endpoint owner, scoped to one full FiLM fit or readout."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

R = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
P = R / 'experiments_iclr/postsubmission_20260930'
H = P / 'native_PyG_FiLM_WikiCS_comparator_execution_root_20261011_v1'
S = P / 'native_PyG_FiLM_WikiCS_comparator_source_20261011_v1'
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
LIMITS = dict(wall_seconds=21600, cuda_bytes=16 * 2**30, rss_bytes=16 * 2**30)
QUEUE_SECONDS = 86400
READOUT_SECONDS = 900
SEEDS = [7301, 7403, 7507]
ARMS = ['FiLM_M1', 'FiLM_shared4', 'FiLM_I4']
SOURCE_MANIFEST_SHA256 = 'f66f2448cd1df8b2a77efa5f8b71b5fe5d9a50d54ed75da01d2c28a308031a0d'
EXECUTION_FILES = ['bind_fullfit_v1.py', 'fullrun_owner_v1.py', 'queue_full_pipeline_v1.py']


def stamp(): return datetime.now(timezone.utc).isoformat()
def read(path): return json.loads(Path(path).read_text())
def item(path):
    path = Path(path).resolve()
    assert path.is_relative_to(R) and path.is_file()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
def identity(pid):
    path = Path('/proc', str(pid), 'stat')
    if not path.exists(): return None
    try: fields = path.read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError: return None
    return dict(PID=pid, start_ticks=int(fields[19]), state=fields[0])
def route():
    assert socket.gethostname() == 'anogena-2-0'
    assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=15).splitlines() == [UUID]
    assert Path.cwd() == R and Path(__file__).resolve().parent == H
def cuda_processes():
    rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'], text=True, timeout=15).splitlines()
    return {int(pid.strip()): int(memory.strip()) * 2**20 for pid, memory in (row.split(',') for row in rows if row.strip())}
def environment(stage):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=UUID if stage == 'fit' else '', OMP_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1', DGLBACKEND='pytorch')
    env.pop('PYTHONPATH', None); env.pop('PYTHONHOME', None)
    return env


def own(stage):
    route()
    config = H / 'SCIENTIFIC_CONFIG_V1.json'; cfg = read(config)
    assert cfg['family_limits'] == LIMITS and cfg['seeds'] == SEEDS
    for row in cfg['source_receipts'].values(): assert item(row['path']) == row
    admission = read(cfg['root_admission']['path'])
    assert item(cfg['root_admission']['path']) == cfg['root_admission']
    assert admission['action'] == 'fixed_family_fit'
    side = H / (stage + '_owner_v1'); side.mkdir(exist_ok=False)
    output = H / ('actual_family_v1' if stage == 'fit' else 'complete_analysis_v1')
    assert not output.exists()
    cap = LIMITS['wall_seconds'] if stage == 'fit' else READOUT_SECONDS
    command = [str(R / '.venv/bin/python'), '-B', str(S / ('run_family.py' if stage == 'fit' else 'analysis.py'))]
    command += ['--config', str(config), '--output', str(output)] if stage == 'fit' else ['--input-spec', str(H / 'COMPLETE_READER_INPUT_SPEC_V1.json'), '--output', str(output)]
    began = time.monotonic(); timed_out = False; failure = None
    peaks = dict(process_RSS_bytes=0, owned_CUDA_bytes=0)
    with (side / 'stdout.log').open('x') as out, (side / 'stderr.log').open('x') as err:
        child = subprocess.Popen(command, cwd=R, env=environment(stage), stdin=subprocess.DEVNULL,
                                 stdout=out, stderr=err, start_new_session=True)
        ident = identity(child.pid) or dict(PID=child.pid, start_ticks=None, state=None)
        try:
            write(side / 'OWNER_START.json', dict(UTC=stamp(), owner=identity(os.getpid()), child=ident,
                stage=stage, command=command, finite_seconds=cap, config_sha256=item(config)['sha256'],
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), no_retry=True,
                caps=LIMITS, source_receipts=cfg['source_receipts'], TEST_access=False))
            while child.poll() is None:
                remaining = cap - (time.monotonic() - began)
                if remaining <= 0:
                    timed_out = True; raise TimeoutError('Finite ' + stage + ' wall cap exceeded')
                try: child.wait(timeout=min(2, remaining))
                except subprocess.TimeoutExpired: pass
                if child.poll() is not None: break
                status = Path('/proc', str(child.pid), 'status')
                try:
                    rss = max([int(line.split()[1]) * 1024 for line in status.read_text().splitlines()
                               if line.startswith(('VmRSS:', 'VmHWM:'))] or [0])
                except FileNotFoundError: rss = 0
                cuda = cuda_processes().get(child.pid, 0)
                peaks['process_RSS_bytes'] = max(peaks['process_RSS_bytes'], rss)
                peaks['owned_CUDA_bytes'] = max(peaks['owned_CUDA_bytes'], cuda)
                if rss > LIMITS['rss_bytes'] or cuda > (LIMITS['cuda_bytes'] if stage == 'fit' else 0):
                    raise RuntimeError('Finite ' + stage + ' process RSS/CUDA cap exceeded')
        except BaseException as error:
            failure = dict(type=type(error).__name__, message=str(error))
        finally:
            if child.poll() is None:
                try: os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError: pass
                try: child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    try: os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError: pass
                    child.wait(timeout=15)
            code = child.wait()
    absent = identity(child.pid) is None; cuda_absent = child.pid not in cuda_processes()
    terminal = output / ('COMPLETE_FAMILY.json' if stage == 'fit' else 'COMPLETE_ANALYSIS_SUMMARY.json')
    try: full = read(terminal) if terminal.exists() else {}
    except (ValueError, OSError) as error:
        full = {}; failure = failure or dict(type=type(error).__name__, message=str(error))
    expected = dict(complete=True, groups=9, fit_units=18, optimizer_acquisition_bundles=18,
                    native_body_fit_records=18, full_same_runtime_stage_complete=True,
                    completed_seed_block=SEEDS, fixed_arm_roster=ARMS) if stage == 'fit' else dict(
                    complete=True, banks=9, optimizer_acquisition_bundles=18, native_body_fit_records=18,
                    all45_calibration_attempts_terminal_before_interpretation=True)
    complete = all(full.get(key) == value for key, value in expected.items())
    if stage == 'readout': complete = complete and full.get('calibration_costs', {}).get('attempted_scalar_fits') == 45
    success = code == 0 and complete and absent and cuda_absent and failure is None and not timed_out
    result = dict(UTC=stamp(), status='COMPLETE_EXIT_0' if success else 'FAILED_RETAINED', stage=stage,
        exit_code=code, timed_out=timed_out, failure=failure, direct_child_wait=True, child=ident,
        child_pid_absent=absent, owned_cuda_pid_absent=cuda_absent, seconds=time.monotonic()-began,
        study_kind='native_PyG_FiLM_WikiCS', config_sha256=item(config)['sha256'],
        training_source_sha256=cfg['source_receipts']['training_family']['sha256'],
        runtime_fingerprint_sha256=admission['runtime_fingerprint_sha256'], hostname=socket.gethostname(),
        gpu_uuid=UUID, scientific_success=success, TEST_access=False, no_retry=True,
        observed_external_peaks=peaks, terminal_receipt=item(terminal) if terminal.exists() else None,
        cost_scope='Normal sole allocation GPU; remaining cohorts may overlap; no isolated speed claim.')
    if stage == 'fit': result.update(complete_family_sha256=item(terminal)['sha256'] if terminal.exists() else None,
        banks=full.get('groups'), optimizer_bundles=full.get('optimizer_acquisition_bundles'),
        native_body_records=full.get('native_body_fit_records'))
    else: result.update(calibration_attempts=full.get('calibration_costs', {}).get('attempted_scalar_fits'),
        complete_analysis=success, native_fits_added=0, native_forward_calls=0)
    write(side / 'OWNER_END.json', result)
    return 0 if success else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--stage', choices=['fit', 'readout'], required=True)
    raise SystemExit(own(parser.parse_args().stage))
