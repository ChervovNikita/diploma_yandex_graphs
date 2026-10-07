"""Run the fixed selected-prediction analysis once after Wiki24 and CPU readout close."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import socket
import subprocess
import time

HERE = Path(__file__).resolve().parent
R = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
P = R / 'experiments_iclr/postsubmission_20260930'
CPU = P / 'Wiki24_analysis_after_closure_execution_root_20261007_v1'
QUALIFIER = P / 'allocation_controls_molhiv_OIPG_GPU_representative_activation_root_20261007_v2/receipts/run01'
COLLECTOR = P / 'internal_BE_Wiki24_selected_prediction_analysis_source_20261007_v1'
E = P / 'Wiki24_selected_analysis_after_closure_execution_root_20261007_v1'
PY = P / 'native_ncn_runtime_20261005_v1/.venv/bin/python'
GPU = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
WAIT, ACTIVE, CLEANUP = 86400, 1800, 10


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def binding(path):
    path = Path(path).resolve()
    return dict(path=str(path.relative_to(P.resolve())), bytes=path.stat().st_size, sha256=sha(path))


def write(name, value):
    path = E / name
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temporary, path)


def identity(pid):
    try:
        s = Path('/proc/' + str(pid) + '/stat').read_text()
    except FileNotFoundError:
        return None
    f = s[s.rfind(')') + 2:].split()
    return dict(PID=pid, start_ticks=int(f[19]), state=f[0])


def live(pid, ticks=None):
    item = identity(pid)
    return item is not None and item['state'] not in ('Z', 'X') and (ticks is None or item['start_ticks'] == ticks)


def interrupted(signum, frame):
    raise InterruptedError('Owned analysis interrupted by signal ' + str(signum))


def gpu_free():
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.free',
                                   '--format=csv,noheader,nounits'], text=True, timeout=10).splitlines()
    assert len(rows) == 1 and rows[0].split(',')[0].strip() == GPU
    return int(rows[0].split(',')[1]) * 1048576


def prerequisites():
    if (CPU / 'FAILURE.json').is_file():
        raise RuntimeError('Original CPU analysis failed; no outcome opening or retry')
    if not (CPU / 'COMPLETE.json').is_file():
        assert live(518522, 6017908116), 'CPU reader owner ended without its COMPLETE/FAILURE'
        return None
    assert read(CPU / 'COMPLETE.json')['complete'] is True
    if live(518522, 6017908116):
        return None
    saved = read(CPU / 'TERMINAL_EVIDENCE.json')
    assert saved['owner_and_recorded_children_conservatively_terminal'] is True
    observations = []
    for row in saved['observed_owned_handles']:
        observations.append(dict(row, current_identity=identity(row['PID']),
                                 owned_identity_live=live(row['PID'], row['start_ticks'])))
    if any(x['owned_identity_live'] for x in observations):
        return None
    for stage in ('EXTRACT', 'READ'):
        terminal = read(CPU / (stage + '_TERMINAL.json'))
        assert terminal['reaped'] is True and terminal['exit_code'] == 0 and not terminal['timed_out']
        handle = terminal['identity']
        if live(handle['PID'], handle['start_ticks']):
            return None
    if not (QUALIFIER / 'TERMINAL.json').is_file():
        assert live(521566, 6018676170), 'Existing molecular qualifier ended without terminal receipt'
        return None
    terminal = read(QUALIFIER / 'TERMINAL.json')
    if terminal['GPU_child_started']:
        assert terminal['reaped'] is True
    if live(521566, 6018676170):
        return None
    owner = read(QUALIFIER / 'OWNER.json')
    if owner.get('worker_PID') is not None and live(owner['worker_PID'], owner['worker_start_ticks']):
        return None
    return saved, observations


def run_child(command, env):
    began = time.monotonic()
    child, handle = None, None
    code, error, cleanup_error, timed_out = None, None, None, False

    def stop(sig):
        if child.poll() is not None:
            return
        assert live(child.pid, handle['start_ticks']) and os.getpgid(child.pid) == child.pid
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            if child.poll() is None:
                raise

    try:
        with (E / 'COLLECT.log').open('x') as log:
            child = subprocess.Popen(command, cwd=R, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            handle = identity(child.pid)
            write('CHILD_OWNER.json', dict(identity=handle, command=command,
                                          active_seconds=ACTIVE, cleanup_seconds=CLEANUP))
            try:
                code = child.wait(timeout=max(0., began + ACTIVE - time.monotonic()))
            except subprocess.TimeoutExpired:
                timed_out = True
    except Exception as exc:
        error = dict(type=type(exc).__name__, message=str(exc))
    finally:
        if child is not None and child.poll() is None:
            try:
                stop(signal.SIGTERM)
                try:
                    child.wait(timeout=max(0., min(time.monotonic() + 5., began + ACTIVE + CLEANUP) - time.monotonic()))
                except subprocess.TimeoutExpired:
                    stop(signal.SIGKILL)
                    child.wait(timeout=max(0., began + ACTIVE + CLEANUP - time.monotonic()))
            except Exception as exc:
                cleanup_error = dict(type=type(exc).__name__, message=str(exc))
        code = child.poll() if child is not None else None
        terminal = dict(exit_code=code, child_started=child is not None,
                        reaped=child is not None and code is not None, identity=handle,
                        timed_out=timed_out, elapsed_seconds=time.monotonic() - began,
                        hard_seconds=ACTIVE + CLEANUP, error=error, cleanup_error=cleanup_error,
                        automatic_retry=False, other_processes_signalled=False)
        write('CHILD_TERMINAL.json', terminal)
    assert terminal['reaped'] and code == 0 and not timed_out and error is None and cleanup_error is None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--approval', type=Path, required=True)
    parser.add_argument('--approval-sha256', required=True)
    args = parser.parse_args()
    assert sha(args.approval) == args.approval_sha256
    approval = read(args.approval)
    assert approval['approved'] is True and approval['root_execution_authorized'] is True
    assert approval['source_manifest_sha256'] == sha(HERE / 'MANIFEST.json')
    for row in read(HERE / 'MANIFEST.json')['files']:
        assert sha(HERE / row['path']) == row['sha256']
    assert sha(COLLECTOR / 'MANIFEST.json') == approval['collector_manifest_sha256']
    assert Path.cwd().resolve() == R.resolve() and socket.gethostname() == 'anogena-2-0'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    gpu_free()
    os.umask(0o077)
    E.mkdir(exist_ok=False)
    started = time.monotonic()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    write('OWNER.json', dict(identity=identity(os.getpid()), source=binding(HERE / 'MANIFEST.json'),
                            waiting_GPU_visible=False, wait_seconds=WAIT, automatic_retry=False))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        while True:
            assert time.monotonic() - started < WAIT, 'Finite analysis wait expired; no science restarted'
            ready = prerequisites()
            free = gpu_free()
            if ready is not None and free >= 8 * 1073741824:
                break
            write('WAIT.json', dict(elapsed_seconds=time.monotonic() - started,
                                   original_CPU_analysis_complete=(CPU / 'COMPLETE.json').is_file(),
                                   qualifier_terminal=(QUALIFIER / 'TERMINAL.json').is_file(),
                                   minimum_free_GPU_bytes=8 * 1073741824, actual_free_GPU_bytes=free,
                                   numerical_child_started=False, predictive_values_opened=False))
            time.sleep(60)
        saved, observations = ready
        export_path = CPU / 'extraction/METADATA_EXPORT.json'
        export = read(export_path)
        assert saved['owner'] == export['gate']['owner'] and saved['closure'] == export['gate']['closure']
        write('TERMINAL_EVIDENCE.json', dict(owner_and_children_terminal=True,
              owner=saved['owner'], closure=saved['closure'], current_observed_handles=observations,
              observation_scope='Exact saved identities where available; conservative PID-only for historical fits missing start ticks.',
              previous_terminal_evidence=binding(CPU / 'TERMINAL_EVIDENCE.json'), UTC=datetime.now(timezone.utc).isoformat()))
        release = dict(schema='internal-be-Wiki24-selected-prediction-release-v1',
              stage_enabled=True, root_execution_authorized=True, source_review_approved=True,
              collector_manifest_sha256=approval['collector_manifest_sha256'], owner_and_children_terminal=True,
              terminal_evidence=binding(E / 'TERMINAL_EVIDENCE.json'), metadata_export=binding(export_path),
              metadata_extraction_cost=binding(CPU / 'extraction/EXTRACTION_COST.json'),
              trusted_checkpoint_deserialization_authorized=True, prediction_collection_and_analysis_cost_charged=True,
              maximum_member_forwards=78, output_directory=str(E / 'predictions'), TEST_access=False, automatic_retry=False)
        write('COLLECT_RELEASE.json', release)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                   OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2',
                   PYTHONPATH=str(P / 'native_ncn_dependency_overlay_20261005_v1') + os.pathsep + str(R / '.venv/lib/python3.11/site-packages'))
        env.pop('PYTHONHOME', None)
        run_child([str(PY), '-B', str(COLLECTOR / 'collect.py'), '--release', str(E / 'COLLECT_RELEASE.json'),
                   '--release-sha256', sha(E / 'COLLECT_RELEASE.json')], env)
        summary = read(E / 'predictions/compact/COLLECTION.json')
        assert summary['whole24_accounted'] is True and summary['status'] in ('complete', 'complete_with_retained_failures')
        write('COMPLETE.json', dict(complete=True, collection_status=summary['status'],
              error_summary=binding(E / 'predictions/compact/ERROR_SUMMARY.json'),
              raw_arrays_server_only=True, new_fits=0, TEST_access=False))
    except Exception as exc:
        write('FAILURE.json', dict(complete=False, error_type=type(exc).__name__, error=str(exc), automatic_retry=False))
        raise
    finally:
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        write('OWNER_COST.json', dict(inclusive_wait_and_stage_seconds=time.monotonic() - started,
              child_CPU_user_seconds=after.ru_utime - before.ru_utime,
              child_CPU_system_seconds=after.ru_stime - before.ru_stime,
              cumulative_child_peak_RSS_bytes=after.ru_maxrss * 1024,
              new_scientific_fits=0, original_science_changed=False))


if __name__ == '__main__':
    main()
