"""Run the fixed nine cells; preserve failures and never inspect partial scores."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import signal
import socket
import subprocess
import time

REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
HERE = Path(__file__).resolve().parent
GPU = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'


def identity(pid):
    text = (Path('/proc') / str(pid) / 'stat').read_text()
    fields = text[text.rfind(')') + 2:].split()
    return dict(pid=pid, start_ticks=int(fields[19]), state=fields[0])


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def write(name, value, replace=False):
    path = HERE / name
    if replace:
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(json.dumps(value, indent=2) + '\n')
        os.replace(temporary, path)
    else:
        with path.open('x') as handle:
            json.dump(value, handle, indent=2)
            handle.write('\n')


def stop_owned(child, saved, signals):
    assert child.poll() is None and identity(child.pid)['start_ticks'] == saved['start_ticks']
    os.killpg(child.pid, signal.SIGTERM)
    signals.append('SIGTERM_OWNED_CHILD_GROUP')
    try:
        return child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        assert child.poll() is None and identity(child.pid)['start_ticks'] == saved['start_ticks']
        os.killpg(child.pid, signal.SIGKILL)
        signals.append('SIGKILL_OWNED_CHILD_GROUP')
        return child.wait(timeout=5)


def main():
    assert Path.cwd() == REPO and HERE.is_relative_to(PHASE)
    assert socket.gethostname() == 'anogena-2-0'
    assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines() == [GPU]
    schedule = json.loads((HERE / 'SCHEDULE.json').read_text())
    assert len(schedule['cells']) == 9 and schedule['automatic_retry'] is False
    for row in schedule['sealed_files']:
        assert sha(PHASE / row['path']) == row['sha256']
    write('PARENT_OWNER.json', identity(os.getpid()))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
        PYTHONPATH=str(PHASE / 'native_ncn_dependency_overlay_20261005_v1') + ':' + str(REPO / '.venv/lib/python3.11/site-packages'))
    completed = []
    try:
        for row in schedule['cells']:
            cell_id = row['cell_id']
            release_path = HERE / 'cells' / (cell_id + '.json')
            assert sha(release_path) == row['release_sha256']
            release = json.loads(release_path.read_text())
            wait_started = time.monotonic()
            while True:
                free = int(subprocess.check_output(['nvidia-smi', '--id=' + GPU, '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip()) * 1024**2
                if free >= release['minimum_fresh_free_GPU_bytes']:
                    break
                if time.monotonic() - wait_started >= 1800:
                    raise RuntimeError('Fresh-memory admission timeout before ' + cell_id)
                time.sleep(10)
            argv = [str(PHASE / 'native_ncn_runtime_20261005_v1/.venv/bin/python'), '-B', str(HERE / 'run_cell.py'), str(release_path)]
            started = time.monotonic()
            signals = []
            with (HERE / 'logs' / (cell_id + '.stdout.log')).open('x') as out, (HERE / 'logs' / (cell_id + '.stderr.log')).open('x') as err:
                child = subprocess.Popen(argv, cwd=REPO, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True)
                saved = identity(child.pid)
                write('logs/' + cell_id + '.OWNER.json', dict(saved, argv=argv, fresh_free_GPU_bytes=free))
                write('CURRENT.json', dict(cell_id=cell_id, child=saved, complete_cells=len(completed), scores_read=False), replace=True)
                timeout = False
                try:
                    exit_code = child.wait(timeout=release['external_active_seconds'])
                except subprocess.TimeoutExpired:
                    timeout = True
                    exit_code = stop_owned(child, saved, signals)
            receipt = dict(cell_id=cell_id, child=saved, exit_code=exit_code, exit_authority='subprocess.Popen.wait',
                child_reaped=True, timeout=timeout, signals=signals, elapsed_seconds=time.monotonic()-started,
                resource_wait_seconds=started-wait_started, release_sha256=row['release_sha256'], scores_read=False)
            write('logs/' + cell_id + '.EXIT.json', receipt)
            assert exit_code == 0 and not timeout, 'Original cell failed; no retry: ' + cell_id
            output = PHASE / release['output']
            finish = json.loads((output / 'COMPLETE.json').read_text())
            assert finish['complete'] is True and finish['epochs'] == finish['steps'] == 1100
            assert finish['arm'] == release['method'] and finish['seed'] == release['seed'] and finish['TEST_scoring'] is False
            assert sha(output / 'selected.pt') == finish['selected_sha256']
            completed.append(dict(cell_id=cell_id, completion_sha256=sha(output / 'COMPLETE.json'), exit_receipt=receipt))
            write('PROGRESS.json', dict(completed=completed, fixed_total=9, scores_read=False), replace=True)
        write('FAMILY_CLOSURE.json', dict(complete=True, completed=completed, fixed_total=9,
            original_cells_only=True, automatic_retry=False, scores_read=False, stage2_admitted=False,
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    except BaseException as error:
        write('OWNER_FAILURE.json', dict(error_type=type(error).__name__, error=str(error),
            completed=completed, automatic_retry=False, scores_read=False))
        raise


if __name__ == '__main__':
    main()
