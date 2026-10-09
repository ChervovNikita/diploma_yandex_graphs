"""One finite original-reader invocation on the original closed molecular roster."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

GPU = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
REPO = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def process_identity(pid):
    try:
        fields = Path('/proc', str(pid), 'stat').read_text().rsplit(') ', 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
        return None
    return dict(pid=pid, start_ticks=int(fields[19]), group=int(fields[2]),
                session=int(fields[3]), state=fields[0])

def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')

def main():
    assert socket.gethostname() == 'anogena-2-0'
    assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid',
           '--format=csv,noheader'], text=True).splitlines() == [GPU]
    repo = Path(REPO)
    phase = repo / 'experiments_iclr/postsubmission_20260930'
    here = Path(__file__).resolve().parent
    assert here.parent == phase and Path.cwd().resolve() == repo
    assert not (here / 'ATTEMPT.json').exists()
    release = here / 'RELEASE.json'
    cfg = json.loads(release.read_text())
    assert cfg['enabled'] and cfg['authorized_work']['gpu_uuid'] == GPU
    output = phase / cfg['output']
    assert output.resolve().is_relative_to(phase) and not output.exists()
    closure_path = phase / cfg['engineering_closure']['path']
    terminal_path = phase / cfg['terminal_evidence']['path']
    assert digest(closure_path) == cfg['engineering_closure']['sha256']
    assert digest(terminal_path) == cfg['terminal_evidence']['sha256']
    closure = json.loads(closure_path.read_text())
    terminal = json.loads(terminal_path.read_text())
    assert closure['closed'] and len(closure['cells']) == 18 and closure['complete_cells'] == 17
    assert terminal['owner_and_children_terminal']
    handles = list(terminal['handles'].values())
    handles.append(json.loads((closure_path.parent / 'PARENT_OWNER.json').read_text()))
    for handle in handles:
        actual = process_identity(handle['pid'])
        assert actual is None or actual['start_ticks'] != handle['start_ticks']
    source = phase / 'internal_BE_molhiv_selected_rank_readout_source_20261007_v3'
    assert digest(source / 'MANIFEST.json') == cfg['readout_source_manifest_sha256']
    for row in json.loads((source / 'MANIFEST.json').read_text())['files']:
        path = source / row['path']
        assert path.stat().st_size == row['bytes'] and digest(path) == row['sha256']
    python = phase / 'native_ncn_runtime_20261005_v1/.venv/bin/python'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, INTERNAL_BE_MOLHIV_READOUT_RELEASED='1',
               PYTHONPATH=str(phase / 'native_ncn_dependency_overlay_20261005_v1') + ':' +
                          str(repo / '.venv/lib/python3.11/site-packages'),
               PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2')
    env.pop('PYTHONHOME', None)
    command = [str(python), '-B', str(source / 'collect.py'),
               '--release', str(release), '--release-sha256', digest(release)]
    start = time.monotonic()
    write(here / 'ATTEMPT.json', dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
          owner=process_identity(os.getpid()), release_sha256=digest(release),
          original_closed_roster_sha256=digest(closure_path), command=command,
          expected_complete18_false=True, finite_seconds=900, new_fits=0, TEST_access=False))
    # One helper-semantics check. Its synthetic arrays are never scientific evidence.
    with (here / 'FIXTURE.log').open('x') as log:
        fixture = subprocess.run([str(python), '-B', str(source / 'qualification_fixture.py')],
                    cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=30)
    write(here / 'FIXTURE_RECEIPT.json', dict(exit_code=fixture.returncode,
          elapsed_seconds=time.monotonic()-start, scientific_evidence=False, serving_calls=0))
    if fixture.returncode:
        raise RuntimeError('Rank helper semantics check failed before scientific collection')
    with (here / 'READOUT.log').open('x') as log:
        child = subprocess.Popen(command, cwd=repo, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        handle = process_identity(child.pid)
        assert handle and handle['group'] == handle['session'] == child.pid
        write(here / 'CHILD_STARTED.json', dict(identity=handle, command=command))
        timed_out = False
        try:
            code = child.wait(timeout=max(1, 900-(time.monotonic()-start)))
        except subprocess.TimeoutExpired:
            timed_out = True
            assert process_identity(child.pid) == handle or (
                process_identity(child.pid) and
                process_identity(child.pid)['start_ticks'] == handle['start_ticks'])
            os.killpg(child.pid, signal.SIGTERM)
            try:
                code = child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                actual = process_identity(child.pid)
                assert actual and actual['start_ticks'] == handle['start_ticks']
                os.killpg(child.pid, signal.SIGKILL)
                code = child.wait(timeout=10)
    report = dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), child=handle,
                  collector_exit_code=code, reaped=True, timed_out=timed_out,
                  inclusive_seconds=time.monotonic()-start, whole_family_success=False,
                  expected_failed_fit_retained='P_7307', new_fits=0, TEST_access=False)
    for name in ['ANALYSIS.json', 'COST.json', 'COLLECTION.json']:
        path = output / 'compact' / name
        report[name] = dict(present=path.is_file(),
                            sha256=digest(path) if path.is_file() else None)
    write(here / 'EXECUTION_RECEIPT.json', report)
    print(json.dumps(report), flush=True)

if __name__ == '__main__':
    main()
