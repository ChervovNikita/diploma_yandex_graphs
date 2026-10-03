"""One bounded CPU inference-audit child. Never starts training or selects models."""
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time

REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
ROOT = Path(__file__).resolve().parent
PHASE = ROOT.parent
AUDITOR = PHASE/'graph_heterogeneous_dblp_native15_closed_audit_preparation_20261003_v1'
LIMIT = 16 * 1024**3
RSS_LIMIT = 12 * 1024**3
WALL_LIMIT = 1800


def descriptor(path):
    return dict(path=str(path), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def main():
    assert Path.cwd().resolve() == REPO and ROOT.is_relative_to(REPO)
    assert subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],
        capture_output=True,text=True,check=True).stdout.splitlines() == ['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
    freeze, release = ROOT/'AUDIT_FREEZE.json', ROOT/'AUDIT_RELEASE.json'
    value = json.loads(release.read_text())
    assert value['execution_authorized'] is True and value['training_child_reaped'] is True
    assert value['root_observed_closed_selected15'] is True and value['independent_source_review_observed'] is True
    assert descriptor(freeze)['sha256'] == value['audit_freeze_sha256']
    assert descriptor(AUDITOR/'MANIFEST.json')['sha256'] == value['prepared_auditor_manifest_sha256']
    output = ROOT/'NATIVE15_AUDIT_run01.json'
    assert str(output) == json.loads(freeze.read_text())['expected_output_path'] and not output.exists()
    available = next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    try:
        maximum = Path('/sys/fs/cgroup/memory.max').read_text().strip()
        if maximum != 'max':
            available = min(available, int(maximum)-int(Path('/sys/fs/cgroup/memory.current').read_text()))
    except (OSError, ValueError):
        pass
    assert available >= 24*1024**3, 'Native inference audit deferred for host memory'
    command = [str(REPO/'.venv/bin/python'), '-B', str(AUDITOR/'audit.py'),
               '--freeze', str(freeze), '--admission', str(release), '--output', str(output)]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH='', PYTHONNOUSERSITE='1',
                       PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    record = dict(status='preparing', training_started=False, source=descriptor(Path(__file__)),
                  freeze=descriptor(freeze), release=descriptor(release), argv=command,
                  address_space_limit_bytes=LIMIT, RSS_limit_bytes=RSS_LIMIT,
                  wall_limit_seconds=WALL_LIMIT, child_pid=None, child_reaped=False,
                  exit_code=None, independent_audit=None)
    write(ROOT/'AUDIT_SUPERVISOR_STARTED.json', record)
    child = None
    start = time.monotonic()
    peak = 0
    def bounded():
        resource.setrlimit(resource.RLIMIT_AS, (LIMIT,LIMIT))
    def interrupted(number, frame):
        raise InterruptedError('Audit supervisor interrupted: '+str(number))
    for number in (signal.SIGINT,signal.SIGTERM):
        signal.signal(number, interrupted)
    try:
        with (ROOT/'audit.stdout.log').open('x') as stdout, (ROOT/'audit.stderr.log').open('x') as stderr:
            child = subprocess.Popen(command, cwd=REPO, env=environment, stdout=stdout,
                                     stderr=stderr, preexec_fn=bounded)
            record['child_pid'] = child.pid
            write(ROOT/'AUDIT_OWNED_CHILD.json', dict(pid=child.pid, argv=command))
            while child.poll() is None:
                try:
                    for line in Path('/proc',str(child.pid),'status').read_text().splitlines():
                        if line.startswith(('VmRSS:','VmHWM:')):
                            peak = max(peak,int(line.split()[1])*1024)
                except OSError:
                    pass
                if time.monotonic()-start > WALL_LIMIT or peak > RSS_LIMIT:
                    raise RuntimeError('Inference audit exceeded bounded resource budget')
                time.sleep(.5)
            assert child.returncode == 0, 'Independent native15 inference audit failed'
            result = json.loads(output.read_text())
            assert result['status']=='complete' and result['all15_checkpoint_replays_audited'] is True
            assert result['heldout_labels_closed'] is True and result['originals_preserved'] is True
            record.update(status='native15_replay_complete', independent_audit=descriptor(output))
    except Exception as error:
        record.update(status='native15_replay_failed', error_type=type(error).__name__, reason=str(error))
    finally:
        if child is not None:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            record.update(exit_code=child.returncode, child_reaped=True)
        record.update(wall_seconds=time.monotonic()-start, peak_observed_RSS_bytes=peak)
        write(ROOT/'AUDIT_SUPERVISOR_RECEIPT.json', record)
    print(json.dumps(dict(status=record['status'], exit_code=record['exit_code'],
                         child_reaped=record['child_reaped'], training_started=False)))
    return 0 if record['status']=='native15_replay_complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
