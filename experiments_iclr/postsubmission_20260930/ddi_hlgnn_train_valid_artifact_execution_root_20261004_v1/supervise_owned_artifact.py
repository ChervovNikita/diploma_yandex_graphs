"""Supervise one owned CPU artifact worker; never imports numerical libraries."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'ddi_hlgnn_train_valid_artifact_preparation_20261004_v1'
PIN = 'f0d08f144376f9260f07353e355890494c6a672a6bd811ee64eb03196d120a7b'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1048576), b''):
            h.update(part)
    return h.hexdigest()


def save(name, value):
    with (ROOT / name).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def identity(pid):
    try:
        raw = (Path('/proc') / str(pid) / 'stat').read_text()
    except FileNotFoundError:
        return None
    fields = raw[raw.rfind(')') + 2:].split()
    return dict(PID=pid, start_ticks=int(fields[19]), group=int(fields[2]), session=int(fields[3]), state=fields[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--release-sha256', required=True)
    args = parser.parse_args()
    release = ROOT / 'ROOT_RELEASE.json'
    assert sha(release) == args.release_sha256 and sha(SOURCE / 'MANIFEST.json') == PIN
    assert os.uname().nodename == 'peptide' and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert not (ROOT / 'run01').exists()
    command = [sys.executable, '-B', str(SOURCE / 'qualify_artifact.py'), '--root-release', str(release),
               '--root-release-sha256', args.release_sha256]
    started = time.monotonic()
    peak_rss, stop, error = 0, None, None
    with (ROOT / 'WORKER_STDOUT.txt').open('xb') as out, (ROOT / 'WORKER_STDERR.txt').open('xb') as err:
        child = subprocess.Popen(command, cwd=Path.cwd(), stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                 start_new_session=True)
        birth = identity(child.pid)
        assert birth is not None and birth['group'] == birth['session'] == child.pid
        save('OWNED_CHILD.json', dict(birth, command=command))

        def kill_owned():
            current = identity(child.pid)
            assert current is not None and current['start_ticks'] == birth['start_ticks']
            assert current['group'] == current['session'] == child.pid
            os.killpg(child.pid, signal.SIGKILL)

        try:
            while child.poll() is None:
                current = identity(child.pid)
                assert current is not None and current['start_ticks'] == birth['start_ticks']
                assert current['group'] == current['session'] == child.pid
                try:
                    lines = (Path('/proc') / str(child.pid) / 'status').read_text().splitlines()
                except FileNotFoundError:
                    if child.poll() is not None:
                        break
                    raise
                rss = next((int(line.split()[1]) * 1024 for line in lines if line.startswith('VmRSS:')), 0)
                peak_rss = max(peak_rss, rss)
                if time.monotonic() - started >= 600 or rss > 4294967296:
                    stop = 'OWNED_WALL_CAP' if time.monotonic() - started >= 600 else 'OWNED_RSS_CAP'
                    kill_owned()
                    break
                time.sleep(0.25)
            code = child.wait(timeout=30)
            if time.monotonic() - started > 600 and stop is None:
                stop = 'OWNED_WALL_CAP'
        except BaseException as exc:
            error = dict(type=type(exc).__name__, message=str(exc))
            if child.poll() is None:
                kill_owned()
            code = child.wait(timeout=30)
    report = ROOT / 'run01/QUALIFICATION.json'
    complete = False
    if code == 0 and stop is None and error is None and report.is_file():
        value = json.loads(report.read_text())
        complete = (value['status'] == 'COMPLETE_OFFICIAL_HLGNN_DDI_TRAIN_VALID_ARTIFACT'
                    and value['source_manifest_sha256'] == PIN
                    and value['root_release_sha256'] == args.release_sha256
                    and value['TEST_payload_decoded'] is False and value['training_or_scoring'] is False
                    and value['native_graph_and_fixed_VALID_contract_qualified'] is True
                    and value['training_runtime_qualified'] is False and value['full_training_budget_qualified'] is False)
    save('TERMINAL.json', dict(UTC=datetime.now(timezone.utc).isoformat(),
         status='COMPLETE_TRAIN_VALID_ARTIFACT' if complete else 'FAILED_OR_INCOMPLETE', physical_exit_code=code,
         direct_child_reaped=True, child_proc_absent=identity(child.pid) is None, stop=stop, error=error,
         wall_seconds=time.monotonic() - started, polled_worker_peak_RSS_bytes=peak_rss,
         source_manifest_sha256=PIN, release_sha256=args.release_sha256,
         qualification_sha256=sha(report) if report.is_file() else None,
         TEST_payload_decoded=False, training_or_scoring=False, automatic_retry=False))
    return 0 if complete else 1


if __name__ == '__main__':
    sys.exit(main())
