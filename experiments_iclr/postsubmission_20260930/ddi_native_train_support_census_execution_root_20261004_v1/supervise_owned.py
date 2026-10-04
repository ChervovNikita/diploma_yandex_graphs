"""Supervise a single census child with exclusive process-group ownership."""
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
SOURCE = ROOT.parent / 'ddi_native_train_support_census_preparation_20261004_v1'
PIN = '989f10f25f6b1fd2affa58609c24bd957f2ac1755c6040aa0fdf139aa5546758'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    with (ROOT / name).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def identity(pid):
    try:
        raw = (Path('/proc') / str(pid) / 'stat').read_text()
    except FileNotFoundError:
        return None
    fields = raw[raw.rfind(')')+2:].split()
    return dict(PID=pid, start_ticks=int(fields[19]), group=int(fields[2]), session=int(fields[3]), state=fields[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--release-sha256', required=True)
    args = parser.parse_args()
    release = ROOT / 'ROOT_RELEASE.json'
    assert sha(release) == args.release_sha256 and sha(SOURCE / 'MANIFEST.json') == PIN
    assert os.uname().nodename == 'peptide' and not (ROOT / 'run01').exists()
    command = [sys.executable, '-B', str(SOURCE / 'census.py'), '--root-release', str(release), '--root-release-sha256', args.release_sha256]
    started = time.monotonic()
    with (ROOT / 'WORKER_STDOUT.txt').open('xb') as out, (ROOT / 'WORKER_STDERR.txt').open('xb') as err:
        child = subprocess.Popen(command, stdout=out, stderr=err, cwd=Path.cwd(), start_new_session=True)
        birth = identity(child.pid)
        assert birth is not None and birth['group'] == birth['session'] == child.pid
        save('OWNED_CHILD.json', dict(birth, command=command))
        stop = None
        try:
            code = child.wait(timeout=1860)
        except subprocess.TimeoutExpired:
            current = identity(child.pid)
            assert current is not None and current['start_ticks'] == birth['start_ticks']
            assert current['group'] == current['session'] == child.pid
            os.killpg(child.pid, signal.SIGKILL)
            stop = 'EXCLUSIVE_OWNED_CHILD_WALL_TIMEOUT'
            code = child.wait(timeout=30)
    census = ROOT / 'run01/CENSUS.json'
    complete = False
    if code == 0 and census.exists():
        result = json.loads(census.read_text())
        complete = (result['status'] == 'COMPLETE_DDI_NATIVE_TRAIN_FULL_BATCH_CENSUS'
                    and result['source_manifest_sha256'] == PIN
                    and result['root_release_sha256'] == args.release_sha256
                    and result['full_batches'] == 43
                    and result['optimizer_updates'] == 0
                    and result['VALID_TEST_reads'] is False)
    save('TERMINAL.json', dict(UTC=datetime.now(timezone.utc).isoformat(),
         status='COMPLETE_TRAIN_CENSUS' if complete else 'FAILED_OR_INCOMPLETE',
         physical_exit_code=code, direct_child_reaped=True, child_proc_absent=identity(child.pid) is None,
         stop=stop, wall_seconds=time.monotonic()-started,
         source_manifest_sha256=PIN, release_sha256=args.release_sha256,
         result_sha256=sha(census) if census.exists() else None,
         automatic_retry=False, VALID_TEST_reads=False, model_or_score_work=False))
    return 0 if complete else 1


if __name__ == '__main__':
    sys.exit(main())
