"""Watch one finite, already admitted engineering child on the own allocation."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


def main():
    assert socket.gethostname() == 'anogena-2-0'
    assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines() == ['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root, output = Path(args.root).resolve(), Path(args.output).resolve()
    source = Path(__file__).resolve().parent
    assert output.is_relative_to(root) and source.is_relative_to(root)
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((source / 'MANIFEST.json').read_text())
    script = source / manifest['source']['path']
    assert hashlib.sha256(script.read_bytes()).hexdigest() == manifest['source']['sha256']
    command = [sys.executable, '-B', '-u', str(script), '--root', str(root), '--output', str(output / 'child')]
    started = time.monotonic()
    with (output / 'child.log').open('x') as log:
        child = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
        (output / 'START.json').write_text(json.dumps({
            'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'child_pid': child.pid, 'command': command, 'maximum_seconds': 1200,
            'supervisor_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'qualifier_sha256': manifest['source']['sha256'], 'automatic_retries': 0}, indent=2) + '\n')
        timed_out = False
        try:
            code = child.wait(timeout=1200)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(child.pid, signal.SIGTERM)
            try:
                code = child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                code = child.wait()
    result = {'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'exit_code': code, 'timed_out': timed_out,
              'wall_seconds': time.monotonic() - started, 'automatic_retries': 0}
    (output / 'END.json').write_text(json.dumps(result, indent=2) + '\n')
    return code if not timed_out else 124


if __name__ == '__main__':
    raise SystemExit(main())
