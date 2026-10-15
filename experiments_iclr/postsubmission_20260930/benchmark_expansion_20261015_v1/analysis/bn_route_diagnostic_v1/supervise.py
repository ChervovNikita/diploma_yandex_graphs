"""Three fixed selected-VALID diagnostics, without new training or selection."""
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
    assert subprocess.check_output(
        ['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True
    ).splitlines() == ['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root, output = Path(args.root).resolve(), Path(args.output).resolve()
    source = Path(__file__).resolve().parent
    assert source.is_relative_to(root) and output.is_relative_to(root)
    inputs = json.loads((source / 'INPUTS.json').read_text())
    for item in [inputs['analysis_source'], *inputs['fit_results']]:
        path = Path(item['path']).resolve()
        assert path.is_relative_to(root)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    rows = []
    for index, item in enumerate(inputs['fit_results']):
        command = [sys.executable, '-B', '-u', inputs['analysis_source']['path'],
                   '--project-root', str(root), '--fit', str(Path(item['path']).parent),
                   '--output', str(output / ('fit_' + str(index)))]
        before = time.monotonic()
        with (output / ('fit_' + str(index) + '.log')).open('x') as log:
            child = subprocess.Popen(command, cwd=root, stdout=log,
                                     stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                     start_new_session=True)
            (output / ('START_' + str(index) + '.json')).write_text(json.dumps({
                'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'child_pid': child.pid, 'command': command, 'timeout_seconds': 180,
                'automatic_retries': 0}, indent=2) + '\n')
            timed_out = False
            try:
                code = child.wait(timeout=180)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    code = child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    code = child.wait()
        rows.append({'index': index, 'exit_code': code, 'timed_out': timed_out,
                     'wall_seconds': time.monotonic() - before})
        if code or timed_out:
            break
    result = {'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'status': 'complete' if len(rows) == 3 and not any(
                  r['exit_code'] or r['timed_out'] for r in rows) else 'failed',
              'children': rows, 'wall_seconds': time.monotonic() - started,
              'automatic_retries': 0, 'training_updates': 0, 'TEST_accessed': False}
    (output / 'END.json').write_text(json.dumps(result, indent=2) + '\n')
    return 0 if result['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
