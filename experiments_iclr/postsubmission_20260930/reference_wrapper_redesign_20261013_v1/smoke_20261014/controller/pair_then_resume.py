"""Run the owned two-job probe, then resume the paused owned smoke queue."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from smoke_case import atomic_json, require_allocation


def main():
    require_allocation()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--smoke-root', required=True)
    parser.add_argument('--python', required=True)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    smoke = Path(args.smoke_root).resolve()
    assert smoke.is_relative_to(root)
    pause = json.loads((smoke / 'results/PAIR_QUEUE_PAUSE1.json').read_text())
    pid = pause['dispatcher_pid']
    command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    assert str(smoke / 'controller/dispatch_smokes.py') in command
    assert str(smoke / 'results/nonnode_matrix_attempt1') in command
    receipt = smoke / 'results/PAIR_QUEUE_RESUME1.json'
    assert not receipt.exists()
    result = {'queue_pid': pid, 'engineering_only': True}
    try:
        child = pause['last_case'].get('pid')
        deadline = time.monotonic() + 1800
        while child and Path(f'/proc/{child}/status').exists():
            status = Path(f'/proc/{child}/status').read_text()
            if any(row.startswith('State:') and 'Z' in row for row in status.splitlines()):
                break
            assert time.monotonic() < deadline, 'Owned queue child did not finish'
            time.sleep(1)
        output = smoke / 'results/pair_physics_private_half_ncnc_i4_attempt1'
        assert not output.exists()
        command = [args.python, '-B', '-u', str(smoke / 'controller/pair_memory.py'),
                   '--manifest', str(smoke / 'MANIFEST.json'),
                   '--project-root', str(root), '--output', str(output),
                   '--python', args.python, '--cases',
                   'polyformer-coauthor-physics-mono/boundary_private_half_pool',
                   'collab-ncnc/independent4']
        completed = subprocess.run(command, cwd=root, timeout=1900, check=False)
        result.update(pair_exit_code=completed.returncode, pair_output=str(output))
    except Exception as error:
        result['supervisor_failure'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        os.kill(pid, signal.SIGCONT)
        result.update(queue_resumed=True,
                      UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        atomic_json(receipt, result)


if __name__ == '__main__':
    main()
