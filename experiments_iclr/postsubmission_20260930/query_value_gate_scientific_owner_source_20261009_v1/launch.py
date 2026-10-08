"""Once-only launcher for a detached normal-host query-value-gated owner."""
import argparse
import os
from pathlib import Path
import subprocess
import time
from owned import identity, release_config, require, write

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--release-sha256', required=True)
    parser.add_argument('--authorized', action='store_true')
    args = parser.parse_args()
    began = time.monotonic()
    cfg, pins, phase = release_config(args.release, args.release_sha256, args.authorized)
    activation = phase / cfg['activation_relative']
    require(activation.is_dir(), 'Separate root activation required')
    receipt = activation / 'LAUNCH.json'
    write(receipt, dict(reserved=True, parent=None, automatic_retry=False, scores_read=False), True)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    environment.pop('PYTHONHOME', None)
    argv = ['/usr/bin/python3', '-I', '-S', '-B', str(HERE / 'owned.py'), '--release',
            str(args.release.resolve()), '--release-sha256', args.release_sha256,
            '--authorized', '--launch-started', repr(began)]
    with (activation / 'OWNER.log').open('xb') as log:
        require(time.monotonic() - began < 30, 'Finite launch reservation')
        parent = subprocess.Popen(argv, cwd=pins['runtime']['repository'], env=environment,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    saved = identity(parent.pid)
    require(saved and saved['group'] == saved['session'] == parent.pid, 'Exact detached parent process group')
    write(receipt, dict(reserved=True, parent=saved, argv=argv, release_sha256=args.release_sha256,
          parent_poll=parent.poll(), detached=True, automatic_retry=False, scores_read=False))
    print(str(receipt))


if __name__ == '__main__':
    main()
