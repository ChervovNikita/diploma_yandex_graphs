"""Once-only detached stdlib parent, with all twelve science keys reserved first."""
import argparse
import os
from pathlib import Path
import subprocess
import time
from common import HERE, PHASE, config, read, require, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--release-sha256', required=True)
    parser.add_argument('--authorized', action='store_true')
    args = parser.parse_args()
    began = time.monotonic()
    cfg, pins, custody, routing, route, route_pins, cells = config(args.release, args.release_sha256, args.authorized)
    activation = PHASE / cfg['activation_relative']
    require(not (PHASE / cfg['output_relative']).exists(), 'Fresh lane output')
    receipt = activation / 'LAUNCH.json'
    write(receipt, dict(parent=None, reserved=True, release_sha256=args.release_sha256, automatic_retry=False), True)
    if cfg['mode'] == 'seed_block':
        plan = read(PHASE / cfg['plan']['path'])
        claims = PHASE / plan['fresh_output_relative'] / '.claims'
        claims.mkdir(parents=True, exist_ok=True)
        write(claims / (cfg['route_id'] + '.json'), dict(route_id=cfg['route_id'], cells=cells,
            plan_sha256=cfg['plan']['sha256'], release_sha256=args.release_sha256, automatic_retry=False), True)
    # -I -S removes the script directory, so bootstrap it explicitly in the private parent.
    bootstrap = 'import runpy,sys;sys.path.insert(0,' + repr(str(HERE)) + ');runpy.run_path(' + repr(str(HERE / 'lane.py')) + ',run_name="__main__")'
    argv = ['/usr/bin/python3', '-I', '-S', '-B', '-c', bootstrap, '--release', str(args.release.resolve()),
            '--release-sha256', args.release_sha256, '--authorized', '--launch-started', repr(began)]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONHOME', None)
    with (activation / 'OWNER.log').open('xb') as log:
        require(time.monotonic() < began + cfg['resources']['startup_seconds'], 'Finite launch reservation')
        parent = subprocess.Popen(argv, cwd=route['repository'], env=env, stdin=subprocess.DEVNULL,
                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    saved = custody.identity(parent.pid)
    require(saved and saved['group'] == saved['session'] == parent.pid, 'Actual detached parent identity')
    write(receipt, dict(parent=saved, reserved=True, argv=argv, release_sha256=args.release_sha256,
          detached=True, automatic_retry=False, scores_read=False))
    print(str(receipt))


if __name__ == '__main__': main()
