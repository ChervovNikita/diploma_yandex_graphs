"""Once-only detached queue launcher; requires a separate reviewed root release."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from queue_owner import binding, bound, inside, module, read, require, sha

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True); parser.add_argument('--release-sha256', required=True)
    args = parser.parse_args(); os.umask(0o077)
    require(sha(args.release) == args.release_sha256, 'Exact separate root queue release')
    cfg = read(args.release)
    require(cfg.get('enabled') is True and cfg.get('root_queue_launch_authorized') is True,
            'Source preparation remains disabled')
    pins = read(HERE / 'SOURCE_BINDINGS.json')
    controller = module(bound(pins['controller_program']), '_queue_launch_existing_controller')
    controller.seal(dict(path=str((HERE / 'MANIFEST.json').relative_to(HERE.parent)), sha256=cfg['queue_manifest_sha256']))
    control_pins = read(bound(pins['controller_pins'])); controller.physical(control_pins)
    activation = inside(pins['qualification_activation']); require(activation.is_dir(), 'Separate root queue activation required')
    receipt = activation / 'QUEUE_LAUNCH.json'
    with receipt.open('x') as stream:
        json.dump(dict(launch_reserved=True, actual_queue_observed=False, automatic_retry=False), stream)
    helper = controller.module(control_pins['ownership_helper'], '_queue_launch_existing_ownership')
    argv = [control_pins['runtime']['python'], '-B', str(HERE / 'queue_owner.py'),
            '--release', str(args.release.resolve()), '--release-sha256', args.release_sha256]
    environment = dict(os.environ, PYTHONPATH='', PYTHONDONTWRITEBYTECODE='1'); environment.pop('PYTHONHOME', None)
    with (activation / 'queue_owner.log').open('xb') as log:
        child = subprocess.Popen(argv, cwd=control_pins['runtime']['repository'], env=environment,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    identity = helper.identity(child.pid)
    require(identity is not None and identity['PID'] == identity['pgid'] == identity['sid'], 'Actual detached queue identity')
    receipt.write_text(json.dumps(dict(launch_reserved=True, queue_identity=identity, argv=argv,
        queue_release=binding(args.release), parent_poll=child.poll(), detached=True,
        automatic_retry=False, scores_read=False), indent=2) + '\n')
    require(child.poll() is None, 'Queue exited at launch; preserve reservation, no relaunch')
    print(json.dumps(dict(queue_owner=identity, scores_read=False)))


if __name__ == '__main__':
    main()
