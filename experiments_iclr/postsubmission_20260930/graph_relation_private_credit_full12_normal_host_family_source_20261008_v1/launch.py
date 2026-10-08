"""Once-only detached root owner launcher; disabled templates are not admission."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from controller import bound, binding, inside, module, physical, read, require, seal, sha

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True); parser.add_argument('--release-sha256', required=True)
    args = parser.parse_args(); os.umask(0o077)
    require(sha(args.release) == args.release_sha256, 'Exact separate root launch release')
    cfg = read(args.release); pins = read(HERE / 'SOURCE_BINDINGS.json')
    require(cfg.get('enabled') is True and cfg.get('root_full12_launch_authorized') is True, 'Source preparation is disabled')
    seal(dict(path=str((HERE / 'MANIFEST.json').relative_to(HERE.parent)), sha256=cfg['controller_manifest_sha256']))
    physical(pins)
    activation = inside(cfg['activation_directory']); require(activation.is_dir(), 'Separate root activation directory required')
    receipt = activation / 'LAUNCH.json'
    with receipt.open('x') as stream:
        json.dump(dict(launch_reserved=True, actual_owner_observed=False, automatic_retry=False), stream)
    argv = [pins['runtime']['python'], '-B', str(HERE / 'controller.py'), '--release', str(args.release.resolve()), '--release-sha256', args.release_sha256]
    environment = dict(os.environ, PYTHONPATH='', PYTHONDONTWRITEBYTECODE='1'); environment.pop('PYTHONHOME', None)
    helper = module(pins['ownership_helper'], '_relation12_launch_observation')
    with (activation / 'owner.log').open('xb') as log:
        child = subprocess.Popen(argv, cwd=pins['runtime']['repository'], env=environment,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    identity = helper.identity(child.pid)
    require(identity is not None and identity['PID'] == identity['pgid'] == identity['sid'], 'Actual detached owner identity')
    receipt.write_text(json.dumps(dict(launch_reserved=True, owner_identity=identity,
        parent_poll=child.poll(), argv=argv, release=binding(args.release), detached=True,
        automatic_retry=False, scores_read=False), indent=2) + '\n')
    require(child.poll() is None, 'Owner exited at launch; preserve this attempt, no relaunch')
    print(json.dumps(dict(owner=identity, scores_read=False)))


if __name__ == '__main__':
    main()
