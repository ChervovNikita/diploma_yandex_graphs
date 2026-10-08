"""Once-only detached combined15 owner launcher; separate root release required."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from own_collection import binding, inside, proc, read, require, seal, sha

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--release-sha256', required=True)
    args = parser.parse_args(); os.umask(0o077); began = time.monotonic()
    require(sha(args.release) == args.release_sha256, 'Exact separate root owner release')
    cfg = read(args.release)
    require(cfg.get('schema') == 'Wiki12-SupCon15-external-owner-root-release-v1'
        and all(cfg.get(key) is True for key in ('enabled', 'root_collection_execution_authorized', 'source_review_approved',
            'actual_union_all15_terminal_bindings_supplied', 'actual_runtime_resource_bindings_supplied',
            'cost_and_external_supervision_authorized')),
        'Source preparation remains disabled pending root release')
    require(all(cfg.get(key) is False for key in ('TEST_access', 'training', 'reselection', 'calibration', 'automatic_retry'))
        and cfg['maximum_member_forwards'] == 60, 'Fixed prediction-only scope; no retry')
    pins = read(HERE / 'SOURCE_BINDINGS.json'); runtime = pins['runtime']
    seal(dict(path=str((HERE / 'MANIFEST.json').relative_to(HERE.parent)), sha256=cfg['owner_manifest_sha256']))
    require(Path.cwd().resolve() == Path(runtime['repository']).resolve() and socket.gethostname() == runtime['hostname']
        and str(Path(sys.executable).absolute()) == runtime['python'] and os.environ.get('PYTHONPATH', '') == '',
        'Original normal77 host/runtime')
    activation = inside(pins['root_activation_directory'])
    require(activation.is_dir() and args.release.resolve() == activation / pins['owner_release_name'],
        'Exact separate root activation')
    require(time.monotonic() - began < pins['launch_admission_reserve_seconds'], 'Finite30s launcher/admission reserve')
    receipt = activation / 'COLLECTION_OWNER_LAUNCH.json'
    with receipt.open('x') as stream:
        json.dump(dict(launch_reserved=True, actual_owner_observed=False, automatic_retry=False,
            scores_read=False, launch_started_monotonic=began), stream, indent=2); stream.write('\n')
    argv = [runtime['python'], '-B', str(HERE / 'own_collection.py'), '--release', str(args.release.resolve()),
        '--release-sha256', args.release_sha256, '--launch-started-monotonic', repr(began)]
    environment = dict(os.environ, PYTHONPATH='', PYTHONDONTWRITEBYTECODE='1'); environment.pop('PYTHONHOME', None)
    with (activation / 'collection_owner.log').open('xb') as log:
        require(time.monotonic() - began < pins['launch_admission_reserve_seconds'], 'Finite30s prelaunch reserve')
        child = subprocess.Popen(argv, cwd=runtime['repository'], env=environment, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    identity = proc(child.pid); observed_live = child.poll() is None
    value = dict(launch_reserved=True, actual_owner_observed=identity is not None, owner_identity=identity,
        argv=argv, owner_release=binding(args.release), parent_poll=child.returncode, detached=True,
        launch_started_monotonic=began, launcher_seconds=time.monotonic() - began,
        automatic_retry=False, scores_read=False)
    temporary = activation / 'COLLECTION_OWNER_LAUNCH.actual.tmp'
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')
    os.replace(temporary, receipt)
    require(identity is not None and identity['pid'] == identity['group'] == identity['session']
        and observed_live, 'Detached owner exited or was not observed; preserve reservation, no relaunch')
    print(json.dumps(dict(collection_owner=identity, detached=True, scores_read=False)))


if __name__ == '__main__':
    main()
