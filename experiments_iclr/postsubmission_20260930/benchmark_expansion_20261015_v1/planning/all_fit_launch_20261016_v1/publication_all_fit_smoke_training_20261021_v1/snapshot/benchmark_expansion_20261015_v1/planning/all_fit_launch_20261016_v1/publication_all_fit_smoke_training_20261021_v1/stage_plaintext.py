"""Inspect, or explicitly stage, one frozen publication via ordinary allocation SCP.

No commit, push, checkout, training, encoded payload or legacy staging is used.
Partial/uncertain deliveries remain preserved for operator reconciliation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys
import uuid

from verify_staged import verify

HERE = Path(__file__).resolve().parent
PHASE = HERE.parents[3]
PROJECT = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
REMOTE = PROJECT / 'experiments_iclr/postsubmission_20260930' / HERE.relative_to(PHASE)
TRANSPORT = PHASE / 'benchmark_expansion_20261015_v1/campaign_readable_coordinator_20261021_v2'
PINS = {
    'readable_transport.py': 'efc78cc3859ebfbe23ab8a6c50f1615b96d7f563126026dcee4d7cd965a5835a',
    'remote_backend.py': '0343060a2bda59ef9084cf08bb47602104450e9448918fc23786c9e4139598cd',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def stage(expected_manifest_sha256, dispatch=False):
    local = verify(HERE, expected_manifest_sha256)
    for name, expected in PINS.items():
        require(hashlib.sha256((TRANSPORT / name).read_bytes()).hexdigest() == expected,
                'Reviewed transport source changed: ' + name)
    sys.path.insert(0, str(TRANSPORT))
    from readable_transport import ALLOCATION_SCP, LOGIN, Transport, checked_hash
    tag = uuid.uuid4().hex
    temporary = REMOTE.with_name(REMOTE.name + '.incoming-' + tag)
    audit = HERE.parent / ('publication_stage_20261021_' + tag)
    parent, target, incoming = map(lambda p: shlex.quote(str(p)), (REMOTE.parent, REMOTE, temporary))
    prepare = ('test "$(realpath -m -- ' + parent + ')" = ' + parent + '\n' +
               'test ! -e ' + target + '\ntest ! -L ' + target + '\n' +
               'test ! -e ' + incoming + '\ntest ! -L ' + incoming + '\n' +
               'mkdir -p -- ' + parent + '\n')
    scp = [*ALLOCATION_SCP, '-r', str(HERE), LOGIN + ':' + str(temporary)]
    verifier = temporary / 'verify_staged.py'
    verifier_raw = (HERE / 'verify_staged.py').read_bytes()
    verifier_sha = hashlib.sha256(verifier_raw).hexdigest()
    verify_command = checked_hash(verifier, verifier_sha) + shlex.join([
        str(PROJECT / '.venv/bin/python'), '-B', str(verifier), '--package', str(temporary),
        '--manifest-sha256', expected_manifest_sha256]) + '\n'
    finish = ('test ! -e ' + target + '\ntest ! -L ' + target + '\n' +
              'mv -T -n -- ' + incoming + ' ' + target + '\n' +
              'test ! -e ' + incoming + '\ntest ! -L ' + incoming + '\n')
    plan = {'local_package': local, 'remote_package': str(REMOTE), 'temporary': str(temporary),
            'audit': str(audit), 'prepare_command': prepare, 'scp_argv': scp,
            'verify_command': verify_command, 'finish_command': finish,
            'automatic_resubmission': False, 'dispatch': dispatch}
    if not dispatch:
        return plan
    transport = Transport(audit)
    transport.call('allocation', prepare)
    transport.run(scp, timeout=180)
    readback = transport.call('allocation', checked_hash(verifier, verifier_sha) +
                              'cat -- ' + shlex.quote(str(verifier)) + '\n')
    require(readback == verifier_raw, 'Verifier plaintext readback mismatch')
    result = json.loads(transport.call('allocation', verify_command))
    require(result == local, 'Remote package verification mismatch')
    transport.call('allocation', finish)
    result.pop('remote_mutations')
    result.update(remote_package=str(REMOTE), audit=str(audit),
                  staging_performed=True, git_operations=0, training_launch_performed=False)
    with (audit / 'STAGED.json').open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--stage', action='store_true', help='Dispatch the reviewed SCP plan')
    args = parser.parse_args()
    print(json.dumps(stage(args.manifest_sha256, args.stage), indent=2))
