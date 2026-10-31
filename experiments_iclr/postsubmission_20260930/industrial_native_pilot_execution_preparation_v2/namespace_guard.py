"""Future Linux namespace observation; source checks never invoke this module."""
from __future__ import annotations
import os
from pathlib import Path
import re

DENIED_HOST_PATHS = (
    '/home/jovyan/shares', '/Users/alex/Documents/ChatGPT',
    '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/'
    'postsubmission_20260930/industrial_dataset_acquisition_root_v1/tolokers2_run01/tolokers-2.zip',
    '/inputs/tolokers-2/targets.csv', '/inputs/tolokers-2.zip',
)


def observe_namespace(device):
    if os.name != 'posix' or not Path('/proc/self/mountinfo').is_file():
        raise RuntimeError('Worker requires the concrete Linux namespace launcher')
    mounts = {}
    observed = []
    devices = {'/dev/null', '/dev/zero', '/dev/random', '/dev/urandom'}
    if device == 'cuda:0':
        physical = {str(path) for path in Path('/dev').iterdir() if re.fullmatch(r'nvidia[0-9]+', path.name)}
        if len(physical) != 1:
            raise RuntimeError('Exactly one physical GPU node must be visible')
        devices |= physical | {'/dev/nvidiactl', '/dev/nvidia-uvm', '/dev/nvidia-uvm-tools'}
    expected_mounts = {'/', '/inputs', '/sources', '/outputs', '/tmp', '/proc', '/dev'} | devices
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        left, right = line.split(' - ', 1)
        fields, suffix = left.split(), right.split()
        target, options = fields[4], set(fields[5].split(','))
        mounts[target] = options
        observed.append({'target': target, 'options': sorted(options), 'fs': suffix[0]})
        if target not in expected_mounts:
            raise RuntimeError('Unreviewed worker mount: ' + target)
    if set(mounts) != expected_mounts:
        raise RuntimeError('Required allowlist mounts are missing')
    filesystem = {item['target']: item['fs'] for item in observed}
    if filesystem['/proc'] != 'proc' or any(filesystem[name] != 'tmpfs' for name in ('/tmp', '/dev')):
        raise RuntimeError('Fresh proc/tmp/dev filesystem contract differs')
    if os.getpid() != 1 or os.getuid() != 0:
        raise RuntimeError('Scientific process lacks the expected mapped-root PID namespace')
    for target in ('/', '/inputs', '/sources'):
        if 'ro' not in mounts.get(target, set()):
            raise RuntimeError('Required read-only mount missing: ' + target)
    if 'rw' not in mounts.get('/outputs', set()):
        raise RuntimeError('Dedicated writable output volume missing')
    deny_observations = []
    for path in DENIED_HOST_PATHS:
        if Path(path).exists() or Path(path).is_symlink():
            raise RuntimeError('Denied custodian/host path is visible: ' + path)
        try:
            descriptor = os.open(path, os.O_RDONLY)
        except FileNotFoundError:
            deny_observations.append({'path': path, 'open': 'ENOENT'})
        except PermissionError:
            # A permission-only denial is weaker than an excluded path; stop.
            raise RuntimeError('Denied path exists behind permissions: ' + path)
        else:
            os.close(descriptor)
            raise RuntimeError('Denied path can be opened: ' + path)
    interfaces = [name for _, name in __import__('socket').if_nameindex()]
    if any(name != 'lo' for name in interfaces):
        raise RuntimeError('Worker network namespace exposes a non-loopback interface')
    if Path('/proc/net/route').read_text().splitlines()[1:]:
        raise RuntimeError('Worker namespace has network routes')
    if device == 'cpu' and list(Path('/dev').glob('nvidia*')):
        raise RuntimeError('CPU worker exposes GPU device nodes')
    status = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
    for key in ('CapInh', 'CapPrm', 'CapEff', 'CapBnd', 'CapAmb'):
        if int(status[key].strip(), 16) != 0:
            raise RuntimeError('Scientific worker retains namespace escape capabilities: ' + key)
    if status['NoNewPrivs'].strip() != '1':
        raise RuntimeError('Worker lacks no_new_privs')
    return {'schema': 'observed-worker-namespace-v2', 'mounts': observed,
            'denied_host_paths_absent': list(DENIED_HOST_PATHS), 'deny_path_open_observations': deny_observations,
            'network_interfaces': interfaces,
            'device_scope': device,
            'capabilities_dropped': True, 'no_new_privileges': True,
            'limit': 'Observed namespace verifies launcher effects. Dedicated runtime-image contents and '
                     'output-volume provenance still require custodian review; JSON cannot supply that trust.'}
