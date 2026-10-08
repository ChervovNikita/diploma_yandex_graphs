"""Launch the prospectively fixed complete comparison once on normal host 18.77."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

ROOT = Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
PHASE = ROOT / 'experiments_iclr/postsubmission_20260930'
HERE = PHASE / 'supcon_full3_scientific_activation_root_20261008_v1'
SOURCE = PHASE / 'portable_wikics_supcon_full3_normal_host_family_source_20261008_v1'
PYTHON = ROOT / '.gnnm_runtime/private_transfer_cp311_cu118_20261005_v1/bin/python'
GPUS = ['GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998',
        'GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def process(pid):
    try:
        raw = Path('/proc', str(pid), 'stat').read_text()
    except FileNotFoundError:
        return None
    fields = raw[raw.rfind(')') + 2:].split()
    return dict(pid=pid, start_ticks=int(fields[19]), state=fields[0],
                ppid=int(fields[1]), group=int(fields[2]), session=int(fields[3]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--admission-sha256', required=True)
    args = parser.parse_args()
    os.umask(0o077)
    assert Path.cwd() == ROOT and socket.gethostname() == 'peptide'
    assert Path(__file__).resolve() == HERE / 'launch.py'
    assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid',
                                   '--format=csv,noheader'], text=True).splitlines() == GPUS
    admission = HERE / 'ADMISSION.json'
    assert sha(admission) == args.admission_sha256
    release = json.loads(admission.read_text())
    assert release['enabled'] and release['root_scientific_launch_authorized']
    assert sha(SOURCE / 'MANIFEST.json') == release['controller_manifest_sha256']
    assert not (PHASE / release['output_directory']).exists()
    receipt = HERE / 'LAUNCH.json'
    assert not receipt.exists()
    # Reserve the single launch identity before starting a detached owner.
    with receipt.open('x') as handle:
        json.dump(dict(launch_reserved=True, scientific_child_observed=False), handle)
    environment = dict(os.environ, PYTHONPATH='', PYTHONDONTWRITEBYTECODE='1')
    command = [str(PYTHON), '-B', str(SOURCE / 'controller.py'),
               '--admission', str(admission), '--admission-sha256', args.admission_sha256]
    with (HERE / 'owner.log').open('xb') as log:
        child = subprocess.Popen(command, cwd=ROOT, env=environment,
                                 stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    identity = process(child.pid)
    time.sleep(0.2)
    status = child.poll()
    result = dict(UTC=datetime.now(timezone.utc).isoformat(), owner=identity,
                  parent_poll=status, admission_sha256=args.admission_sha256,
                  controller_manifest_sha256=release['controller_manifest_sha256'],
                  argv=command, detached=True, launch_reserved=True,
                  fixed_complete_fits=3, scientific_child_observed=False,
                  automatic_retry=False, quality_scores_read=False)
    receipt.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    if status is not None:
        raise RuntimeError('Owner exited at launch. Observe this attempt without restarting it.')


if __name__ == '__main__':
    main()
