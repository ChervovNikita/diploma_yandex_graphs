"""Separate once-only launch receipt; requires a later enabled root admission."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
REPO = Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
PYTHON = REPO / '.gnnm_runtime/private_transfer_cp311_cu118_20261005_v1/bin/python'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--admission', type=Path, required=True)
    parser.add_argument('--admission-sha256', required=True)
    args = parser.parse_args(); os.umask(0o077)
    assert Path.cwd() == REPO and sha(args.admission) == args.admission_sha256
    admission = json.loads(args.admission.read_text())
    assert admission['enabled'] and admission['root_launch_authorized']
    assert sha(HERE / 'controller.py') == admission['controller_sha256']
    receipt = HERE / 'LAUNCH.json'
    with receipt.open('x') as stream:
        json.dump(dict(launch_reserved=True, scientific_child_observed=False), stream)
    argv = [str(PYTHON), '-B', str(HERE / 'controller.py'),
            '--admission', str(args.admission.resolve()), '--admission-sha256', args.admission_sha256]
    env = dict(os.environ, PYTHONPATH='', PYTHONDONTWRITEBYTECODE='1'); env.pop('PYTHONHOME', None)
    with (HERE / 'owner.log').open('xb') as log:
        child = subprocess.Popen(argv, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    raw = Path('/proc', str(child.pid), 'stat').read_text().rsplit(') ', 1)[1].split()
    identity = dict(pid=child.pid, start_ticks=int(raw[19]), group=int(raw[2]), session=int(raw[3]))
    time.sleep(.2)
    status = child.poll()
    receipt.write_text(json.dumps(dict(launch_reserved=True, owner=identity, parent_poll=status,
        argv=argv, admission_sha256=args.admission_sha256, detached=True,
        never_started_cells_only=True, automatic_retry=False, scientific_child_observed=False,
        quality_scores_read=False), indent=2) + '\n')
    if status is not None:
        raise RuntimeError('Owner exited at launch; preserve this attempt, do not relaunch')


if __name__ == '__main__':
    main()
