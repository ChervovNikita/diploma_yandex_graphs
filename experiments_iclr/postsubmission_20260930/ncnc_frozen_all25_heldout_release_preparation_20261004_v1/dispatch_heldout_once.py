"""Future root-approved detached launch of one frozen all25 heldout supervisor."""
from argparse import ArgumentParser
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess

REPO = Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
HERE = REPO / 'experiments_iclr/postsubmission_20260930/ncnc_frozen_all25_heldout_release_preparation_20261004_v1'
RUNNER_SHA256 = '3aa4a4f879e860003971e330a11ef396b5bdcb546111e1d065b2d2e246d3b389'


def file_sha(path):
    h = sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--root-release', required=True)
    parser.add_argument('--release-sha256', required=True)
    args = parser.parse_args()
    release_path = Path(args.root_release)
    assert Path.cwd().resolve() == REPO and Path(__file__).resolve() == HERE / 'dispatch_heldout_once.py'
    assert release_path.is_absolute() and release_path.resolve() == release_path and release_path.parent == HERE
    assert file_sha(release_path) == args.release_sha256
    release = json.loads(release_path.read_text())
    assert release['schema'] == 'ncnc-frozen-all25-one-time-heldout-root-release-v1'
    assert release['execution_enabled'] is True and release['root_authorization_reference']
    assert release['authorized_stages'] == ['frozen_all25_heldout_confirmation']
    assert release['authorized_invocations'] == [dict(stage='frozen_all25_heldout_confirmation', output_directory=str(HERE / 'heldout/run01'), cuda_visible_devices='GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998')]
    assert release['TEST_access_authorized'] is True and release['evaluation_once'] is True
    assert not (HERE / 'supervision_run01').exists() and not (HERE / 'heldout/run01').exists() and not (HERE / 'ONE_TIME_TEST_CLAIM.json').exists()
    runner = HERE / 'supervise_heldout_once.py'
    assert runner.resolve() == runner and file_sha(runner) == RUNNER_SHA256
    argv = ['/usr/bin/python3', '-I', '-S', '-B', str(runner), '--root-release', str(release_path), '--release-sha256', args.release_sha256]
    with (HERE / 'RUNNER.stdout.log').open('x') as stdout, (HERE / 'RUNNER.stderr.log').open('x') as stderr:
        child = subprocess.Popen(argv, cwd=REPO, stdout=stdout, stderr=stderr, start_new_session=True)
        stat = Path(f'/proc/{child.pid}/stat').read_text()
        fields = stat[stat.rfind(')') + 2:].split()
        handle = dict(pid=child.pid, start_ticks=int(fields[19]), pgid=int(fields[2]), sid=int(fields[3]))
        assert handle['pgid'] == handle['sid'] == child.pid
    result = dict(schema='ncnc-frozen-all25-heldout-detached-launch-v1', UTC=datetime.now(timezone.utc).isoformat(), status='DETACHED_SUPERVISOR_STARTED_ONCE', runner_handle=handle, argv=argv, root_release_sha256=args.release_sha256, supervisor_sha256=RUNNER_SHA256, resource_recheck_owner='supervisor_immediately_before_numerical_child', physical_GPU_index=0, GPU_UUID=release['cuda_visible_devices'], served_cell_count=25, planned_scorer_slots=40, new_training_updates=0, retry=False, namespaces=False, unrelated_jobs_signaled=False)
    with (HERE / 'DETACHED_LAUNCH.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
