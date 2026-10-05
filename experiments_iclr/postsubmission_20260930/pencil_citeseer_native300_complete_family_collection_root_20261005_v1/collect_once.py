"""Collect the reviewed three-seed scalar summary after complete custody."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
PREP = PHASE / 'pencil_citeseer_native300_complete_family_collection_preparation_20261005_v1'
REPO = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
LOGIN = 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
OBS = PHASE / 'pencil_citeseer_native300_launch_receipts_20261005_v1/metadata_20261005T192744Z'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    assert not (HERE / 'ACTIVATION.json').exists(), 'One collection identity only'
    manifest = json.loads((PREP / 'MANIFEST.json').read_text())
    for row in manifest['files']:
        data = (PREP / row['path']).read_bytes()
        assert len(data) == row['bytes'] and sha(data) == row['sha256']
    review_path = PHASE / 'pencil_citeseer_native300_collector_root_source_review_20261005_v1/ROOT_SOURCE_REVIEW.json'
    review = json.loads(review_path.read_text())
    assert review['accepted_as_disabled_source'] is True
    source = (PREP / 'collect_complete_family.py.txt').read_text()
    assert sha(source.encode()) == review['collector_sha256']
    assert sha((PREP / 'MANIFEST.json').read_bytes()) == review['manifest_sha256']
    observations = {}
    for name in ['COHORT_FREEZE.json', 'TERMINAL_RECEIPT.json']:
        data = (OBS / name).read_bytes()
        value = json.loads(data)
        assert value['complete'] is True and value['failures'] == []
        assert [row['seed'] for row in value['completed_fits']] == [0, 1, 2]
        assert value['host'] == 'anogena-2-0' and value['GPU_UUID'] == UUID
        observations[name] = {'sha256': sha(data), 'bytes': len(data)}
    assert source.count('COLLECTION_ENABLED = False') == 1
    enabled = source.replace('COLLECTION_ENABLED = False', 'COLLECTION_ENABLED = True', 1)
    ast.parse(enabled)
    (HERE / 'collect_complete_family_enabled.py.txt').write_text(enabled)
    guard = (
        'from pathlib import Path\nimport socket, subprocess\n'
        f'assert Path.cwd().resolve() == Path({REPO!r})\n'
        "assert socket.gethostname() == 'anogena-2-0'\n"
        "assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=15).split() == "
        f'[{UUID!r}]\n'
    )
    executed = guard + enabled
    activation = {
        'UTC': datetime.now(timezone.utc).isoformat(),
        'purpose': 'Complete-family descriptive selected VALID scalar collection',
        'review_sha256': sha(review_path.read_bytes()),
        'disabled_source_sha256': sha(source.encode()),
        'enabled_source_sha256': sha(enabled.encode()),
        'executed_source_sha256': sha(executed.encode()),
        'sole_source_change': 'COLLECTION_ENABLED False to True',
        'complete_three_seed_observation': observations,
        'actual_remote_custody_checks_required_before_scalars': True,
        'ssh_destination': LOGIN, 'GPU_UUID': UUID,
        'TEST_access': False, 'new_model_runs': 0,
    }
    (HERE / 'ACTIVATION.json').write_text(json.dumps(activation, indent=2) + '\n')
    command = [
        'ssh', '-T', '-p', '2222', '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt',
        '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes', '-o', 'StrictHostKeyChecking=yes',
        '-o', 'UpdateHostKeys=no', '-o', 'ConnectTimeout=20', LOGIN,
        'cd ' + shlex.quote(REPO) + ' && exec /usr/bin/python3 -I -S -B -',
    ]
    result = subprocess.run(command, input=executed, capture_output=True, text=True, timeout=55)
    receipt = {
        'UTC': datetime.now(timezone.utc).isoformat(), 'exit_code': result.returncode,
        'stderr': result.stderr, 'stdout': result.stdout,
        'executed_source_sha256': sha(executed.encode()),
        'TEST_access': False, 'new_model_runs': 0,
    }
    (HERE / 'TRANSPORT_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
    if result.returncode:
        print(json.dumps({'exit_code': result.returncode, 'stderr': result.stderr, 'stdout': result.stdout}))
        raise SystemExit(result.returncode)
    value = json.loads(result.stdout)
    assert value['collection_completed'] and value['complete_family_custody_passed']
    assert value['TEST_access'] is False and value['raw_payloads_transferred'] is False
    (HERE / 'COMPLETE_FAMILY_SUMMARY.json').write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps({key: value[key] for key in ['selected_VALID', 'descriptive_summary', 'comparison_scope']}))


if __name__ == '__main__':
    main()
