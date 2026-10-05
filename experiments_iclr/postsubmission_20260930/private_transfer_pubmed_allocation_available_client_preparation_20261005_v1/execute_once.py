"""Disabled client for one fresh CPU input check; never restart on timeout."""
from datetime import datetime, timezone
import ast
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
SOURCE = PHASE / 'private_transfer_pubmed_allocation_available_preparation_20261005_v1'
SUPERVISION = PHASE / 'private_transfer_pubmed_allocation_available_supervision_preparation_20261005_v1'
LOGIN = 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
SOURCE_MANIFEST_SHA = 'f77ce5821522a4ee3a472c636a0f597ed28e814e36bd8e0152d0b0f0138d3322'
REMOTE_SOURCE_SHA = 'f936dd02dbf3978ca1642667f237a9ef8f81c61997755f1d3074cf51deb4e234'
EXECUTION_NAME = 'private_transfer_pubmed_allocation_available_execution_root_20261005_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-directory', type=Path, required=True)
    args = parser.parse_args()
    release = args.release_directory.resolve(strict=True)
    if release.parent != PHASE or release.name != EXECUTION_NAME:
        raise ValueError('Exact local project release identity required')
    intent = release / 'LOCAL_INTENT.json'
    if intent.exists():
        raise ValueError('Attempt identity already exists; inspect it, never relaunch')
    remote = (SUPERVISION / 'owned_supervisor.py.txt').read_text()
    if sha(SOURCE / 'SOURCE_MANIFEST.json') != SOURCE_MANIFEST_SHA or hashlib.sha256(remote.encode()).hexdigest() != REMOTE_SOURCE_SHA:
        raise ValueError('Prepared immutable source changed')
    ast.parse(remote)
    admission = json.loads((release / 'OWNED_SUPERVISION_ADMISSION.json').read_text())
    if admission.get('root_approved') is not True or admission.get('remote_source_sha256') != REMOTE_SOURCE_SHA:
        raise ValueError('Prepared client remains disabled without root admission')
    if admission.get('client_sha256') != sha(Path(__file__)) or admission.get('client_independent_review_approved') is not True:
        raise ValueError('Exact independently reviewed launcher required')
    if admission.get('TEST_access') is not False or admission.get('fit_admission') is not False:
        raise ValueError('Input checks only, TEST and fitting closed')
    for row in admission.get('review_evidence', []):
        path = PHASE / row['path']
        if not path.resolve().is_relative_to(PHASE) or path.is_symlink() or sha(path) != row['sha256']:
            raise ValueError('Actual independent review bytes changed')
    if not admission.get('review_evidence'):
        raise ValueError('Actual source/launcher reviews absent')
    files = []
    for row in json.loads((SOURCE / 'MANIFEST.json').read_text())['files']:
        path = SOURCE / row['path']
        if not path.resolve().is_relative_to(SOURCE) or path.is_symlink() or path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError('Prepared source packet changed')
    for path in sorted(SOURCE.iterdir()):
        if path.is_file():
            raw = path.read_bytes()
            files.append({'path': 'source/' + path.name, 'bytes': len(raw),
                'sha256': hashlib.sha256(raw).hexdigest(), 'base64': base64.b64encode(raw).decode()})
    names = ('EXTRACT_RELEASE.json', 'INSPECT_RELEASE.json', 'ROOT_REVIEW.md',
        'INDEPENDENT_SOURCE_REVIEW.md', 'SUPERVISION_SOURCE_REVIEW.md', 'OWNED_SUPERVISION_ADMISSION.json')
    for name in names:
        path = release / name
        if path.is_symlink() or not path.is_file():
            raise ValueError('Concrete reviewed release files required')
        raw = path.read_bytes()
        files.append({'path': name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
            'base64': base64.b64encode(raw).decode()})
    files.append({'path': 'owned_supervisor.py.txt', 'bytes': len(remote.encode()),
        'sha256': REMOTE_SOURCE_SHA, 'base64': base64.b64encode(remote.encode()).decode()})
    payload = {'files': files, 'source_manifest_sha256': SOURCE_MANIFEST_SHA,
        'remote_source_sha256': REMOTE_SOURCE_SHA,
        'release_sha256': {name: sha(release / name) for name in ('EXTRACT_RELEASE.json', 'INSPECT_RELEASE.json')},
        'root_review_sha256': sha(release / 'ROOT_REVIEW.md'),
        'independent_source_review_sha256': sha(release / 'INDEPENDENT_SOURCE_REVIEW.md'),
        'supervision_review_sha256': sha(release / 'SUPERVISION_SOURCE_REVIEW.md')}
    with intent.open('x') as target:
        json.dump({'UTC': datetime.now(timezone.utc).isoformat(), 'single_attempt': True,
            'client_sha256': sha(Path(__file__)), 'remote_source_sha256': REMOTE_SOURCE_SHA,
            'source_manifest_sha256': SOURCE_MANIFEST_SHA, 'TEST_access': False,
            'fit_admission': False, 'relaunch_after_disconnect': False}, target, indent=2)
        target.write('\n')
    ssh = ['ssh', '-T', '-p', '2222', '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt',
        '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes', '-o', 'StrictHostKeyChecking=yes',
        '-o', 'UpdateHostKeys=no', '-o', 'ConnectTimeout=20', LOGIN]
    command = 'cd /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs && exec python3 -I -B -c ' + shlex.quote(remote)
    try:
        result = subprocess.run([*ssh, command], input=json.dumps(payload), capture_output=True, text=True, timeout=1900)
    except subprocess.TimeoutExpired as error:
        def decoded(value):
            return value.decode(errors='replace') if isinstance(value, bytes) else (value or '')
        (release / 'CONSOLE.jsonl').write_text(decoded(error.stdout))
        (release / 'TRANSPORT.json').write_text(json.dumps({'transport_timeout': True,
            'remote_terminal_status': 'UNKNOWN_REPOLL_EXACT_ROOT_DO_NOT_RELAUNCH',
            'stderr': decoded(error.stderr), 'sole_authorized_route': LOGIN}, indent=2) + '\n')
        raise
    (release / 'CONSOLE.jsonl').write_text(result.stdout)
    (release / 'TRANSPORT.json').write_text(json.dumps({'exit_code': result.returncode,
        'stderr': result.stderr, 'remote_source_sha256': REMOTE_SOURCE_SHA,
        'client_sha256': sha(Path(__file__)), 'sole_authorized_route': LOGIN}, indent=2) + '\n')
    if result.returncode:
        raise RuntimeError(result.stderr[-4000:])
    value = json.loads(result.stdout.splitlines()[-1])
    if value.get('completed') is not True:
        raise ValueError('No complete owned result; preserve transport without relaunch')
    for key, name in (('acquisition_manifest', 'ACQUISITION_MANIFEST.json'),
        ('available_inspection', 'AVAILABLE_INSPECTION.json'), ('execution_receipt', 'EXECUTION_RECEIPT.json')):
        with (release / name).open('x') as target:
            json.dump(value[key], target, indent=2)
            target.write('\n')
    print(json.dumps({'completed': True, 'input_geometry_pass': value['available_inspection']['available_geometry_checks_pass'],
        'TEST_access': False, 'fits': 0, 'raw_inputs_fetched_to_Mac': False}))


if __name__ == '__main__':
    main()
