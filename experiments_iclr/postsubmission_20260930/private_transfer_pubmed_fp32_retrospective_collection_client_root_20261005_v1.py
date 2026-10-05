"""Collect already retained numerical qualification evidence; run no models."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json
import shlex
import subprocess

PHASE = Path(__file__).resolve().parent
PREPARATION = PHASE / 'private_transfer_pubmed_fp32_retrospective_collection_preparation_20261005_v3'
REVIEW = PHASE / 'private_transfer_pubmed_fp32_retrospective_collection_independent_review_20261005_v3'
OUT = PHASE / 'private_transfer_pubmed_fp32_retrospective_collection_execution_root_20261005_v1'
LOGIN = 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
REPO = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
COLLECTOR_SHA = '5ae182e2ba1ef054455269b0878d2e8b5d0add1cf1a1a69c725c8ea8328ee47b'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def verify_packet(root):
    manifest_bytes = (root / 'MANIFEST.json').read_bytes()
    seal = json.loads((root / 'SEAL.json').read_bytes())
    assert seal['manifest_sha256'] == sha(manifest_bytes)
    for row in json.loads(manifest_bytes)['files']:
        path = root / row['path']
        assert path.resolve().is_relative_to(root.resolve()) and not path.is_symlink()
        raw = path.read_bytes()
        assert len(raw) == row['bytes'] and sha(raw) == row['sha256']
    return {'manifest_sha256': sha(manifest_bytes),
            'seal_sha256': sha((root / 'SEAL.json').read_bytes()),
            'report_sha256': sha((root / 'REPORT.md').read_bytes())}


def main():
    # The independent report must exist before root activates this read-only source.
    assert REVIEW.is_dir() and not OUT.exists()
    preparation = verify_packet(PREPARATION)
    review = verify_packet(REVIEW)
    disabled = (PREPARATION / 'collect_existing.py.txt').read_bytes()
    assert sha(disabled) == COLLECTOR_SHA
    source = disabled.decode()
    assert source.count('COLLECTION_ENABLED = False') == 1
    tree = ast.parse(source)
    origin = next(ast.literal_eval(n.value) for n in tree.body
                  if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                  and n.targets[0].id == 'LOCAL_ORIGIN_EVIDENCE')
    # Current local evidence and all preparation dependencies retain exact custody.
    bindings = json.loads((PREPARATION / 'INPUT_BINDINGS.json').read_text())
    for row in bindings['files']:
        path = PHASE / row['path']
        assert path.resolve().is_relative_to(PHASE) and not path.is_symlink()
        raw = path.read_bytes()
        assert len(raw) == row['bytes'] and sha(raw) == row['sha256']
    for row in origin.values():
        raw = (PHASE / row['path']).read_bytes()
        assert len(raw) == row['bytes'] and sha(raw) == row['sha256']
    enabled = source.replace('COLLECTION_ENABLED = False', 'COLLECTION_ENABLED = True', 1)
    submitted = (
        'from pathlib import Path\nimport socket,subprocess\n'
        f'assert Path.cwd().resolve()==Path({REPO!r})\n'
        "assert socket.gethostname()=='anogena-2-0'\n"
        "assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],"
        f'text=True,timeout=20).split()==[{UUID!r}]\n'
        f'exec(compile({enabled!r}, "reviewed-read-only-retrospective-collector", "exec"))\n'
    )
    OUT.mkdir()
    (OUT / 'COLLECTOR_DISABLED.py.txt').write_bytes(disabled)
    (OUT / 'COLLECTOR_ENABLED.py.txt').write_text(enabled)
    (OUT / 'SUBMITTED_SOURCE.py.txt').write_text(submitted)
    (OUT / 'ROOT_ADMISSION.json').write_text(json.dumps({
        'UTC': datetime.now(timezone.utc).isoformat(),
        'purpose': 'read_only_retrospective_existing_evidence_collection',
        'preparation': preparation, 'independent_review': review,
        'disabled_collector_sha256': COLLECTOR_SHA,
        'enabled_collector_sha256': sha(enabled.encode()),
        'submitted_source_sha256': sha(submitted.encode()),
        'only_collector_activation_change': 'COLLECTION_ENABLED False to True',
        'root_approved_read_only_collection': True,
        'ssh_destination': LOGIN, 'expected_host': 'anogena-2-0', 'gpu_uuid': UUID,
        'new_model_runs': 0, 'fits_authorized': False, 'VALID_TEST_values_access': False,
        'original_workflow_success': False, 'original_failure_preserved': True,
        'qualification_admission_before_collection': False,
        'stdout_limit_bytes': 65536, 'original_output_limit_bytes': 67108864,
        'original_remote_files_written': False,
    }, indent=2) + '\n')
    command = ['ssh', '-T', '-p', '2222', '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt',
               '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes',
               '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=20', LOGIN,
               'cd ' + shlex.quote(REPO) + ' && exec python3 -I -S -B -']
    started = datetime.now(timezone.utc).isoformat()
    result = subprocess.run(command, input=submitted.encode(), capture_output=True, timeout=45)
    (OUT / 'TRANSPORT.json').write_text(json.dumps({
        'start_UTC': started, 'terminal_UTC': datetime.now(timezone.utc).isoformat(),
        'exit_code': result.returncode, 'stderr': result.stderr.decode(),
        'stdout_sha256': sha(result.stdout), 'stdout_bytes': len(result.stdout),
        'client_sha256': sha(Path(__file__).read_bytes()),
        'submitted_source_sha256': sha(submitted.encode()),
        'ssh_destination': LOGIN, 'new_model_runs': 0, 'read_only': True,
    }, indent=2) + '\n')
    assert len(result.stdout) <= 65536
    (OUT / 'COLLECTION.json').write_bytes(result.stdout)
    assert result.returncode == 0, result.stderr.decode()
    receipt = json.loads(result.stdout)
    assert receipt['collection_completed'] is True and receipt['retrospective_assertions_passed'] is True
    assert receipt['original_workflow_success'] is False and receipt['original_transport']['exit_code'] == 1
    assert receipt['qualification_admission'] is False and receipt['fit_admission'] is False
    assert receipt['new_model_runs'] == receipt['fits'] == 0 and receipt['VALID_TEST_values_access'] is False
    assert receipt['current_existing_root_output_bytes'] <= 67108864
    print(json.dumps({'UTC': receipt['UTC'], 'collection_completed': True,
                      'root_output_bytes': receipt['current_existing_root_output_bytes'],
                      'inventory_entries': len(receipt['complete_owned_root_inventory']),
                      'all_three_numerical_architectures_passed': True,
                      'original_workflow_success': False, 'new_model_runs': 0}))


if __name__ == '__main__':
    main()
