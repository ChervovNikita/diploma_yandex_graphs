"""Receive only exact literature archive bytes in a fresh authorized directory."""
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess
import sys
import tarfile

repo = Path(sys.argv[1])
expected_gpu, expected_size, expected_archive, expected_manifest = sys.argv[2:]
expected_size = int(expected_size)
gpu = subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                     capture_output=True, text=True, check=True, timeout=15)
assert gpu.stdout.strip().splitlines() == [expected_gpu], 'Authorized one-GPU route differs'
assert repo.resolve() == repo
git = subprocess.run(['git', 'rev-parse', '--show-toplevel'], cwd=repo,
                     capture_output=True, text=True, check=True, timeout=10)
assert Path(git.stdout.strip()).resolve() == repo
root = repo / 'experiments_iclr/postsubmission_20260930/literature_raw_asset_archive_20261004_v1'
assert root.resolve() == root and root.is_relative_to(repo) and not root.exists()
root.mkdir()


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    sync_directory(path.parent)


archive = root / 'ASSETS.tar.gz'
received = 0
digest = hashlib.sha256()
with archive.open('xb') as stream:
    while True:
        block = sys.stdin.buffer.read(1048576)
        if not block:
            break
        received += len(block)
        assert received <= expected_size
        digest.update(block)
        stream.write(block)
    stream.flush()
    os.fsync(stream.fileno())
assert received == expected_size and digest.hexdigest() == expected_archive
sync_directory(root)
with tarfile.open(archive, 'r:gz') as tar:
    members = tar.getmembers()
    names = [member.name for member in members]
    assert len(names) == len(set(names))
    metadata = tar.extractfile('ARCHIVE_MANIFEST.json').read()
    assert hashlib.sha256(metadata).hexdigest() == expected_manifest
    manifest = json.loads(metadata)
    assert manifest['archive_directory'] == str(root)
    assert manifest['repository'] == str(repo) and manifest['GPU_UUID'] == expected_gpu
    expected = {row['archive_path']: row for row in manifest['files']}
    assert len(expected) == len(manifest['files'])
    assert set(names) == set(expected) | {'ARCHIVE_MANIFEST.json'}
    for member in members:
        name = PurePosixPath(member.name)
        assert member.isfile() and not name.is_absolute() and '..' not in name.parts
        path = root / member.name
        assert path.resolve().is_relative_to(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tar.extractfile(member) as source, path.open('xb') as target:
            for block in iter(lambda: source.read(1048576), b''):
                target.write(block)
            target.flush()
            os.fsync(target.fileno())
        if member.name == 'ARCHIVE_MANIFEST.json':
            assert sha(path) == expected_manifest
        else:
            row = expected[member.name]
            assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
for directory in sorted((p for p in root.rglob('*') if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
    sync_directory(directory)
sync_directory(root)
verified = []
for row in manifest['files']:
    path = root / row['archive_path']
    assert path.is_file() and not path.is_symlink()
    assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
    verified.append({'archive_path': row['archive_path'], 'bytes': row['bytes'],
                     'sha256': row['sha256'], 'match': True})
assert archive.stat().st_size == expected_size and sha(archive) == expected_archive
verification = {'schema': 'literature-exact-archive-remote-verification-v1',
                'status': 'FULL_EXACT_RAW_AND_PROVENANCE_ARCHIVE_VERIFIED',
                'UTC': datetime.now(timezone.utc).isoformat(), 'GPU_UUIDs': [expected_gpu],
                'repository': str(repo), 'archive_directory': str(root),
                'archive_bytes': expected_size, 'archive_sha256': expected_archive,
                'archive_manifest_sha256': expected_manifest, 'files': verified,
                'raw_assets': len(manifest['raw_assets_to_remove_locally']),
                'raw_bytes': sum(row['bytes'] for row in manifest['raw_assets_to_remove_locally']),
                'all_files_and_directories_fsynced': True, 'scientific_access_or_compute': False,
                'active_run_files_or_unrelated_paths_mutated': False}
write(root / 'ARCHIVE_VERIFICATION.json', verification)
seal = {'schema': 'literature-remote-archive-seal-v1', 'archive_sha256': expected_archive,
        'archive_manifest_sha256': expected_manifest,
        'archive_verification_sha256': sha(root / 'ARCHIVE_VERIFICATION.json'),
        'directory': str(root), 'durable_exact_archive_verified': True}
write(root / 'ARCHIVE_SEAL.json', seal)
sync_directory(root.parent)
assert sha(root / 'ARCHIVE_MANIFEST.json') == expected_manifest
assert sha(root / 'ARCHIVE_VERIFICATION.json') == seal['archive_verification_sha256']
print(json.dumps({**seal, 'archive_seal_sha256': sha(root / 'ARCHIVE_SEAL.json'),
                  'files_verified': len(verified), 'archive_member_files': len(members),
                  'raw_assets': verification['raw_assets'], 'raw_bytes': verification['raw_bytes']}, sort_keys=True))
