"""Verify archive bytes, exact member set and every unique evidence object."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

MANIFEST_SHA = '73f73778f64b25f0ef4a0f2c954f63f7f52c136409fd0be7be69258e2741a487'


def verify(path):
    archive_sha = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            archive_sha.update(block)
    manifest = None; observed = set(); objects = {}; small_files = {}
    with tarfile.open(path, mode='r|gz') as packed:
        for member in packed:
            if not member.isfile() or member.name in observed:
                raise RuntimeError('Unexpected non-file or duplicate archive member')
            observed.add(member.name); stream = packed.extractfile(member)
            if member.name == 'EVIDENCE_MANIFEST.json':
                data = stream.read()
                if hashlib.sha256(data).hexdigest() != MANIFEST_SHA:
                    raise RuntimeError('Manifest identity differs')
                manifest = json.loads(data)
            elif member.name in {'recover_evidence.py','RECOVERY.md'}:
                data = stream.read(); small_files[member.name] = hashlib.sha256(data).hexdigest()
            elif member.name.startswith('objects/'):
                digest = member.name.removeprefix('objects/')
                calculated = hashlib.sha256(); count = 0
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    count += len(block); calculated.update(block)
                if calculated.hexdigest() != digest or count != member.size:
                    raise RuntimeError('Evidence object payload hash/size differs')
                objects[digest] = count
            else:
                raise RuntimeError('Unexpected archive member')
    if manifest is None or set(objects) != set(manifest['objects']):
        raise RuntimeError('Exact object set differs')
    for digest, row in manifest['objects'].items():
        if objects[digest] != row['bytes']:
            raise RuntimeError('Object size differs from preserved manifest')
    for row in manifest['files']:
        if row['sha256'] not in objects or objects[row['sha256']] != row['bytes']:
            raise RuntimeError('Original file does not resolve to verified object')
    if set(small_files) != {'recover_evidence.py','RECOVERY.md'}:
        raise RuntimeError('Recovery material missing')
    return dict(schema='gnnm-evidence-archive-verified-v1', status='ALL_MEMBERS_AND_EVERY_OBJECT_SHA256_VERIFIED',
        archive_path=str(path), archive_bytes=path.stat().st_size, archive_sha256=archive_sha.hexdigest(),
        manifest_sha256=MANIFEST_SHA, original_file_count=manifest['original_file_count'],
        original_bytes=manifest['original_bytes'], unique_object_count=len(objects),
        unique_object_bytes=sum(objects.values()), archive_member_count=len(observed), recovery_hashes=small_files)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('archive')
    args = parser.parse_args(); print(json.dumps(verify(Path(args.archive)),indent=2))
