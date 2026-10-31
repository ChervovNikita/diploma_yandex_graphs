"""Verify the complete compressed archive and its exact five regular members."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

MANIFEST_SHA = '61c5ca6d6c744aa0a5d2ed7d0aa886fa57e839f5673da4b8c9c7698ae258d5bb'


def verify(path):
    before = path.stat(); h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            h.update(block)
    observed = {}; recovery = {}; manifest = None
    with tarfile.open(path, mode='r|gz') as packed:
        for member in packed:
            if not member.isfile() or member.name in observed:
                raise RuntimeError('Unexpected or duplicate archive member')
            source = packed.extractfile(member); sha = hashlib.sha256(); count = 0
            if member.name == 'EXACT_FILE_MANIFEST.json':
                data = source.read(); sha.update(data); count = len(data)
                if sha.hexdigest() != MANIFEST_SHA:
                    raise RuntimeError('Manifest identity differs')
                manifest = json.loads(data)
            else:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
                    sha.update(block); count += len(block)
            assert count == member.size
            observed[member.name] = dict(bytes=count, sha256=sha.hexdigest())
            if member.name in {'recover_large_audits.py', 'RECOVERY.txt'}:
                recovery[member.name] = sha.hexdigest()
    assert manifest is not None
    expected = {'EXACT_FILE_MANIFEST.json', 'recover_large_audits.py', 'RECOVERY.txt'} | {'files/' + Path(row['path']).name for row in manifest['files']}
    assert set(observed) == expected and len(observed) == 5
    for row in manifest['files']:
        assert observed['files/' + Path(row['path']).name] == dict(bytes=row['bytes'], sha256=row['sha256'])
    after = path.stat()
    assert (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns) == (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns)
    return dict(status='ALL_ARCHIVE_BYTES_MEMBERS_AND_TWO_PAYLOAD_SHA256_VERIFIED', archive_path=str(path), archive_bytes=after.st_size,
                archive_sha256=h.hexdigest(), manifest_sha256=MANIFEST_SHA, member_count=5, file_count=2, original_bytes=manifest['bytes'],
                recovery_hashes=recovery, archive_stable_during_verification=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('archive')
    args = parser.parse_args(); print(json.dumps(verify(Path(args.archive)), indent=2))
