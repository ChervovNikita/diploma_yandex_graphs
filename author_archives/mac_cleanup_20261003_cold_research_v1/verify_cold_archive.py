"""Independently verify archive bytes, exact members and all recovery mappings."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

MANIFEST_SHA = 'd361fa9498d646793ac8944b17ccae5c3d5aed498b9d6cfab24a50eea70c1654'


def verify(path):
    before = path.stat(); calculated = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            calculated.update(block)
    observed = set(); objects = {}; recovery = {}; manifest = None
    with tarfile.open(path, mode='r|gz') as packed:
        for member in packed:
            if not member.isfile() or member.name in observed:
                raise RuntimeError('Non-file or duplicate archive member')
            observed.add(member.name); source = packed.extractfile(member)
            if member.name == 'COLD_FILE_MANIFEST.json':
                data = source.read()
                if hashlib.sha256(data).hexdigest() != MANIFEST_SHA:
                    raise RuntimeError('Manifest identity differs')
                manifest = json.loads(data)
            elif member.name in {'recover_cold_files.py', 'RECOVERY.txt'}:
                recovery[member.name] = hashlib.sha256(source.read()).hexdigest()
            elif member.name.startswith('objects/'):
                digest = member.name.removeprefix('objects/'); sha = hashlib.sha256(); count = 0
                for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
                    sha.update(block); count += len(block)
                if sha.hexdigest() != digest or count != member.size:
                    raise RuntimeError('Object SHA256/size differs')
                objects[digest] = count
            else:
                raise RuntimeError('Unexpected archive member')
    if manifest is None or set(objects) != set(manifest['objects']) or set(recovery) != {'recover_cold_files.py', 'RECOVERY.txt'}:
        raise RuntimeError('Exact archive member set differs')
    for digest, row in manifest['objects'].items():
        assert objects[digest] == row['bytes']
    for row in manifest['files']:
        assert objects[row['sha256']] == row['bytes']
    after = path.stat()
    assert (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    return dict(status='ALL_MEMBERS_OBJECT_SHA256_AND_48_RECOVERY_MAPPINGS_VERIFIED', archive_path=str(path),
                archive_bytes=after.st_size, archive_sha256=calculated.hexdigest(), manifest_sha256=MANIFEST_SHA,
                original_file_count=len(manifest['files']), original_bytes=sum(row['bytes'] for row in manifest['files']),
                unique_object_count=len(objects), unique_object_bytes=sum(objects.values()), archive_member_count=len(observed),
                recovery_hashes=recovery, archive_stable_during_verification=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('archive')
    args = parser.parse_args(); print(json.dumps(verify(Path(args.archive)), indent=2))
