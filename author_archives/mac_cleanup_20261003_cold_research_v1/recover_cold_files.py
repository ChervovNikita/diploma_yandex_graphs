"""Recover exactly 48 verified historical files into a NEW project directory."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile

MANIFEST_SHA = 'd361fa9498d646793ac8944b17ccae5c3d5aed498b9d6cfab24a50eea70c1654'


def restore(archive, destination):
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink() or destination.parent.resolve() != destination.parent:
        raise RuntimeError('Choose a new directory under a canonical existing parent')
    destination.mkdir(mode=0o700)
    with tempfile.TemporaryDirectory(prefix='.cold-objects-', dir=destination) as temporary:
        objects = Path(temporary); observed = set(); manifest = None
        with tarfile.open(archive, mode='r|gz') as packed:
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
                    source.read()
                elif member.name.startswith('objects/'):
                    digest = member.name.removeprefix('objects/')
                    if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                        raise RuntimeError('Unsafe object name')
                    calculated = hashlib.sha256()
                    with (objects / digest).open('xb') as output:
                        for block in iter(lambda: source.read(1024 * 1024), b''):
                            output.write(block); calculated.update(block)
                    if calculated.hexdigest() != digest:
                        raise RuntimeError('Object hash differs')
                else:
                    raise RuntimeError('Unexpected archive member')
        if manifest is None or {path.name for path in objects.iterdir()} != set(manifest['objects']):
            raise RuntimeError('Exact object set differs')
        for row in manifest['files']:
            relative = PurePosixPath(row['path'])
            if relative.is_absolute() or '..' in relative.parts or str(relative) != row['path']:
                raise RuntimeError('Unsafe recovery path')
            target = destination / str(relative); source = objects / row['sha256']
            if source.stat().st_size != row['bytes']:
                raise RuntimeError('Object size differs')
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open('rb') as input_file, target.open('xb') as output_file:
                shutil.copyfileobj(input_file, output_file, 1024 * 1024)
            os.chmod(target, row['mode']); os.utime(target, ns=(row['mtime_ns'], row['mtime_ns']))
    print(json.dumps(dict(restored_files=manifest['original_file_count'], restored_bytes=manifest['original_bytes'],
                          destination=str(destination), every_object_SHA256_verified=True)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive'); parser.add_argument('new_project_directory')
    args = parser.parse_args(); restore(args.archive, args.new_project_directory)
