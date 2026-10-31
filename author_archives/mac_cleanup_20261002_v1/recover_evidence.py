"""Recover a verified content-addressed evidence archive into a NEW directory.

Usage: python3 -B recover_evidence.py ARCHIVE.tar.gz NEW_EVIDENCE_DIRECTORY
No archive member is executed. No existing destination is overwritten.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile

MANIFEST_SHA = '73f73778f64b25f0ef4a0f2c954f63f7f52c136409fd0be7be69258e2741a487'


def safe_relative(value):
    p = PurePosixPath(value)
    if p.is_absolute() or '..' in p.parts or (value != '.' and str(p) != value):
        raise RuntimeError('Unsafe stored relative path')
    return p


def restore(archive, destination):
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink() or destination.parent.resolve() != destination.parent:
        raise RuntimeError('Choose a new destination under a canonical existing parent')
    destination.mkdir(mode=0o700)
    with tempfile.TemporaryDirectory(prefix='.objects-', dir=destination) as folder:
        objects = Path(folder)
        manifest = None
        observed = set()
        with tarfile.open(archive, mode='r|gz') as packed:
            for member in packed:
                if not member.isfile() or member.name in observed:
                    raise RuntimeError('Non-file or duplicate archive member')
                observed.add(member.name)
                stream = packed.extractfile(member)
                if member.name == 'EVIDENCE_MANIFEST.json':
                    data = stream.read()
                    if hashlib.sha256(data).hexdigest() != MANIFEST_SHA:
                        raise RuntimeError('Manifest identity differs')
                    manifest = json.loads(data)
                elif member.name in {'recover_evidence.py', 'RECOVERY.md'}:
                    stream.read()
                elif member.name.startswith('objects/'):
                    digest = member.name.removeprefix('objects/')
                    if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                        raise RuntimeError('Unexpected object name')
                    calculated = hashlib.sha256()
                    with (objects / digest).open('xb') as output:
                        for block in iter(lambda: stream.read(1024 * 1024), b''):
                            calculated.update(block); output.write(block)
                    if calculated.hexdigest() != digest:
                        raise RuntimeError('Object hash differs')
                else:
                    raise RuntimeError('Unexpected archive member')
        if manifest is None or {p.name for p in objects.iterdir()} != set(manifest['objects']):
            raise RuntimeError('Object set incomplete')
        for row in sorted(manifest['directories'], key=lambda row: len(safe_relative(row['path']).parts)):
            (destination / str(safe_relative(row['path']))).mkdir(parents=True, exist_ok=True)
        for row in manifest['files']:
            target = destination / str(safe_relative(row['path']))
            source = objects / row['sha256']
            if source.stat().st_size != row['bytes']:
                raise RuntimeError('Stored object size differs')
            with source.open('rb') as input_file, target.open('xb') as output_file:
                shutil.copyfileobj(input_file, output_file, 1024 * 1024)
            os.chmod(target, row['mode'])
            os.utime(target, ns=(row['mtime_ns'], row['mtime_ns']))
        for row in sorted(manifest['directories'], key=lambda row: len(safe_relative(row['path']).parts), reverse=True):
            os.chmod(destination / str(safe_relative(row['path'])), row['mode'])
    print(json.dumps(dict(restored_files=manifest['original_file_count'], restored_bytes=manifest['original_bytes'],
                          destination=str(destination), every_object_SHA256_verified=True)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive'); parser.add_argument('destination')
    args = parser.parse_args(); restore(args.archive, args.destination)
