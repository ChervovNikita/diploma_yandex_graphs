"""Recover the two verified audit files into a NEW phase directory."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tarfile

MANIFEST_SHA = '61c5ca6d6c744aa0a5d2ed7d0aa886fa57e839f5673da4b8c9c7698ae258d5bb'


def restore(archive, destination):
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink() or destination.parent.resolve() != destination.parent:
        raise RuntimeError('Choose a new directory under an existing canonical parent')
    destination.mkdir(mode=0o700); manifest = None; observed = set(); restored = set()
    with tarfile.open(archive, mode='r|gz') as packed:
        for member in packed:
            if not member.isfile() or member.name in observed:
                raise RuntimeError('Unexpected or duplicate archive member')
            observed.add(member.name); source = packed.extractfile(member)
            if member.name == 'EXACT_FILE_MANIFEST.json':
                data = source.read()
                if hashlib.sha256(data).hexdigest() != MANIFEST_SHA:
                    raise RuntimeError('Manifest identity differs')
                manifest = json.loads(data)
            elif member.name in {'recover_large_audits.py', 'RECOVERY.txt'}:
                source.read()
            elif member.name.startswith('files/') and manifest is not None:
                name = member.name.removeprefix('files/')
                row = next(row for row in manifest['files'] if Path(row['path']).name == name)
                relative = PurePosixPath(row['phase_relative_path'])
                if relative.is_absolute() or '..' in relative.parts or str(relative) != row['phase_relative_path']:
                    raise RuntimeError('Unsafe recovery path')
                target = destination / str(relative); target.parent.mkdir(parents=True, exist_ok=True)
                sha = hashlib.sha256(); count = 0
                with target.open('xb') as output:
                    for block in iter(lambda: source.read(1024 * 1024), b''):
                        output.write(block); sha.update(block); count += len(block)
                if count != row['bytes'] or sha.hexdigest() != row['sha256']:
                    raise RuntimeError('Recovered bytes or SHA256 differ')
                os.chmod(target, row['mode']); os.utime(target, ns=(row['mtime_ns'], row['mtime_ns']))
                restored.add(name)
            else:
                raise RuntimeError('Unexpected archive member')
    if manifest is None or restored != {Path(row['path']).name for row in manifest['files']}:
        raise RuntimeError('Recovery file set differs')
    print(json.dumps(dict(restored_files=len(restored), restored_bytes=manifest['bytes'], every_SHA256_verified=True, destination=str(destination))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('archive'); parser.add_argument('new_phase_directory')
    args = parser.parse_args(); restore(args.archive, args.new_phase_directory)
