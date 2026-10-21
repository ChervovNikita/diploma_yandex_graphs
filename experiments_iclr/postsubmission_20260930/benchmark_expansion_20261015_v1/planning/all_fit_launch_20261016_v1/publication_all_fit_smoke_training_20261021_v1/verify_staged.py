"""Verify a plaintext publication package without importing any staged executable."""
import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath


def require(ok, message):
    if not ok:
        raise ValueError(message)


def verify(package, expected_manifest_sha256):
    package = Path(package).resolve()
    manifest_raw = (package / 'PACKAGE_MANIFEST.json').read_bytes()
    require(hashlib.sha256(manifest_raw).hexdigest() == expected_manifest_sha256, 'Manifest identity mismatch')
    manifest = json.loads(manifest_raw)
    rows = manifest['files']
    require(len(rows) == len({row['path'] for row in rows}), 'Duplicate package path')
    expected = {row['path'] for row in rows} | {'PACKAGE_MANIFEST.json'}
    actual = {path.relative_to(package).as_posix() for path in package.rglob('*') if path.is_file()}
    require(actual == expected, 'Unexpected or missing package file')
    total = 0
    for row in rows:
        relative = PurePosixPath(row['path'])
        require(str(relative) == row['path'] and not relative.is_absolute() and
                not any(part in {'.', '..', '.git'} for part in relative.parts), 'Unsafe package path')
        path = package / relative
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(package),
                'Missing, linked or escaped package file')
        raw = path.read_bytes()
        require(len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256'],
                'Package file identity mismatch: ' + row['path'])
        text = raw.decode('utf-8')
        if path.suffix == '.py':
            ast.parse(text, filename=str(path))
        elif path.suffix == '.json':
            json.loads(text)
        total += len(raw)
    return {'status': 'VERIFIED', 'files': len(rows), 'bytes': total,
            'manifest_sha256': expected_manifest_sha256, 'remote_mutations': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', required=True)
    parser.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.package, args.manifest_sha256)))
