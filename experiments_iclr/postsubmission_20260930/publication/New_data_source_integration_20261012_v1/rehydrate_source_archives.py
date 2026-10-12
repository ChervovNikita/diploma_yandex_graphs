"""Verify or restore exact source metadata; never overwrite different bytes.

Run from any directory. By default this only verifies the source archives.
--restore reconstructs missing metadata under the project phase directory.
No code, model, dataset or scientific job is executed.
"""
import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath


def confined(root, relative):
    rel = PurePosixPath(relative)
    if rel.is_absolute() or '..' in rel.parts or not rel.parts:
        raise ValueError('Require a project-relative source path')
    path = root.joinpath(*rel.parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Source path escapes project')
    cursor = root
    for part in rel.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError('Source symlinks are not supported')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--restore', action='store_true')
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    root = here.parent.parent
    index = json.loads((here / 'RESTORATION_INDEX.json').read_text())
    prepared = []
    total = 0
    for entry in index['archives']:
        wrapper_path = confined(root, entry['archive_project_relative_path'])
        wrapper_bytes = wrapper_path.read_bytes()
        if hashlib.sha256(wrapper_bytes).hexdigest() != entry['archive_sha256']:
            raise ValueError('Source archive digest mismatch')
        wrapper = json.loads(wrapper_bytes)
        if wrapper['schema'] != 'exact-project-source-archive-v1' or wrapper['encoding'] != 'base64-gzip':
            raise ValueError('Unknown source archive encoding')
        for key in ('original_project_relative_path', 'original_bytes', 'original_sha256'):
            if wrapper[key] != entry[key]:
                raise ValueError('Source archive index mismatch')
        raw = gzip.decompress(base64.b64decode(wrapper['payload'], validate=True))
        if len(raw) != entry['original_bytes'] or hashlib.sha256(raw).hexdigest() != entry['original_sha256']:
            raise ValueError('Restored source bytes mismatch')
        target = confined(root, entry['original_project_relative_path'])
        exists = target.exists()
        if exists and (not target.is_file() or target.read_bytes() != raw):
            raise ValueError('Refuse to overwrite different source bytes: ' + str(target))
        prepared.append((target, raw, exists))
        total += len(raw)
    # Check all sources before creating any missing file.
    restored = 0
    if args.restore:
        for target, raw, exists in prepared:
            if not exists:
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as stream:
                    stream.write(raw)
                restored += 1
    print(json.dumps({'verified_archives': len(prepared), 'exact_source_bytes': total,
                      'restored_files': restored, 'scientific_jobs_started': 0}))


if __name__ == '__main__':
    main()
