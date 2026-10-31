"""UNEXECUTED library-only allowlist image assembler; no install/import/compile.

Called only after root separately provisions/reviews a CPython standalone tree
and the candidate venv. Copies those exact library trees, selected system/driver
library paths, and a minimal passwd file into a new repo-owned image. It never
copies host root, project, home, archives, inputs, credentials or output trees.
Relocation and native shared-library closure remain mandatory runtime gates.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


def checked_prefix(value):
    path = Path(value)
    if path.is_symlink() or not path.is_dir():
        raise ValueError('Runtime prefix must be a dedicated actual directory')
    path = path.resolve()
    if path in (Path('/'), Path('/usr'), Path('/home'), Path('/root')):
        raise ValueError('A broad host directory cannot be a runtime prefix')
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--python-prefix', required=True)
    parser.add_argument('--environment', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--library-tree', action='append', default=[])
    parser.add_argument('--library-file', action='append', default=[])
    args = parser.parse_args()
    source_root = Path(__file__).resolve().parent
    image = Path(args.image).resolve()
    image.relative_to(source_root.parent)
    if image.exists() or image.is_symlink() or source_root in image.parents:
        raise ValueError('Image must be a new directory under this project, outside source packet')
    python = checked_prefix(args.python_prefix)
    environment = checked_prefix(args.environment)
    if any(path == image or path in image.parents or image in path.parents for path in (python, environment)):
        raise ValueError('Source prefixes and image destination must be disjoint')
    if not (python / 'bin/python3.12').exists() or not (environment / 'pyvenv.cfg').is_file():
        raise ValueError('Standalone CPython3.12 and dedicated candidate venv required')
    if (python / 'bin').is_symlink() or (environment / 'bin').is_symlink() or (environment / 'pyvenv.cfg').is_symlink():
        raise ValueError('Interpreter/config launch directories must not be symlinked source prefixes')
    if any(name in {path.name for path in environment.iterdir()} for name in ('data', 'targets.csv', 'checkpoints')):
        raise ValueError('Candidate environment contains project/data artifacts')
    image.mkdir(mode=0o700)
    for name in ('opt', 'etc', 'inputs', 'sources', 'outputs', 'tmp', 'dev', 'proc', 'home', 'root', 'var/empty'):
        (image / name).mkdir(parents=True, exist_ok=True)
    shutil.copytree(python, image / 'opt/python', symlinks=True)
    shutil.copytree(environment, image / 'opt/native', symlinks=True)
    # Only interpreter launch metadata is relocated. Package/wheel bytes stay unchanged.
    for name in ('python', 'python3', 'python3.12'):
        path = image / 'opt/native/bin' / name
        if path.exists() or path.is_symlink():
            path.unlink()
        path.symlink_to('/opt/python/bin/python3.12')
    cfg = image / 'opt/native/pyvenv.cfg'
    lines = cfg.read_text().splitlines()
    rewritten = []
    for line in lines:
        key = line.split('=', 1)[0].strip()
        if key == 'home':
            line = 'home = /opt/python/bin'
        elif key == 'executable':
            line = 'executable = /opt/python/bin/python3.12'
        elif key == 'include-system-site-packages':
            line = 'include-system-site-packages = false'
        rewritten.append(line)
    cfg.write_text('\n'.join(rewritten) + '\n')
    allowed = ('/lib', '/lib64', '/usr/lib', '/usr/lib64', '/usr/local/nvidia/lib', '/usr/local/nvidia/lib64')
    records = []
    for raw in args.library_tree + args.library_file:
        path = Path(raw).absolute()
        if '..' in path.parts:
            raise ValueError('Library paths must not contain parent traversal')
        if not any(str(path) == prefix or str(path).startswith(prefix + '/') for prefix in allowed):
            raise ValueError('Only explicit system/driver library paths may be copied')
        if not path.exists():
            raise ValueError('Required library source path is absent: ' + str(path))
        target = image / path.relative_to('/')
        try:
            target.parent.resolve().relative_to(image)
        except ValueError as error:
            raise ValueError('Library destination traverses an absolute image symlink; copy its canonical target path instead') from error
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            raise ValueError('Overlapping runtime library targets require a revised explicit list')
        if path.is_symlink():
            target.symlink_to(os.readlink(path))
        elif path.is_dir():
            shutil.copytree(path, target, symlinks=True)
        else:
            shutil.copy2(path, target)
        records.append({'source': str(path), 'target': str(path), 'symlink': path.is_symlink()})
    (image / 'etc/passwd').write_text('root:x:0:0:closed-worker:/var/empty:/bin/false\n')
    (image / 'etc/group').write_text('root:x:0:\n')
    payload = []
    for path in sorted(image.rglob('*')):
        if path.is_symlink():
            payload.append({'path': path.relative_to(image).as_posix(), 'symlink': os.readlink(path)})
        elif path.is_file():
            digest = hashlib.sha256()
            with path.open('rb') as file:
                for block in iter(lambda: file.read(1024 * 1024), b''):
                    digest.update(block)
            payload.append({'path': path.relative_to(image).as_posix(), 'bytes': path.stat().st_size,
                            'sha256': digest.hexdigest()})
        elif not path.is_dir():
            raise ValueError('Runtime image contains a special file')
    (image / 'RUNTIME_IMAGE_MANIFEST.json').write_text(json.dumps({
        'schema': 'candidate-allowlist-runtime-image-v2', 'python_source': str(python),
        'environment_source': str(environment), 'system_library_copies': records, 'payload': payload,
        'installation_performed': False, 'model_import_performed': False,
        'relocation_runtime_qualified': False, 'no_project_or_dataset_tree_copy': True}, indent=2) + '\n')


if __name__ == '__main__':
    main()
