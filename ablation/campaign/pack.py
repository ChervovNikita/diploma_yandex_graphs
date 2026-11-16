"""Prepare checksum-verified repository and conda archives after source freeze.

Default execution prints the proposed package contents and commands. --execute
writes archives inside the repository. The repository archive contains only
manifest-bound source/data/cache files and explicitly selected text receipts.
"""

import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile

from .offline import REPOSITORY, digest_file, inside_repository, record
from .schema import relative_path
from .store import atomic_json

RECEIPT_SUFFIXES = {'.md', '.json', '.csv', '.txt', '.log'}


def collect_files(manifest_path, includes):
    manifest_path = inside_repository(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema') != 'gnnm-campaign-offline-v1':
        raise ValueError('Expected a frozen campaign offline manifest')
    files = dict(manifest['files'])
    for name, expected in files.items():
        path = inside_repository(REPOSITORY / relative_path(name))
        if not path.is_file() or record(path) != expected:
            raise ValueError(f'Manifest-bound input changed or is missing: {name}')
    files[str(manifest_path.relative_to(REPOSITORY))] = record(manifest_path)
    for requested in includes:
        path = inside_repository(requested)
        if not path.exists():
            raise FileNotFoundError(f'Supplemental receipt is missing: {path}')
        if path.is_file() and path.suffix not in RECEIPT_SUFFIXES:
            raise ValueError(f'Supplemental receipts must be text evidence: {path}')
        candidates = [path] if path.is_file() else sorted(path.rglob('*'))
        for candidate in candidates:
            if not candidate.is_file() or candidate.suffix not in RECEIPT_SUFFIXES:
                continue
            if '.git' in candidate.parts or '__pycache__' in candidate.parts:
                continue
            safe = inside_repository(candidate)
            files[str(candidate.relative_to(REPOSITORY))] = record(safe)
    return dict(sorted(files.items()))


def verify_repository_archive(path, files, prefix):
    expected = {prefix + '/' + name: value for name, value in files.items()}
    seen = set()
    with tarfile.open(path, 'r:gz') as archive:
        for member in archive:
            if not member.isfile() or member.name not in expected or member.name in seen:
                raise ValueError(f'Unexpected archive member: {member.name}')
            stream = archive.extractfile(member)
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                digest.update(block)
            actual = {'bytes': member.size, 'sha256': digest.hexdigest()}
            if actual != expected[member.name]:
                raise ValueError(f'Archived input differs from its manifest: {member.name}')
            seen.add(member.name)
    if seen != expected.keys():
        raise ValueError('Archive is missing manifest-bound inputs')
    return len(seen)


def repository_archive(destination, files, prefix):
    destination = inside_repository(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f'Archive already exists: {destination}')
    with tempfile.NamedTemporaryFile(prefix='.repository-', suffix='.tar.gz',
                                     dir=destination.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with tarfile.open(temporary, 'w:gz', compresslevel=1, dereference=True) as archive:
            for name in files:
                path = inside_repository(REPOSITORY / relative_path(name))
                info = archive.gettarinfo(str(path), arcname=prefix + '/' + name)
                info.uid = info.gid = 0
                info.uname = info.gname = ''
                with path.open('rb') as stream:
                    archive.addfile(info, stream)
        members = verify_repository_archive(temporary, files, prefix)
        temporary.rename(destination)
        return dict(path=str(destination.relative_to(REPOSITORY)), members=members, **record(destination))
    finally:
        temporary.unlink(missing_ok=True)


def environment_archive(environment, destination, threads):
    import conda_pack
    destination = inside_repository(destination)
    if destination.exists():
        raise FileExistsError(f'Environment archive already exists: {destination}')
    temporary = destination.with_name('.' + destination.name + '.building')
    if temporary.exists():
        raise FileExistsError(f'An earlier environment packing output exists: {temporary}')
    try:
        conda_pack.pack(prefix=str(environment), output=str(temporary), format='tar.gz',
                        n_threads=threads, compress_level=1, force=False)
        # Read every member to verify gzip/tar integrity before publishing.
        count = 0
        names = set()
        with tarfile.open(temporary, 'r:gz') as archive:
            for member in archive:
                names.add(member.name)
                if member.isfile():
                    stream = archive.extractfile(member)
                    for _ in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                        pass
                count += 1
        if not any(name.startswith('conda-meta/') for name in names) or 'bin/conda-unpack' not in names:
            raise ValueError('Packed environment lacks conda metadata or relocation helper')
        temporary.rename(destination)
        return dict(path=str(destination.relative_to(REPOSITORY)), members=count, **record(destination))
    finally:
        temporary.unlink(missing_ok=True)


def main(args):
    # Detect packaging dependencies before creating either large archive.
    import conda_pack
    manifest_path = inside_repository(args.manifest)
    destination = inside_repository(args.output_dir)
    environment = args.environment.resolve()
    if not (environment / 'conda-meta').is_dir() or not (environment / 'bin/python').is_file():
        raise ValueError('Select the existing conda environment prefix')
    if args.threads < 1:
        raise ValueError('Packing thread count must be positive')
    files = collect_files(manifest_path, args.include)
    proposed = {
        'schema': 'gnnm-campaign-transfer-package-v1', 'executed': args.execute,
        'manifest': str(manifest_path.relative_to(REPOSITORY)),
        'manifest_sha256': digest_file(manifest_path),
        'repository_files': len(files), 'repository_input_bytes': sum(row['bytes'] for row in files.values()),
        'repository_archive': str((destination / 'gnnm-rebuttal-repository.tar.gz').relative_to(REPOSITORY)),
        'environment_archive': str((destination / 'gnnm-rebuttal-conda.tar.gz').relative_to(REPOSITORY)),
        'environment_prefix': str(environment), 'packing_threads': args.threads,
    }
    print(json.dumps(proposed, indent=2, sort_keys=True), flush=True)
    if not args.execute:
        return 0
    destination.mkdir(parents=True, exist_ok=True)
    receipt_path = destination / 'transfer_receipt.json'
    if receipt_path.exists():
        existing = json.loads(receipt_path.read_text())
        if existing['manifest_sha256'] != proposed['manifest_sha256']:
            raise ValueError('Existing package receipt belongs to another frozen manifest')
        for kind in ('repository', 'environment'):
            saved = existing[kind]
            if record(inside_repository(REPOSITORY / relative_path(saved['path']))) != {
                    key: saved[key] for key in ('bytes', 'sha256')}:
                raise ValueError(f'Existing {kind} archive differs from its receipt')
        print(json.dumps(existing, indent=2, sort_keys=True))
        return 0
    atomic_json(destination / 'repository_inputs.json', files)
    repository_path = destination / 'gnnm-rebuttal-repository.tar.gz'
    if repository_path.exists():
        members = verify_repository_archive(repository_path, files, REPOSITORY.name)
        repository = dict(path=str(repository_path.relative_to(REPOSITORY)), members=members,
                          **record(repository_path))
    else:
        repository = repository_archive(repository_path, files, REPOSITORY.name)
    atomic_json(destination / 'repository_receipt.json', repository)
    environment_record = environment_archive(environment, destination / 'gnnm-rebuttal-conda.tar.gz', args.threads)
    receipt = dict(proposed, repository=repository, environment=environment_record,
                   destination_driver_qualification='pending', destination_cross_host_lock_qualification='pending')
    atomic_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--manifest', type=Path, required=True)
    result.add_argument('--environment', type=Path, required=True)
    result.add_argument('--output-dir', type=Path, default=REPOSITORY / 'transfer_packages')
    result.add_argument('--include', type=Path, nargs='*', default=[])
    result.add_argument('--threads', type=int, default=4)
    result.add_argument('--execute', action='store_true')
    return result


if __name__ == '__main__':
    raise SystemExit(main(parser().parse_args()))
