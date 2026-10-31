"""Freeze a compact local source publication once; perform no remote or Git action."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

HERE = Path(__file__).resolve().parent
PHASE = HERE.parents[3]
PREFIX = PurePosixPath('experiments_iclr/postsubmission_20260930')
PARENT = '6adbcf17b706f73a1cd878c20e046ba3f22fee39'
LIVE_HEAD = '48329b19941ea869c439e5a220a3df0e30aa1237'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def identity(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def safe_path(value):
    path = PurePosixPath(value)
    require(str(path) == value and not path.is_absolute() and
            not any(part in {'.', '..', '.git'} for part in path.parts) and
            not any(c in value for c in '\x00\n\r\\'), 'Unsafe selected path')
    return path


def prepare():
    require(not (HERE / 'snapshot').exists(), 'Existing immutable snapshot must be preserved')
    scope_raw = (HERE / 'SELECTED_SCOPE.json').read_bytes()
    scope = json.loads(scope_raw)
    require(scope['expected_parent'] == PARENT, 'Unexpected publication parent')
    paths = scope['phase_sources']
    require(paths and len(paths) == len(set(paths)), 'Unique nonempty selection required')
    values = []
    for value in sorted(paths):
        relative = safe_path(value)
        require(relative.suffix in {'.py', '.md', '.json', '.txt', '.csv', '.patch'}, 'Compact text only')
        require(not any(part in {'results', 'datasets', 'checkpoints', 'provider_cache', 'runs', 'smokes'}
                        or part.startswith('generated') for part in relative.parts), 'Runtime payload selected')
        source = PHASE / relative
        require(source.is_file() and not source.is_symlink() and source.resolve().is_relative_to(PHASE),
                'Missing, linked or escaped source: ' + value)
        raw = source.read_bytes()
        require(len(raw) <= 2_000_000 and raw.endswith(b'\n'), 'Oversized or unterminated text: ' + value)
        text = raw.decode('utf-8')
        require(b'PRIVATE ' + b'KEY-----' not in raw, 'Private key selected')
        require(not re.search(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})', raw),
                'Access token selected')
        if relative.suffix == '.py':
            ast.parse(text, filename=value)
        elif relative.suffix == '.json':
            json.loads(text)
        values.append((relative, raw))
    rows, staging = [], []
    for relative, raw in values:
        destination = HERE / 'snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as stream:
            stream.write(raw)
        destination.chmod(0o444)
        source_relative = destination.relative_to(PHASE).as_posix()
        row = {'path': str(PREFIX / relative), 'source_path': str(PREFIX / source_relative), **identity(raw)}
        rows.append(row)
        staging.append({'phase_source': source_relative, 'target_relative': row['path'], **identity(raw)})
    auth_target = str(PREFIX / 'benchmark_expansion_20261015_v1/planning/all_fit_launch_20261016_v1/USER_GO.json')
    auth_row = next(row for row in rows if row['path'] == auth_target)
    authorization = {k: auth_row[k] for k in ('path', 'bytes', 'sha256')}
    selection = {'reviewed': True, 'publication_only': True, 'conditional_full_training_GO': True,
                 'branch': 'codex/all-fit-smoke-training-20261016', 'expected_parent': PARENT,
                 'expected_branch_tip': PARENT, 'expected_live_head': LIVE_HEAD,
                 'allowed_target_prefix': str(PREFIX), 'authorization': authorization,
                 'message': 'Preserve verified all-fit launch sources and compact October 21 evidence',
                 'files': rows}
    write_json(HERE / 'SELECTION.json', selection)
    write_json(HERE / 'STAGING_MAP.json', {'files': staging, 'source_snapshot_immutable': True})
    checks = {'prepared_utc': datetime.now(timezone.utc).isoformat(), 'files': len(rows),
              'bytes': sum(row['bytes'] for row in rows), 'selection': identity((HERE / 'SELECTION.json').read_bytes()),
              'scope': identity(scope_raw), 'python_parse_and_json_decode': 'PASS',
              'remote_operations': 0, 'git_operations': 0, 'training_launch_performed': False,
              'prior_snapshots_modified': False, 'snapshot_files_read_only': True}
    write_json(HERE / 'PREPARED_SOURCES.json', checks)
    package = []
    for path in sorted(HERE.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and path.name != 'PACKAGE_MANIFEST.json':
            require(not path.is_symlink(), 'Package symlink rejected')
            package.append({'path': path.relative_to(HERE).as_posix(), **identity(path.read_bytes())})
    write_json(HERE / 'PACKAGE_MANIFEST.json', {'schema': 'compact-plaintext-publication-v1', 'files': package})
    return checks


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(prepare(), indent=2))
