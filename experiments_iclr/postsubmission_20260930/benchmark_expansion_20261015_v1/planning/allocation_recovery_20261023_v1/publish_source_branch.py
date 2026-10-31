"""Publish reviewed source from an explicit parent; preserve the live checkout.

Root invokes this on the verified project. No checkout/reset/add/push/model
operation is performed. The commit parent and expected live HEAD are distinct.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_relative(value):
    require(isinstance(value, str) and value and not any(c in value for c in '\x00\n\r\\'), 'Safe relative POSIX path required')
    path = PurePosixPath(value)
    require(not path.is_absolute() and str(path) == value and not any(p in ('.', '..', '.git') for p in path.parts), 'Normalized project path required')
    return path


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def publish(root, selection, receipt):
    root = Path(root).resolve()
    selection = Path(selection).resolve()
    receipt = Path(receipt).resolve()
    require(selection.is_relative_to(root) and receipt.is_relative_to(root), 'Project-local control files required')
    require(not receipt.exists(), 'Preserve previous publication receipt')
    selection_raw = selection.read_bytes()
    value = json.loads(selection_raw)
    require(value.get('reviewed') is True and value.get('publication_only') is True, 'Reviewed source publication required')
    require(value.get('conditional_full_training_GO') is True, 'Record the existing conditional user GO honestly')
    branch = value['branch']
    require(branch.startswith('codex/'), 'Distinct codex/ branch required')
    parent = value['expected_parent']
    expected_live = value['expected_live_head']
    require(all(re.fullmatch('[0-9a-f]{40}', item) for item in (parent, expected_live)), 'Exact full commit IDs required')
    prefix = safe_relative(value['allowed_target_prefix'])
    rows = value['files']
    require(rows and len(rows) == len({row['path'] for row in rows}), 'Nonempty unique selection required')
    snapshots = []
    for row in rows:
        target = safe_relative(row['path'])
        source = safe_relative(row.get('source_path', row['path']))
        require(target.is_relative_to(prefix), 'Target outside authorized publication scope')
        require(target.suffix in {'.py', '.md', '.json', '.txt', '.csv', '.patch'}, 'Source/compact text only')
        require(not any(part in {'results', 'datasets', 'checkpoints', 'provider_cache', 'runs', 'smokes'} or part.startswith('generated') for part in target.parts), 'No runtime payloads in source publication')
        path = root / source
        require(not path.is_symlink() and path.resolve().is_relative_to(root), 'No symlink or escaped source')
        raw = path.read_bytes()
        require(len(raw) == row['bytes'] and sha(raw) == row['sha256'] and len(raw) <= 2_000_000, 'Reviewed bytes changed or too large: ' + str(source))
        raw.decode('utf-8')
        require(b'PRIVATE ' + b'KEY-----' not in raw, 'Private key in selection')
        require(not re.search(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})', raw), 'Access token in selection')
        if target.suffix == '.py':
            ast.parse(raw.decode(), filename=str(target))
        elif target.suffix == '.json':
            json.loads(raw)
        snapshots.append((str(target), raw))
    authorization = value['authorization']
    require(any(row['path'] == authorization['path'] and all(row[k] == authorization[k] for k in ('bytes', 'sha256')) for row in rows), 'Selected exact user authorization required')
    auth = json.loads(dict(snapshots)[authorization['path']])
    require(auth.get('training_authorized_after_all_fit_smokes_pass') is True and auth.get('expected_training_fits') == 6460, 'Preserve conditional all-fit user authorization')

    def git(*args, env=None, data=None):
        result = subprocess.check_output(['git', *args], cwd=root, env=env, input=data, text=not isinstance(data, bytes))
        return (result.decode() if isinstance(result, bytes) else result).rstrip('\n')

    git('check-ref-format', 'refs/heads/' + branch)
    head = git('rev-parse', 'HEAD')
    symbolic = git('symbolic-ref', 'HEAD')
    require(head == expected_live and symbolic != 'refs/heads/' + branch, 'Live HEAD changed or destination is live')
    require(git('rev-parse', parent + '^{commit}') == parent, 'Explicit publication parent missing or changed')
    existing = subprocess.run(['git', 'show-ref', '--verify', '--quiet', 'refs/heads/' + branch], cwd=root)
    require(existing.returncode == 1, 'A fresh publication branch is required')
    index = Path(git('rev-parse', '--git-path', 'index'))
    if not index.is_absolute():
        index = root / index
    index_before = sha(index.read_bytes()) if index.exists() else None
    temporary = receipt.with_suffix('.index')
    require(not temporary.exists(), 'Preserve existing temporary index')
    env = dict(os.environ, GIT_INDEX_FILE=str(temporary))
    git('read-tree', parent, env=env)
    # Commit the verified in-memory snapshot; later file mutation cannot enter it.
    for relative, raw in snapshots:
        blob = git('hash-object', '-w', '--stdin', data=raw)
        git('update-index', '--add', '--cacheinfo', '100644,' + blob + ',' + relative, env=env)
    tree = git('write-tree', env=env)
    names = [name for name in git('diff-tree', '--no-commit-id', '--name-only', '-r', '-z', parent, tree).split('\x00') if name]
    require(names and set(names).issubset({row['path'] for row in rows}), 'Unexpected tree changes or empty commit')
    commit = git('commit-tree', tree, '-p', parent, data=value['message'] + '\n')
    require(git('rev-parse', 'HEAD') == head and git('symbolic-ref', 'HEAD') == symbolic, 'Live HEAD changed during publication')
    index_after = sha(index.read_bytes()) if index.exists() else None
    require(index_before == index_after, 'Live index changed during publication')
    git('update-ref', 'refs/heads/' + branch, commit, '0' * 40)
    result = {'commit': commit, 'branch': branch, 'parent': parent, 'live_head': head,
        'live_head_unchanged': True, 'live_index_unchanged': True, 'working_tree_checkout_performed': False,
        'changed_files': names, 'conditional_full_training_GO': True, 'publication_only': True,
        'training_launch_performed': False, 'selection_sha256': sha(selection_raw)}
    with receipt.open('x') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write('\n')
    temporary.unlink()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'selection', 'receipt'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    print(json.dumps(publish(args.root, args.selection, args.receipt)))
