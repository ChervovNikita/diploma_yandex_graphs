"""Exercise exact builder cache/registration AST with tiny standard-library files."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile

OUT = Path(__file__).resolve().parents[1]
tree = ast.parse((OUT / 'runtime/build_final_master_sync.py').read_text())
main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
cached = next(n for n in main.body if isinstance(n, ast.FunctionDef) and n.name == 'cached')
registration = next(n for n in main.body if isinstance(n, ast.For) and 'external_allowed' in ast.unparse(n))


def checked(row):
    return {key: row[key] for key in ('path', 'bytes', 'sha256')}


def receipt(root, relative):
    rel = Path(relative)
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError('Confined receipt required')
    raw = (Path(root) / rel).read_bytes()
    return {'path': str(relative), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def reject(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError('Undeclared root/path/provider was accepted')


with tempfile.TemporaryDirectory(dir=OUT / 'tests') as tmp:
    repository = Path(tmp) / 'repo'
    root = repository / 'phase'
    root.mkdir(parents=True)
    provider = '.venv/lib/python3.11/site-packages/ogb/fixture_provider.py'
    path = repository / provider
    path.parent.mkdir(parents=True)
    path.write_bytes(b'opaque tiny provider fixture; never imported\n')
    expected = receipt(repository, provider)
    (root / 'phase_source.py').write_bytes(b'phase fixture\n')
    scope = {'root': root, 'Path': Path, 'cache': {}, 'calls': Counter(), 'external_allowed': {}, 'external_cache': {},
             'receipt': receipt, 'checked': checked,
             'studies': [{'runs': [{'kind': 'author_cli_group', 'cli': {'execution_repository': str(repository), 'runtime_provider_files': [expected]}}]}]}
    exec(compile(ast.Module(body=[cached, registration], type_ignores=[]), 'actual_builder_cache_AST', 'exec'), scope)
    assert scope['cached'](repository, provider) == expected
    assert scope['cached'](repository, provider) == expected
    assert scope['cached'](root, 'phase_source.py') == receipt(root, 'phase_source.py')
    assert scope['cached'](root, 'phase_source.py') == receipt(root, 'phase_source.py')
    assert max(scope['calls'].values()) == 1 and len(scope['calls']) == 2
    reject(lambda: scope['cached'](Path(tmp) / 'arbitrary_root', provider))
    reject(lambda: scope['cached'](repository, '.venv/lib/unlisted_provider.py'))
    changed = dict(expected, sha256='0' * 64)
    scope['external_allowed'][(str(repository), provider)] = changed
    reject(lambda: scope['cached'](repository, provider))
    scope['studies'] = [{'runs': [{'kind': 'author_cli_group', 'cli': {'execution_repository': str(Path(tmp) / 'foreign'), 'runtime_provider_files': [expected]}}]}]
    reject(lambda: exec(compile(ast.Module(body=[registration], type_ignores=[]), 'actual_registration_AST', 'exec'), scope))
    scope['studies'] = [{'runs': [{'kind': 'author_cli_group', 'cli': {'execution_repository': str(repository), 'runtime_provider_files': [dict(expected, path='nonvenv.py')]}}]}]
    reject(lambda: exec(compile(ast.Module(body=[registration], type_ignores=[]), 'actual_registration_AST', 'exec'), scope))
print(json.dumps({'actual_builder_cache_and_allowlist_registration_AST_exercised': True,
    'phase_and_exact_author_provider_each_hashed_once': True, 'arbitrary_roots_rejected': True,
    'unlisted_provider_paths_rejected': True, 'changed_receipt_rejected': True,
    'foreign_repository_and_nonvenv_provider_rejected': True, 'fixtures': 7,
    'numerical_imports': 0, 'scientific_arrays': 0, 'models': 0, 'fits': 0, 'GO': False}, sort_keys=True))
