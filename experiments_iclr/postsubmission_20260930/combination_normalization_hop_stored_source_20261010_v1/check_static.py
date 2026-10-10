"""Stdlib source/metadata checks only; never import the diagnostic worker."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    tree = ast.parse((HERE/'collect.py').read_text())
    funcs = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [item.name for item in node.names] if isinstance(node, ast.Import) else [node.module]
            assert not any(name in ('numpy', 'torch') for name in names)
    text = (HERE/'collect.py').read_text()
    assert 'ENABLED = False' in text
    assert 'normread.closed_science(' in text and 'diag.closed18(' in text and 'diag.selected(' in text
    assert 'ic.closed_stage(' in text and 'paired.compare(' in text and 'reader.masks(' in text
    assert all(word not in text for word in ('Session(', 'optimizer.step(', 'torch.load(', 'Popen(', 'def own_one'))
    calls = [n for n in funcs['main'].body if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)]
    assert any(isinstance(n.value.func, ast.Name) and n.value.func.id == 'admit' for n in calls)
    assert any(isinstance(n, ast.Import) and any(alias.name == 'numpy' for alias in n.names) for n in funcs['execute'].body)
    fixed = json.loads((HERE/'FIXED_SPEC.json').read_text())
    assert fixed['enabled'] is False and fixed['execution_authorized'] is False
    assert fixed['seeds'] == [9101, 9203, 9307]
    assert fixed['banks'] == {'baseline': 'shared4_own', 'A': 'shared4_own_M_normalized', 'B': 'private_missinghop', 'controls': ['full_aux', 'common_nonfull']}
    assert fixed['eligibility']['minimum_nodes_each_seed'] == 4
    assert fixed['eligibility']['minimum_persistent_nodes'] == 1
    assert fixed['eligibility']['minimum_persistent_support'] == 2
    for file in HERE.glob('*TEMPLATE_DISABLED.json'):
        value = json.loads(file.read_text())
        assert value.get('enabled', value.get('approved')) is False
    for file in HERE.glob('*.json'):
        json.loads(file.read_text())
    bindings = json.loads((HERE/'SOURCE_BINDINGS.json').read_text())
    for row in bindings['files']:
        path = (PHASE/row['path']).resolve(strict=True)
        assert path.is_relative_to(PHASE) and sha(path) == row['sha256'] and path.stat().st_size == row['bytes']
    assert sha(HERE/'FIXED_SPEC.json') == bindings['fixed_spec_sha256']
    manifest_path = HERE/'SOURCE_MANIFEST.json'
    if manifest_path.exists():
        for row in json.loads(manifest_path.read_text())['files']:
            path = HERE/row['path']
            assert sha(path) == row['sha256'] and path.stat().st_size == row['bytes']
    print(json.dumps(dict(passed=True, checks='AST/JSON/hash only', numerical_imports=False,
                         numerical_execution=False, payload_reads=False, jobs=False)))


if __name__ == '__main__':
    main()
