"""AST/JSON/hash bookkeeping only; no numerical imports or execution."""
import ast
import hashlib
import json
from pathlib import Path

D = Path(__file__).resolve().parent
checks = []
for path in sorted(D.glob('*.py')):
    tree = ast.parse(path.read_text(), filename=str(path))
    checks.append({'file': path.name, 'AST': 'PASS'})
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = (node.module or '') if isinstance(node, ast.ImportFrom) else ','.join(a.name for a in node.names)
            assert not any(name in module.split(',') for name in ('numpy', 'torch', 'torch_geometric'))
    if path.name in ('export_neighbors.py', 'study.py', 'operators.py', 'support.py'):
        for node in tree.body:
            assert not isinstance(node, ast.Expr) or isinstance(node.value, ast.Constant), 'No top-level runtime calls'
support = json.loads((D / 'FROZEN_SUPPORT.json').read_text())
assert len(support['families']) == 3
assert sum(len(f['banks']) for f in support['families']) == 45
assert sum(len(b['checkpoints']) for f in support['families'] for b in f['banks']) == 99
assert sum(b['members'] for f in support['families'] for b in f['banks']) == 126
assert sum((2 if b['members'] == 1 else 5) * 5 for f in support['families'] for b in f['banks']) == 855
for row in support['source_files']:
    assert hashlib.sha256((D.parent / row['path']).read_bytes()).hexdigest() == row['sha256']
for row in json.loads((D / 'SOURCE_BINDINGS.json').read_text())['inputs']:
    assert hashlib.sha256((D.parent / row['path']).read_bytes()).hexdigest() == row['sha256']
result = {'PASS': True, 'checks': checks, 'support_hashes': 'PASS',
          'banks': 45, 'selected_checkpoint_units': 99, 'fullgraph_member_trajectories': 126,
          'nonduplicate_small_head_fits': 855, 'maximum_small_head_updates': 427500,
          'scope': 'Static syntax/roster/hash validation only; numerical integration, restoration, inference and optimization remain unexecuted.'}
(D / 'STATIC_CHECK.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
