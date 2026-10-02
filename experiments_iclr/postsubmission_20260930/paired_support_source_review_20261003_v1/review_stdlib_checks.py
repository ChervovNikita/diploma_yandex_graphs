"""Independent read-only source checks; writes receipts only beside this file."""
import ast
from contextlib import redirect_stdout
from fractions import Fraction as Q
import hashlib
import io
import json
from pathlib import Path

review = Path(__file__).resolve().parent
packet = review.parent / 'graph_full_node_cotangent_paired_alpha_v1'
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
checks = []


def check(name, condition):
    assert condition, name
    checks.append({'name': name, 'passed': True})


check('requested_manifest_sha256', sha(packet / 'MANIFEST.json') ==
      'd3410a4594d0b1c483aa4bb2b8e02a739eb020fbceefa68deab1738afbe60877')
check('requested_paired_source_sha256', sha(packet / 'prototype/paired_shared_alpha_initializer.py') ==
      'e3ff6f134bae639269f284c4d8b0d290a31672787fe662ee829033465ed147d0')

# Reproduce authored source/symbolic checks in memory, removing only their
# STDLIB_RESULTS write. No subject import, Torch or reviewed-file mutation.
fixture = packet / 'fixtures/stdlib_checks.py'
tree = ast.parse(fixture.read_bytes(), filename=str(fixture))
removed = []
for node in list(tree.body):
    if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == 'write_text'):
        removed.append(node.lineno)
        tree.body.remove(node)
check('only_authored_receipt_write_removed', removed == [119])
namespace = {'__file__': str(fixture), '__name__': '__source_review_checks__'}
captured = io.StringIO()
with redirect_stdout(captured):
    exec(compile(tree, str(fixture), 'exec'), namespace)
check('authored_read_only_checks_reproduced', len(namespace['checks']) == 107
      and all(row['passed'] for row in namespace['checks']))

# Exact independent row-map check. Output row r maps to graph node nodes[r].
# Use a nonidentity permutation and a nonlocal output Jacobian; no subject code.
nodes = [2, 0, 3, 1]
train_rows = [0, 3]
residual = [Q(2), Q(0), Q(0), Q(-1)]
full_residual = [Q(0)] * 4
for row, graph_node in enumerate(nodes):
    full_residual[graph_node] = residual[row]
check('nonidentity_scatter_and_gather_inverse',
      [full_residual[node] for node in nodes] == residual)
identity = [[Q(int(i == j)) for j in range(4)] for i in range(4)]
shift = [[Q(int(j == (i + 1) % 4), 8) for j in range(4)] for i in range(4)]
bands = [shift, [[-x for x in row] for row in shift], identity,
         [[Q(0)] * 4 for _ in range(4)]]
matvec = lambda matrix, vector: [sum((a * b for a, b in zip(row, vector)), Q(0))
                               for row in matrix]
graph_fields = [matvec(band, full_residual) for band in bands]
output_fields = [[field[node] for node in nodes] for field in graph_fields]
remasked_fields = [[value if row in train_rows else Q(0)
                   for row, value in enumerate(field)] for field in output_fields]
jacobian = [[Q(1), Q(2)], [Q(-1), Q(3)], [Q(4), Q(-2)], [Q(2), Q(1)]]
pullback = lambda field: [sum((jacobian[row][coordinate] * field[row]
                              for row in range(4)), Q(0)) for coordinate in range(2)]
gradient = pullback(residual)
full_h, remasked_h = list(map(pullback, output_fields)), list(map(pullback, remasked_fields))
check('nonidentity_full_gradient_sum',
      [sum(column, Q(0)) for column in zip(*full_h)] == gradient)
check('nonidentity_remasked_gradient_sum',
      [sum(column, Q(0)) for column in zip(*remasked_h)] == gradient)
check('nonlocal_pullback_support_delta', full_h != remasked_h)

manifest = json.loads((packet / 'MANIFEST.json').read_bytes())
source_hashes = {'MANIFEST.json': sha(packet / 'MANIFEST.json'),
                 'SEAL.json': sha(packet / 'SEAL.json')}
for row in manifest['payload']:
    check('payload_unchanged_after_review:' + row['path'],
          sha(packet / row['path']) == row['sha256'])
    source_hashes[row['path']] = sha(packet / row['path'])
bindings = json.loads((packet / 'NATIVE_SOURCE_BINDINGS.json').read_bytes())
bound_source_hashes = []
for row in bindings['source_files']:
    actual = sha(review.parent / row['path'])
    check('native_bound_source_hash:' + row['path'], actual == row['sha256'])
    bound_source_hashes.append({'path': row['path'], 'sha256': actual})

result = {'schema': 'independent-paired-source-review-stdlib-v1',
          'python_mode': '/usr/bin/python3 -I -S -B',
          'checks': checks, 'authored_read_only_checks': namespace['checks'],
          'authored_check_stdout': captured.getvalue(),
          'source_hashes': source_hashes, 'native_bound_source_hashes': bound_source_hashes,
          'source_imported': False, 'torch_imported': False,
          'dataset_model_checkpoint_logits_or_GPU_access': False}
(review / 'REVIEW_CHECKS.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
print(json.dumps({'passed': True, 'independent_checks': len(checks),
                  'authored_read_only_checks': len(namespace['checks']),
                  'receipt': str(review / 'REVIEW_CHECKS.json')}))
