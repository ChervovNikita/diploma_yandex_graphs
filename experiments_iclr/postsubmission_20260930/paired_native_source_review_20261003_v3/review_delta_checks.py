"""Targeted final v3 stdlib delta review; no numeric originals or native work."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys

review = Path(__file__).resolve().parent
phase = review.parent
v2 = phase / 'graph_paired_native_qualification_preparation_v2'
v3 = phase / 'graph_paired_native_qualification_preparation_v3'
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
checks = []


def check(name, condition):
    assert condition, name
    checks.append({'name': name, 'passed': True})


prior_review_hashes = {}
for version in ('v1', 'v2'):
    folder = phase / ('paired_native_source_review_20261003_' + version)
    for path in folder.iterdir():
        if path.is_file():
            prior_review_hashes[str(path.relative_to(phase))] = sha(path)
check('requested_v3_script_hash', sha(v3 / 'prototype/qualify_squirrel17.py') ==
      '3d38e56dd2b87fea3507e9b96ae5987c72283582321d123f58da2ee3df3febb1')
check('requested_v3_manifest_hash', sha(v3 / 'MANIFEST.json') ==
      'fba0de7e94eb3974fd5df40cd996bdccb816a0748c6834925c3eca3881a743c7')
for name in ('BOUND_INPUTS.json', 'RESOURCE_EVIDENCE.json'):
    check('byte_identical_to_v2:' + name, (v2 / name).read_bytes() == (v3 / name).read_bytes())
t2 = ast.parse((v2 / 'prototype/qualify_squirrel17.py').read_bytes())
t3 = ast.parse((v3 / 'prototype/qualify_squirrel17.py').read_bytes())
f2 = {n.name: n for n in t2.body if isinstance(n, ast.FunctionDef)}
f3 = {n.name: n for n in t3.body if isinstance(n, ast.FunctionDef)}
dump = lambda node: ast.dump(node, include_attributes=False)
check('only_observer_and_return_status_helpers_added', set(f3) - set(f2) ==
      {'observed_forward', 'paired_return_status'} and set(f2) - set(f3) == set())
for name in sorted(set(f2) - {'main'}):
    check('unchanged_existing_function_AST:' + name, dump(f2[name]) == dump(f3[name]))


class NormalizeMainDelta(ast.NodeTransformer):
    def __init__(self):
        self.removed = []
    def visit_Constant(self, node):
        if node.value in ('Squirrel17-native-paired-source-qualification-v2',
                          'Squirrel17-native-paired-source-qualification-v3'):
            node.value = 'schema-normalized-for-delta-review'
        return node
    def visit_FunctionDef(self, node):
        if node.name == 'observed':
            node.body = [ast.Pass()]
            return node
        return self.generic_visit(node)
    def visit_Assign(self, node):
        if any(isinstance(t, ast.Name) and t.id in ('observed_resource_failures', 'paired_status')
               for t in node.targets):
            self.removed.append(node.targets[0].id)
            return None
        return self.generic_visit(node)
    def visit_Call(self, node):
        node = self.generic_visit(node)
        for kw in list(node.keywords):
            if kw.arg in ('permutation', 'observed_resource_failures'):
                self.removed.append(kw.arg + '_receipt')
                node.keywords.remove(kw)
        return node
    def visit_If(self, node):
        if (isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Name)
                and node.test.left.id == 'paired_status'):
            self.removed.append('resource_precedence_branch')
            return self.visit(node.orelse[0])
        return self.generic_visit(node)


n2, n3 = NormalizeMainDelta(), NormalizeMainDelta()
m2, m3 = n2.visit(copy.deepcopy(f2['main'])), n3.visit(copy.deepcopy(f3['main']))
check('only_declared_v3_main_delta_normalized', n2.removed == [] and sorted(n3.removed) == sorted([
    'observed_resource_failures', 'paired_status', 'permutation_receipt',
    'resource_precedence_branch', 'observed_resource_failures_receipt']))
check('remaining_main_exact_AST', dump(m2) == dump(m3))

# Check that extracting the observer left ordinary forwarding/counting intact.
old_observed = next(n for n in ast.walk(t2) if isinstance(n, ast.FunctionDef) and n.name == 'observed')
new_observed = copy.deepcopy(f3['observed_forward'])
handler = next(n for n in new_observed.body if isinstance(n, ast.Try)).handlers[0]
check('new_observer_handler_delta_shape', len(handler.body) == 5)
del handler.body[2:4]
class NormalizeObserver(ast.NodeTransformer):
    def visit_Name(self, node):
        if node.id == 'counts':
            return ast.copy_location(ast.Name(id='closure_counts', ctx=node.ctx), node)
        if node.id == 'sink':
            return ast.copy_location(ast.Attribute(value=ast.Name(id='ledger', ctx=ast.Load()),
                                                  attr='sink', ctx=node.ctx), node)
        return node
new_body = NormalizeObserver().visit(ast.Module(body=new_observed.body, type_ignores=[]))
check('ordinary_observer_body_exact_AST', dump(new_body) ==
      dump(ast.Module(body=old_observed.body, type_ignores=[])))

spec = importlib.util.spec_from_file_location('independent_native_v3_subject',
                                             v3 / 'prototype/qualify_squirrel17.py')
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)
check('stdlib_source_import_no_Torch', 'torch' not in sys.modules)
paired_path = phase / 'graph_full_node_cotangent_paired_alpha_v1/prototype/paired_shared_alpha_initializer.py'
paired_tree = ast.parse(paired_path.read_bytes())
candidate_try = next(n for n in ast.walk(paired_tree) if isinstance(n, ast.Try) and n.lineno == 223)
post_status = next(n for n in ast.walk(t3) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'paired_status' for t in n.targets))
post_branch = next(n for n in ast.walk(t3) if isinstance(n, ast.If)
                   and isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Name)
                   and n.test.left.id == 'paired_status')
post_code = compile(ast.Module(body=[post_status, post_branch], type_ignores=[]),
                    str(v3 / 'prototype/qualify_squirrel17.py'), 'exec')

allocator = type('OutOfMemoryError', (RuntimeError,), {})
class DerivedAllocator(allocator):
    pass
wrapped_host = RuntimeError('opaque host wrapper'); wrapped_host.__cause__ = MemoryError('host cause')
wrapped_device = RuntimeError('opaque device wrapper'); wrapped_device.__cause__ = allocator('device cause')
implicit = RuntimeError('suppressed context'); implicit.__context__ = MemoryError('implicit host')
implicit.__suppress_context__ = True
cases = [('direct_host', MemoryError('direct host'), True),
         ('direct_device', allocator('direct device'), True),
         ('derived_device', DerivedAllocator('derived device'), True),
         ('wrapped_host', wrapped_host, True), ('wrapped_device', wrapped_device, True),
         ('implicit_host', implicit, True), ('nonresource', ValueError('invalid'), False)]
mock_results = []
for name, error, is_resource in cases:
    counts = {'calls_started': 0, 'calls_completed': 0, 'calls_failed': 0}
    events, failures = [], []
    def sink(event, record):
        events.append({'event': event, 'record': record})
    def unavailable(theta, captured_error=error):
        raise captured_error
    def observed(theta):
        return subject.observed_forward(unavailable, theta, counts, sink, failures)
    sandbox = {'logits_fn': observed, 'row': object(), 'outputs': [], 'errors': {}, 'name': 'full_node'}
    exec(compile(ast.Module(body=[candidate_try], type_ignores=[]), str(paired_path), 'exec'), sandbox)
    check('caught_candidate_counts:' + name, sandbox['outputs'] == [None] and
          counts == {'calls_started': 1, 'calls_completed': 0, 'calls_failed': 1})
    check('resource_sentinel_distinction:' + name, len(failures) == int(is_resource)
          and (not is_resource or failures[0]['call'] == 1))
    # A later successful closure must not clear a prior allocator receipt.
    marker = object()
    returned = subject.observed_forward(lambda theta: marker, object(), counts, sink, failures)
    check('later_success_preserves_receipt:' + name, returned is marker and
          counts == {'calls_started': 2, 'calls_completed': 1, 'calls_failed': 1}
          and len(failures) == int(is_resource))
    outcomes = []
    for outcome, slices in [('failure', None), ('acceptance', {'mock_arm': object()})]:
        expected_status = 'resource_deferred' if is_resource else (
            'joint_failure' if slices is None else 'joint_accepted')
        check('paired_return_status:' + name + ':' + outcome,
              subject.paired_return_status(slices, failures) == expected_status)
        if is_resource or slices is None:
            inner_report = {'status': 'joint_failure' if slices is None else 'joint_accepted',
                            'candidate_errors': sandbox['errors']}
            result = {'paired': inner_report, 'installation': {}}
            post_namespace = {'paired_return_status': subject.paired_return_status, 'slices': slices,
                              'observed_resource_failures': failures, 'exit_code': 0, 'result': result}
            # No paired/rt/native symbol exists: entering installation would fail.
            exec(post_code, post_namespace)
            check('actual_post_helper_precedence:' + name + ':' + outcome,
                  result['status'] == expected_status and result['installation'] == {}
                  and result['paired'] is inner_report
                  and post_namespace['exit_code'] == (1 if is_resource else 0))
            outcomes.append({'helper_outcome': outcome, 'outer_status': result['status'],
                             'exit_code': post_namespace['exit_code'], 'installation_entered': False})
    mock_results.append({'case': name, 'resource_failures': failures, 'counts_after_later_success': counts,
                         'candidate_error': sandbox['errors']['full_node'][0], 'post_helper_outcomes': outcomes})

# Verify the actual new receipt expression with a stdlib stand-in for a small
# unlabeled int64 vector. No real tensor, array or graph is used.
vector = [2, 0, 3, 1]
vector_bytes = struct.pack('=4q', *vector)
class FakePermutation:
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self
    def tobytes(self): return vector_bytes
    def tolist(self): return list(vector)
topology_assignment = next(n for n in ast.walk(t3) if isinstance(n, ast.Assign)
    and any(isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant)
            and t.slice.value == 'topology_control' for t in n.targets))
topology_namespace = {'result': {}, 'bound': {'topology_permutation_seed': 80017},
                      'hashlib': hashlib, 'permutation': FakePermutation()}
exec(compile(ast.Module(body=[topology_assignment], type_ignores=[]),
             str(v3 / 'prototype/qualify_squirrel17.py'), 'exec'), topology_namespace)
topology_receipt = topology_namespace['result']['topology_control']
check('saved_vector_hash_correspondence_mock', topology_receipt['permutation'] == vector and
      topology_receipt['permutation_sha256'] == hashlib.sha256(vector_bytes).hexdigest())

source_hashes = {name: sha(v3 / name) for name in ('MANIFEST.json', 'SEAL.json')}
for row in json.loads((v3 / 'MANIFEST.json').read_bytes())['payload']:
    actual = sha(v3 / row['path'])
    check('v3_payload_unchanged:' + row['path'], actual == row['sha256'])
    source_hashes[row['path']] = actual
for path, digest in prior_review_hashes.items():
    check('prior_review_preserved:' + path, sha(phase / path) == digest)
bound = json.loads((v3 / 'BOUND_INPUTS.json').read_bytes())
numeric = [r for r in bound['original_records'] if Path(r['path']).suffix in ('.npy','.npz','.pt','.pth')]
check('unchanged_151_descriptors_seven_numeric', len(bound['original_records']) == 151 and len(numeric) == 7)
check('TRAIN_only_runtime_labels', set(bound['context']['source_labels']) == {'train'})
check('no_Torch_import_after_checks', 'torch' not in sys.modules)
receipt = {'schema': 'independent-native-v3-final-delta-review-v1', 'checks': checks,
           'candidate_and_precedence_mocks': mock_results, 'vector_hash_mock': topology_receipt,
           'source_hashes': source_hashes, 'prior_review_hashes': prior_review_hashes,
           'paired_candidate_source_sha256': sha(paired_path),
           'native_GPU_remote_execution': False, 'torch_imported': False,
           'numeric_originals_opened_or_hashed': False,
           'full_authored_suite_rerun': False}
(review / 'REVIEW_CHECKS.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
print(json.dumps({'passed': True, 'targeted_checks': len(checks),
                  'B2_closed': True, 'permutation_vector_note_closed': True,
                  'normal_native_main_AST_preserved': True}))
