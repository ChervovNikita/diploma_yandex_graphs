"""Targeted stdlib v2 delta checks; no numeric/native/GPU/remote execution."""
import ast
from contextlib import redirect_stdout
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

review = Path(__file__).resolve().parent
phase = review.parent
v1 = phase / 'graph_paired_native_qualification_preparation_v1'
v2 = phase / 'graph_paired_native_qualification_preparation_v2'
prior_review = phase / 'paired_native_source_review_20261003_v1'
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
checks = []


def check(name, condition):
    assert condition, name
    checks.append({'name': name, 'passed': True})


prior_review_hashes = {path.name: sha(path) for path in prior_review.iterdir() if path.is_file()}
check('requested_v2_manifest_hash', sha(v2 / 'MANIFEST.json') ==
      '07665a4d75613dcfc6c7f6b7582982d5a297de0a7e98150cefe549ab12d2bde8')
check('requested_v2_script_hash', sha(v2 / 'prototype/qualify_squirrel17.py') ==
      '489ef0d0976ec7e58644afea17f807a4da3976fc35952448da9fc72aba19923e')
check('preserved_v1_script_hash', sha(v1 / 'prototype/qualify_squirrel17.py') ==
      '7e76d69a61aaa68dad93463a26de84fdf03da3f8f9625f3868f731dd92f05356')
check('BOUND_INPUTS_exact_bytes_preserved', (v1 / 'BOUND_INPUTS.json').read_bytes() ==
      (v2 / 'BOUND_INPUTS.json').read_bytes())
check('RESOURCE_EVIDENCE_exact_bytes_preserved', (v1 / 'RESOURCE_EVIDENCE.json').read_bytes() ==
      (v2 / 'RESOURCE_EVIDENCE.json').read_bytes())

t1 = ast.parse((v1 / 'prototype/qualify_squirrel17.py').read_bytes())
t2 = ast.parse((v2 / 'prototype/qualify_squirrel17.py').read_bytes())
f1 = {node.name: node for node in t1.body if isinstance(node, ast.FunctionDef)}
f2 = {node.name: node for node in t2.body if isinstance(node, ast.FunctionDef)}
dump = lambda node: ast.dump(node, include_attributes=False)
check('only_resource_classifier_added', set(f2) - set(f1) == {'resource_failure_in_chain'}
      and set(f1) - set(f2) == set())
for name in sorted(set(f1) - {'main'}):
    check('unchanged_function_AST:' + name, dump(f1[name]) == dump(f2[name]))

# Remove exactly the reviewed classification change and normalize the schema
# string. Equality then proves the entire normal main branch is unchanged.
m1, m2 = copy.deepcopy(f1['main']), copy.deepcopy(f2['main'])
for tree in (m1, m2):
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value in (
                'Squirrel17-native-paired-source-qualification-v1',
                'Squirrel17-native-paired-source-qualification-v2'):
            node.value = 'Squirrel17-native-paired-source-qualification-review-normalized'
try1 = next(node for node in m1.body if isinstance(node, ast.Try) and node.handlers)
try2 = next(node for node in m2.body if isinstance(node, ast.Try) and node.handlers)
check('expected_handler_delta_shape', len(try1.handlers[0].body) == 4
      and len(try2.handlers[0].body) == 5)
del try1.handlers[0].body[2]
del try2.handlers[0].body[2:4]
check('normal_main_and_remaining_failure_receipts_exact_AST', dump(m1) == dump(m2))

spec = importlib.util.spec_from_file_location('independent_native_v2_delta_subject',
                                             v2 / 'prototype/qualify_squirrel17.py')
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)
check('subject_import_stdlib_only', 'torch' not in sys.modules)

allocator_type = type('OutOfMemoryError', (RuntimeError,), {})
class DerivedAllocator(allocator_type):
    pass
class DerivedHostMemory(MemoryError):
    pass


def wrapper(cause=None, context=None):
    value = RuntimeError('opaque wrapper')
    value.__cause__, value.__context__ = cause, context
    return value


cases = [
    ('direct_host', MemoryError('host'), 'MemoryError'),
    ('host_subclass', DerivedHostMemory('host subclass'), 'MemoryError'),
    ('direct_allocator', allocator_type('device'), 'OutOfMemoryError'),
    ('allocator_subclass', DerivedAllocator('device subclass'), 'OutOfMemoryError'),
    ('cause_host', wrapper(MemoryError('host cause')), 'MemoryError'),
    ('cause_allocator', wrapper(allocator_type('device cause')), 'OutOfMemoryError'),
    ('context_host', wrapper(context=MemoryError('host context')), 'MemoryError'),
    ('deep_cause', wrapper(wrapper(MemoryError('deep'))), 'MemoryError'),
    ('context_after_nonresource_cause', wrapper(ValueError('invalid'), MemoryError('context')), 'MemoryError'),
    ('legacy_cuda_message', RuntimeError('cUdA OuT oF MeMoRy: legacy'), 'CUDA_out_of_memory_message'),
    ('nonresource', ValueError('invalid'), None),
    ('wrapped_nonresource', wrapper(ValueError('invalid')), None),
]
suppressed = wrapper(context=MemoryError('suppressed'))
suppressed.__suppress_context__ = True
cases.append(('suppressed_context', suppressed, 'MemoryError'))
cycle_outer, cycle_inner = wrapper(), wrapper()
cycle_outer.__cause__, cycle_inner.__context__ = cycle_inner, cycle_outer
cases.append(('cycle_without_resource', cycle_outer, None))
shared = ValueError('shared')
cases.append(('shared_cause_context', wrapper(shared, shared), None))
classifier_results = []
for name, error, expected_reason in cases:
    match = subject.resource_failure_in_chain(error)
    check('classifier:' + name, (None if match is None else match['reason']) == expected_reason)
    classifier_results.append({'case': name, 'match': match})
cycle_inner.__cause__ = MemoryError('resource within cycle')
check('cycle_with_resource', subject.resource_failure_in_chain(cycle_outer)['reason'] == 'MemoryError')

paired = phase / 'graph_full_node_cotangent_paired_alpha_v1/prototype/paired_shared_alpha_initializer.py'
paired_tree = ast.parse(paired.read_bytes())
exception_class = next(node for node in paired_tree.body if isinstance(node, ast.ClassDef)
                       and node.name == 'PairedGeometryError')
exception_namespace = {}
exec(compile(ast.Module(body=[exception_class], type_ignores=[]), str(paired), 'exec'), exception_namespace)
PairedGeometryError = exception_namespace['PairedGeometryError']
main_mocks = []
for name, inner, expected_status in [
        ('wrapped_host_memory', MemoryError('mock host memory exhaustion'), 'resource_deferred'),
        ('wrapped_device_allocator', allocator_type('mock device allocator exhaustion'), 'resource_deferred'),
        ('wrapped_nonresource', ValueError('mock invalid geometry'), 'qualification_failed')]:
    with tempfile.TemporaryDirectory(dir=review) as directory:
        fake = Path(directory)
        original = fake / 'fake_source.txt'
        original.write_text('fake source only\n')
        bound = {'original_records': [{'path': str(original), 'sha256': sha(original)}],
                 'canonical_research_root': str(fake)}
        (fake / 'BOUND_INPUTS.json').write_text(json.dumps(bound))
        (fake / 'RESOURCE_EVIDENCE.json').write_text('{}')
        rows = [{'path': n, 'sha256': sha(fake / n)} for n in ('BOUND_INPUTS.json', 'RESOURCE_EVIDENCE.json')]
        manifest_bytes = json.dumps({'payload': rows}).encode()
        (fake / 'MANIFEST.json').write_bytes(manifest_bytes)
        (fake / 'SEAL.json').write_text(json.dumps({
            'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest()}))
        error = PairedGeometryError('common_VJP_primal', inner, {'accepted_alpha': None})
        error.__cause__ = inner
        with (patch.object(subject, 'PACKET', fake), patch.object(subject, 'resource_preflight',
                return_value={'status': 'resource_preflight_passed'}),
                patch.object(subject, 'load', side_effect=error), redirect_stdout(io.StringIO())):
            code = subject.main(['--run-name', name])
        receipt = json.loads((fake / 'runs' / name / 'QUALIFICATION.json').read_bytes())
        check('main_status:' + name, receipt['status'] == expected_status and code == 1)
        check('retained_geometry_and_preservation:' + name, receipt['paired_geometry_abort'] == error.report
              and receipt['original_preservation_before'] == receipt['original_preservation_after']
              and original.read_text() == 'fake source only\n')
        check('retained_scope_and_resource_match:' + name, not receipt['native_invocation_performed']
              and not receipt['scientific_merit_assessed']
              and ('resource_failure_exception' in receipt) == (expected_status == 'resource_deferred'))
        main_mocks.append({'case': name, 'status': receipt['status'], 'exit_code': code,
                           'resource_match': receipt.get('resource_failure_exception'),
                           'geometry_receipt_retained': True, 'original_preserved': True})

# Exercise the exact v2 observer and sealed candidate try/except as stdlib
# snippets. The mocked closure raises before any tensor method can be reached.
observed_def = next(node for node in ast.walk(t2) if isinstance(node, ast.FunctionDef)
                    and node.name == 'observed')
candidate_try = next(node for node in ast.walk(paired_tree) if isinstance(node, ast.Try)
                     and node.lineno == 223)
check('observer_has_no_resource_sentinel', not any(isinstance(node, ast.Call)
      and isinstance(node.func, ast.Name) and node.func.id == 'resource_failure_in_chain'
      for node in ast.walk(observed_def)))
candidate_mocks = []
opaque = wrapper(MemoryError('mock hidden allocator cause'))
for name, error in [('direct_candidate_memory', MemoryError('mock candidate allocation')),
                    ('opaque_candidate_memory_cause', opaque)]:
    events = []
    class FakeLedger:
        def sink(self, event, receipt):
            events.append({'event': event, 'receipt': receipt})
    def unavailable(theta, captured_error=error):
        raise captured_error
    observed_namespace = {'closure_counts': {'calls_started': 0, 'calls_completed': 0, 'calls_failed': 0},
                          'ledger': FakeLedger(), 'closure': unavailable}
    exec(compile(ast.Module(body=[observed_def], type_ignores=[]), str(v2 / 'prototype/qualify_squirrel17.py'),
                 'exec'), observed_namespace)
    candidate_namespace = {'logits_fn': observed_namespace['observed'], 'row': object(),
                           'outputs': [], 'errors': {}, 'name': 'full_node'}
    exec(compile(ast.Module(body=[candidate_try], type_ignores=[]), str(paired), 'exec'), candidate_namespace)
    counters = observed_namespace['closure_counts']
    recorded = candidate_namespace['errors']['full_node'][0]
    check('candidate_resource_exception_swallowed:' + name,
          candidate_namespace['outputs'] == [None]
          and counters == {'calls_started': 1, 'calls_completed': 0, 'calls_failed': 1}
          and recorded['error_type'] == type(error).__name__)
    check('candidate_resource_chain_not_retained:' + name,
          set(recorded) == {'member', 'error_type', 'error_message'}
          and subject.resource_failure_in_chain(error)['reason'] == 'MemoryError')
    candidate_mocks.append({'case': name, 'observer_counters': counters,
                            'candidate_error': recorded, 'observer_events': events,
                            'exception_propagated_out_of_candidate_try': False,
                            'resource_match_if_classified_before_swallow': subject.resource_failure_in_chain(error)})

source_hashes = {n: sha(v2 / n) for n in ('MANIFEST.json', 'SEAL.json')}
manifest = json.loads((v2 / 'MANIFEST.json').read_bytes())
for row in manifest['payload']:
    actual = sha(v2 / row['path'])
    check('v2_payload_unchanged:' + row['path'], actual == row['sha256'])
    source_hashes[row['path']] = actual
for name, digest in prior_review_hashes.items():
    check('v1_review_preserved:' + name, sha(prior_review / name) == digest)
bound = json.loads((v2 / 'BOUND_INPUTS.json').read_bytes())
numeric = [row for row in bound['original_records'] if Path(row['path']).suffix in ('.npy','.npz','.pt','.pth')]
check('151_original_descriptors_unchanged', len(bound['original_records']) == 151)
check('seven_numeric_descriptors_not_opened_or_hashed', len(numeric) == 7)
check('TRAIN_only_runtime_context', set(bound['context']['source_labels']) == {'train'})
check('no_Torch_import_after_mocks', 'torch' not in sys.modules)
receipt = {'schema': 'independent-native-v2-delta-review-v1', 'checks': checks,
           'classifier_results': classifier_results, 'main_mocks': main_mocks,
           'swallowed_candidate_resource_mocks': candidate_mocks,
           'source_hashes': source_hashes, 'prior_v1_review_hashes': prior_review_hashes,
           'v1_native_script_sha256': sha(v1 / 'prototype/qualify_squirrel17.py'),
           'paired_exception_class_source_sha256': sha(paired),
           'numeric_descriptors_not_opened_or_hashed': 7,
           'native_GPU_remote_execution': False, 'torch_imported': False,
           'full_authored_check_suite_rerun': False}
(review / 'REVIEW_CHECKS.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
print(json.dumps({'passed': True, 'targeted_checks': len(checks),
                  'v1_B1_wrapped_memory_defect_closed': main_mocks[0]['status'] == 'resource_deferred',
                  'new_B2_candidate_resource_path_confirmed': True,
                  'normal_main_AST_unchanged_after_allowed_delta': True}))
