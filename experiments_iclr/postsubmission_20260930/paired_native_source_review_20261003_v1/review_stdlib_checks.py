"""Review-only stdlib checks/mocks; writes only inside this review folder."""
import ast
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

review = Path(__file__).resolve().parent
packet = review.parent / 'graph_paired_native_qualification_preparation_v1'
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
checks = []


def check(name, condition):
    assert condition, name
    checks.append({'name': name, 'passed': True})


check('requested_manifest_sha256', sha(packet / 'MANIFEST.json') ==
      'f21826f941d39dd22a8f12cfb28168067d052c3541612e9de97d5835bbd923e9')
check('requested_native_script_sha256', sha(packet / 'prototype/qualify_squirrel17.py') ==
      '7e76d69a61aaa68dad93463a26de84fdf03da3f8f9625f3868f731dd92f05356')

# Reproduce the authored stdlib mocks with their scratch directory redirected
# here and their packet-receipt write removed. Reviewed files stay read-only.
fixture = packet / 'fixtures/stdlib_checks.py'
tree = ast.parse(fixture.read_bytes(), filename=str(fixture))
modified = []
for node in list(tree.body):
    if (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'scratch'
                                            for t in node.targets)):
        node.value = ast.parse("review / 'scratch'", mode='eval').body
        modified.append(('scratch_redirect', node.lineno))
    if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == 'write_text'):
        tree.body.remove(node)
        modified.append(('receipt_write_removed', node.lineno))
check('only_scratch_and_receipt_write_adapted', modified ==
      [('scratch_redirect', 124), ('receipt_write_removed', 157)])
namespace = {'__file__': str(fixture), '__name__': '__source_review_checks__', 'review': review}
captured = io.StringIO()
with redirect_stdout(captured):
    exec(compile(ast.fix_missing_locations(tree), str(fixture), 'exec'), namespace)
check('authored_343_checks_reproduced', len(namespace['checks']) == 343
      and all(row['passed'] for row in namespace['checks']))
subject = namespace['subject']

# Extract only the sealed helper's stdlib exception class, without importing
# the helper, Torch, its base module, or any model/input.
paired_path = review.parent / 'graph_full_node_cotangent_paired_alpha_v1/prototype/paired_shared_alpha_initializer.py'
paired_tree = ast.parse(paired_path.read_bytes())
exception_class = next(n for n in paired_tree.body if isinstance(n, ast.ClassDef)
                       and n.name == 'PairedGeometryError')
exception_namespace = {}
exec(compile(ast.Module(body=[exception_class], type_ignores=[]), str(paired_path), 'exec'),
     exception_namespace)
PairedGeometryError = exception_namespace['PairedGeometryError']

failure_mocks = []
for inner_message, expected_status in [('mock host memory exhaustion', 'qualification_failed'),
                                       ('CUDA out of memory: mocked', 'resource_deferred')]:
    with tempfile.TemporaryDirectory(dir=review / 'scratch') as directory:
        fake_packet = Path(directory)
        bound = {'original_records': [], 'canonical_research_root': str(fake_packet)}
        (fake_packet / 'BOUND_INPUTS.json').write_text(json.dumps(bound))
        (fake_packet / 'RESOURCE_EVIDENCE.json').write_text('{}')
        payload = [{'path': name, 'sha256': sha(fake_packet / name)}
                   for name in ('BOUND_INPUTS.json', 'RESOURCE_EVIDENCE.json')]
        manifest_bytes = json.dumps({'payload': payload}).encode()
        (fake_packet / 'MANIFEST.json').write_bytes(manifest_bytes)
        (fake_packet / 'SEAL.json').write_text(json.dumps({
            'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest()}))
        inner = MemoryError(inner_message)
        wrapped = PairedGeometryError('common_VJP_primal', inner, {'accepted_alpha': None})
        wrapped.__cause__ = inner
        with (patch.object(subject, 'PACKET', fake_packet), patch.object(
                subject, 'resource_preflight', return_value={'status': 'resource_preflight_passed'}),
                patch.object(subject, 'load', side_effect=wrapped), redirect_stdout(io.StringIO())):
            code = subject.main(['--run-name', 'wrapped_resource_mock'])
        receipt = json.loads((fake_packet / 'runs/wrapped_resource_mock/QUALIFICATION.json').read_bytes())
        check('wrapped_resource_classification:' + expected_status,
              receipt['status'] == expected_status and code == 1
              and receipt['paired_geometry_abort']['geometry_error']['error_type'] == 'MemoryError')
        failure_mocks.append({'underlying_error_type': 'MemoryError',
                              'underlying_error_message': inner_message,
                              'outer_error_type': type(wrapped).__name__,
                              'observed_status': receipt['status'],
                              'exit_code': code,
                              'native_invocation_performed': receipt['native_invocation_performed'],
                              'original_preservation_after': receipt['original_preservation_after']})

manifest = json.loads((packet / 'MANIFEST.json').read_bytes())
source_hashes = {'MANIFEST.json': sha(packet / 'MANIFEST.json'),
                 'SEAL.json': sha(packet / 'SEAL.json')}
for row in manifest['payload']:
    actual = sha(packet / row['path'])
    check('payload_unchanged_after_review:' + row['path'], actual == row['sha256'])
    source_hashes[row['path']] = actual
bound = json.loads((packet / 'BOUND_INPUTS.json').read_bytes())
nonnumeric_hashes = []
numeric_descriptors = []
for row in bound['original_records']:
    if Path(row['path']).suffix in ('.npy', '.npz', '.pt', '.pth'):
        numeric_descriptors.append(row)
        continue
    relative = Path(row['path']).relative_to(bound['canonical_research_root'])
    actual = sha(review.parent / relative)
    check('nonnumeric_original_unchanged:' + str(relative), actual == row['sha256'])
    nonnumeric_hashes.append({'path': str(relative), 'sha256': actual})
check('exact_seven_numeric_descriptors', len(numeric_descriptors) == 7)
check('only_TRAIN_label_descriptor', [Path(r['path']).name for r in numeric_descriptors
      if '/labels/' in r['path']] == ['seed17_split0_train.npz'])

import sys
check('no_Torch_import', 'torch' not in sys.modules)
result = {'schema': 'independent-native-paired-source-review-stdlib-v1',
          'checks': checks, 'authored_read_only_checks': namespace['checks'],
          'authored_check_stdout': captured.getvalue(), 'failure_mocks': failure_mocks,
          'source_hashes': source_hashes, 'nonnumeric_original_hashes': nonnumeric_hashes,
          'numeric_descriptors_not_opened_or_hashed': numeric_descriptors,
          'torch_imported': False, 'native_or_GPU_or_remote_execution': False,
          'validation_or_final_label_arrays_opened_or_content_hashed': False,
          'reviewed_sources_changed': False}
(review / 'REVIEW_CHECKS.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
print(json.dumps({'passed': True, 'review_checks': len(checks),
                  'authored_read_only_checks': len(namespace['checks']),
                  'wrapped_MemoryError_status': failure_mocks[0]['observed_status']}))
