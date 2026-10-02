"""Stdlib/source/mock checks only. No remote/GPU/native/array operation."""
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

packet = Path(__file__).resolve().parents[1]
source_path = packet/'prototype/qualify_squirrel17.py'
source_bytes = source_path.read_bytes()
tree = ast.parse(source_bytes)
sha = lambda data: hashlib.sha256(data).hexdigest()
checks = []


def check(name, condition):
    assert condition, name
    checks.append(dict(name=name, passed=True))


check('source_syntax', isinstance(tree, ast.Module))
text = source_bytes.decode()
for name, snippet in [
    ('pinned_runtime', 'driver.load_runtime(context, source_directory)'),
    ('TRAIN_only_loader', 'driver.source_inputs(rt, context, ledger, validation=False)'),
    ('pinned_restore', "result['native_invocation_performed'] = True\n            native, optimizer = measured('pinned_native_restore_no_updates', lambda:\n                rt.integration.restore_native(rt.adapter, checkpoint, rt.device)"),
    ('pinned_clone', 'rt.integration.clone_boundary(native, rt.boundary, 4)'),
    ('pinned_binding', "rt.method.bind_common_model(k1, 'PolyFormer-Mono'"),
    ('active_precision', 'precision.qualify_gradient_interface(observed, theta0, train.nodes, train.labels,'),
    ('sealed_paired', 'paired.initialize_paired_four_arms(observed, theta0, S, S_permuted, targets,'),
    ('explicit_full_contract', 'homogeneous_full_node_outputs=True'),
    ('Pi_S_PiT', 'rt.torch.equal(expected.indices(), S_permuted.indices())'),
    ('fresh_arm_copy', "fresh = measured('fresh_K4_clone_'+arm"),
    ('install_equality', 'rt.integration.difference(actual, expected_logits, 1e-6, 1e-5)'),
    ('joint_failure_retained', "if slices is None:\n                result['status'] = 'joint_failure'"),
    ('FP32_native', 'value.dtype == rt.torch.float32'),
    ('scheduled_completed_distinction', 'completed product count unknown for an aborted bank'),
    ('fixed_UUID_probe', "'--id='+uuid"),
    ('fresh_output', 'out.mkdir(parents=True, exist_ok=False)'),
    ('preserve_before_after', "stdlib_stage('original_fingerprints_after'"),
    ('no_bytecode_original_write', 'sys.dont_write_bytecode = True'),
]:
    check(name, snippet in text)
calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
for forbidden in ['step', 'backward', 'save', 'rng_restore', 'warm_native', 'phase_freeze', 'main_phase']:
    check('no_'+forbidden+'_call', not any(isinstance(n.func, ast.Attribute) and n.func.attr == forbidden
                                        for n in calls))
check('no_direct_remote_command', all(word not in text for word in ['ssh ', 'scp ', 'rsync ']))
check('no_direct_environment_mutation', not any(
    isinstance(n, (ast.Assign, ast.AugAssign)) and any(isinstance(t, ast.Subscript)
    and isinstance(t.value, ast.Attribute) and t.value.attr == 'environ'
    for t in (n.targets if isinstance(n, ast.Assign) else [n.target])) for n in ast.walk(tree)))
bound = json.loads((packet/'BOUND_INPUTS.json').read_text())
evidence = json.loads((packet/'RESOURCE_EVIDENCE.json').read_text())
check('exact_Squirrel17', (bound['context']['graph'], bound['context']['backbone'], bound['context']['seed'],
                          bound['nodes'], bound['classes'], bound['private_dimensions'])
      == ('Squirrel', 'polyformer_mono', 17, 2223, 5, 512))
check('only_TRAIN_descriptor_in_runtime_context', set(bound['context']['source_labels']) == {'train'})
check('exact_warm_checkpoint', bound['warm_checkpoint']['sha256'] ==
      'e67a0ab44c966f4980c709d0256a10a4a6916daa7dc564a07c98fc1a867a57b2')
check('exact_shared_source', bound['paired_source_sha256'] ==
      'e3ff6f134bae639269f284c4d8b0d290a31672787fe662ee829033465ed147d0')
check('no_validation_label_original_descriptor', all(
    not row['path'].endswith('seed17_split0_validation.npz') for row in bound['original_records']))
check('only_seven_declared_numeric_inputs', len([row for row in bound['original_records']
      if Path(row['path']).suffix in ('.npy','.npz','.pt','.pth')]) == 7)
check('only_one_TRAIN_label_path', [Path(row['path']).name for row in bound['original_records']
      if '/labels/' in row['path']] == ['seed17_split0_train.npz'])
check('outcome_free_evidence_only', all(set(row) <= {'operation','status','seconds','peak_allocated_bytes',
    'peak_reserved_bytes','process_maxrss_native_units'} for row in evidence['prior_operations']))
check('conservative_resource_allowance', evidence['required_free_bytes'] >=
      2*evidence['prior_max_reserved_bytes']+2*2**30)


def verify_local_source_mirrors(when):
    for row in bound['original_records']:
        if Path(row['path']).suffix in ('.npy','.npz','.pt','.pth'):
            continue  # No original numeric input is opened by this fixture.
        relative = Path(row['path']).relative_to(bound['canonical_research_root'])
        local = packet.parent/relative
        check(when+':source_metadata_unchanged:'+str(relative),
              sha(local.read_bytes()) == row['sha256'])


verify_local_source_mirrors('before')

spec = importlib.util.spec_from_file_location('native_preparation_stdlib_subject', source_path)
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)
check('source_import_uses_no_Torch', 'torch' not in sys.modules)
check('resource_pass', subject.resource_decision(evidence['required_free_bytes'], evidence, True)['status']
      == 'resource_preflight_passed')
check('memory_deferred', subject.resource_decision(evidence['required_free_bytes']-1, evidence, True)['status']
      == 'resource_deferred')
check('UUID_mapping_deferred', subject.resource_decision(evidence['required_free_bytes'], evidence, False)['status']
      == 'resource_deferred')
check('wall_budget_deferred', subject.resource_decision(evidence['required_free_bytes'],
      dict(evidence, planned_wall_seconds=601), True)['status'] == 'resource_deferred')
try:
    subject.verify_originals([dict(path='/never/open/labels/final.npz',sha256='0'*64,
                                  kind='compact_TRAIN_label_descriptor')])
except ValueError as error:
    check('final_label_path_refused_before_open', 'Only exact TRAIN' in str(error))
else:
    check('final_label_path_refused_before_open', False)
uuid = bound['expected_gpu_uuid']
with patch.object(subject.subprocess, 'run', return_value=SimpleNamespace(
        stdout=f'{uuid}, NVIDIA A100-SXM4-80GB, 81920, 60000\n')), patch.dict(subject.os.environ,
        {'CUDA_VISIBLE_DEVICES':uuid}):
    probe = subject.resource_preflight(bound, evidence)
    check('mock_only_GPU_probe_pass', probe['status'] == 'resource_preflight_passed')
    check('mock_probe_uuid_exact', probe['uuid'] == uuid and '--id='+uuid in probe['command'])
with patch.object(subject.subprocess, 'run', side_effect=RuntimeError('mock unavailable')):
    check('unavailable_probe_deferred', subject.resource_preflight(bound, evidence)['status'] == 'resource_deferred')

# Exercise both early output paths with fake source bytes in a scratch directory
# inside this new packet. No original numeric input or actual GPU probe is used.
scratch = packet/'fixtures/scratch'
scratch.mkdir(exist_ok=True)
for mode in ['resource_deferred', 'resource_preflight_passed']:
    with tempfile.TemporaryDirectory(dir=scratch) as directory:
        root = Path(directory)
        original = root/'fake_source.txt'; original.write_text('source fixture only\n')
        fake_bound = dict(original_records=[dict(path=str(original),sha256=sha(original.read_bytes()))])
        (root/'BOUND_INPUTS.json').write_text(json.dumps(fake_bound))
        (root/'RESOURCE_EVIDENCE.json').write_text(json.dumps(evidence))
        rows = [dict(path=name,sha256=sha((root/name).read_bytes())) for name in
                ['BOUND_INPUTS.json','RESOURCE_EVIDENCE.json']]
        manifest_bytes = json.dumps(dict(payload=rows)).encode()
        (root/'MANIFEST.json').write_bytes(manifest_bytes)
        (root/'SEAL.json').write_text(json.dumps(dict(manifest_sha256=sha(manifest_bytes))))
        with patch.object(subject, 'PACKET', root), patch.object(subject, 'resource_preflight',
                return_value=dict(status=mode)), contextlib.redirect_stdout(io.StringIO()):
            code = subject.main(['--run-name','fixture','--preflight-only'])
        result = json.loads((root/'runs/fixture/QUALIFICATION.json').read_text())
        check('mock_main_exit:'+mode, code == 0)
        check('mock_main_status:'+mode, result['status'] == (
            'resource_deferred' if mode == 'resource_deferred' else 'resource_preflight_complete'))
        check('mock_main_no_native:'+mode, result['native_invocation_performed'] is False
              and result['closure_forward_calls']['calls_started'] == 0)
        check('mock_main_original_preserved:'+mode, original.read_text() == 'source fixture only\n'
              and result['original_preservation_before'] == result['original_preservation_after'])
check('all_mock_paths_leave_no_Torch', 'torch' not in sys.modules)
verify_local_source_mirrors('after')

result = dict(schema='native-preparation-stdlib-source-mock-checks-v1', checks=checks,
              executed_stdlib_only=True, native_script_imported_stdlib_only=True,
              native_or_remote_or_GPU_execution=False, actual_GPU_probe_run=False,
              actual_numeric_graph_checkpoint_or_label_arrays_opened=False,
              source_sha256=sha(source_bytes), original_records_bound=len(bound['original_records']))
(packet/'STDLIB_RESULTS.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(passed=True,checks=len(checks),source_sha256=sha(source_bytes),
                      actual_native_or_GPU_execution=False)))
