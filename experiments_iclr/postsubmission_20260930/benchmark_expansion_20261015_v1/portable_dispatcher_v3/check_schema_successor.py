"""Narrow checks for the observed schema/config-loop repair; no native work."""
import ast
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch
import descriptor_binding as binding
import host_binding
import prepare_projection

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'portable_dispatcher_v2'


def main():
    audit = json.loads((HERE / 'FULL_SCHEMA_AUDIT.json').read_text())
    assert audit['fits_scanned'] == 6460 and audit['descriptor_read_passes'] == 1
    assert audit['all_descriptor_file_job_and_science_hashes_verified'] is True
    assert set(audit['unclassified_runtime_pointers']) == {'/recipe/_native_callback'}
    assert audit['unclassified_runtime_pointers']['/recipe/_native_callback']['count'] == 700
    assert audit['runtime_path_normalization_or_confinement_violations'] == []
    assert binding.allowed('/recipe/_native_callback')
    assert not binding.allowed('/recipe/_unclassified_callback')
    example = audit['unclassified_runtime_pointers']['/recipe/_native_callback']['examples'][0]['value']
    root, phase = host_binding.ALLOCATION_ROOT, host_binding.PHASE_RELATIVE
    target = '/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git'
    # A compact typed-path fixture, not a scientific descriptor/job or data artifact.
    fixture = {'id': 'metadata-schema-fixture', 'callback_module': example, 'data_root': root + '/data',
               'recipe': {'_repository_root': root, '_redesign_root': root + '/' + phase + '/reference_wrapper_redesign_20261013_v1',
                          '_native_callback': example, '_native_callback_sha256': 'a' * 64},
               'source_plan_entry': {'unchanged_path': example}, 'descriptor_preparation': {'unchanged_path': example}}
    bound, receipt = binding.rebind_job(fixture, root, phase, target, 'postsubmission_research_20260930')
    expected = example.replace(root + '/' + phase, target + '/postsubmission_research_20260930', 1)
    assert bound['recipe']['_native_callback'] == expected
    assert bound['recipe']['_native_callback_sha256'] == fixture['recipe']['_native_callback_sha256']
    assert bound['source_plan_entry'] == fixture['source_plan_entry']
    assert bound['descriptor_preparation'] == fixture['descriptor_preparation']
    assert receipt['canonical_runtime_job_hash'] == binding.digest(fixture)
    binding.audit_runtime_paths(bound, target, 'postsubmission_research_20260930')
    def funcs(path):
        return {n.name: ast.dump(n, include_attributes=False) for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.FunctionDef)}
    assert funcs(HERE / 'host_binding.py')['validate_host_job'] == funcs(PARENT / 'host_binding.py')['validate_host_job']
    assert 'from descriptor_binding import audit_runtime_paths' in (HERE / 'host_binding.py').read_text()
    records = []
    for path in sorted((HERE / 'hosts').glob('*.json')):
        records.append({'path': str(path.relative_to(HERE)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size})
    allocation_sha = next(r['sha256'] for r in records if r['path'] == 'hosts/allocation_gpu0.json')
    with patch.dict(os.environ, {'BENCHMARK_HOST_CONFIG_SHA256': allocation_sha}), \
         patch('socket.gethostname', side_effect=AssertionError('No host probe permitted')), \
         patch('subprocess.check_output', side_effect=AssertionError('No subprocess probe permitted')):
        configs = prepare_projection.metadata_host_configs({'files': records})
        assert configs['server77/gpu0']['host_enabled'] is configs['server77/gpu1']['host_enabled'] is False
        assert os.environ['BENCHMARK_HOST_CONFIG_SHA256'] == allocation_sha
    source = (HERE / 'prepare_projection.py').read_text()
    assert source.index('descriptor_audit = prewrite_descriptor_audit(') < source.index('    output.mkdir()')
    assert source.index("require(descriptor_audit['status'] == 'PASS'") < source.index('    output.mkdir()')
    native_files = [p for group in ('runtime_link', 'runtime_node_molecular', 'runtime_I4') for p in (HERE / group).glob('*.py')]
    assert len(native_files) == 16 and all(p.read_bytes() == (PARENT / p.relative_to(HERE)).read_bytes() for p in native_files)
    core = ['engine.py', 'custody.py', 'host_ledger.py', 'ownership.py', 'stop_transport.py', 'contracts.py', 'qualified_python_entry.py', 'run_gpu.py']
    assert all((HERE / name).read_bytes() == (PARENT / name).read_bytes() for name in core)
    for path in HERE.rglob('*.py'):
        compile(path.read_text(), str(path), 'exec')
    result = {'schema': 'portable-dispatcher-v3-narrow-schema-source-checks', 'status': 'PASS',
              'checks': ['actual6460audit_has_one_missing_pointer700occurrences_and_zero_path_violations',
                         'exact_native_callback_pointer_relocated_with_hash_and_provenance_preserved',
                         'preparer_and_unchanged_runtime_guard_share_same_typed_classifier',
                         'per_config_manifest_pin_loop_preserves_allocation_environment_and_disabled77',
                         'full_prewrite_descriptor_audit_and_PASS_gate_precede_first_projection_write',
                         'all16native_modules_and_eight_core_modules_byte_identical_to_v2',
                         'all_successor_sources_parse'],
              'parent30checks_reused_not_rerun': True, 'native_jobs_started': 0, 'tensor_reads': 0,
              'new_projection_executed': False, 'local_check_hardware_or_network_probes': 0, 'server77_accessed': False}
    (HERE / 'CHECKS.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
