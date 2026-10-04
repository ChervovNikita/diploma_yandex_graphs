"""Bounded v2-to-v3 source/hash checks and synthetic sampler regression only."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
V2 = PHASE / 'pencil_collab_paired_predictive_preparation_20261004_v2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def functions(path):
    return {node.name: node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)}


def check():
    parent_manifest = json.loads((V2 / 'MANIFEST.json').read_text())
    assert sha(V2 / 'MANIFEST.json') == 'c937462c776f48ef3ce84d428ede3f3d5ccc4837c4874d54bd8961638a74bbc4'
    assert json.loads((V2 / 'SEAL.json').read_text())['manifest_sha256'] == sha(V2 / 'MANIFEST.json')
    for row in parent_manifest['files']:
        path = V2 / row['path']
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], row['path']
        assert path.suffix not in ('.pt', '.npy', '.npz', '.gz', '.pkl')
    changed = []
    for row in parent_manifest['files']:
        if (HERE / row['path']).read_bytes() != (V2 / row['path']).read_bytes(): changed.append(row['path'])
    assert set(changed) == {'AUTHOR_SOURCE_CHECK.json', 'INPUT_BINDINGS.json', 'PLAN.json',
                           'PROPOSED_COMMAND.json', 'README.md', 'ROOT_RELEASE_TEMPLATE.json',
                           'static_check.py', 'supervise.py'}, changed

    before, after = functions(V2 / 'supervise.py'), functions(HERE / 'supervise.py')
    assert set(after) - set(before) == {'parse_proc_stat_rss', 'process_identity_with_rss'}
    assert set(before) <= set(after)
    for name, node in before.items():
        if name != 'members_of_session': assert ast.dump(node) == ast.dump(after[name]), name
    for name in ('worker.py', 'common.py', 'data_adapter.py', 'NATIVE_SOURCE_BINDINGS.json',
                 'ACTUAL_DEPENDENCY_BINDING.json', 'OFFICIAL_CONFIG_REFERENCE.yaml',
                 'metadata/config.json', 'metadata/DATA_AUTHORITY.json', 'metadata/RUNTIME_AUTHORITY.json'):
        assert (HERE / name).read_bytes() == (V2 / name).read_bytes(), name
    native = json.loads((HERE / 'NATIVE_SOURCE_BINDINGS.json').read_text())['files']
    assert len(native) == 20
    for row in native:
        path = HERE / row['path']
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
        assert path.read_bytes() == (V2 / row['path']).read_bytes(), row['path']

    prior_plan = json.loads((V2 / 'PLAN.json').read_text())
    plan = json.loads((HERE / 'PLAN.json').read_text())
    restored = dict(plan)
    restored.pop('v2_failure_metadata'); restored.pop('supervisor_RSS_sampling')
    restored['execution_directory'] = prior_plan['execution_directory']
    restored['protocol_differences'] = list(plan['protocol_differences'])
    restored['protocol_differences'][-1] = prior_plan['protocol_differences'][-1]
    assert restored == prior_plan
    assert plan['status'] == 'DISABLED_UNEXECUTED' and plan['scientific_fit_released'] is False
    assert plan['workload'] == prior_plan['workload'] and plan['caps'] == prior_plan['caps']
    assert plan['allocator_policy'] == prior_plan['allocator_policy']
    assert plan['seeds'] == [0, 1, 2] and plan['workload']['native_epochs'] == 20
    assert plan['execution_directory'] == 'pencil_collab_paired_predictive_execution_root_20261004_v3'
    release = json.loads((HERE / 'ROOT_RELEASE_TEMPLATE.json').read_text())
    restored_release = dict(release, plan_sha256=sha(V2 / 'PLAN.json'))
    assert restored_release == json.loads((V2 / 'ROOT_RELEASE_TEMPLATE.json').read_text())
    assert release['status'] == 'DISABLED_TEMPLATE_NOT_AUTHORIZATION' and release['root_authorization_reference'] is None
    assert release['plan_sha256'] == sha(HERE / 'PLAN.json')
    assert (HERE / 'ROOT_RELEASE_DISABLED_CANDIDATE.json').read_bytes() == (HERE / 'ROOT_RELEASE_TEMPLATE.json').read_bytes()
    command = json.loads((HERE / 'PROPOSED_COMMAND.json').read_text())
    old_command = json.loads((V2 / 'PROPOSED_COMMAND.json').read_text())
    assert command['environment'] == old_command['environment']
    assert command['numeric_launch_authorized'] is False and command['supplied_client'] is False
    assert command['execution_order'] == [0, 1, 2] and command['maximum_concurrent_fits'] == 1
    restored_command = dict(command, scientific_stage=old_command['scientific_stage'], commands=[])
    for seed, row in enumerate(command['commands']):
        assert row['seed'] == seed and row['argv'][-2:] == ['--seed', str(seed)]
        restored_command['commands'].append(dict(row, argv=[value.replace(HERE.name, V2.name).replace(
            plan['execution_directory'], prior_plan['execution_directory']) for value in row['argv']]))
    assert restored_command == old_command

    inputs = json.loads((HERE / 'INPUT_BINDINGS.json').read_text())
    assert inputs['inputs'] == json.loads((V2 / 'INPUT_BINDINGS.json').read_text())['inputs']
    for row in inputs['v3_repair_inputs']:
        path = PHASE / row['path']
        assert path.resolve().is_relative_to(PHASE) and not path.is_symlink()
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], row['path']
    receipt = json.loads((HERE / 'evidence/V2_FAILURE_RESOURCE_RECEIPTS.json').read_text())
    physical = next(row['metadata'] for row in receipt['receipts'] if row['path'].endswith('PHYSICAL_TERMINAL.json'))
    assert physical['source_manifest_sha256'] == sha(V2 / 'MANIFEST.json')
    assert physical['physical_session_closed'] and physical['direct_child_reaped'] and not physical['unresolved_cleanup']
    assert physical['errors'] == ['RuntimeError: Owned live RSS observation missing'] and physical['physical_exit_code'] == -9
    assert (HERE / 'evidence/V2_FAILURE_RESOURCE_RECEIPTS.json').read_bytes() == (PHASE / 'pencil_v2_failure_diagnosis_20261004_v1/FAILURE_RESOURCE_RECEIPTS.json').read_bytes()
    assert (HERE / 'evidence/V2_ROOT_MONITOR_0004.json').read_bytes() == (PHASE / 'pencil_collab_paired_predictive_execution_root_20261004_v2/MONITOR_0004.json').read_bytes()
    for name in ('V1_WORKER_MONITOR_FAILURE.json', 'V1_FAILURE_METADATA.json'):
        assert (HERE / 'evidence' / name).read_bytes() == (V2 / 'evidence' / name).read_bytes()

    python_files = sorted(HERE.rglob('*.py'))
    for path in python_files: ast.parse(path.read_text(), filename=str(path))
    import sampler_regression
    sampler = sampler_regression.check()
    if (HERE / 'MANIFEST.json').exists():
        rows = json.loads((HERE / 'MANIFEST.json').read_text())['files']
        assert {row['path'] for row in rows} == {str(path.relative_to(HERE)) for path in HERE.rglob('*')
            if path.is_file() and path.name not in ('MANIFEST.json', 'SEAL.json')}
        for row in rows:
            path = HERE / row['path']
            assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
    return dict(status='AUTHOR_SOURCE_CHECKS_PASS_ONLY', AST_parsed_files=len(python_files),
                parent_v2_payloads_verified=len(parent_manifest['files']), changed_v2_payloads=sorted(changed),
                only_modified_existing_supervisor_function='members_of_session',
                new_sampler_helpers=['parse_proc_stat_rss', 'process_identity_with_rss'],
                all_other_supervisor_functions_AST_identical=True,
                worker_common_data_adapter_and_native_byte_identical=True, native_byte_identical_files=20,
                scientific_plan_and_caps_unchanged=True, fresh_scratch_seeds=[0, 1, 2], epochs_each=20,
                allocator_policy_unchanged=True, v1_v2_failures_preserved=True,
                v2_physical_session_closed_and_reaped=True, v2_failing_task_identity_unlogged=True,
                repair_input_pins_verified=len(inputs['v3_repair_inputs']), sampler_regression=sampler,
                disabled_future_release_candidate=True, execution_client_prepared=False,
                additional_scientific_qualification_requested=False, numerical_or_native_imports=False,
                graph_score_checkpoint_payload_reads=False, server_actions=False, launch=False,
                signals_sent=False, automatic_retry=False, independent_source_review=False,
                numeric_execution_authorized=False,
                limitation='Source checks establish neither actual proc lifecycle execution nor scientific feasibility or predictive quality.')


if __name__ == '__main__':
    print(json.dumps(check(), indent=2, sort_keys=True))
