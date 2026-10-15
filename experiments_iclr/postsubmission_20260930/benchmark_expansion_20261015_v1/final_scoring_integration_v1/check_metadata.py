"""Narrow source/metadata checks only; never calls host/runtime/ledger APIs."""
import ast
import copy
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from final_scoring_integration_v1.authority import check_host_decision
from final_scoring_integration_v1.common import digest
from final_scoring_integration_v1.engine import Engine
from final_scoring_integration_v1.offline_seal import role_semantics, unpack
from final_scoring_integration_v1.projection import frozen_task, project_ownership, project_release
from final_scoring_integration_v1.preparation import make_capsule, validate_capsule, verify_own_scalar_chain
from final_scoring_integration_v1.projection import build_request
from final_scoring_integration_v1.collect import collect_profile


def refuses(function, *args):
    try:
        function(*args)
    except (ValueError, KeyError):
        return True
    raise AssertionError('Expected metadata refusal')


def main():
    checks = []
    before = set(sys.modules)
    for path in HERE.glob('*.py'):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(not item.name.startswith(('torch', 'numpy', 'torch_geometric')) for item in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert not (node.module or '').startswith(('torch', 'numpy', 'torch_geometric'))
    checks.append('all source parses; no direct model/data imports')
    original = ast.parse((HERE.parent / 'portable_dispatcher_v3/engine.py').read_text())
    successor = ast.parse((HERE / 'engine.py').read_text())
    def method(tree, name):
        return next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
    for name in ('launch', 'drain', 'output', 'folder', 'immutable_claim', 'stop', 'latest_stops'):
        assert ast.dump(method(original, name), include_attributes=False) == ast.dump(method(successor, name), include_attributes=False)
    checks.append('original launch/drain/leases-adjacent custody methods unchanged')
    fits = [{'id': 'fit/' + str(i), 'kind': 'selected_fit_TEST', 'training_fit_id': 'fit/' + str(i),
             'dependencies': ['fit/' + str(i)], 'selection_rule': 'existing_own_VALID_selected_checkpoint_only',
             'output_relative': 'selected/' + str(i)} for i in range(6460)]
    i4 = [{'id': 'I4/' + str(i), 'kind': 'I4_TEST_aggregate', 'training_I4_id': 'I4/' + str(i),
           'dependencies': ['TEST/fit/' + str(4*i+j) for j in range(4)],
           'selection_rule': 'four_own_VALID_selected_states', 'output_relative': 'I4/' + str(i)} for i in range(270)]
    # Physical final IDs are always TEST/original, including the I4 prefix.
    for task in fits + i4:
        task['id'] = 'TEST/' + task['id']
    roster = {'schema': 'frozen-final-selected-scoring-roster-v2', 'tasks': fits + i4, 'output_root': '/fixture/final'}
    owners = []
    for task in fits + i4:
        origin = task.get('training_fit_id', task.get('training_I4_id'))
        owners.append({'id': origin, 'worker': 'allocation/gpu0', 'science_contract_hash': 'a'*64,
                       'cohort': 'fixture', 'dependencies': [] if task['kind'] == 'selected_fit_TEST' else
                           [key.removeprefix('TEST/') for key in task['dependencies']]})
    owner = {'assignments': owners, 'campaign': 'fixture', 'hosts': {'allocation': {'project_root': '/fixture'}}}
    projected = project_ownership(owner, roster)
    assert len(projected['assignments']) == 6730
    assert sum(not row['dependencies'] for row in projected['assignments']) == 6460
    release = {'host': 'allocation', 'tasks': [{'id': row['id']} for row in owners]}
    bindings = {row['id']: {} for row in roster['tasks']}
    final = project_release({}, roster, {}, release, owner, bindings, source_manifest={})
    assert all(frozen_task(row) == original for row, original in zip(final['tasks'], roster['tasks']))
    assert final['additional_training_fits'] == 0 and final['launch_permission'] is False
    broken = copy.deepcopy(roster); broken['tasks'].pop()
    refuses(project_ownership, owner, broken)
    checks.append('6730-task projection retains6460 external fit deps/270 exact I4 deps; reversible; incomplete inventory refused')
    contract = {'hosts': {'allocation': {}}, 'fixture': True}
    instruction = {'path': '/fixture/instruction.json', 'bytes': 1, 'sha256': 'a'*64}
    decision = {'schema': 'portable-global-human-decision-v1', 'user_explicit_GO': True,
                'human_instruction_evidence': instruction, 'global_final_contract_sha256': digest(contract)}
    go = {'decision_view_schema': 'deterministic-host-campaign-decision-view-v1', 'decision_host': 'allocation',
          'parent_global_decision_sha256': digest(decision), 'human_instruction_evidence': instruction,
          'new_user_action_claimed': False, 'user_explicit_GO': True, 'final_execution': {'global_contract': {}}}
    template = {'schema': 'prepared-host-campaign-decision-template-v1', 'host': 'allocation',
                'execution_view': {key: copy.deepcopy(value) for key, value in go.items() if key != 'parent_global_decision_sha256'}}
    template['execution_view'].update(user_explicit_GO=False, human_instruction_evidence=None)
    template['execution_view']['final_execution'].pop('global_contract')
    check_host_decision(go, decision, contract, 'allocation', instruction, template)
    broken = dict(go, decision_host='server77')
    refuses(check_host_decision, broken, decision, contract, 'allocation', instruction, template)
    broken = dict(decision, user_explicit_GO=False)
    refuses(check_host_decision, go, broken, contract, 'allocation', instruction, template)
    checks.append('one-decision host view accepted; altered host/pending decision refused; no cyclic file hashes')
    class MetadataBackend:
        def validate_result(self, task, raw):
            pass
    engine = Engine.__new__(Engine); engine.backend = MetadataBackend()
    task = {'kind': 'selected_fit_TEST', 'id': 'TEST/fit/0', 'training_fit_id': 'fit/0', 'job_hash': 'a'*64}
    raw = {'schema': 'final-selected-fit-scoring-result-v2', 'task_id': task['id'], 'training_fit_id': 'fit/0',
           'job_hash': 'a'*64, 'role': 'TEST', 'physical_training_fits': 0, 'optimizer_updates': 0,
           'automatic_retries': 0, 'status': 'complete', 'TEST_accessed': True}
    engine.validate_terminal(task, raw)
    refuses(engine.validate_terminal, task, dict(raw, TEST_accessed=False))
    refuses(engine.validate_terminal, task, dict(raw, physical_training_fits=1))
    refuses(engine.validate_terminal, task, dict(raw, schema='native-full-epoch-fit-result-v1'))
    checks.append('literal final TEST terminal accepted; VALID schema/training work/hidden TEST access refused')
    import hashlib
    exact = '{"a": 1}\r\n'
    origin = {'path': '/foreign/only-origin.json', 'bytes': len(exact.encode()), 'sha256': hashlib.sha256(exact.encode()).hexdigest()}
    assert unpack({'origin': origin, 'exact_json_text': exact}) == {'a': 1}
    refuses(unpack, {'origin': origin, 'exact_json_text': exact + ' '})
    checks.append('copied exact JSON authority accepted without foreign-file read; changed bytes refused')
    class LookupOnly:
        def __init__(self, values):
            self.values = values
        def __getitem__(self, key):
            return self.values[key]
        def __contains__(self, key):
            return key in self.values
        def __iter__(self):
            raise AssertionError('Child must not iterate unrelated campaign inventory')
    final_index = {row['id']: row for row in final['tasks']}
    training_index = {row['id']: row for row in owners}
    outcome = {'state': instruction, 'terminal': instruction, 'training_result': instruction}
    preparation_pin = instruction
    preparation = {'schema': 'fully-validated-final-supervisor-preparation-v1', 'worker': 'allocation/gpu0',
                   'all_campaign_checks_performed_by_supervisor': True, 'training_seal': instruction}
    calls = []
    def metadata_only_verifier(value, root, **kwargs):
        calls.append(value)
    for selected in (final['tasks'][0], final['tasks'][-1]):
        origin = selected.get('training_fit_id', selected.get('training_I4_id'))
        seal = {'outcomes': LookupOnly({origin: outcome})}
        capsule = make_capsule(selected, final['output_root'], LookupOnly(final_index), LookupOnly(training_index), seal, instruction, preparation_pin)
        expected = build_request(selected, final, instruction, instruction, seal, {'source_manifest': instruction})
        if selected['kind'] == 'I4_TEST_aggregate':
            expected['component_results'] = [{'path': str(Path(final['output_root']) / row['output_relative'] / 'RESULT.json')}
                                             for row in capsule['dependency_tasks']]
        counts = validate_capsule(preparation, capsule, expected, {'final_scoring': {'source_manifest': instruction}})
        assert counts['training_tasks_checked'] == 1 and counts['final_dependency_tasks_checked'] in (0, 4)
        assert counts['campaign_inventory_iterations'] == 0
        assert verify_own_scalar_chain(capsule, '/fixture', metadata_only_verifier) == 3
    assert len(calls) == 6
    child = (HERE / 'runtime_entry.py').read_text()
    assert 'training_contracts.authority' not in child and 'merge_host_shards' not in child
    checks.append('child capsule checks1own training chain+0/4deps,3scalar pins; campaign-iteration sentinel passes; no child full authority')
    # Explicit missing-role rows must survive normalization and accepted report
    # validation, including exact M1/body0, four components and mean diagnostic.
    profile = {'fit_specs': {}, 'I4_specs': {}}
    roles, expected_rows, cohorts = [], [], []
    contrasts = [{'id': 'shared_vs_I4', 'lhs': {'method_id': 'shared', 'readout': 'actual'},
                  'rhs': {'method_id': 'native_I4', 'readout': 'actual'}}]
    for seed in range(10):
        cohort = {'seed': seed, 'split_index': seed, 'literal_split_seed': seed, 'cohort_id': str(seed)}
        cohorts.append(cohort)
        def identity(method, fit_id, actual_seed, procedure='native_single', readout='actual'):
            return {'dataset': 'metadata-fixture', 'preset_id': 'fixture', 'track': 'matched_extended',
                'cohort_id': str(seed), 'cohort_seed': seed, 'split_index': seed, 'literal_split_seed': seed,
                'scientific_method_id': method, 'runtime_procedure_id': procedure, 'readout': readout,
                'fit_id': fit_id, 'actual_seed': actual_seed, 'endpoint': {'name': 'accuracy', 'units': 'fraction', 'direction': 'max'},
                'bindings': {field: 'a'*64 for field in ('native_source', 'data', 'controller', 'evaluator', 'readout')}}
        bodies = []
        for body, offset in enumerate((0, 100003, 200003, 300007)):
            fit_id = 'native/' + str(seed) + '/' + str(body)
            value = identity('native_M1' if body == 0 else 'native_body' + str(body), fit_id, seed + offset)
            profile['fit_specs'][fit_id] = {'identity': value, 'job_hash': digest(fit_id), 'output_relative': fit_id}
            roles.append({'id': 'TEST/' + fit_id, 'kind': 'selected_fit_TEST', 'training_fit_id': fit_id,
                          'preset_id': 'fixture', 'output_relative': 'selected/' + digest(fit_id)})
            bodies.append(fit_id)
            if body == 0:expected_rows.append(value)
        fit_id = 'shared/' + str(seed)
        actual = identity('shared', fit_id, seed, 'gnnm_boundary_native')
        mean = identity('shared', fit_id, seed, 'gnnm_boundary_native', 'same_selected_bank_mean')
        profile['fit_specs'][fit_id] = {'identity': actual, 'same_bank_mean_identity': mean,
            'job_hash': digest(fit_id), 'output_relative': fit_id}
        roles.append({'id': 'TEST/' + fit_id, 'kind': 'selected_fit_TEST', 'training_fit_id': fit_id,
                      'preset_id': 'fixture', 'output_relative': 'selected/' + digest(fit_id)})
        expected_rows += [actual, mean]
        task_id = 'I4/' + str(seed)
        value = identity('native_I4', task_id, seed, 'native_I4')
        value['component_fit_ids'] = bodies
        profile['I4_specs'][task_id] = {'identity': value, 'component_fit_ids': bodies}
        roles.append({'id': 'TEST/' + task_id, 'kind': 'I4_TEST_aggregate', 'training_I4_id': task_id,
                      'preset_id': 'fixture', 'output_relative': task_id})
        expected_rows.append(value)
    profile['registry'] = {'schema': 'expanded-ten-cohort-report-registry-v1', 'expected_predictors': expected_rows,
                          'presets': [{'id': 'fixture', 'contrasts': contrasts, 'matched_cohorts': cohorts}]}
    profile['contrast_plan'] = {'schema': 'expanded-prescribed-contrast-plan-v1', 'presets': {'fixture': contrasts},
                               'cohort_keys': {'fixture': [{key: row[key] for key in ('seed', 'split_index', 'literal_split_seed')} for row in cohorts]}}
    absent = HERE / 'metadata_fixture_absent_outputs'
    report = collect_profile(profile, {'tasks': roles}, absent / 'training', absent / 'final', HERE.parent.parent.parent,
                             {}, {}, host='allocation')
    assert len(report['raw_rows']) == len(report['all_expected_predictor_statuses']) == 40
    assert all(row['observed'] is False for row in report['all_expected_predictor_statuses'])
    assert all(row['descriptive_marginal_95pct_paired_percentile_interval'] is None for row in report['contrasts'])
    assert report['cost_summary']['attempts_with_unknown_wall'] == 1
    checks.append('accepted reporter retains40explicit missing TEST rows including I4/mean; all intervals withheld; absent TRAIN costs unknown')
    assert 'torch' not in sys.modules and 'torch_geometric' not in sys.modules
    result = {'schema': 'final-scoring-integration-metadata-checks-v1', 'status': 'PASS', 'checks': checks,
              'new_training_fits': 0, 'models': 0, 'forwards': 0, 'optimizer_updates': 0,
              'host_or_resource_probes': 0, 'remote_calls': 0, 'TEST_accessed': False,
              'native_execution_qualified': False, 'full_campaign_GO': False}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
