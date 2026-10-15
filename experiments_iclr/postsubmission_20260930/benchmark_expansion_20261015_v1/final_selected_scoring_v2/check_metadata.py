"""Narrow source/metadata checks only. No torch, checkpoint, dataset or TEST read."""
import ast
import copy
import json
from pathlib import Path
import sys
from .contracts import readout_policy, selected_role_contract, validate_role_contract

HERE = Path(__file__).resolve().parent


def rejected(call):
    try:
        call()
    except ValueError:
        return True
    raise AssertionError('Expected explicit metadata refusal')


def main():
    assert 'torch' not in sys.modules
    expansion = HERE.parent
    facts = json.loads((expansion / 'benchmark_review_protocol_proposal_v1/FACTS.json').read_text())
    preset_family = {p['id']: 'node' if p['id'].startswith('polyformer-') else
                     'ogb' if p['id'].startswith('ogb-') else 'link' for p in facts['presets']}
    covered = 0
    for method in facts['methods']:
        for preset in method['presets']:
            policy = readout_policy(method['id'], preset_family[preset])
            assert policy['members'] == 4
            covered += 1
    aliases = 0
    for name in ('REGISTRY_v1.json', 'OVERLAY_v2.json', 'OVERLAY_v3.json', 'OVERLAY_v4.json'):
        source = expansion / 'expanded_job_binding_v1' / name
        value = json.loads(source.read_text())
        for row in value['entries']:
            if row.get('runtime_procedure'):
                readout_policy(row['runtime_procedure'], preset_family[row['cell_id']])
                aliases += 1
    assert readout_policy('native_single', 'node')['members'] == 1
    assert readout_policy('boundary_global_half_pool', 'node')['kind'] == 'mean'
    assert readout_policy('aggregation_conditional_globalJ_v1', 'link')['kind'] == 'restored_gate'
    assert readout_policy('interaction_input_conditionalJ', 'link')['gate_leaf'] == '_interaction_readout'
    assert rejected(lambda: readout_policy('unknown_method', 'node'))
    # In-memory schema fixture only; its placeholder pins are not real inputs
    # and cannot pass the actual source/GO/provider byte-verification APIs.
    contract = {'schema': 'native-final-TEST-role-contract-v2', 'role': 'TEST', 'family': 'node',
                'preset_id': 'fixture', 'dataset': 'fixture', 'topology': 'unchanged_native_transductive_graph',
                'index_source': 'same_literal_split_test_mask', 'calibration': 'none',
                'no_TEST_driven_selection': True, 'rtol': 1e-5, 'atol': 1e-6,
                'populations': {str(i): {'records': 2, 'ids_sha256': '1' * 64,
                                         'provider_evidence': {}, 'literal_split_seed': 42 + i} for i in range(10)}}
    validate_role_contract(contract)
    identity = {'dataset': 'fixture', 'preset_id': 'fixture', 'split_index': 7, 'literal_split_seed': 49}
    assert selected_role_contract(contract, identity)['population']['literal_split_seed'] == 49
    assert rejected(lambda: selected_role_contract(contract, dict(identity, literal_split_seed=50)))
    missing = copy.deepcopy(contract); missing['populations'].pop('7')
    assert rejected(lambda: validate_role_contract(missing))
    calibrated = dict(contract, calibration='temperature')
    assert rejected(lambda: validate_role_contract(calibrated))
    source_trees = {}
    for path in HERE.glob('*.py'):
        tree = ast.parse(path.read_text(), filename=str(path)); source_trees[path.name] = tree
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in {'train_epoch', 'train_update', 'backward', 'Popen', 'system'}
        for node in tree.body:
            if isinstance(node, ast.Import):
                assert all(not x.name.startswith(('torch', 'numpy', 'scipy', 'ogb')) for x in node.names)
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or '').startswith(('torch', 'numpy', 'scipy', 'ogb'))
    pool = next(n for n in source_trees['role_serving.py'].body if isinstance(n, ast.FunctionDef) and n.name == 'pool')
    assert not any(isinstance(n, ast.Name) and n.id == 'labels' for n in ast.walk(pool))
    assert 'torch' not in sys.modules
    report = {'schema': 'final-selected-scoring-source-metadata-checks-v2', 'status': 'PASS',
              'scientific_method_preset_routes': covered, 'registry_overlay_runtime_routes': aliases,
              'unknown_readout_refused': True, 'ten_literal_node_TEST_populations_required': True,
              'wrong_split_and_CAL_refused': True, 'prediction_aggregation_has_no_labels_input': True,
              'top_level_model_imports': 0, 'training_calls': 0, 'datasets_opened': 0,
              'checkpoints_opened': 0, 'TEST_payloads_opened': 0, 'native_executions': 0,
              'numerical_or_source_parity_qualified': False}
    (HERE / 'LOCAL_CHECKS.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
