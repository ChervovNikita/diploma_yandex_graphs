"""One retained scalar-state fixture plus exact source checks; no native work."""
import ast
from collections import OrderedDict
import copy
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
EXPANSION = HERE.parent
PARENT = EXPANSION / 'composable_normalization_runtime_v1/final_scoring'


def function(tree, name):
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def gate_config(task_kind, seed):
    # Evaluate only the retained metadata expression, never its model factory.
    tree = ast.parse((EXPANSION / 'conditional_aggregation_v1/core.py').read_text())
    constants = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
                 if isinstance(node, ast.Assign) and len(node.targets) == 1
                 and isinstance(node.targets[0], ast.Name)}
    factory = function(tree, 'make_aggregator')
    config = next(node.value for node in factory.body if isinstance(node, ast.Assign)
                  and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'config')
    cls = next(node for node in factory.body if isinstance(node, ast.ClassDef) and node.name == 'Aggregator')
    assert ast.unparse(function(cls, 'get_extra_state').body[0]) == 'return copy.deepcopy(config)'
    scope = {**constants, 'task_kind': task_kind, 'seed': seed, 'mode': 'conditional',
             'reference': SimpleNamespace(dtype='torch.float32'), 'str': str}
    return eval(compile(ast.Expression(config), '<retained-gate-config-only>', 'eval'),
                {'__builtins__': {}}, scope)


def main():
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    repair = json.loads((HERE / 'REPAIR.json').read_text())
    for pin in repair['source_evidence']:
        data = (EXPANSION / pin['path_relative_to_expansion']).read_bytes()
        assert len(data) == pin['bytes'] and hashlib.sha256(data).hexdigest() == pin['sha256']
    for name in repair['unchanged_members']:
        assert (HERE / name).read_bytes() == (PARENT / name).read_bytes()
    for path in HERE.glob('*.py'):
        compile(path.read_text(), str(path), 'exec')
    package_name = '_composed_final_scalar_check'
    spec = importlib.util.spec_from_file_location(package_name, HERE / '__init__.py',
                                                  submodule_search_locations=[str(HERE)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = package
    spec.loader.exec_module(package)
    backend = importlib.import_module(package_name + '.backend')
    assert backend.EXPANSION == EXPANSION
    backend.source_authority(HERE.parents[2], backend.receipt(HERE / 'MANIFEST.json', HERE.parents[2]))

    old = ast.parse((PARENT / 'backend.py').read_text())
    new = ast.parse((HERE / 'backend.py').read_text())
    v5 = ast.parse((EXPANSION / 'final_selected_scoring_v5/backend.py').read_text())
    assert ast.dump(function(new, '_state_fingerprint')) == ast.dump(function(v5, '_state_fingerprint'))
    old_model, new_model = function(old, 'model_fingerprint'), function(new, 'model_fingerprint')
    assert ast.dump(old_model.body[0]) == ast.dump(new_model.body[0])
    helper = function(new, '_state_fingerprint')
    tensor_branch = next(node for node in helper.body if isinstance(node, ast.If))
    old_leaf = old_model.body[-1].value.value
    assert ast.dump(old_leaf) == ast.dump(tensor_branch.body[0].value)
    comprehension = new_model.body[-1].value
    assert not comprehension.generators[0].ifs
    assert ast.unparse(comprehension.generators[0].iter) == 'model.state_dict().items()'
    assert ast.unparse(comprehension.value) == '_state_fingerprint(value, torch.is_tensor)'
    unchanged_functions = []
    for node in old.body:
        if not isinstance(node, ast.FunctionDef) or node.name == 'model_fingerprint':
            continue
        current = copy.deepcopy(function(new, node.name))
        if node.name == 'source_authority':
            for item in ast.walk(current):
                if isinstance(item, ast.Constant) and item.value == 'composed-normalization-final-scoring-source-v2':
                    item.value = 'composed-normalization-final-scoring-source-v1'
        assert ast.dump(node) == ast.dump(current), node.name
        unchanged_functions.append(node.name)
    score = function(new, 'score_fit')
    assert len([node for node in ast.walk(score) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name) and node.func.id == 'model_fingerprint']) == 3

    # One scalar metadata fixture contains the actual node and NCNC gate configs.
    # No tensor, fake Torch module, native model, array, or checkpoint is created.
    state = OrderedDict([('interaction_readout._extra_state', gate_config('multiclass', 101)),
                         ('_interaction_readout._extra_state', gate_config('binary', 0))])
    fingerprint = lambda value: backend._state_fingerprint(value, lambda _: False)
    baseline = fingerprint(state)
    assert baseline == fingerprint(copy.deepcopy(state))
    json.dumps(baseline, allow_nan=False)
    old_failures, changes = 0, 0
    for name, config in state.items():
        try:
            eval(compile(ast.Expression(old_leaf), '<original-tensor-leaf>', 'eval'),
                 {'list': list, 'str': str}, {'value': config})
        except AttributeError as error:
            assert "'dict' object has no attribute 'shape'" in str(error)
            old_failures += 1
        else:
            raise AssertionError('Original tensor-only leaf unexpectedly accepted dict state')
        for field, replacement in [('seed', True), ('features', ['changed_feature'])]:
            changed = copy.deepcopy(state)
            changed[name][field] = replacement
            assert fingerprint(changed) != baseline
            changes += 1
    assert fingerprint(state) == baseline

    # Exercise the original composed owner's scalar recursion on the same fixture.
    # It already supports these finite scalar/config types; its format is retained.
    owner_tree = ast.parse((EXPANSION / 'composable_normalization_v2/adapter.py').read_text())
    owner_class = next(node for node in owner_tree.body if isinstance(node, ast.ClassDef)
                       and node.name == 'NormalizedAdapter')
    owner_method = copy.deepcopy(function(owner_class, '_state_signature'))
    scope = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[owner_method], type_ignores=[])),
                 '<original-owner-scalar-signature>', 'exec'), scope)
    owner_type = type('ScalarSignatureFixture', (), {'_state_signature': scope['_state_signature']})
    owner = owner_type()
    owner.t = SimpleNamespace(is_tensor=lambda _: False)
    assert owner._state_signature(state) == owner._state_signature(copy.deepcopy(state))
    changed = copy.deepcopy(state)
    changed['interaction_readout._extra_state']['seed'] += 1
    assert owner._state_signature(state) != owner._state_signature(changed)
    marker = {'composed_owner_signature': 'preserved without model access'}
    dispatch = SimpleNamespace(composed_producer=True, state_signature=lambda: marker)
    assert backend.model_fingerprint(dispatch) is marker
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    return {'status': 'PASS', 'source_receipts_verified': len(repair['source_evidence']),
            'dependency_receipts_verified': len(json.loads((HERE / 'SOURCE_BINDINGS.json').read_text())['files']),
            'unchanged_functions': unchanged_functions, 'scalar_fixtures': 1,
            'retained_gate_configurations': 2, 'original_dict_shape_failures_reproduced': old_failures,
            'scalar_and_nested_changes_detected': changes, 'scalar_fixture_unchanged': True,
            'helper_AST_identical_to_final_selected_scoring_v5': True,
            'tensor_leaf_AST_identical_to_predecessor': True,
            'composed_owner_branch_AST_identical': True,
            'original_composed_owner_accepts_retained_scalar_config': True,
            'all_three_score_fit_fingerprint_calls_retained': True,
            'serving_runtime_and_BN_owner_sources_unchanged': True,
            'native_model_array_or_tensor_execution': 0, 'Torch_imported': False,
            'TEST_accessed': False, 'remote_operations': 0, 'Git_operations': 0}


if __name__ == '__main__':
    print(json.dumps(main(), indent=2, sort_keys=True))
