"""Focused stdlib regression for recursive extra state; no tensor/model execution."""
import ast
from collections import OrderedDict
import copy
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
from .backend import _state_fingerprint

HERE = Path(__file__).resolve().parent


def gate_config(task_kind, seed):
    """Evaluate only the retained config dictionary expression, not its factory."""
    tree = ast.parse((HERE.parent/'conditional_aggregation_v1/core.py').read_text())
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            constants[node.targets[0].id] = ast.literal_eval(node.value)
    factory = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'make_aggregator')
    config = next(node.value for node in factory.body if isinstance(node, ast.Assign)
                  and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'config')
    cls = next(node for node in factory.body if isinstance(node, ast.ClassDef) and node.name == 'Aggregator')
    getter = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'get_extra_state')
    assert ast.unparse(getter.body[0]) == 'return copy.deepcopy(config)'
    scope = {**constants, 'task_kind': task_kind, 'seed': seed, 'mode': 'conditional',
             'reference': SimpleNamespace(dtype='torch.float32'), 'str': str}
    return eval(compile(ast.Expression(config), '<retained-gate-config-only>', 'eval'), {'__builtins__': {}}, scope)


def check_recursive_fingerprint():
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    fingerprint = lambda value: _state_fingerprint(value, lambda _: False)
    changes = 0
    for task_kind, seed in [('multiclass', 101), ('binary', 0)]:
        config = gate_config(task_kind, seed)
        state = OrderedDict([('interaction_readout._extra_state', config)])
        baseline = fingerprint(state)
        assert baseline == fingerprint(copy.deepcopy(state))
        for key in config:
            changed = copy.deepcopy(state)
            changed['interaction_readout._extra_state'][key] = ['changed', config[key]]
            assert fingerprint(changed) != baseline
            changes += 1
        changed = copy.deepcopy(state)
        changed['interaction_readout._extra_state']['features'][0] = 'changed_feature'
        assert fingerprint(changed) != baseline
        assert fingerprint(state) == baseline  # Fingerprinting did not change the state.
        json.dumps(baseline, allow_nan=False)
    for left, right in [(True, 1), (1, 1.0), (1, '1'), ([1], (1,)), (None, 'None'),
                        ({1: 'x'}, {'1': 'x'}), ({'a': 1}, OrderedDict([('a', 1)])),
                        (b'abc', 'abc'), (0.0, -0.0)]:
        assert fingerprint(left) != fingerprint(right)
    nan = struct.unpack('>d', bytes.fromhex('7ff8000000000001'))[0]
    other_nan = struct.unpack('>d', bytes.fromhex('7ff8000000000002'))[0]
    assert fingerprint(nan) == fingerprint(nan) != fingerprint(other_nan)
    json.dumps(fingerprint([float('inf'), float('-inf'), nan]), allow_nan=False)
    for unsupported in [object(), {1, 2}]:
        try:
            fingerprint({'nested': unsupported})
        except TypeError:
            pass
        else:
            raise AssertionError('Unsupported extra-state object was silently accepted')
    parent = ast.parse((HERE.parent/'final_selected_scoring_v4/backend.py').read_text())
    current = ast.parse((HERE/'backend.py').read_text())
    old_model = next(n for n in parent.body if isinstance(n, ast.FunctionDef) and n.name == 'model_fingerprint')
    helper = next(n for n in current.body if isinstance(n, ast.FunctionDef) and n.name == '_state_fingerprint')
    tensor_branch = next(n for n in helper.body if isinstance(n, ast.If))
    assert ast.dump(old_model.body[0].value.value) == ast.dump(tensor_branch.body[0].value)
    model = next(n for n in current.body if isinstance(n, ast.FunctionDef) and n.name == 'model_fingerprint')
    comprehension = model.body[1].value
    assert not comprehension.generators[0].ifs and comprehension.value.func.id == '_state_fingerprint'
    assert ast.unparse(comprehension.generators[0].iter) == 'model.state_dict().items()'
    assert ast.unparse(comprehension.value.args[1]) == 'torch.is_tensor'
    score = next(n for n in current.body if isinstance(n, ast.FunctionDef) and n.name == 'score_fit')
    assert len([n for n in ast.walk(score) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == 'model_fingerprint']) == 3
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    return {'status': 'PASS', 'retained_gate_configurations': 2, 'config_field_mutations_detected': changes,
            'nested_feature_mutation_detected': True, 'scalar_key_and_container_types_preserved': True,
            'signed_zero_and_nonfinite_bits_preserved': True, 'unsupported_objects_refused': True,
            'tensor_leaf_representation_AST_identical_to_v4': True,
            'actual_score_fit_uses_recursive_full_state_at_all_three_checks': True,
            'native_or_tensor_execution': 0, 'scope': 'stdlib metadata and source regression; numerical witness pending'}


if __name__ == '__main__':
    print(json.dumps(check_recursive_fingerprint(), indent=2, sort_keys=True))
