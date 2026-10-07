"""Reproducible stdlib-only integrity, identity, guard, and routing checks."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PARENT = HERE.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def main():
    bindings = json.loads((HERE / 'SOURCE_BINDINGS.json').read_text())
    dependencies = []
    for entry in bindings['sealed_dependencies']:
        root = PARENT / entry['directory']
        assert sha(root / 'MANIFEST.json') == entry['manifest_sha256']
        rows = json.loads((root / 'MANIFEST.json').read_text())['files']
        for row in rows:
            path = root / row['path']
            assert sha(path) == row['sha256'] and path.stat().st_size == row['bytes']
        dependencies.append({'directory': entry['directory'], 'file_count': len(rows), 'all_sealed_files_unchanged': True})
    trees = {}
    for path in sorted(HERE.glob('*.py')):
        text = path.read_text()
        compile(text, str(path), 'exec')  # Syntax only; writes no pyc and executes no source.
        trees[path.name] = ast.parse(text)
    numeric_before = {n for n in sys.modules if n.split('.')[0] in ('torch', 'numpy', 'ogb', 'torch_geometric')}
    control = load(HERE / 'control.py', '_message_own_source_check')
    identity = control.source_identity()
    assert identity['control_id'] == 'be_init__allocation_I_message_own'
    assert identity['charged_reverse_calls_per_update'] == 3
    assert identity['additional_reverse_calls_over_I'] == 1
    assert identity['full_source_training_reverse_collections'] == 3 * 25800
    assert identity['full_source_training_member_forwards'] == 8 * 25800
    assert identity['source_preparation_adopts_execution'] is False

    class UnreadablePath:
        def __fspath__(self):
            raise AssertionError('Disabled guard touched an input/output path')

    for admission in (None, {}, {'fixed18_closed': True, 'message_question_warranted': True,
        'admitted_by': 'root', 'decision_reference': ''}):
        try:
            control.run_complete(UnreadablePath(), UnreadablePath(), UnreadablePath(), admission=admission)
        except ValueError:
            pass
        else:
            raise AssertionError('Missing/invalid admission did not fail closed')

    # Routing only: replace the entry to the full driver with a stdlib sentinel.
    # No Session, auditor runtime, tensor, data, gradient, or training is invoked.
    actual_load = control._module
    seen = {}

    def stubbed_load(path, name):
        value = actual_load(path, name)
        assert path.name == 'train.py'
        seen['original_adapter'] = value.method.PolicyAdapter

        def sentinel(*args, **kwargs):
            seen['args'], seen['kwargs'] = args, kwargs
            bound = value.control_identity('I', 'be_init', .5)
            assert bound['control_id'] == identity['control_id'] and bound['charged_reverse_calls_per_update'] == 3
            assert value.method.REVERSE_CALLS['I'] == 3
            assert value.method.PolicyAdapter is not seen['original_adapter']
            seen['bound_identity'] = bound
            return {'source_routing_only': True}

        value.run_complete = sentinel
        return value

    control._module = stubbed_load
    admission = {'fixed18_closed': True, 'message_question_warranted': True,
        'admitted_by': 'root', 'decision_reference': 'source-check sentinel; NOT a root adoption decision'}
    result = control.run_complete('unread TRAIN sentinel', 'unread VALID sentinel', 'uncreated output sentinel',
        admission=admission, seed=6101, device='cpu')
    assert result == {'source_routing_only': True}
    assert seen['args'] == ('molhiv', 'I', 'be_init', 'unread TRAIN sentinel',
        'unread VALID sentinel', 'uncreated output sentinel', .5)
    assert seen['kwargs'] == {'seed': 6101, 'device': 'cpu'}
    assert numeric_before == {n for n in sys.modules if n.split('.')[0] in ('torch', 'numpy', 'ogb', 'torch_geometric')}

    method_tree = trees['method.py']
    calls = [n for n in ast.walk(method_tree) if isinstance(n, ast.Call)]
    assert sum(isinstance(n.func, ast.Name) and n.func.id == 'native' for n in calls) == 3
    assert not any(isinstance(n.func, ast.Attribute) and n.func.attr in ('step', 'backward', 'zero_grad', 'forward') for n in calls)
    assert sum(isinstance(n.func, ast.Attribute) and n.func.attr == 'train_step' for n in calls) == 1
    assert not any(isinstance(n, ast.Import) and any(a.name.split('.')[0] in ('torch', 'numpy', 'ogb') for a in n.names)
        for t in trees.values() for n in ast.walk(t))
    text = (HERE / 'method.py').read_text()
    assert "self.counters['autograd_grad_calls'] += 1" in text
    assert 'replacement_loss = self.own + .05 * self.alignment' in text
    assert 'gradients[self.message_index] = replacement' in text
    assert 'parameter is session.model.models[0].message_factors' in text
    assert 'setattr(self.raw, name, value)' in text
    catalog = json.loads((PARENT / 'public_internal_be_private_steering_adapter_20261007_v1' / 'BACKBONE_AUDIT.json').read_text())['tasks']['molhiv']
    (name, entry), = catalog['extra_private'].items()
    assert entry == {'role': 'internal_private', 'shape': [5, 4, 256]}
    assert catalog['internal_private_tensor_count'] == 37

    report = {'schema': 'molhiv-message-own-source-check-v1', 'complete': True,
        'scope': 'stdlib source checks only; no numerical/runtime qualification',
        'dependency_integrity': dependencies, 'syntax_without_source_execution_passed': True,
        'disabled_guard_rejects_before_input_output_path_access': True,
        'stdlib_identity_and_full_delegate_routing_sentinel_passed': True,
        'catalog_message_name_read': name, 'catalog_message_shape': entry['shape'],
        'catalog_internal_tensor_count': 37, 'only_one_catalog_extra_private_tensor': True,
        'actual_reverse_call_sites': 3, 'delegated_update_call_sites': 1,
        'new_optimizer_or_model_or_epoch_loop': False,
        'numerical_modules_imported_models_constructed_data_read_gradients_run_fits_run': 0,
        'runtime_verified': False, 'role_data_runtime_qualified': False,
        'full_fit': False, 'selected_readout_qualified': False,
        'source_checks_are_not_runtime_evidence': True, 'source_preparation_adopts_execution': False,
        'current_families_modified': False, 'TEST_scoring': False}
    (HERE / 'SOURCE_CHECK.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'complete': True, 'source_only': True, 'runtime_verified': False,
        'source_check': str(HERE / 'SOURCE_CHECK.json')}))


if __name__ == '__main__':
    main()
