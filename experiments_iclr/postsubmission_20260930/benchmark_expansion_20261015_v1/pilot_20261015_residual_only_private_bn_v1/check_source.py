"""Local standard-library descriptor/composition checks; no scientific execution."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
E = HERE.parent
P = E.parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check():
    spec = json.loads((HERE / 'SPEC.json').read_text())
    modspec = importlib.util.spec_from_file_location('_missing_arm_source_check', HERE / 'session_extension.py')
    extension = importlib.util.module_from_spec(modspec)
    modspec.loader.exec_module(extension)
    original = json.loads((E / 'pilot_20261015_bn_geometry_correction_v1/SPEC.json').read_text())
    bn = json.loads((E / 'pilot_20261015_bn_state_v1/SPEC.json').read_text())
    for pin in spec['local_inputs']:
        path = P / pin['path_relative_to_phase']
        assert path.is_file() and sha(path) == pin['sha256'], str(path)
    for relative, pin in extension.PINS.items():
        assert sha(E / relative) == pin, relative
    assert spec['new_fits'] == 3 and spec['scientific_launch_enabled'] is False
    jobs = [json.loads((HERE / row['file']).read_text()) for row in spec['jobs']]
    assert [j['seed'] for j in jobs] == [101, 202, 303]
    invariant = ('schema', 'family', 'callback_module', 'callback_sha256', 'data_root', 'policy',
                 'TEST_access', 'split', 'device', 'torch_threads', 'base_procedure', 'body')
    sys.path.insert(0, str(E / 'runtime'))
    from run_fit import validate_job
    for row, job in zip(spec['jobs'], jobs):
        assert row['sha256'] == sha(HERE / row['file'])
        assert all(job[k] == original['common_job'][k] == bn['jobs'][0][k] for k in invariant)
        recipe = copy.deepcopy(job['recipe'])
        assert recipe.pop('_repository_root') == spec['project_root']
        assert recipe == original['common_job']['recipe'] == bn['jobs'][0]['recipe']
        assert job['method_settings'] == extension.settings_for()
        assert job['session_transform']['sha256'] == sha(HERE / 'session_extension.py')
        validate_job(job)
    assert len(spec['reused_wrapper_controls']) == 9
    assert {(r['seed'], r['control']) for r in spec['reused_wrapper_controls']} == \
        {(s, c) for s in [101, 202, 303] for c in ['original_G0', 'private_BN_base', 'private_BN_BE_plus_residual']}
    assert len(spec['reused_native_reference_scalars']) == 6
    for path in HERE.glob('*.py'):
        ast.parse(path.read_text())
    # Mock only imported existing components. Exercise the new composition's
    # call order, settings, identity/accounting and checkpoint binding law.
    class Parameter:
        def __init__(self, count): self.count = count
        def numel(self): return self.count
        def element_size(self): return 4
        def count_nonzero(self): return SimpleNamespace(item=lambda: 0)
    native_p = Parameter(1000)
    native = SimpleNamespace(parameters=lambda: (native_p,), buffers=lambda: (Parameter(20),))
    old = SimpleNamespace(native=native, private_parameters=lambda: (Parameter(4300),))
    private = (Parameter(9600), Parameter(9600))
    new = SimpleNamespace(native=native, private_parameters=lambda: private,
                          parameters=lambda: (native_p,) + private,
                          input_residual=SimpleNamespace(up=private[1]),
                          assert_source_identity=lambda: None)
    session = SimpleNamespace(model=old, optimizer=object(), updates=0,
                              config=extension.NATIVE_CONFIG, procedure='gnnm_boundary_native',
                              metadata=lambda: {}, state=lambda: {'source': True}, load_state=lambda value: None)
    calls = []
    own_update = lambda: None
    def loader(root, relative):
        def transform(s, job):
            calls.append((relative, copy.deepcopy(job)))
            if relative.startswith('portable'):
                assert job['procedure'] == 'portable_input_residual_only_v1'
                s.model = new
                s.portable_private_input_metadata = {'BE_retained': False}
            elif relative.startswith('molecular'):
                assert job['method_settings']['auxiliary'] is False and job['method_settings']['input_residual'] is False
                s.train_update = own_update
            else:
                assert job['base_procedure'] == 'gnnm_boundary_native'
                new.route_bn_state = SimpleNamespace(metadata=lambda: {'bank_bytes': 320})
            return s
        return SimpleNamespace(transform=transform)
    extension._load = loader
    extension.transform(session, jobs[0])
    assert [c[0] for c in calls] == ['portable_private_input_v2/extension.py',
                                   'molecular/session_extension.py', 'private_bn_state_v3/session_extension.py']
    account = session.metadata()['candidate_extension']['parameter_and_state_accounting']
    assert account['total_trainable_parameters'] == 20200 and account['private_trainable_parameters'] == 19200
    assert account['removed_BE_parameters'] == 4300 and account['extra_Adam_two_moment_bytes'] == 153600
    state = session.state()
    session.load_state(state)
    broken = copy.deepcopy(state)
    broken['residual_only_private_bn_binding']['seed'] = 202
    try: session.load_state(broken)
    except ValueError: pass
    else: raise AssertionError('Changed checkpoint seed accepted')
    for key, bad in [('seed', 404), ('TEST_access', True), ('policy', {**jobs[0]['policy'], 'patience': 51})]:
        broken = copy.deepcopy(jobs[0]); broken[key] = bad
        try: extension.transform(session, broken)
        except ValueError: pass
        else: raise AssertionError('Changed frozen descriptor accepted: ' + key)
    return {'status': 'passed', 'jobs': 3, 'local_input_hashes': len(spec['local_inputs']),
            'source_hashes': len(extension.PINS), 'parent_recipe_policy_data_callback_equal': True,
            'new_composition_mock_order_state_accounting_guards': 'passed',
            'TEST_accessed': False, 'new_model_constructions': 0, 'new_optimizer_steps': 0,
            'numerical_composition_qualification': 'pending root-reviewed engineering execution',
            'scientific_execution_started': False}


if __name__ == '__main__':
    print(json.dumps(check(), indent=2))
