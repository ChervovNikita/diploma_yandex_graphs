"""Source/namespace regression only; no tensors, arrays, data, models or remote work."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
E = HERE.parents[1]
REDESIGN = E.parent/'reference_wrapper_redesign_20261013_v1'


def parse(path): return ast.parse(path.read_text(), filename=str(path))
def function(tree, name): return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def hydration(path, native, callback):
    body = function(parse(path), 'buddy_dataset').body
    start = next(i for i, n in enumerate(body) if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'args')
    end = next(i for i, n in enumerate(body) if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'cache')
    scope = {'copy': copy, 'n': native, 'callback': callback}
    exec(compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(body[start:end]), type_ignores=[])), str(path), 'exec'), scope)
    return scope['args']


def rejected(operation):
    try: operation()
    except ValueError: return True
    return False


def check():
    assert not any(name in sys.modules for name in ('torch', 'numpy', 'scipy'))
    callbacks = parse(REDESIGN/'smoke_20261014/link_port/callbacks.py')
    scope = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function(callbacks, '_settings'), function(callbacks, '_fixed_config')], type_ignores=[])), '<frozen scalar author configuration>', 'exec'), scope)
    callback = SimpleNamespace(**scope); config = callback._fixed_config('BUDDY')
    cls = next(n for n in callbacks.body if isinstance(n, ast.ClassDef) and n.name == 'BUDDYSession')
    ctor = function(cls, '__init__')
    assignment = next(n for n in ctor.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr == 'args' for t in n.targets))
    args = eval(compile(ast.Expression(assignment.value), '<frozen constructor namespace>', 'eval'), {'SimpleNamespace': SimpleNamespace})
    native = SimpleNamespace(args=args, recipe={'native_fixed_config': config}); original = copy.deepcopy(vars(args))
    dataset = next(n for n in parse(REDESIGN/'native_sources/buddy/buddy_src_datasets_elph.py').body if isinstance(n, ast.ClassDef) and n.name == 'HashDataset')
    required = {n.attr for n in ast.walk(dataset) if isinstance(n, ast.Attribute) and
                ((isinstance(n.value, ast.Name) and n.value.id == 'args') or
                 (isinstance(n.value, ast.Attribute) and isinstance(n.value.value, ast.Name) and n.value.value.id == 'self' and n.value.attr == 'args'))}
    old_args = hydration(E/'composable_normalization_final_scoring_v2/native_link.py', native, callback)
    assert required-set(vars(old_args)) == {'subgraph_feature_batch_size'}
    try: old_args.subgraph_feature_batch_size
    except AttributeError: pass
    else: raise AssertionError('Original failure did not reproduce')
    pairs = [('composable_normalization_final_scoring_v2', 'composable_normalization_final_scoring_v3', 'composed-normalization-final-scoring-source-v2', 'composed-normalization-final-scoring-source-v3'),
             ('final_selected_scoring_v5', 'final_selected_scoring_v6', 'final-selected-scoring-source-v5', 'final-selected-scoring-source-v6')]
    for old, new, old_schema, new_schema in pairs:
        args_out = hydration(E/new/'native_link.py', native, callback)
        assert not required-set(vars(args_out)) and args_out.subgraph_feature_batch_size == 11000000
        assert vars(native.args) == original and args_out is not native.args
        assert {k:v for k,v in vars(args_out).items() if k != 'subgraph_feature_batch_size'} == vars(old_args)
        for bad in (None, 0, 65536, True, 11000000.0):
            fixture = copy.deepcopy(native); fixture.recipe['native_fixed_config']['subgraph_feature_batch_size'] = bad
            assert rejected(lambda: hydration(E/new/'native_link.py', fixture, callback))
        fixture = copy.deepcopy(native); del fixture.recipe['native_fixed_config']['subgraph_feature_batch_size']
        assert rejected(lambda: hydration(E/new/'native_link.py', fixture, callback))
        fixture = copy.deepcopy(native); fixture.args.subgraph_feature_batch_size = 65536
        assert rejected(lambda: hydration(E/new/'native_link.py', fixture, callback))
        fixture.args.subgraph_feature_batch_size = 11000000
        assert hydration(E/new/'native_link.py', fixture, callback).subgraph_feature_batch_size == 11000000
        for name in ('__init__.py', 'contracts.py', 'role_serving.py', 'seal.py', 'SOURCE_BINDINGS.json'):
            assert (E/old/name).read_bytes() == (E/new/name).read_bytes()
        assert (E/old/'backend.py').read_text().replace(old_schema, new_schema) == (E/new/'backend.py').read_text()
        for name in ('load_split', 'load_collab'):
            assert ast.dump(function(parse(E/old/'native_link.py'), name)) == ast.dump(function(parse(E/new/'native_link.py'), name))
        for row in json.loads((E/old/'MANIFEST.json').read_text())['files']:
            raw = (E/old/row['path']).read_bytes(); assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']
    assert (E/pairs[0][1]/'native_link.py').read_bytes() == (E/pairs[1][1]/'native_link.py').read_bytes()
    replay = parse(HERE/'replay.py'); compile(replay, str(HERE/'replay.py'), 'exec')
    calls = [n for n in ast.walk(replay) if isinstance(n, ast.Call)]
    assert not any(isinstance(n.func, ast.Attribute) and n.func.attr in {'run', 'train_epoch', 'train_update', 'train_diagnostic', 'backward', 'step', 'validate', 'score_fit'} for n in calls)
    predictions = [n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == 'predict']
    assert len(predictions) == 1 and isinstance(predictions[0].args[3], ast.Constant) and predictions[0].args[3].value == 'VALID'
    inputs = json.loads((HERE/'INPUTS.json').read_text()); retained = json.loads((E/'buddy_seed1_RUNTIME_CHECK_31.json').read_text())
    fit = retained['result']['invocations'][-1]['result']
    assert inputs['selected_checkpoint'] == fit['selected_checkpoint'] and inputs['selected_VALID_export'] == fit['selected_VALID_export']
    assert inputs['failed_qualification'] == retained['child_result_receipt'] and not any(fit['optimizer_calls_this_invocation']['completed'])
    assert inputs['TRAIN_updates'] == inputs['TEST_calls'] == inputs['automatic_retries'] == 0
    assert not any(name in sys.modules for name in ('torch', 'numpy', 'scipy'))
    report = {'status': 'passed', 'scope': 'Scalar namespaces from actual frozen constructor/configuration and AST of HashDataset; no native execution',
              'original_missing_attribute_reproduced': True, 'HashDataset_required_args': sorted(required), 'frozen_batch_size': 11000000,
              'all_HashDataset_argument_reads_supplied': True, 'producer_namespace_unchanged': True, 'missing_wrong_type_wrong_value_conflict_rejected': True,
              'both_successor_helpers_identical': True, 'unrelated_serving_functions_and_frozen_predecessors_unchanged': True,
              'one_literal_VALID_call_no_fit_TRAIN_TEST_calls': True, 'exact_failed_attempt_selected_state_and_export_bound': True,
              'model_tensor_array_data_checkpoint_execution': False, 'numerical_parity_claimed': False}
    print(json.dumps(report, indent=2))


if __name__ == '__main__': check()
