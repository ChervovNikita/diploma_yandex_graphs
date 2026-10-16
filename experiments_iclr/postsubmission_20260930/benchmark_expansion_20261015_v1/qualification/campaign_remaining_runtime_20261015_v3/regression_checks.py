"""Actual host JSON and exact native epoch-loop AST regressions; scalar execution only."""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from runtime_guards import projection_host, guard_training_transition, native_update_call

HERE = Path(__file__).resolve().parent
E = HERE.parents[1]


def expect_value_error(function):
    try:
        function()
    except ValueError:
        return
    raise AssertionError('Expected finite guard rejection')


def run():
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    host_path = E/'server51_campaign_port_v1/hosts/allocation_gpu0.json'
    raw = host_path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == 'c5f0937180ed2a24b61471d46848125d7c3f01a39c825c805091ecd922eec94b'
    config = json.loads(raw)
    assert 'host' not in config
    adapted = projection_host(config)
    assert adapted == {**config, 'host': 'allocation'} and 'host' not in config
    expect_value_error(lambda: projection_host({**config, 'host': 'server51'}))
    expect_value_error(lambda: projection_host({**config, 'worker': 'server51/gpu0'}))

    native_path = E/'portable_dispatcher_v3/runtime_node_molecular/adapters.py'
    source = native_path.read_bytes()
    assert hashlib.sha256(source).hexdigest() == 'b066c8ca696b734f345701e5778bc8af4444945d4c64c1e4d151c34967a05fc7'
    parsed = ast.parse(source)
    exception = next(n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name == 'EpochComplete')
    adapter = next(n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name == 'OGBAdapter')
    train = next(n for n in adapter.body if isinstance(n, ast.FunctionDef) and n.name == 'train_epoch')
    loop = next(n for n in ast.walk(train) if isinstance(n, ast.While))
    # Execute the unchanged scalar control loop, not copied/reimplemented loop logic.
    # Dataset iteration, tensors, forwards and optimizers are never imported/executed.
    wrapper = ast.parse('def drive(self):\n    losses, work = [], []\n    return losses, work\n').body[0]
    wrapper.body.insert(1, copy.deepcopy(loop))
    assert ast.dump(wrapper.body[1]) == ast.dump(loop)
    tree = ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(exception), wrapper], type_ignores=[]))
    scope = {'math': math}
    exec(compile(tree, str(native_path)+' [exact scalar loop only]', 'exec'), scope)
    epoch_complete, drive = scope['EpochComplete'], scope['drive']

    def fixture(mode='normal'):
        work = {'TRAIN_update_attempts': 0, 'TRAIN_updates': 0, 'TRAIN_callback_calls': 0,
            'TRAIN_exhaustion_probe_attempts': 0, 'TRAIN_exhaustion_probes_completed': 0,
            'optimizer_attempts': 0, 'optimizer_completed': 0,
            'native_forwards_attempted': 0, 'native_forwards_completed': 0, 'forbidden_calls': 0}
        original_calls = 0
        def forbid(message):
            work['forbidden_calls'] += 1
            raise ValueError(message)
        def callback():
            nonlocal original_calls
            original_calls += 1
            if mode == 'early' and original_calls == 6:
                raise epoch_complete()
            if original_calls == 39:
                if mode == 'extra_forward':
                    guard_training_transition(work, True, forbid)
                    work['native_forwards_attempted'] += 1
                if mode == 'returns_extra':
                    return {'loss': .25}
                raise epoch_complete()
            guard_training_transition(work, True, forbid)
            # Scalar counters only; not a native or tensor execution witness.
            work['native_forwards_attempted'] += 4
            work['native_forwards_completed'] += 4
            work['optimizer_attempts'] += 1
            work['optimizer_completed'] += 1
            return {'loss': .25, 'work': {'metadata_only': True}}
        def update():
            return native_update_call(callback, work, True, epoch_complete, lambda: None, forbid)
        return work, lambda: original_calls, update, callback, forbid

    work, calls, update, callback, forbid = fixture()
    losses, records = drive(SimpleNamespace(session=SimpleNamespace(train_update=update)))
    assert len(losses) == len(records) == 38 and calls() == 39
    assert work == {'TRAIN_update_attempts': 38, 'TRAIN_updates': 38, 'TRAIN_callback_calls': 39,
        'TRAIN_exhaustion_probe_attempts': 1, 'TRAIN_exhaustion_probes_completed': 1,
        'optimizer_attempts': 38, 'optimizer_completed': 38,
        'native_forwards_attempted': 152, 'native_forwards_completed': 152, 'forbidden_calls': 0}
    expect_value_error(update)
    assert calls() == 39 and work['TRAIN_updates'] == work['optimizer_completed'] == 38
    for mode in ('extra_forward', 'returns_extra', 'early'):
        state, count, guarded, _, _ = fixture(mode)
        expect_value_error(lambda: drive(SimpleNamespace(session=SimpleNamespace(train_update=guarded))))
        assert state['forbidden_calls'] == 1 and state['optimizer_completed'] <= 38
        assert state['native_forwards_attempted'] == 4*state['TRAIN_updates']
        assert state['TRAIN_exhaustion_probes_completed'] == 0
    state, count, _, original, reject = fixture()
    expect_value_error(lambda: native_update_call(original, state, False, epoch_complete, lambda: None, reject))
    assert count() == state['TRAIN_callback_calls'] == state['optimizer_completed'] == 0
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    return {'status': 'PASS', 'actual_bundled_host_without_host_key_adapted': True,
        'actual_native_loop_AST_executed_with_scalar_callbacks': True,
        'successful_scalar_updates': 38, 'native_callback_calls_in_scalar_fixture': 39,
        'separate_exhaustion_probe': 1, 'extra_forward_extra_return_early_exhaustion_post_epoch_rejected': True,
        'outside_TRAIN_callback_rejected': True, 'native_model_tensor_array_execution': 0,
        'scientific_TRAIN_updates': 0}


if __name__ == '__main__':
    print(json.dumps(run(), indent=2, sort_keys=True))
