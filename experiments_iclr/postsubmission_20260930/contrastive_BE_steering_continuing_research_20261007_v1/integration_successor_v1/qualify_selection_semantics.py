"""Stdlib checkpoint-dispatch fixture; no framework, scientific data or fit."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
from context_dispatch import selection_policy, local_transition, restore_own_best_bank


class Body:
    def __init__(self):
        self.state = None
        self.global_flag = False

    def load_state_dict(self, state):
        self.state = state

    def set_global(self, value):
        self.global_flag = value


class Optimizer:
    def __init__(self):
        self.history = None

    def load_state_dict(self, value):
        self.history = value


class JointSelector:
    def __init__(self):
        self.calls = []

    def local_transition(self, model, optimizers, load, ordinary_independent):
        self.calls.append(ordinary_independent)
        assert ordinary_independent is False
        load('selected_local.pt')


def session_fixture(arm, mode=None):
    count = 1 if arm.startswith('single') else 4
    model = SimpleNamespace(arm=arm, members=count,
        independent=arm.startswith(('single', 'independent4')),
        contrastive=arm.endswith('_contrastive'), models=[Body() for _ in range(count)])
    model.set_global = lambda flag: [body.set_global(flag) for body in model.models]
    facade = SimpleNamespace(mode=mode) if mode is not None else None
    joint = JointSelector()
    config = {'contrastive': {'alignment_weight': .05, 'max_objects': 512,
                              'residual_weight': 0., 'temperature': .2}}
    session = SimpleNamespace(arm=arm, model=model,
        optimizers=[Optimizer() for _ in range(count if model.independent else 1)],
        core={'objectives': facade, 'selection': joint}, config=config,
        streams=('unchanged end-local member stream markers',))
    return session, facade, joint


def main():
    checked = []
    for method, arm, mode in (
        ('independent4_native', 'independent4', None),
        ('independent4_route_context', 'independent4_contrastive', 'route')):
        session, facade, joint = session_fixture(arm, mode)
        policy = selection_policy(method, session, facade)
        names = []
        def load(name):
            names.append(name)
            member = int(name.rsplit('_', 1)[1].split('.')[0])
            return {'model': 'own model '+str(member), 'optimizer': 'own Adam '+str(member),
                    'global': bool(member % 2), 'epoch': 31+member}
        local_transition(session, load, policy, facade)
        assert names == ['own_local_'+str(m)+'.pt' for m in range(4)]
        assert not joint.calls and session.model.arm == arm
        assert all(body.state == 'own model '+str(m) and body.global_flag is True
                   for m, body in enumerate(session.model.models))
        assert [opt.history for opt in session.optimizers] == ['own Adam '+str(m) for m in range(4)]
        assert session.streams == ('unchanged end-local member stream markers',)
        names.clear()
        epochs = restore_own_best_bank(session, load, policy, facade)
        assert epochs == [31, 32, 33, 34]
        assert names == ['own_best_'+str(m)+'.pt' for m in range(4)]
        assert [body.global_flag for body in session.model.models] == [False, True, False, True]
        assert [opt.history for opt in session.optimizers] == ['own Adam '+str(m) for m in range(4)]
        checked.append(method+': own local histories, unchanged streams, own evaluation flags')
    for method, arm, mode in (
        ('shared_common', 'be_unit_contrastive', 'common'),
        ('single_common_context', 'single_contrastive', 'common'),
        ('single_native', 'single', None)):
        session, facade, joint = session_fixture(arm, mode)
        policy = selection_policy(method, session, facade)
        names = []
        local_transition(session, lambda name: names.append(name), policy, facade)
        assert names == ['selected_local.pt'] and joint.calls == [False]
        checked.append(method+': original joint local selector delegated')
    session, facade, _ = session_fixture('independent4_contrastive', 'route')
    session.config['contrastive']['residual_weight'] = .05
    try:
        selection_policy('independent4_route_context', session, facade)
    except ValueError:
        checked.append('coupled residual objective rejected before own-bank splicing')
    else:
        raise AssertionError('legacy coupled contrast must be rejected')
    session, facade, _ = session_fixture('independent4_contrastive', 'route')
    session.model.arm = 'independent4'
    try:
        selection_policy('independent4_route_context', session, facade)
    except ValueError:
        checked.append('hidden underlying-arm normalization rejected')
    else:
        raise AssertionError('arm alias must be rejected')
    here = Path(__file__).resolve().parent
    for name in ('context_dispatch.py', 'train_context.py', 'context_recompute.py'):
        ast.parse((here/name).read_text())
    driver = (here/'train_context.py').read_text()
    replay = (here/'context_recompute.py').read_text()
    assert "ordinary_independent = args.arm" not in driver
    assert 'if policy.own_selected_four:' in driver and 'metric > best' in driver
    assert 'per[member] > own_best[member]' in driver
    assert 'own.sum() + members * auxiliary if session.model.independent else own.mean() + auxiliary' in replay
    checked.append('materialized full driver/replay AST and explicit strict-first/scaling sites')
    report = dict(status='STDLIB_DISPATCH_FIXTURE_ONLY', checks=checked,
        framework_imports=0, scientific_data_access=False, scientific_fits=0,
        numerical_native_or_CUDA_qualification=False)
    (here/'SELECTION_SOURCE_QUALIFICATION.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
