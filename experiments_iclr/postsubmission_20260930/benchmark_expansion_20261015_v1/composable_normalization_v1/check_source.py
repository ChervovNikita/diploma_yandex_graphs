"""One stdlib protocol fixture plus source/contract checks; no tensor runtime.

Scalar fixtures exercise constructor seams and ownership, not BN mathematics,
real candidate algorithms, native numeric parity, resources or accuracy.
"""
import ast
from contextlib import contextmanager, nullcontext
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import weakref

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import adapter
import contracts


class Buffer:
    shape, dtype, device = (1,), 'scalar-fixture', 'cpu-fixture'
    def __init__(self, value=0): self.value = value
    def detach(self): return self
    def clone(self): return Buffer(self.value)
    def data_ptr(self): return id(self)


class Hooks:
    def __init__(self): self.hooks = {'forward': [], 'export': [], 'load': []}; self.training = True
    def hook(self, name, fn):
        self.hooks[name].append(fn)
        return SimpleNamespace(remove=lambda: self.hooks[name].remove(fn))
    def register_forward_pre_hook(self, fn): return self.hook('forward', fn)
    def register_state_dict_pre_hook(self, fn): return self.hook('export', fn)
    def register_load_state_dict_pre_hook(self, fn): return self.hook('load', fn)
    def dispatch(self, name):
        for fn in self.hooks[name]: fn(self, ())


class Native(Hooks):
    def __init__(self): super().__init__(); self.buffer = Buffer(); self.parameter = object()
    def named_parameters(self): return [('weight', self.parameter)]
    def named_buffers(self): return [('running', self.buffer)]
    def modules(self): return [self]
    def eval(self): self.training = False
    def __call__(self):
        self.dispatch('forward')
        if self.training: self.buffer.value += 1
        return 'unchanged-parent-prediction'
    def state_dict(self): self.dispatch('export'); return {'running': self.buffer.value}
    def load_state_dict(self, value): self.dispatch('load'); self.buffer.value = value['running']


class Banks(Hooks):
    def __init__(self, native):
        super().__init__(); self.native = native; self.original = native.buffer
        self.values = [Buffer() for _ in range(4)]
        self._active = None; self._refs = [weakref.ref(native)]
    def _native_ref(self): return self.native
    def assert_native_identity(self): assert self.native.buffer is self.original
    def named_buffers(self): return [(str(i), b) for i, b in enumerate(self.values)]
    @contextmanager
    def route(self, member):
        assert self._active is None
        self._active = member; self.native.buffer = self.values[member]
        try: yield
        finally: self._active = None; self.native.buffer = self.original
    def state_dict(self): self.dispatch('export'); return [x.value for x in self.values]
    def load_state_dict(self, value, strict):
        self.dispatch('load'); assert strict and len(value) == 4
        for bank, stored in zip(self.values, value): bank.value = stored
    def metadata(self): return {'fixture_only': True, 'additional_learned_parameters': 0}


class Core:
    def __init__(self, native): self.native = native; self.training = True
    def named_parameters(self): return self.native.named_parameters()
    def modules(self): return [self, self.native]
    def eval(self): self.training = self.native.training = False
    def run_member(self, member, batch): return self.native()


class Parent:
    def __init__(self, family):
        self.torch = SimpleNamespace(equal=lambda x, y: x.value == y.value, no_grad=nullcontext)
        native = Native(); self.family = family
        self.model = Core(native) if family == 'ogb' else native
        self.optimizers = [SimpleNamespace(param_groups=[{'params': [native.parameter]}])]
        self.session = SimpleNamespace(torch=self.torch, updates=0)
        self.completed_epochs = 0; self.calls = []; self.closed = False
        if family == 'link':
            @contextmanager
            def route(member, training): yield
            self.session.native = SimpleNamespace(model=native, _route=route)
        self.native = native
    def forwards(self, training):
        self.model.training = self.native.training = training
        for member in range(4):
            if self.family == 'ogb': self.model.run_member(member, None)
            else:
                with self.session.native._route(member, training): self.native()
    def train_epoch(self):
        self.calls.append('parent-train'); self.forwards(True)
        self.completed_epochs += 1; self.session.updates += 1
        return {'parent_loss_and_gradient_rule': 'unchanged scalar-fixture marker'}
    def validate(self):
        self.calls.append('parent-valid'); self.forwards(False)
        return {'same_call_mean_provider': 'unchanged scalar-fixture marker'}
    def state(self):
        return {'model': self.native.state_dict(), 'epochs': self.completed_epochs, 'updates': self.session.updates}
    def load_state(self, value):
        self.native.load_state_dict(value['model'])
        self.completed_epochs = value['epochs']; self.session.updates = value['updates']
    def assert_identity(self): pass
    def metadata(self): return {'parent': 'scalar ownership fixture, not a native model'}
    def close(self): self.closed = True


def refuses(fn):
    try: fn()
    except (ValueError, RuntimeError): return
    raise AssertionError('Expected retained ownership refusal')


def main():
    bindings = json.loads((HERE / 'PARENT_BINDINGS.json').read_text())
    for row in bindings['files']:
        raw = (HERE.parent.parent / row['path_relative_to_phase']).read_bytes()
        assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']
    roster = json.loads((HERE.parent / 'expanded_study_design_v1/PROPOSED_ROSTER.json').read_text())
    for name in ('contracts.py', 'ownership.py', 'adapter.py', 'check_source.py'):
        tree = ast.parse((HERE / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import): assert all(not x.name.startswith(('torch', 'numpy')) for x in node.names)
            if isinstance(node, ast.ImportFrom): assert not (node.module or '').startswith(('torch', 'numpy'))
    cases = [('ogb', 'ogb-ogbg-molbace-gin-drop0.5', 'interaction_geometry_input'),
             ('ogb', 'ogb-ogbg-molhiv-gcn-virtual-drop0.5', 'interaction_input_conditionalJ'),
             ('link', 'collab-buddy', 'portable_input_residual_only_v1')]
    for family, preset, method in cases:
        parent_job = {'id': 'fixture/' + method, 'scientific_method_id': method, 'recipe_id': preset,
            'family': family, 'TEST_access': False, 'body': 0, 'policy': {'track': 'long_convergence'},
            'native_loss_and_optimizer': 'immutable fixture marker'}
        declaration = {'schema': contracts.SCHEMA, 'policy': contracts.POLICY,
            'source_manifest_sha256': '0'*64, 'banks_sha256': '1'*64,
            'parent_descriptor': {'fixture_only': True}, 'parent_job_sha256': contracts.digest(parent_job),
            'matched_control_method_id': contracts.CONTROL}
        job = contracts.migrated_job(parent_job, declaration, roster)
        assert parent_job['id'] + contracts.SUFFIX == job['id']
        assert job['native_loss_and_optimizer'] == parent_job['native_loss_and_optimizer']
        bad = copy.deepcopy(parent_job); bad['scientific_method_id'] = 'gnnm_boundary_native'
        refuses(lambda: contracts.eligible(bad, roster))
        inner = Parent(family); manager = Banks(inner.native)
        original = inner.train_epoch
        composed = adapter.NormalizedAdapter(inner, job, manager, {'fixture_only': True})
        assert inner.train_epoch == original and composed.optimizers is inner.optimizers
        refuses(inner.native)
        refuses(manager.state_dict)
        composed.train_epoch()
        assert [x.value for x in manager.values] == [1]*4 and inner.native.buffer.value == 0
        state = composed.state(); manager.values[0].value = 999
        composed.load_state(state)
        assert [x.value for x in manager.values] == [1]*4
        assert composed.validate()['same_call_mean_provider'].startswith('unchanged')
        inner.model.training = True
        before = list(inner.calls); modes = [x.training for x in composed.model.modules()]
        try:
            with composed.inference_operation('TEST'):
                refuses(composed.state); refuses(composed.train_epoch)
                assert inner.calls == before
                inner.forwards(False)
                raise LookupError('fixture exception')
        except LookupError: pass
        assert modes == [x.training for x in composed.model.modules()]
        bad_state = copy.deepcopy(state); bad_state['normalization_producer']['job_sha256'] = 'changed'
        refuses(lambda: composed.load_state(bad_state))
        if family == 'link':
            def mutate_inference_bank():
                with composed.inference_operation('VALID'): manager.values[0].value += 1
            refuses(mutate_inference_bank)
            assert composed._invalid is not None
            refuses(composed.state)
        composed.close(); assert inner.closed
    mapping = {'ingredient': 'ingredient'+contracts.SUFFIX, 'candidate': 'candidate'+contracts.SUFFIX}
    old = {'fit_id': 'candidate', 'controls': [
        {'resolved_control': 'gnnm_boundary_native', 'fit_ids': ['original']},
        {'resolved_control': 'ingredient', 'fit_ids': ['ingredient']},
        {'resolved_control': 'native_I4', 'fit_ids': ['body0','body1','body2','body3']}]}
    new = contracts.migrated_controls(old, mapping, 'private-BN-control')
    assert new['normalization_attribution_reference']['fit_ids'] == ['private-BN-control']
    assert new['controls'][0]['fit_ids'] == ['original']
    assert new['controls'][1]['fit_ids'] == [mapping['ingredient']]
    assert new['controls'][2] == old['controls'][2]
    assert not any(name.startswith(('torch', 'numpy')) for name in sys.modules)
    print(json.dumps({'schema': 'composable-normalization-source-check-v1', 'status': 'PASS',
        'scalar_protocol_cases': len(cases), 'source_pins_verified': len(bindings['files']),
        'state_restore_role_exception_and_control_link_checks': True,
        'real_native_compositions_qualified': False, 'models': 0, 'native_forwards': 0,
        'optimizer_updates': 0, 'TEST_payload_accessed': False, 'remote_calls': 0,
        'roster_regenerated': False, 'campaign_GO': False}, indent=2))


if __name__ == '__main__': main()
