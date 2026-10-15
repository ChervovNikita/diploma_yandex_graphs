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
import tempfile
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
        self.torch = SimpleNamespace(equal=lambda x, y: x.value == y.value, no_grad=nullcontext,
                                     is_tensor=lambda value: False)
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


def actual_parent_class():
    """Execute pinned parent lifecycle bodies, without importing its runtime."""
    path = HERE.parent / 'portable_dispatcher_v3/runtime_link/adapters.py'
    parsed = ast.parse(path.read_text())
    cls = next(node for node in parsed.body if isinstance(node, ast.ClassDef)
               and node.name == 'LinkSameCallMeanAdapter')
    keep = {'completed_epochs', '__getattr__', '_invalidate', '_state_signature',
            'state', 'load_state', 'bind_loaded_selection', 'selected_mean_export'}
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in keep]
    namespace = {'copy': copy, 'digest': contracts.digest, 'receipt': adapter.receipt,
                 'inside': lambda value, root: Path(value).resolve()}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(path), 'exec'), namespace)
    return namespace['LinkSameCallMeanAdapter']


def lifecycle_fixture():
    """Regression for actual nested restore, selected envelope and mean export."""
    cls = actual_parent_class()
    with tempfile.TemporaryDirectory(prefix='scalar_lifecycle_', dir=HERE) as temp:
        root = Path(temp)
        def setup(implementation, *, preload=False):
            bare = Parent('link'); bare.completed_epochs = 1; bare.session.updates = 1
            bare.session.native.updates = 1
            verifier = cls.__new__(cls)
            verifier.inner = bare; verifier.torch = bare.torch; verifier.root = root
            verifier.session, verifier.model, verifier.optimizers = bare.session, bare.model, bare.optimizers
            verifier._generation = verifier._serial = 0
            verifier._cache = verifier._loaded_state_signature = verifier._selection = None
            verifier._check_runtime = lambda binding: None  # Explicit fixture; no runtime qualification.
            verifier._tensor_identity = lambda value: {'scalar_fixture': value.value}
            verifier._source_sidecar = lambda actual: {'scalar_fixture': True}
            job = {'id':'scalar-checkpoint-fixture', 'family':'link', 'normalization_composition':{'policy':contracts.POLICY}}
            verifier._job_authority = copy.deepcopy(job)
            if preload:
                verifier.load_state(bare.state())
            banks = Banks(bare.native)
            composed = implementation.NormalizedAdapter(verifier, job, banks, {'fixture_only':True})
            return bare, verifier, banks, composed, job
        # Reproduce v1's actual nested export failure without changing v1.
        import importlib.util
        spec = importlib.util.spec_from_file_location('_preserved_normalization_v1', HERE.parent/'composable_normalization_v1/adapter.py')
        legacy = importlib.util.module_from_spec(spec); spec.loader.exec_module(legacy)
        bare, verifier, banks, old, job = setup(legacy)
        state = old.state()
        try: old.load_state(state)
        except RuntimeError as error:
            assert str(error) == 'Checkpoint must use the idle composed adapter state API'
        else: raise AssertionError('Original nested-load blocker must reproduce')
        old.close()
        # Independently reproduce envelope mismatch after a genuine parent-only
        # restore, rather than forging its loaded-state verification result.
        bare, verifier, banks, old, job = setup(legacy, preload=True)
        state = old.state()
        # Actual parent bind compares the full envelope to its parent-state signature.
        checkpoint = root/'legacy-envelope.json'; checkpoint.write_text(json.dumps(state))
        runtime = {'scalar_fixture_runtime':'not-qualified'}
        payload = {'adapter':state,'selector':{'selected_epoch':1},'job_hash':contracts.digest(job),'runtime_binding':runtime}
        try: old.bind_loaded_selection(payload,selected_checkpoint=adapter.receipt(checkpoint),job_hash=contracts.digest(job),runtime_binding=runtime)
        except ValueError as error:
            assert str(error) == 'Selected payload must be the exact source state just loaded for this JOB'
        else: raise AssertionError('Original selected-envelope blocker must reproduce')
        old.close()

        bare, verifier, banks, current, job = setup(adapter)
        before_methods = (bare.state, bare.load_state)
        state = current.state()
        with current.owner.state('load'):
            refuses(bare.native.state_dict)  # No blanket export permission during load.
        current.load_state(state)
        assert (bare.state,bare.load_state) == before_methods
        checkpoint = root/'genuine-composed-fixture-envelope.json'; checkpoint.write_text(json.dumps(state))
        selected = adapter.receipt(checkpoint)
        payload = {'adapter':state,'selector':{'selected_epoch':1},'job_hash':contracts.digest(job),'runtime_binding':runtime}
        bad_payload = copy.deepcopy(payload); bad_payload['adapter']['private_BN_banks'][0] += 1
        refuses(lambda: current.bind_loaded_selection(bad_payload,selected_checkpoint=selected,job_hash=contracts.digest(job),runtime_binding=runtime))
        current.bind_loaded_selection(payload,selected_checkpoint=selected,job_hash=contracts.digest(job),runtime_binding=runtime)
        assert payload['adapter'] is state and 'private_BN_banks' in payload['adapter']
        authority = {'state_generation':verifier._generation,'validation_serial':verifier._serial,'source_updates':1,
                     'ids':{'scalar_fixture':1},'labels':{'scalar_fixture':2},'query_endpoints':{'scalar_fixture':3},
                     'actual_logits':{'scalar_fixture':4},'mean_logits':{'scalar_fixture':5}}
        cache = {'authority':authority,'ids':Buffer(1),'labels':Buffer(2),'query_endpoints':Buffer(3),
                 'outputs':Buffer(5),'loss':0.5,'metrics':{'fixture':0.5}}
        verifier._cache = cache
        actual = {'same_bank_mean_VALID':cache,'ids':Buffer(1),'labels':Buffer(2),'query_endpoints':Buffer(3),'outputs':Buffer(4)}
        before_calls = list(bare.calls)
        exported = current.selected_mean_export(actual,job_hash=contracts.digest(job),selected_epoch=1,
            selected_checkpoint=selected,runtime_binding=runtime)
        assert bare.calls == before_calls and exported['extra_forward_calls'] == 0
        assert 'normalization_selected_state' in exported
        banks.values[0].value += 1
        refuses(lambda: current.selected_mean_export(actual,job_hash=contracts.digest(job),selected_epoch=1,
            selected_checkpoint=selected,runtime_binding=runtime))
        current.close()
    return {'actual_parent_AST_lifecycle_exercised':True,'legacy_nested_load_blocker_reproduced':True,
        'legacy_selected_envelope_blocker_reproduced':True,'composed_restore_selected_bind_mean_export_passed':True,
        'unscoped_load_export_still_refused':True,'mutated_BN_selected_payload_and_live_bank_refused':True,
        'zero_forward_mean_export_preserved':True,'native_runtime_or_role_data_qualified':False}


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
    lifecycle = lifecycle_fixture()
    print(json.dumps({'schema': 'composable-normalization-source-check-v2', 'status': 'PASS',
        'scalar_protocol_cases': len(cases), 'source_pins_verified': len(bindings['files']),
        'state_restore_role_exception_and_control_link_checks': True, 'selected_lifecycle_regression':lifecycle,
        'real_native_compositions_qualified': False, 'models': 0, 'native_forwards': 0,
        'optimizer_updates': 0, 'TEST_payload_accessed': False, 'remote_calls': 0,
        'roster_regenerated': False, 'campaign_GO': False}, indent=2))


if __name__ == '__main__': main()
