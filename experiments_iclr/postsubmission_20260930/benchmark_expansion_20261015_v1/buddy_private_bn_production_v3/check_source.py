"""Source/metadata and scalar ownership fixtures; no model/tensor/data imports."""
import ast
import copy
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent


def load_source(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def refused(call, kind=(ValueError, RuntimeError)):
    try: call()
    except kind: return True
    raise AssertionError('Expected source guard refusal')


class Buffer:
    shape, dtype, device = (1,), 'scalar-int', 'fixture-host'
    def __init__(self, value): self.value = value
    def data_ptr(self): return id(self)
    def detach(self): return self
    def clone(self): return Buffer(self.value)


class TorchFixture:
    @staticmethod
    def equal(a, b): return a.value == b.value
    @staticmethod
    @contextmanager
    def no_grad(): yield


class Module:
    def __init__(self): self.training = True
    def modules(self): return [self]
    def eval(self): self.training = False
    def named_buffers(self): return [('canonical', self.buffer)]


def fixture(library):
    owner = object.__new__(library.BUDDYPrivateBN)
    owner._operation = owner._state_operation = None
    owner._inference_role = owner._inference_calls = None
    model = Module(); model.buffer = Buffer(3)
    manager = SimpleNamespace(_active=None, names=['bn_feats', 'bn_labels', 'bn_RA'])
    buffers = [('active_route_bank', Buffer(4)), ('inactive_route_bank', Buffer(5))]
    manager.named_buffers = lambda: iter(buffers)
    owner.native = SimpleNamespace(model=model)
    owner.manager, owner.t = manager, TorchFixture
    owner.method_metadata = {'schema':'native-BUDDY-private-BN-production-source-v1',
        'method_id':'gnnm_private_bn_state_v3', 'port':'buddy_private_bn_production_v2',
        'source_receipts':['genuine-producer-fixture']}
    owner._calls = {role: {name: [0]*4 for name in manager.names} for role in ('TRAIN', 'VALID')}
    owner._TRAIN_provider_checks, owner._VALID_provider_checks = 11, 2
    owner.assert_identity = lambda: None
    return owner


def main():
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    source = load_source(HERE/'session.py', '_private_BN_inference_source_check')
    parent_tree = ast.parse((HERE.parent/'buddy_private_bn_production_v2/session.py').read_text())
    tree = ast.parse((HERE/'session.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    old_cls = next(n for n in parent_tree.body if isinstance(n, ast.ClassDef))
    methods = {n.name:n for n in cls.body if isinstance(n, ast.FunctionDef)}
    old = {n.name:n for n in old_cls.body if isinstance(n, ast.FunctionDef)}
    retained = ['assert_identity','_snapshot','_counts','_audit_TRAIN','train_update','train_epoch','validate','metadata']
    for name in retained:
        assert ast.dump(methods[name], include_attributes=False) == ast.dump(old[name], include_attributes=False), name
    state_assign = lambda t: next(n.value for n in t.body if isinstance(n, ast.Assign) and any(isinstance(x, ast.Name) and x.id=='STATE_SCHEMA' for x in n.targets))
    assert ast.dump(state_assign(tree)) == ast.dump(state_assign(parent_tree))
    # Persistent state/save/load bodies are identical after the added live-
    # inference refusal. No saved TEST ledger, schema conversion or source pin
    # rewriting is introduced.
    for name in ('state','load_state'):
        a, b = copy.deepcopy(methods[name]), copy.deepcopy(old[name])
        a.body.pop(0); b.body.pop(0)
        assert ast.dump(a, include_attributes=False) == ast.dump(b, include_attributes=False), name
    report = {}
    for role in ('VALID','TEST'):
        owner = fixture(source); ledger = copy.deepcopy(owner._calls); meta = copy.deepcopy(owner.method_metadata)
        with owner.inference_operation(role) as audit:
            assert owner._inference_role == role and owner._operation is None
            assert owner.native.model.training is False and audit['role'] == role
            assert refused(lambda: owner.state())
            assert refused(lambda: owner.load_state({}))
            assert refused(lambda: owner.close())
            assert refused(lambda: owner.inference_operation(role).__enter__())
            assert refused(lambda: owner.operation('VALID').__enter__())
            owner._inference_calls['bn_feats'][2] += 1
        assert owner._inference_role is None and owner.native.model.training is True
        assert owner._calls == ledger and owner.method_metadata == meta
        assert audit['branch_calls_by_route']['bn_feats'] == [0,0,1,0]
        assert audit['saved_training_ledger_unchanged'] and audit['banks_unchanged']
        report[role] = 'literal role, eval, separate diagnostics, idle cleanup, no saved ledger change'
    assert refused(lambda: fixture(source).inference_operation('TRAIN').__enter__())
    assert refused(lambda: fixture(source).operation('TEST').__enter__())
    def bank_mutation():
        owner = fixture(source)
        with owner.inference_operation('TEST'):
            dict(owner.manager.named_buffers())['inactive_route_bank'].value += 1
    assert refused(bank_mutation)
    def canonical_mutation():
        owner = fixture(source)
        with owner.inference_operation('VALID'): owner.native.model.buffer.value += 1
    assert refused(canonical_mutation)
    def ledger_mutation():
        owner = fixture(source)
        with owner.inference_operation('TEST'): owner._calls['VALID']['bn_feats'][0] += 1
    assert refused(ledger_mutation)
    owner = fixture(source)
    try:
        with owner.inference_operation('TEST'): raise LookupError('source fixture exception')
    except LookupError: pass
    else: raise AssertionError('Source exception did not propagate')
    assert owner._inference_role is None and owner.native.model.training is True
    # Both closure paths retain original source route and private manager; eval
    # does not use the TRAIN sampler or source hydrate provider.
    install = methods['_install_ownership_guards']
    assert any(isinstance(n, ast.With) for n in ast.walk(install))
    for path in HERE.glob('*.py'):
        parsed = ast.parse(path.read_text())
        for node in parsed.body:
            if isinstance(node, ast.Import): assert all(not x.name.startswith(('torch','numpy','ogb')) for x in node.names)
            if isinstance(node, ast.ImportFrom): assert not (node.module or '').startswith(('torch','numpy','ogb'))
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    out = {'schema':'BUDDY-explicit-inference-source-checks-v1','status':'PASS',
        'unchanged_TRAIN_VALID_methods':retained,'checkpoint_schema_and_payload_layout_unchanged':True,
        'saved_producer_metadata_rewritten':False,'literal_inference_roles':report,
        'private_including_inactive_and_native_bank_mutation_refused':True,
        'saved_ledger_mutation_refused':True,'live_state_access_and_nested_owners_refused':True,
        'exception_releases_owner_and_restores_mode':True,
        'runtime_overlay_compatibility_numerically_qualified':False,
        'model_tensor_dataset_checkpoint_or_TEST_payload_reads':0,'native_executions':0}
    (HERE/'LOCAL_CHECKS.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out))


if __name__ == '__main__': main()
