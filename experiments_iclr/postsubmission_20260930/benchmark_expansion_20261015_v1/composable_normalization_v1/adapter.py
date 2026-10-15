"""Construction-time normalization over a fully assembled parent candidate.

Only create_adapter loads the pinned, unchanged native BN bank law. Importing
this module performs no model/library/data work. This is a new producer API;
it is not attached to any existing job, live session or saved checkpoint.
"""
from contextlib import contextmanager
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent
EXPANSION = HERE.parent
PHASE = EXPANSION.parent
STATE_SCHEMA = 'composed-native-route-BN-adapter-state-v1'


def sibling(name):
    path = HERE / (name + '.py')
    key = '_composable_normalization_' + hashlib.sha256(path.read_bytes()).hexdigest()
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, path)
        module = importlib.util.module_from_spec(spec); sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


def receipt(path):
    path = Path(path).resolve(); raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def verify(receipt_value, root):
    path = Path(receipt_value['path']).resolve()
    if not path.is_relative_to(root) or receipt(path) != receipt_value:
        raise ValueError('Exact repository-confined parent/source receipt required')
    return path


class NormalizedAdapter:
    """Preserve parent algorithm and mean provider; own only normalization."""
    def __init__(self, inner, job, manager, source_binding):
        self.inner, self.job, self.manager = inner, copy.deepcopy(job), manager
        self.t = getattr(inner, 'torch', None) or getattr(inner.session, 'torch', None) or inner.session.t
        self.model, self.optimizers, self.session = inner.model, inner.optimizers, inner.session
        self.owner = sibling('ownership').Ownership()
        self.binding = {'schema': 'composed-native-route-BN-producer-v1',
            'job_sha256': sibling('contracts').digest(job),
            'declaration': copy.deepcopy(job['normalization_composition']),
            'sources': copy.deepcopy(source_binding)}
        self._handles, self._closed, self._invalid = [], False, None
        self._parameters = tuple(self.model.named_parameters())
        self._optimizer_parameters = tuple(p for opt in self.optimizers for group in opt.param_groups for p in group['params'])
        self._native = manager._native_ref()
        self._canonical = {name: (value, value.detach().clone()) for name, value in self._native.named_buffers()}
        self._route_owner = self.model if job['family'] == 'ogb' else self.session.native
        self._route_name = 'run_member' if job['family'] == 'ogb' else '_route'
        self._original_route = getattr(self._route_owner, self._route_name)
        self._install()

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def _install(self):
        def before_forward(module, inputs):
            if self.owner.member is None or self.manager._active != self.owner.member \
                    or bool(module.training) != (self.owner.operation_role == 'TRAIN'):
                raise RuntimeError('Native BN forward bypassed its normalization owner')
        def state_guard(expected):
            def before_state(module, *args):
                if self.owner.state_role != expected or self.owner.member is not None \
                        or self.owner.operation_role is not None or self.owner.inference_role is not None:
                    raise RuntimeError('Checkpoint must use the idle composed adapter state API')
            return before_state
        try:
            for module in (self._native, *(ref() for ref in self.manager._refs)):
                self._handles.append(module.register_forward_pre_hook(before_forward))
            for module in (self._native, self.manager, *(ref() for ref in self.manager._refs)):
                self._handles.append(module.register_state_dict_pre_hook(state_guard('export')))
                public = getattr(module, 'register_load_state_dict_pre_hook', None)
                self._handles.append(public(state_guard('load')) if public is not None else
                    module._register_load_state_dict_pre_hook(state_guard('load'), with_module=True))
            if self.job['family'] == 'ogb':
                def run_member(owner, member, batch):
                    training = bool(self.model.training)
                    with self.owner.route(member, training), self.manager.route(member):
                        return self._original_route(member, batch)
                setattr(self._route_owner, self._route_name, types.MethodType(run_member, self._route_owner))
            else:
                @contextmanager
                def route(owner, member, training):
                    with self.owner.route(member, training), self._original_route(member, training), self.manager.route(member):
                        yield
                setattr(self._route_owner, self._route_name, types.MethodType(route, self._route_owner))
            self._installed_route = getattr(self._route_owner, self._route_name)
            self.assert_identity()
        except BaseException:
            setattr(self._route_owner, self._route_name, self._original_route)
            for handle in self._handles: handle.remove()
            self._handles = []
            raise

    def assert_identity(self):
        if self._invalid is not None or self._closed:
            raise ValueError('Composed adapter is invalid or closed')
        self.manager.assert_native_identity()
        self.inner.assert_identity()
        if tuple((name, id(value)) for name, value in self.model.named_parameters()) != \
                tuple((name, id(value)) for name, value in self._parameters):
            raise ValueError('Normalization changed candidate learned parameters')
        owners = tuple(p for opt in self.optimizers for group in opt.param_groups for p in group['params'])
        if tuple(map(id, owners)) != tuple(map(id, self._optimizer_parameters)):
            raise ValueError('Normalization changed optimizer ownership')
        current = dict(self._native.named_buffers())
        if set(current) != set(self._canonical) or any(current[name] is not old or not self.t.equal(old, initial)
                for name, (old, initial) in self._canonical.items()):
            raise ValueError('Canonical native buffers changed outside private routes')

    def _buffers(self):
        return {(kind, name): value for kind, obj in (('native', self._native), ('banks', self.manager))
                for name, value in obj.named_buffers()}

    @contextmanager
    def _operation(self, role, *, inference=False):
        self.assert_identity()
        readonly = role != 'TRAIN'
        before = {key: (value, value.data_ptr(), value.detach().clone()) for key, value in self._buffers().items()} if readonly else {}
        modes = [(module, bool(module.training)) for module in self.model.modules()] if inference else []
        try:
            with self.owner.operation(role, inference=inference):
                if inference:
                    self.model.eval()
                    with self.t.no_grad(): yield
                else: yield
        finally:
            for module, training in modes: module.training = training
            self.assert_identity()
            if readonly:
                after = self._buffers()
                if set(after) != set(before) or any(after[key] is not old or after[key].data_ptr() != pointer
                        or after[key].shape != saved.shape or after[key].dtype != saved.dtype
                        or after[key].device != saved.device or not self.t.equal(after[key], saved)
                        for key, (old, pointer, saved) in before.items()):
                    self._invalid = 'Evaluation modified normalization buffers'
                    raise ValueError('Evaluation modified native/private normalization buffers')

    def train_epoch(self):
        with self._operation('TRAIN'): result = self.inner.train_epoch()
        return self._annotate(result)

    def train_update(self):
        with self._operation('TRAIN'):
            result = self.session.train_update()
        return self._annotate(result)

    def validate(self):
        with self._operation('VALID'): result = self.inner.validate()
        return self._annotate(result)

    def _annotate(self, result):
        result = dict(result)
        result['normalization_execution'] = {'policy': self.job['normalization_composition']['policy'],
            'native_BN_affine_and_operators_shared': True, 'native_shared_BN_running_state': False,
            'normalization_added_learned_parameters': 0}
        if isinstance(result.get('work'), dict):
            result['work'] = dict(result['work'])
            result['work']['parent_reported_native_shared_BN'] = result['work'].get('native_shared_BN')
            result['work']['native_shared_BN'] = False
            result['work']['route_private_BN_running_state'] = True
        return result

    def train_diagnostic(self):
        with self._operation('TRAIN_DIAGNOSTIC'): return self.inner.train_diagnostic()

    @contextmanager
    def inference_operation(self, role):
        with self._operation(role, inference=True): yield

    def state(self):
        self.assert_identity()
        with self.owner.state('export'):
            return {'schema': STATE_SCHEMA, 'normalization_producer': copy.deepcopy(self.binding),
                'parent_adapter_state': self.inner.state(),
                'private_BN_banks': copy.deepcopy(self.manager.state_dict())}

    def load_state(self, value):
        if value.get('schema') != STATE_SCHEMA or value.get('normalization_producer') != self.binding:
            raise ValueError('Only the same genuine composed producer can restore this state')
        self.assert_identity()
        self.owner.idle()
        try:
            with self.owner.state('load'):
                self.inner.load_state(copy.deepcopy(value['parent_adapter_state']))
                self.manager.load_state_dict(copy.deepcopy(value['private_BN_banks']), strict=True)
        except BaseException:
            self._invalid = 'Composed restore failed after admission'
            raise
        self.assert_identity()

    def metadata(self):
        return {**self.inner.metadata(), 'normalization_composition': copy.deepcopy(self.binding),
            'normalization_bank_state': self.manager.metadata(),
            'normalization_state_owner': STATE_SCHEMA, 'normalization_added_learned_parameters': 0}

    def close(self):
        self.owner.idle()
        if self._closed: raise RuntimeError('Composed adapter already closed')
        if self._invalid is None: self.assert_identity()
        if getattr(self._route_owner, self._route_name) != self._installed_route:
            raise ValueError('Another component changed the installed route after composition')
        setattr(self._route_owner, self._route_name, self._original_route)
        for handle in self._handles: handle.remove()
        self._handles = []; self._closed = True
        self.inner.close()


def create_adapter(torch, job, policy, root, *, parent_factory):
    """Called only by a future genuine producer at fresh construction.

    The parent factory receives this exact new JOB unchanged, so existing
    same-call providers and job/selection hashes retain their own full identity.
    Frozen runtime entrypoints do not call this API and remain unchanged.
    """
    root = Path(root).resolve()
    if not HERE.is_relative_to(root): raise ValueError('Actual project source root required')
    declaration = job.get('normalization_composition', {})
    own = receipt(HERE / 'MANIFEST.json')
    if own['sha256'] != declaration.get('source_manifest_sha256'):
        raise ValueError('Exact composable-normalization source manifest required')
    manifest = json.loads((HERE / 'MANIFEST.json').read_text())
    for row in manifest['files']:
        path = (HERE / row['path']).resolve()
        if not path.is_relative_to(HERE): raise ValueError('Source closure escapes package')
        actual = receipt(path)
        if any(actual[key] != row[key] for key in ('bytes', 'sha256')): raise ValueError('Composition source changed')
    bindings = json.loads((HERE / 'PARENT_BINDINGS.json').read_text())
    for row in bindings['files']:
        path = (PHASE / row['path_relative_to_phase']).resolve()
        if not path.is_relative_to(root): raise ValueError('Parent dependency escapes repository')
        actual = receipt(path)
        if any(actual[key] != row[key] for key in ('bytes', 'sha256')): raise ValueError('Frozen composition dependency changed')
    roster_path = EXPANSION / 'expanded_study_design_v1/PROPOSED_ROSTER.json'
    parent_path = verify(declaration['parent_descriptor'], root)
    parent = json.loads(parent_path.read_text())
    roster = json.loads(roster_path.read_text())
    expected = sibling('contracts').migrated_job(parent, declaration, roster)
    if job != expected: raise ValueError('Composition changed a parent scientific field')
    factory_path = EXPANSION / 'portable_dispatcher_v3' / \
        ('runtime_node_molecular' if job['family'] == 'ogb' else 'runtime_link') / 'adapters.py'
    if getattr(parent_factory, '__name__', None) != 'create_adapter' \
            or not hasattr(parent_factory, '__code__') \
            or Path(parent_factory.__code__.co_filename).resolve() != factory_path:
        raise ValueError('Only the pinned existing native adapter factory may construct the parent')
    source = EXPANSION / 'private_bn_state_v3/banks.py'
    bank_pin = receipt(source)
    if bank_pin['sha256'] != declaration['banks_sha256']:
        raise ValueError('Pin the unchanged private-BN law')
    inner = parent_factory(torch, copy.deepcopy(job), policy, root)
    try:
        authority = getattr(inner, '_job_authority', getattr(inner, 'job', None))
        if authority is not None and authority != job:
            raise ValueError('Parent factory must be built from the exact new immutable JOB')
        if int(getattr(inner, 'completed_epochs', -1)) != 0 or int(getattr(inner.session, 'updates', -1)) != 0:
            raise ValueError('Attach at fresh construction before training or restoration')
        spec = importlib.util.spec_from_file_location('_composed_BN_law_' + bank_pin['sha256'], source)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        native = inner.model.native if job['family'] == 'ogb' else inner.session.native.model
        manager = module.RouteBatchNormState(native, members=4)
        if job['family'] == 'link' and getattr(inner.session, 'family', None) != 'BUDDY':
            raise ValueError('Only native tracked-BN BUDDY is an eligible link preset')
        return NormalizedAdapter(inner, job, manager, {'composition_manifest': own,
            'unchanged_bank_law': bank_pin, 'parent_descriptor': declaration['parent_descriptor']})
    except BaseException:
        inner.close()
        raise
