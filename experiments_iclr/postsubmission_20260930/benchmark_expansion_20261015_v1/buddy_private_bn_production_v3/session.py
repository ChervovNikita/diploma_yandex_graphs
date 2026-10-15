"""Native BUDDY port of frozen v3 private running state/shared affine BN.

Banks are owned by the wrapped session, so the original native model retains
exact Parameter and buffer objects and unchanged optimizer ownership.
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
METHOD = 'gnnm_private_bn_state_v3'
STATE_SCHEMA = 'native-BUDDY-private-BN-production-state-v1'
KEYS = ('running_mean', 'running_var', 'num_batches_tracked')
EXPECTED_BN = {'bn_feats': 1024, 'bn_labels': 16, 'bn_RA': 1}
ACTIVE_BN = {'bn_feats', 'bn_labels'}


def receipt(path):
    path = Path(path).resolve(); raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def load(path):
    path = Path(path).resolve()
    name = '_buddy_private_BN_'+hashlib.sha256(str(path).encode()+path.read_bytes()).hexdigest()
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec); sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def bindings(root, expected):
    root = Path(root).resolve()
    if not HERE.is_relative_to(root): raise ValueError('Bind actual repository root')
    actual = receipt(HERE/'MANIFEST.json')
    if actual['sha256'] != expected: raise ValueError('Pin the exact BUDDY private-BN source manifest')
    pins = [actual]
    for row in json.loads((HERE/'MANIFEST.json').read_text())['files']:
        path = (HERE/row['path']).resolve()
        if not path.is_relative_to(HERE): raise ValueError('Own manifest escapes package')
        pin = receipt(path)
        if any(pin[k] != row[k] for k in ('bytes', 'sha256')): raise ValueError('Prepared source changed: '+row['path'])
        pins.append(pin)
    for row in json.loads((HERE/'SOURCES.json').read_text())['source_files']:
        path = (PHASE/row['path_relative_to_phase']).resolve()
        if not path.is_relative_to(root): raise ValueError('Frozen source escapes repository')
        pin = receipt(path)
        if any(pin[k] != row[k] for k in ('bytes', 'sha256')): raise ValueError('Frozen dependency changed: '+str(path))
        pins.append(pin)
    return pins


def same(torch, actual, expected):
    return actual.shape == expected.shape and actual.dtype == expected.dtype and torch.equal(actual.detach().cpu(), expected)


class BUDDYPrivateBN:
    def __init__(self, base, job, pins):
        self.base, self.native, self.job = base, base.native, copy.deepcopy(job)
        n = self.native; self.t = n.t
        if getattr(base,'metadata_record',{}).get('provider_mode') != 'production_native':
            raise ValueError('Attach only over the explicit compact production-native provider')
        if base.family != 'BUDDY' or n.procedure != 'base' or n.members != 4 or n.updates != 0 or base.candidate_id is not None:
            raise ValueError('Fresh standalone native BUDDY four-route own-loss session required')
        if n.args.batch_size != 1024 or n.args.year != 2007 or n.args.sign_k != 0 or n.args.use_RA:
            raise ValueError('Only the frozen native Collab BUDDY recipe is prepared')
        if n.model.node_embedding is not None or not n.model.use_feature:
            raise ValueError('Fixed feature branch and absent embedding branch required')
        self._operation = self._state_operation = None
        self._inference_role = self._inference_calls = None
        self._last_ids = None
        self._TRAIN_provider_checks = self._VALID_provider_checks = 0
        self._original_route, self._original_next, self._original_hydrate = n._route, n._next_batch, n._hydrate
        self._handles = []
        bank_path = EXPANSION/'private_bn_state_v3/banks.py'
        bank_pin = receipt(bank_path)
        if bank_pin['sha256'] != job['method_settings']['banks_sha256']:
            raise ValueError('Pin the frozen reviewed private-BN v3 bank law')
        self.manager = load(bank_path).RouteBatchNormState(n.model, members=4)
        modules = {name: ref() for name, ref in zip(self.manager.names, self.manager._refs)}
        if {name: module.num_features for name,module in modules.items()} != EXPECTED_BN:
            raise ValueError('Native BUDDY BatchNorm branches differ from the fixed source')
        self._canonical = {name+'.'+key: (value, value.detach().clone())
            for name, values in zip(self.manager.names, self.manager._original) for key,value in zip(KEYS, values)}
        self._inactive_initial = [(value,value.detach().clone())
            for name,banks in zip(self.manager.names,self.manager.banks) if name not in ACTIVE_BN
            for bank in banks for value in bank.buffers()]
        self._calls = {role: {name: [0]*4 for name in self.manager.names} for role in ('TRAIN', 'VALID')}
        self.method_metadata = {'schema': 'native-BUDDY-private-BN-production-source-v1', 'method_id': METHOD,
            'native_family': 'BUDDY', 'port': 'buddy_private_bn_production_v3', 'source_receipts': pins,
            'bank_source': bank_pin, 'production_native_provider': {'provider_mode':job['method_settings']['provider_mode'],'provider_manifest_sha256':job['method_settings']['provider_manifest_sha256'],'static_provider_sha256':job['method_settings']['static_provider_sha256']},
            'definition': 'frozen v3 route-private running state; original shared native affine Parameters',
            'manager': self.manager.metadata(), 'active_BN_modules': sorted(ACTIVE_BN),
            'inactive_BN_modules': ['bn_RA'], 'embedding_BN_absent': True,
            'native_task': 'original four-route joint BCE minibatch mean; original native _step unchanged',
            'native_VALID_metric_and_loss': 'unchanged complete-source evaluator and original source diagnostic loss',
            'native_sampler_hydration_graph_forwards_and_RNG': 'unchanged source operators',
            'state_owner': 'wrapped session.state/load_state includes native integration and all route banks',
            'direct_model_only_save_load': 'rejected to prevent incomplete checkpoints',
            'extra_learned_parameters': 0, 'source_Parameter_replacements': 0,
            'source_BN_module_or_affine_replacements': 0, 'source_buffer_objects_outside_route': 'original canonical objects and initial values',
            'activation_recomputation_SyncBN_concurrent_routes_or_post_attach_migration': 'unsupported',
            'input_scope': 'actual native fit seed/year2007; native sampler and first/second/tail plus whole VALID lineage',
            'native_qualification': 'pending bounded real-data smoke; source prepared only', 'TEST_accessed': False}

        self._install_ownership_guards()
        self.inference_metadata = {'schema': 'bound-BUDDY-private-BN-inference-api-v1',
            'producer_port': self.method_metadata['port'],
            'inference_port': 'buddy_private_bn_production_v3',
            'source_receipts': copy.deepcopy(pins), 'producer_metadata_changed': False}

    def _install_ownership_guards(self):
        n = self.native
        modules = {name: ref() for name, ref in zip(self.manager.names, self.manager._refs)}
        def require_route(module, inputs):
            if (self._operation not in ('TRAIN', 'VALID') and self._inference_role not in ('VALID', 'TEST')) \
                    or self.manager._active is None or int(n.core.route) != self.manager._active:
                raise RuntimeError('BUDDY private-BN forward requires its owned source route/task context')
        def require_BN(name):
            def check(module, inputs):
                require_route(module, inputs)
                if bool(module.training) != (self._operation == 'TRAIN'):
                    raise RuntimeError('Native BN mode and task role differ')
                calls = self._inference_calls if self._inference_role is not None else self._calls[self._operation]
                calls[name][self.manager._active] += 1
            return check
        def state_export(module, *args):
            if self.manager._active is not None or self._operation is not None or self._inference_role is not None or self._state_operation != 'export':
                raise RuntimeError('Private-BN checkpoint must use the idle wrapped session.state')
        def state_load(module, *args):
            if self.manager._active is not None or self._operation is not None or self._inference_role is not None or self._state_operation != 'load':
                raise RuntimeError('Private-BN restore must use the idle wrapped session.load_state')
        def before_load(module):
            public = getattr(module, 'register_load_state_dict_pre_hook', None)
            return public(state_load) if public is not None else module._register_load_state_dict_pre_hook(state_load, with_module=True)
        self._handles.append(n.model.register_forward_pre_hook(require_route))
        self._handles.extend(module.register_forward_pre_hook(require_BN(name)) for name,module in modules.items())
        for module in (n.model, self.manager, *modules.values()):
            self._handles.extend((module.register_state_dict_pre_hook(state_export), before_load(module)))

        @contextmanager
        def route(owner, member, training):
            if self._inference_role is not None:
                if self._inference_role not in ('VALID', 'TEST') or bool(training):
                    raise RuntimeError('Owned final inference requires the literal VALID/TEST role and eval routes')
            elif self._operation not in ('TRAIN', 'VALID') or bool(training) != (self._operation == 'TRAIN'):
                raise RuntimeError('Only the unchanged source TRAIN/VALID route dispatch is supported')
            with self._original_route(member, training):
                with self.manager.route(member):
                    yield
        def next_batch(owner):
            if self._operation != 'TRAIN': raise RuntimeError('Native sampler requires the owned TRAIN task')
            ids = self._original_next(); self._last_ids = ids.detach().cpu().clone(); return ids
        def hydrate(owner, links):
            if self._operation not in ('TRAIN', 'VALID'): raise RuntimeError('Hydration requires the owned source task')
            values = self._original_hydrate(links)
            if self._operation == 'TRAIN':
                if self._last_ids is None: raise RuntimeError('Original sampler IDs are unavailable')
                if not same(self.t,links,n.links[self._last_ids].detach().cpu()):
                    raise ValueError('Actual source hydration must match actual native sampler IDs')
                self._TRAIN_provider_checks += 1
            else:
                self._VALID_provider_checks += 1
            return values
        n._route, n._next_batch, n._hydrate = types.MethodType(route,n), types.MethodType(next_batch,n), types.MethodType(hydrate,n)
        self.assert_identity()

    def __getattr__(self, name): return getattr(self.base, name)


    def assert_identity(self):
        self.base.assert_identity(); self.manager.assert_native_identity()
        for name, (original, initial) in self._canonical.items():
            if not self.t.equal(original,initial): raise ValueError('Canonical native BN state changed outside route banks: '+name)
        if any(not self.t.equal(value,initial) for value,initial in self._inactive_initial):
            raise ValueError('Inactive native BUDDY BN branch state changed')
        pointers = [value.data_ptr() for module in self.manager.banks for bank in module for value in bank.buffers()]
        if len(pointers) != len(set(pointers)): raise ValueError('Route BN state banks alias')

    @contextmanager
    def operation(self, role):
        if role not in ('TRAIN', 'VALID'):
            raise ValueError('Training operations admit only the unchanged TRAIN/VALID roles')
        if self._operation is not None or self._inference_role is not None or self._state_operation is not None or self.manager._active is not None:
            raise RuntimeError('One nonnested source task at a time')
        self.assert_identity(); self._operation = role
        try: yield
        finally:
            self._operation = None; self.assert_identity()


    def _inference_buffers(self):
        return {(scope, name): value for scope, owner in
                (('route_bank', self.manager), ('native', self.native.model))
                for name, value in owner.named_buffers()}

    @contextmanager
    def inference_operation(self, role):
        """Own literal final VALID/TEST eval without changing saved TRAIN state.

        Call diagnostics are returned locally, never added to the checkpointed
        TRAIN/VALID ledger. Native route/BN guards remain active in this owner.
        This source API neither authorizes TEST nor constructs any role data.
        """
        if role not in ('VALID', 'TEST'):
            raise ValueError('One literal final VALID or TEST role required')
        if self._operation is not None or self._inference_role is not None or self._state_operation is not None or self.manager._active is not None:
            raise RuntimeError('Final inference requires an idle operation/state/route owner')
        self.assert_identity()
        buffers = self._inference_buffers()
        before = {key: (value, value.data_ptr(), value.detach().clone()) for key, value in buffers.items()}
        ledger = copy.deepcopy(self._calls)
        providers = (self._TRAIN_provider_checks, self._VALID_provider_checks)
        modes = [(module, bool(module.training)) for module in self.native.model.modules()]
        self._inference_role = role
        self._inference_calls = {name: [0]*4 for name in self.manager.names}
        audit = {'schema': 'owned-BUDDY-private-BN-final-inference-v1', 'role': role,
                 'inference_port': 'buddy_private_bn_production_v3',
                 'producer_port': self.method_metadata['port'],
                 'bank_nonmutation_required': True, 'saved_training_ledger_unchanged': False}
        try:
            self.native.model.eval()
            with self.t.no_grad():
                yield audit
        finally:
            calls = copy.deepcopy(self._inference_calls)
            self._inference_role = self._inference_calls = None
            for module, training in modes:
                module.training = training
            after = self._inference_buffers()
            if set(after) != set(before) or any(after[key] is not original or after[key].data_ptr() != pointer
                    or after[key].shape != saved.shape or after[key].dtype != saved.dtype
                    or after[key].device != saved.device or not self.t.equal(after[key], saved)
                    for key, (original, pointer, saved) in before.items()):
                raise ValueError('Owned final inference changed native or private BN buffers')
            if self._calls != ledger or (self._TRAIN_provider_checks, self._VALID_provider_checks) != providers:
                raise ValueError('Final inference changed the saved TRAIN/VALID logical ledger')
            self.assert_identity()
            audit.update(branch_calls_by_route=calls, banks_unchanged=True,
                         saved_training_ledger_unchanged=True)


    def _snapshot(self):
        return {name: value.detach().clone() for name,value in self.manager.named_buffers()}

    def _counts(self):
        return {name: [int(bank.num_batches_tracked) for bank in banks] for name,banks in zip(self.manager.names,self.manager.banks)}

    def _audit_TRAIN(self, before_calls, before_counts, maps):
        counts = self._counts(); delta = {}
        for name in self.manager.names:
            actual = [a-b for a,b in zip(self._calls['TRAIN'][name],before_calls[name])]
            expected = maps if name in ACTIVE_BN else 0
            if actual != [expected]*4 or [a-b for a,b in zip(counts[name],before_counts[name])] != actual:
                raise ValueError('Native BUDDY route-private BN counter ownership differs: '+name)
            delta[name] = actual
        if any(not self.t.isfinite(value).all() for value in self.manager.buffers()): raise FloatingPointError('Nonfinite route BN state')
        return {'branch_calls_by_route':delta,'route_bank_counters':counts,'canonical_native_buffers_unchanged':True,
                'shared_native_affine_Parameter_IDs_preserved':True,'joint_source_TRAIN_BCE_unchanged':True}

    def train_update(self):
        before_calls, before_counts = copy.deepcopy(self._calls['TRAIN']), self._counts()
        with self.operation('TRAIN'): result = self.base.train_update()
        audit = self._audit_TRAIN(before_calls,before_counts,1)
        result.setdefault('work',{}).update(source_native_shared_BN_flag=result['work'].get('native_shared_BN'),
            native_shared_BN=False,native_BN_modules_and_affine_shared=True,route_private_BN_running_state=True)
        result['private_BN_state'] = audit
        return result

    def train_epoch(self):
        before_calls, before_counts = copy.deepcopy(self._calls['TRAIN']), self._counts()
        with self.operation('TRAIN'): result = self.base.train_epoch()
        result['private_BN_state'] = self._audit_TRAIN(before_calls,before_counts,result['optimizer_maps'])
        return result

    def validate(self):
        before = self._snapshot(); before_calls = copy.deepcopy(self._calls['VALID']); self._VALID_provider_checks = 0
        with self.operation('VALID'): result = self.base.validate()
        if self._VALID_provider_checks != 2 or any(not self.t.equal(value,dict(self.manager.named_buffers())[name]) for name,value in before.items()):
            raise ValueError('Complete VALID context or read-only BN-bank ownership failed')
        calls = {name:[a-b for a,b in zip(self._calls['VALID'][name],before_calls[name])] for name in self.manager.names}
        if any(row != ([2]*4 if name in ACTIVE_BN else [0]*4) for name,row in calls.items()):
            raise ValueError('Native BUDDY VALID branch dispatch differs')
        result.update(
            production_native_VALID_inputs_verified=True,private_BN_eval_state_unchanged=True,
            private_BN_branch_calls_by_route=calls,same_bank_mean_outputs=result['outputs'],
            same_bank_mean_loss=result['loss'],same_bank_mean_metrics=copy.deepcopy(result['metrics']))
        result['work'].update(native_shared_BN=False,native_BN_modules_and_affine_shared=True,route_private_BN_running_state=True)
        return result

    def metadata(self):
        return {**self.base.metadata(),'candidate_extension':copy.deepcopy(self.method_metadata),
            'boundary_BN_policy_override':'route-private running state, original shared native affine',
            'private_BN_branch_ledger_scope':'checkpointed logical calls, not cumulative actual qualification work'}

    def state(self):
        if self._operation is not None or self._inference_role is not None or self._state_operation is not None: raise RuntimeError('Checkpoint only outside live task')
        self.assert_identity(); self._state_operation = 'export'
        try:
            result = {'schema':STATE_SCHEMA,'method':copy.deepcopy(self.method_metadata),'native_integration':self.base.state(),
                'private_BN_banks':{name:value.detach().clone() for name,value in self.manager.state_dict().items()},
                'logical_branch_calls':copy.deepcopy(self._calls),'TRAIN_provider_checks':self._TRAIN_provider_checks}
        finally:self._state_operation = None
        return result

    def load_state(self, value):
        if self._operation is not None or self._inference_role is not None or self._state_operation is not None or value.get('schema') != STATE_SCHEMA or value.get('method') != self.method_metadata:
            raise ValueError('Idle exact method/source/context checkpoint required')
        self.assert_identity(); self._state_operation = 'load'
        try:
            self.base.load_state(value['native_integration'])
            self.manager.load_state_dict(value['private_BN_banks'],strict=True)
            self._calls = copy.deepcopy(value['logical_branch_calls']); self._TRAIN_provider_checks = int(value['TRAIN_provider_checks'])
        finally:self._state_operation = None
        self.assert_identity()

    def close(self):
        if self._operation is not None or self._inference_role is not None or self._state_operation is not None or self.manager._active is not None:
            raise RuntimeError('Close only outside live task/state operation')
        self.assert_identity()
        self.native._route,self.native._next_batch,self.native._hydrate = self._original_route,self._original_next,self._original_hydrate
        for handle in self._handles:handle.remove()
        self._handles=[]; self.base.close()


def attach_inference_owner(session, job, root, expected_manifest_sha256):
    """Bind this inference API to a genuine restored v2 producer session.

    The original producer source is authenticated independently. Its method
    metadata, callback/runtime binding, checkpoint layout and private state
    remain genuine v2; this separately bound overlay changes ownership guards
    only. A failed attachment invalidates the worker and must not be retried.
    """
    root = Path(root).resolve()
    pins = bindings(root, expected_manifest_sha256)
    parent_path = EXPANSION/'buddy_private_bn_production_v2/session.py'
    parent = sys.modules.get(type(session).__module__)
    if parent is None or Path(getattr(parent, '__file__', '')).resolve() != parent_path \
            or type(session) is not parent.BUDDYPrivateBN:
        raise ValueError('Inference overlay requires the exact authenticated v2 producer class')
    metadata = session.method_metadata
    if metadata.get('schema') != 'native-BUDDY-private-BN-production-source-v1' \
            or metadata.get('method_id') != METHOD or metadata.get('port') != 'buddy_private_bn_production_v2':
        raise ValueError('Genuine v2 producer metadata required')
    if job != session.job or job.get('procedure') != METHOD or job.get('TEST_access') is not False:
        raise ValueError('Overlay job must be the exact closed-TEST genuine producer descriptor')
    callback = EXPANSION/'buddy_private_bn_production_v2/callback.py'
    declared_callback = Path(job['callback_module'])
    declared_callback = (declared_callback if declared_callback.is_absolute() else root/declared_callback).resolve()
    if declared_callback != callback or job.get('callback_sha256') != receipt(callback)['sha256']:
        raise ValueError('Overlay cannot replace or relabel the v2 training callback')
    parent_pins = parent.bindings(root, job['method_settings']['manifest_sha256'])
    if metadata.get('source_receipts') != parent_pins:
        raise ValueError('Restored producer source closure differs from genuine v2')
    if '_inference_role' in session.__dict__ or 'inference_metadata' in session.__dict__:
        raise ValueError('Attach one reviewed inference API only')
    if session._operation is not None or session._state_operation is not None or session.manager._active is not None:
        raise RuntimeError('Attach inference only to an idle canonical restored session')
    session.assert_identity()
    expected_handles = 1+len(session.manager.names)+2*(2+len(session.manager.names))
    if len(session._handles) != expected_handles:
        raise ValueError('Authentic complete producer ownership guards required before replacement')
    for name in ('_route', '_next_batch', '_hydrate'):
        if getattr(getattr(getattr(session.native, name), '__func__', None), '__module__', None) != parent.__name__:
            raise ValueError('Producer native ownership method source changed')
    producer_metadata = copy.deepcopy(metadata)
    session._inference_role = session._inference_calls = None
    # Replace the producer-owned guards as one explicit inference overlay. The
    # unchanged native model/optimizer/manager and original route remain owned.
    for handle in session._handles:
        handle.remove()
    session._handles = []
    for name in ('_inference_buffers', 'inference_operation', 'operation', 'state', 'load_state', 'close'):
        setattr(session, name, types.MethodType(getattr(BUDDYPrivateBN, name), session))
    BUDDYPrivateBN._install_ownership_guards(session)
    if session.method_metadata != producer_metadata:
        raise ValueError('Inference overlay must not rewrite saved producer metadata')
    session.inference_metadata = {'schema': 'bound-BUDDY-private-BN-inference-api-v1',
        'producer_port': 'buddy_private_bn_production_v2',
        'inference_port': 'buddy_private_bn_production_v3',
        'producer_manifest': receipt(callback.with_name('MANIFEST.json')),
        'source_receipts': copy.deepcopy(pins), 'producer_metadata_changed': False}
    return session.inference_metadata


def bind_inference_owner(session, job, root, expected_manifest_sha256):
    """Bind a genuine v2 overlay or verify this successor's genuine own API."""
    if session.method_metadata.get('port') != 'buddy_private_bn_production_v3':
        return attach_inference_owner(session, job, root, expected_manifest_sha256)
    pins = bindings(root, expected_manifest_sha256)
    actual = sys.modules.get(type(session).__module__)
    if actual is None or Path(getattr(actual, '__file__', '')).resolve() != HERE/'session.py' \
            or type(session) is not actual.BUDDYPrivateBN or job != session.job \
            or session.method_metadata.get('source_receipts') != pins \
            or session.inference_metadata.get('source_receipts') != pins:
        raise ValueError('Genuine v3 producer and separately bound inference source required')
    session.assert_identity()
    return copy.deepcopy(session.inference_metadata)
