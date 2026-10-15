"""Explicit fresh-native NCN component factory; no standalone transform chain."""
from __future__ import annotations
import copy
import hashlib
import importlib.util
from importlib import import_module
import json
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent
EXPANSION = HERE.parent
PHASE = EXPANSION.parent
METHODS = ('interaction_geometry_input', 'interaction_responsibility_private_pool',
           'interaction_input_meanJ', 'interaction_input_conditionalJ')
STATE_SCHEMA = 'explicit-NCN-interaction-state-v2'


def receipt(path):
    path = Path(path).resolve(); raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def load(path, package=False):
    path = Path(path).resolve(); name = '_NCN_interaction_'+hashlib.sha256(str(path).encode()+path.read_bytes()).hexdigest()
    if name not in sys.modules:
        options = {'submodule_search_locations': [str(path.parent)]} if package else {}
        spec = importlib.util.spec_from_file_location(name, path, **options)
        m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return sys.modules[name]


def verify_package(folder, expected):
    folder = Path(folder).resolve(); pin = receipt(folder/'MANIFEST.json')
    if pin['sha256'] != expected:
        raise ValueError('Pin actual immutable interaction/ingredient manifest')
    pins = [pin]
    declared_files = json.loads((folder/'MANIFEST.json').read_text())['files']
    if isinstance(declared_files, dict):
        if any(not isinstance(name,str) or not isinstance(sha,str) or len(sha)!=64
               for name,sha in declared_files.items()):
            raise ValueError('Explicit pinned SHA-only dictionary manifest required')
        rows = [{'path':name,'sha256':sha} for name,sha in declared_files.items()]
    elif isinstance(declared_files,list):
        rows = declared_files
    else:
        raise ValueError('Unsupported immutable ingredient manifest format')
    for row in rows:
        path = (folder/row['path']).resolve()
        if not path.is_relative_to(folder):
            raise ValueError('Frozen manifest escapes ingredient package')
        value = receipt(path)
        if any(value[k] != row[k] for k in ('bytes', 'sha256') if k in row):
            raise ValueError('Frozen ingredient/source changed: '+str(path))
        pins.append(value)
    return pins


def bindings(root, expected):
    root = Path(root).resolve()
    if not HERE.is_relative_to(root) or not PHASE.is_relative_to(root):
        raise ValueError('Bind actual absolute repository containing native sources')
    pins = verify_package(HERE, expected)
    declaration = json.loads((HERE/'BINDINGS.json').read_text())
    for row in declaration['packages']:
        pins += verify_package(EXPANSION/row['path'], row['manifest_sha256'])
    for row in declaration['native_sources']:
        path = (PHASE/row['path_relative_to_phase']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Native dependency escapes actual repository')
        value = receipt(path)
        if any(value[k] != row[k] for k in ('bytes', 'sha256')):
            raise ValueError('Original native dependency changed')
        pins.append(value)
    portable = load(EXPANSION/'portable_private_input_v2/extension.py')
    pins += portable._source_bindings('link', root,
        next(row['manifest_sha256'] for row in declaration['packages'] if row['path'] == 'portable_private_input_v2'))
    return pins, declaration


def selected_mean_sidecar(session):
    value = session._last_VALID
    if value is None or value['source_updates'] != session.native.updates:
        raise ValueError('Fresh whole VALID of currently loaded source state required')
    return copy.deepcopy(value)


class NCNInteractionSession:
    def __init__(self, base, job, pins, declaration):
        self.base, self.native, self.job = base, base.native, copy.deepcopy(job)
        n = self.native; self.t = n.t; self.method = job['procedure']
        if base.family != 'NCN' or base.candidate_id is not None or n.procedure != 'base' \
                or n.members != 4 or n.updates != 0 or n.batch_size != 65536:
            raise ValueError('Fresh original four-route own-loss NCN with full native source batch required')
        if type(job['seed']) is not int or int(n.seed) != job['seed']:
            raise ValueError('Actual native fit seed and component seed must agree')
        if getattr(n, '_portable_private_input_attached', False) or any(hasattr(base, key) for key in ('interaction_metadata', 'context_affinity_metadata')):
            raise ValueError('Explicit native factory cannot wrap another method')
        self.references = dict(n.source_parameter_references)
        self.library = load(EXPANSION/'link/__init__.py', package=True)
        responsibility = load(EXPANSION/'structured_responsibility_control_v2/__init__.py', package=True)
        self.responsibility = import_module(responsibility.__name__+'.objectives')
        self.sampler = import_module(responsibility.__name__+'.sampler')
        self.aggregation = load(EXPANSION/'conditional_aggregation_v1/core.py')
        self.objectives = load(HERE/'objectives.py')
        self.portable = load(EXPANSION/'portable_private_input_v2/core.py')
        mapping = load(EXPANSION/'portable_private_input_v2/extension.py')
        owner, native, core, sites, output = mapping.source_mapping(base, 'link')
        if owner is not n or native is not n.model or core is not n.core or len(sites) != 1 \
                or sites[0][0] != 'encoder.convs.0.lin' or sites[0][2] != 64 \
                or tuple(output[1].weight.shape) != (1, 64):
            raise ValueError('Only declared actual native NCN logical boundaries are admitted')
        before_groups = tuple(n.optimizer.param_groups)
        options = [{k: copy.deepcopy(v) for k, v in g.items() if k != 'params'} for g in before_groups]
        self._handles = []; self.residuals = None; self.gate = None
        self._original_private = core.private_parameters
        private_extra, input_sites = [], []
        if self.method != 'interaction_responsibility_private_pool':
            self.residuals = self.t.nn.ModuleList(self.portable.build_residual(self.t, anchor, width=width,
                seed=int(job['seed'])+901003+104729*i) for i, (_, _, width, anchor) in enumerate(sites))
            core.add_module('_interaction_input_residuals', self.residuals)
            for (path, module, width, anchor), residual in zip(sites, self.residuals):
                def post(source, arguments, encoded, adapter=residual):
                    return adapter(encoded, int(core.route))
                self._handles.append(module.register_forward_hook(post))
                group = self.portable.assign_private_parameters(n.optimizer, anchor, residual.parameters())
                input_sites.append({'site': path, 'position': 'after retained native logical affine+BE output',
                    'optimizer_group': group, **residual.metadata()})
            private_extra += list(self.residuals.parameters())
        gate_group = None
        if self.method in ('interaction_input_meanJ', 'interaction_input_conditionalJ'):
            self.gate = self.aggregation.make_aggregator(self.t,
                mode='conditional' if self.method == 'interaction_input_conditionalJ' else 'mean',
                task_kind='binary', seed=int(job['seed']), reference=output[1].weight)
            core.add_module('_interaction_readout', self.gate)
            gate_group = self.portable.assign_private_parameters(n.optimizer, output[1].weight, self.gate.parameters())
            private_extra += list(self.gate.parameters())
        extra_parameters = tuple(private_extra)
        def private_parameters(instance):
            return tuple(self._original_private())+extra_parameters
        core.private_parameters = types.MethodType(private_parameters, core)
        if len(n.optimizer.param_groups) != len(before_groups) or any(g is not old or
                {k:v for k,v in g.items() if k != 'params'} != old_options
                for g,old,old_options in zip(n.optimizer.param_groups,before_groups,options)):
            raise ValueError('Original native source optimizer groups/options changed')
        self.assert_identity()
        self.capture = self.library.TaskPreheadCapture(output[1], lambda: True)
        self._active = False; self._depth = 0; self._raw = None; self._work = None
        self._last_recipe = None; self._last_VALID = None; self._closed = False
        self._original_predictor = n.predictor.multidomainforward; self._original_step = n._step
        self.interaction_metadata = {'schema': 'explicit-NCN-interaction-v1', 'method_id': self.method,
            'source_receipts': pins, 'ingredient_bindings': declaration, 'fit_seed': int(job['seed']),
            'source_recipe': copy.deepcopy(job['recipe']), 'native_binding': copy.deepcopy(base.native_binding),
            'input_residual': input_sites, 'aggregation': None if self.gate is None else self.gate.metadata(),
            'readout_optimizer_group': gate_group, 'native_optimizer_group_options': options,
            'new_optimizer_groups': 0, 'source_options_changed': False,
            'credit': 'sharedF;privateF+.05A_endpoint' if self.method == 'interaction_geometry_input' else
                'sharedF;private.5W_endpoint_v2+.5L_mean' if self.method == 'interaction_responsibility_private_pool' else
                'all-block.5F+.5L_actual',
            'private_parameters': 'original BE plus actual input residual/readout only',
            'native_reduction': 'mean of4originalroute[positive mean+negative mean] over65536+65536sourcequeries',
            'query_scope': self.objectives.QUERY_SCOPE, 'source_sampler_mask_operator_RNG_preserved': True,
            'native_parameters_replaced': 0, 'native_backbone_copies': 1, 'extra_native_forwards': 0,
            'source_reference': False, 'TEST_accessed': False, 'novelty_or_accuracy_claim': False,
            'native_qualification': 'pending separate finite lifecycle receipts',
            'NCNC_and_BUDDY': 'unsupported by this port', 'final_TEST_reconstruction': 'pending',
            'mean_sidecar': 'same-call zero-forward cache/getter; production selected-checkpoint binding pending'}
        def predictor(instance, *args, **kwargs):
            if not self._active:
                return self._original_predictor(*args, **kwargs)
            if self._depth:
                raise ValueError('Recursive predictor rejected by NCN-only interaction factory')
            member = int(core.route)
            if member not in range(4) or len(self.capture.rows[member]) >= 2:
                raise ValueError('Exactly two original task calls per valid route required')
            observed = self.capture.rows[member]
            edges = args[2] if len(args) > 2 else kwargs.get('tar_ei')
            perm = n.permutation[n.cursor-n.batch_size:n.cursor]
            expected = (n.positive if not observed else n.negative)[:, perm]
            if edges is None or not self.t.equal(edges, expected):
                raise ValueError('Actual original source query identities/order changed')
            groups = [1 if not observed else 0]*n.batch_size
            self._depth += 1
            try:
                with self.capture.task(member, edges.t().detach().cpu().tolist(), groups):
                    result = self._original_predictor(*args, **kwargs)
            finally:
                self._depth -= 1
            if result.numel() != n.batch_size:
                raise ValueError('Original NCN task output row count changed')
            self._raw[member].append(result.reshape(-1)); return result
        n.predictor.multidomainforward = types.MethodType(predictor, n.predictor)
        def step(instance, losses, pool):
            return self._step(losses, pool)
        n._step = types.MethodType(step, n)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def assert_identity(self):
        self.base.assert_identity()
        current = dict(self.native.model.named_parameters())
        if any(current.get(name) is not p for name,p in self.references.items()):
            raise ValueError('Original native Parameter identity changed')
        shared = tuple(self.references.values()); private = tuple(self.native.core.private_parameters())
        owners = [id(p) for g in self.native.optimizer.param_groups for p in g['params']]
        both = [id(p) for p in shared+private]
        if len(both) != len(set(both)) or set(both) != set(owners):
            raise ValueError('Complete disjoint original shared/private optimizer partition required')

    def _finite_gradients(self):
        if any(p.grad is not None and not self.t.isfinite(p.grad).all().item() for p in self.native.model.parameters()):
            raise FloatingPointError('Nonfinite native interaction gradient before source Adam transition')

    def _recipe_receipt(self, keys, groups, *, contrast_plan=None, raw_weights=None):
        planning = sys.modules[self.library.__name__+'.planning']
        out = {'source_query_authentication': planning.authenticate(keys, groups, 'TRAIN'),
            'source_records': len(keys), 'source_positive_records': groups.count(1), 'source_negative_records': groups.count(0)}
        if self.method == 'interaction_geometry_input':
            out['endpoint_contrast_plan'] = contrast_plan if contrast_plan is not None else planning.contrast_plan(
                keys, groups, int(self.native.seed), 'endpoint', 1024, 16)
        if self.method == 'interaction_responsibility_private_pool':
            batch = self.sampler.make_query_batch([key[:2] for key in keys], [key[2] for key in keys],
                scope=self.objectives.QUERY_SCOPE, directed=False)
            if raw_weights is None:
                raw = self.sampler.raw_responsibilities(batch, int(self.native.seed), mode='endpoint', floor=.25)
                weights = self.t.tensor(raw, dtype=self.native.predictor.lin[-1].weight.dtype)
            else:
                weights = raw_weights.detach().cpu()
            out['v2_authentication'] = self.sampler.authenticate(batch, groups, role='TRAIN')
            out['raw_weight_shape'] = list(weights.shape)
            out['raw_weights_sha256'] = hashlib.sha256(weights.contiguous().numpy().tobytes()).hexdigest()
        return out

    def _step(self, native_losses, native_pool):
        t,n = self.t,self.native
        if not self._active or any(len(self.capture.rows[m]) != 2 or len(self._raw[m]) != 2 for m in range(4)):
            raise ValueError('Four complete original positive/negative task captures required')
        aligned = self.capture.aligned(t)
        if tuple(aligned['preheads'].shape) != (4,131072,64):
            raise ValueError('Keep actual full source-sized NCN prehead observations')
        raw = [t.cat(row) for row in self._raw]
        terms = self.objectives.build(t, method=self.method, link=self.library, responsibility=self.responsibility,
            sampler=self.sampler, aggregation=self.aggregation, aligned=aligned, raw_logits=raw,
            native_losses=native_losses, native_pool=native_pool, gate=self.gate,
            seed=int(n.seed), actual_route_rows=[self.capture.rows[m] for m in range(4)])
        if any(not t.isfinite(terms[key]).item() for key in ('shared_objective','private_objective')):
            raise FloatingPointError('Nonfinite declared native interaction credit')
        planning = sys.modules[self.library.__name__+'.planning']
        keys = planning.query_keys(aligned['endpoints'], aligned['relation_ids'])
        self._last_recipe = self._recipe_receipt(keys, aligned['groups'],
            contrast_plan=terms.get('diagnostics',{}).get('contrast_plan'), raw_weights=terms.get('raw_weights'))
        if terms['scalar_update']:
            terms['shared_objective'].backward(); self._finite_gradients(); n.optimizer.step()
            counts = {'reverse_collections':1,'optimizer_transitions':1,'all_gradients_from_old_state':True}
        else:
            counts = self.library.step_blocks(t,n.optimizer,tuple(self.references.values()),n.core.private_parameters(),
                terms,clip=self._finite_gradients)
        n.updates += 1; self._work = counts
        gradients = {'input_residual':[] if self.residuals is None else
                [{'shape':list(p.shape),'present':p.grad is not None,'L2':None if p.grad is None else float(p.grad.detach().norm())} for p in self.residuals.parameters()],
            'readout':[] if self.gate is None else
                [{'shape':list(p.shape),'present':p.grad is not None,'L2':None if p.grad is None else float(p.grad.detach().norm())} for p in self.gate.parameters()]}
        return {'loss':float(terms['logged_objective'].detach()),'native_own_loss':float(terms['native_own_loss'].detach()),
            'pool_loss':float(native_pool.detach()),'shared_criterion':float(terms['shared_objective'].detach()),
            'private_criterion':float(terms['private_objective'].detach()),
            'loss_semantics':'joint scalar' if terms['scalar_update'] else 'private block credit; shared F retained separately',
            'weighted_private_task_loss':float(terms['weighted_private_task_loss'].detach()) if 'weighted_private_task_loss' in terms else None,
            'query_auxiliary_loss':float(terms['query_auxiliary_loss'].detach()) if 'query_auxiliary_loss' in terms else None,
            'actual_served_loss':float(terms['actual_served_loss'].detach()) if 'actual_served_loss' in terms else None,
            'component_gradients':gradients,'interaction_diagnostics':terms['interaction_diagnostics'],
            'actual_source_recipe_receipt':copy.deepcopy(self._last_recipe),'optimizer_updates':1,'lossfinite':True}

    def train_update(self):
        if self._active or self._closed:
            raise ValueError('No nested/closed native interaction update')
        self._last_VALID = None; self.capture.clear(); self._raw = [[] for _ in range(4)]; self._active = True
        try:
            result = self.base.train_update()
            result['work'].update(self._work,extra_source_forward_views=0,captured_source_query_records=131072)
            self.assert_identity(); return result
        finally:
            self._active = False; self.capture.clear(); self._raw = None

    def train_epoch(self):
        before = self.base.epoch_progress(); rows = []
        for _ in range(before['maps_to_next_completed_epoch']):
            rows.append(self.train_update())
        after = self.base.epoch_progress()
        if not after['at_epoch_boundary'] or after['completed_epochs'] != before['completed_epochs']+1:
            raise ValueError('Original native epoch boundary changed')
        complete = self.base.epoch_accumulator['observed_maps'] == self.base.layout['optimizer_maps_per_epoch']
        return {'epoch':after['completed_epochs'],'complete_native_TRAIN':True,'optimizer_maps':len(rows),
            'native_epoch_optimizer_maps':self.base.layout['optimizer_maps_per_epoch'],
            'epoch_progress':after,'mean_update_loss':sum(row['loss'] for row in rows)/len(rows),
            'native_source_epoch_loss':self.base.epoch_accumulator['native_loss_sum']/self.base.layout['optimizer_maps_per_epoch'] if complete else None,
            'native_source_epoch_loss_complete':complete,
            'native_work_totals':{key:sum(row['work'].get(key,0) for row in rows) for key in ('optimizer_transitions','encoder_forwards','reverse_collections')},
            'last_extension_diagnostics':rows[-1]['interaction_diagnostics']}

    def validate(self):
        if self._active or self._closed:
            raise ValueError('VALID outside complete open native interaction updates only')
        result = self.base.validate(); mean,mean_loss,mean_metrics = result['outputs'],result['loss'],copy.deepcopy(result['metrics'])
        if self.method == 'interaction_input_conditionalJ':
            n,t = self.native,self.t
            with t.no_grad():
                readout = self.gate([row.to(n.predictor.lin[-1].weight) for row in result['member_outputs']])
                served = readout['served_raw_logits'].cpu()
                result.update(n._validate_result(served[:60084],served[60084:],result['work']))
                result['mean_route_weights'] = readout['weights'].detach().mean(1).cpu().tolist()
        result.update(same_bank_mean_outputs=mean,same_bank_mean_loss=mean_loss,same_bank_mean_metrics=mean_metrics,
            selection_uses='actual declared served VALID Hits@50 only')
        self._last_VALID = {'schema':'same-call-NCN-interaction-VALID-sidecar-v1','role':'VALID','method_id':self.method,
            'source_updates':self.native.updates,'ids':result['ids'].clone(),'labels':result['labels'].clone(),
            'query_endpoints':result['query_endpoints'].clone(),'actual_served_raw_logits':result['outputs'].detach().cpu().clone(),
            'actual_loss':result['loss'],'actual_metrics':copy.deepcopy(result['metrics']),
            'same_bank_mean_raw_logits':mean.detach().cpu().clone(),'same_bank_mean_loss':mean_loss,'same_bank_mean_metrics':mean_metrics,
            'selected_checkpoint_bound':False,'caller_must_bind_loaded_selected_epoch_checkpoint':True,'extra_forward_calls':0,'TEST_accessed':False}
        self.assert_identity(); return result

    def metadata(self):
        return {**self.base.metadata(),'interaction_NCN':copy.deepcopy(self.interaction_metadata)}

    def state(self):
        if self._active or self._depth or self._closed:
            raise ValueError('Checkpoint only between complete open native updates')
        self.assert_identity()
        return {'schema':STATE_SCHEMA,'method':copy.deepcopy(self.interaction_metadata),
            'last_recipe':copy.deepcopy(self._last_recipe),'native_integration':self.base.state()}

    def load_state(self,value):
        if self._active or self._closed or value.get('schema') != STATE_SCHEMA or value.get('method') != self.interaction_metadata:
            raise ValueError('Explicit combined source/recipe/initialization/component identity differs')
        saved = value['last_recipe']; native_state = value['native_integration']['native']
        if saved is not None:
            sampler = native_state['sampler']; cursor = sampler['cursor']
            index = sampler['permutation'][cursor-self.native.batch_size:cursor].detach().cpu()
            positive,negative = self.native.positive,sampler['negative']
            endpoints = self.t.cat((positive.index_select(1,index.to(positive.device)).detach().cpu(),
                negative.index_select(1,index.to(negative.device)).detach().cpu()),dim=1)
            planning = sys.modules[self.library.__name__+'.planning']
            keys = planning.query_keys(endpoints.t().tolist())
            if self._recipe_receipt(keys,[1]*65536+[0]*65536) != saved:
                raise ValueError('Saved original query identity/contrast plan/responsibility law differs')
        self.base.load_state(value['native_integration']); self._last_recipe = copy.deepcopy(saved)
        self._last_VALID = None; self.assert_identity()

    def close(self):
        if self._active:
            raise ValueError('Cannot close during native task')
        if self._closed:
            return
        self.native.predictor.multidomainforward = self._original_predictor
        self.native._step = self._original_step
        self.native.core.private_parameters = self._original_private
        self.capture.close()
        for handle in self._handles:
            handle.remove()
        self._handles = []; self._last_VALID = None; self._closed = True; self.base.close()
