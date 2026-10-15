"""Explicit fresh-native BUDDY component factory; no standalone transform chain."""
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
STATE_SCHEMA = 'explicit-BUDDY-interaction-state-v1'


def receipt(path):
    path = Path(path).resolve(); raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def load(path, package=False):
    path = Path(path).resolve(); name = '_BUDDY_interaction_'+hashlib.sha256(str(path).encode()+path.read_bytes()).hexdigest()
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


class BUDDYInteractionSession:
    def __init__(self, base, job, pins, declaration):
        self.base, self.native, self.job = base, base.native, copy.deepcopy(job)
        n = self.native; self.t = n.t; self.method = job['procedure']
        if base.family != 'BUDDY' or base.candidate_id is not None or n.procedure != 'base' \
                or n.members != 4 or n.updates != 0 or n.args.batch_size != 1024 or n.args.year != 2007:
            raise ValueError('Fresh original four-route own-loss BUDDY with full native source batch required')
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
        if owner is not n or native is not n.model or core is not n.core or len(sites) != 2 \
                or [(site[0],site[2]) for site in sites] != [('label_lin_layer',16),('lin_feat',1024)] \
                or output[0] != 'lin' or tuple(output[1].weight.shape) != (1,1040):
            raise ValueError('Only declared actual native BUDDY logical boundaries are admitted')
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
        self._original_forward = n.model.forward; self._original_step = n._step
        self._original_next_batch,self._original_hydrate = n._next_batch,n._hydrate
        self._batch_ids = self._links = self._groups = self._source_inputs = None
        self._VALID_context_checks = 0
        self.context,self.context_pin = load(EXPANSION/'link_expansion_ports_BUDDY_v2/session.py').native_context(n,base,job)
        self.interaction_metadata = {'schema': 'explicit-BUDDY-interaction-v1', 'method_id': self.method, 'native_family':'BUDDY',
            'source_receipts': pins, 'ingredient_bindings': declaration, 'fit_seed': int(job['seed']),
            'source_recipe': copy.deepcopy(job['recipe']), 'native_binding': copy.deepcopy(base.native_binding),
            'input_residual': input_sites, 'aggregation': None if self.gate is None else self.gate.metadata(),
            'readout_optimizer_group': gate_group, 'native_optimizer_group_options': options,
            'new_optimizer_groups': 0, 'source_options_changed': False,
            'credit': 'sharedF;privateF+.05A_endpoint' if self.method == 'interaction_geometry_input' else
                'sharedF;private.5W_endpoint_v2+.5L_mean' if self.method == 'interaction_responsibility_private_pool' else
                'all-block.5F+.5L_actual',
            'private_parameters': 'original BE plus actual input residual/readout only',
            'native_reduction': 'mean of four original joint BCE losses over each complete native mixed1024 query batch',
            'query_scope': self.objectives.QUERY_SCOPE, 'source_sampler_mask_operator_RNG_preserved': True,
            'native_parameters_replaced': 0, 'native_backbone_copies': 1, 'extra_native_forwards': 0,
            'source_reference': False, 'TEST_accessed': False, 'novelty_or_accuracy_claim': False,
            'native_qualification': 'pending separate finite lifecycle receipts',
            'other_link_ports': {'NCN':'separate NCN interaction package','NCNC':'separate NCNC interaction package'}, 'final_TEST_reconstruction': 'pending',
            'materialized_context':self.context_pin,
            'context_scope':'authenticated seed0/year2007 original first two TRAIN batches and both whole VALID roles',
            'native_VALID_criterion':'unchanged source evaluator and source diagnostic loss for actual and same-bank mean logits',
            'native_BN':'original shared running-state and affine unchanged; private BN is a separate candidate',
            'mean_sidecar': 'same-call zero-forward cache/getter; production selected-checkpoint binding pending'}
        def next_batch(instance):
            ids = self._original_next_batch()
            if self._active:
                self._batch_ids = ids.detach().cpu().clone()
                self._links = n.links[ids].detach().cpu().clone()
                self._groups = n.labels[ids].detach().cpu().clone()
            return ids
        def hydrate(instance, links):
            values = self._original_hydrate(links)
            if self._active:
                if self._links is None or not self.t.equal(links.detach().cpu(),self._links):
                    raise ValueError('Actual source BUDDY hydration rows differ from original sampler')
                self._source_inputs = values
                if n.updates < 2:
                    expected = self.context['TRAIN_batches'][n.updates]
                    if not self.t.equal(self._batch_ids,expected['sampler_ids']) or not self.t.equal(self._links,expected['links']) or not self.t.equal(self._groups,expected['labels']):
                        raise ValueError('Original BUDDY sampler rows/labels differ from authenticated native context')
                    self._check_inputs(values,expected['inputs'])
            else:
                expected = next((row for row in self.context['VALID_inputs'].values()
                    if links.shape==row['links'].shape and self.t.equal(links.detach().cpu(),row['links'])),None)
                if expected is None: raise ValueError('Unexpected native BUDDY VALID role/query population')
                self._check_inputs(values,expected['inputs']);self._VALID_context_checks += 1
            return values
        def forward(instance, *args, **kwargs):
            if not self._active: return self._original_forward(*args,**kwargs)
            member = int(core.route)
            if self._depth or member not in range(4) or self._raw[member] is not None:
                raise ValueError('Exactly one original source call per BUDDY route required')
            if self._source_inputs is None or len(args)<4 or any(actual is not expected for actual,expected in zip(args[:4],self._source_inputs)):
                raise ValueError('Route did not receive actual source-hydrated input objects')
            self._depth += 1
            try:
                with self.capture.task(member,self._links.tolist(),self._groups.tolist()):
                    result = self._original_forward(*args,**kwargs)
            finally:self._depth -= 1
            if result.numel()!=len(self._batch_ids):raise ValueError('Native BUDDY task row count changed')
            self._raw[member]=result.reshape(-1)
            return result
        n._next_batch,n._hydrate = types.MethodType(next_batch,n),types.MethodType(hydrate,n)
        n.model.forward = types.MethodType(forward,n.model)
        def step(instance, losses, pool):
            return self._step(losses, pool)
        n._step = types.MethodType(step, n)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def _check_inputs(self,actual,expected):
        if len(actual)!=4 or len(expected)!=4 or any(not self.t.equal(a.detach().cpu(),b) for a,b in zip(actual,expected)):
            raise ValueError('Actual native BUDDY hydration differs from authenticated source inputs')

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

    def _recipe_receipt(self, keys, groups, *, sampler_ids, contrast_plan=None, raw_weights=None):
        planning = sys.modules[self.library.__name__+'.planning']
        out = {'source_sampler_ids':sampler_ids.detach().cpu().tolist(), 'source_query_authentication': planning.authenticate(keys, groups, 'TRAIN'),
            'source_records': len(keys), 'source_positive_records': groups.count(1), 'source_negative_records': groups.count(0)}
        if self.method == 'interaction_geometry_input':
            out['endpoint_contrast_plan'] = contrast_plan if contrast_plan is not None else planning.contrast_plan(
                keys, groups, int(self.native.seed), 'endpoint', 1024, 16)
        if self.method == 'interaction_responsibility_private_pool':
            batch = self.sampler.make_query_batch([key[:2] for key in keys], [key[2] for key in keys],
                scope=self.objectives.QUERY_SCOPE, directed=False)
            if raw_weights is None:
                raw = self.sampler.raw_responsibilities(batch, int(self.native.seed), mode='endpoint', floor=.25)
                weights = self.t.tensor(raw, dtype=self.native.model.lin.weight.dtype)
            else:
                weights = raw_weights.detach().cpu()
            out['v2_authentication'] = self.sampler.authenticate(batch, groups, role='TRAIN')
            out['raw_weight_shape'] = list(weights.shape)
            out['raw_weights_sha256'] = hashlib.sha256(weights.contiguous().numpy().tobytes()).hexdigest()
        return out

    def _step(self, native_losses, native_pool):
        t,n = self.t,self.native
        if not self._active or any(len(self.capture.rows[m]) != 1 or self._raw[m] is None for m in range(4)):
            raise ValueError('Four complete original native mixed-query task captures required')
        aligned = self.capture.aligned(t)
        if tuple(aligned['preheads'].shape) != (4,len(self._batch_ids),1040):
            raise ValueError('Keep actual full source-sized BUDDY prehead observations')
        raw = list(self._raw)
        terms = self.objectives.build(t, method=self.method, link=self.library, responsibility=self.responsibility,
            sampler=self.sampler, aggregation=self.aggregation, aligned=aligned, raw_logits=raw,
            native_losses=native_losses, native_pool=native_pool, gate=self.gate,
            seed=int(n.seed), actual_route_rows=[self.capture.rows[m] for m in range(4)])
        if any(not t.isfinite(terms[key]).item() for key in ('shared_objective','private_objective')):
            raise FloatingPointError('Nonfinite declared native interaction credit')
        planning = sys.modules[self.library.__name__+'.planning']
        keys = planning.query_keys(aligned['endpoints'], aligned['relation_ids'])
        self._last_recipe = self._recipe_receipt(keys, aligned['groups'],sampler_ids=self._batch_ids,
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
        if self._active or self._closed:raise ValueError('No nested/closed native interaction update')
        self._last_VALID = None;self.capture.clear();self._raw = [None]*4
        self._batch_ids = self._links = self._groups = self._source_inputs = None
        self._active = True
        try:
            result = self.base.train_update()
            result['work'].update(self._work,extra_source_forward_views=0,captured_source_query_records=len(self._batch_ids))
            result['interaction_diagnostics']['source_group_counts']={'1':int((self._groups==1).sum()),'0':int((self._groups==0).sum())}
            result['interaction_diagnostics']['four_route_query_signatures']=self.capture.aligned(self.t)['route_query_signatures']
            self.assert_identity();return result
        finally:
            self._active=False;self.capture.clear();self._raw=self._source_inputs=None

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
            'native_source_epoch_loss':self.base.epoch_accumulator['native_loss_sum']*1024/self.base.layout['task_query_records_per_epoch'] if complete else None,
            'native_source_epoch_loss_complete':complete,
            'native_work_totals':{key:sum(row['work'].get(key,0) for row in rows) for key in ('optimizer_transitions','native_BUDDY_forwards','query_records','reverse_collections')},
            'last_extension_diagnostics':rows[-1]['interaction_diagnostics']}

    def validate(self):
        if self._active or self._closed:
            raise ValueError('VALID outside complete open native interaction updates only')
        self._VALID_context_checks = 0
        result = self.base.validate()
        if self._VALID_context_checks!=2 or self._depth:
            raise ValueError('Both full official native BUDDY VALID input roles must authenticate')
        result['materialized_VALID_context_verified']=True
        mean,mean_loss,mean_metrics = result['outputs'],result['loss'],copy.deepcopy(result['metrics'])
        if self.method == 'interaction_input_conditionalJ':
            n,t = self.native,self.t
            with t.no_grad():
                readout = self.gate([row.to(n.model.lin.weight) for row in result['member_outputs']])
                served = readout['served_raw_logits'].cpu()
                result.update(n._validate_result(served[:60084],served[60084:],result['work']))
                result['mean_route_weights'] = readout['weights'].detach().mean(1).cpu().tolist()
        result.update(same_bank_mean_outputs=mean,same_bank_mean_loss=mean_loss,same_bank_mean_metrics=mean_metrics,
            selection_uses='actual declared served VALID Hits@50 only')
        self._last_VALID = {'schema':'same-call-BUDDY-interaction-VALID-sidecar-v1','role':'VALID','method_id':self.method,
            'source_updates':self.native.updates,'ids':result['ids'].clone(),'labels':result['labels'].clone(),
            'query_endpoints':result['query_endpoints'].clone(),'actual_served_raw_logits':result['outputs'].detach().cpu().clone(),
            'actual_loss':result['loss'],'actual_metrics':copy.deepcopy(result['metrics']),
            'same_bank_mean_raw_logits':mean.detach().cpu().clone(),'same_bank_mean_loss':mean_loss,'same_bank_mean_metrics':mean_metrics,
            'selected_checkpoint_bound':False,'caller_must_bind_loaded_selected_epoch_checkpoint':True,'extra_forward_calls':0,'TEST_accessed':False}
        self.assert_identity(); return result

    def metadata(self):
        return {**self.base.metadata(),'interaction_BUDDY':copy.deepcopy(self.interaction_metadata)}

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
            sampler=native_state['sampler'];indices=sampler['sample_indices'];cursor=int(sampler['batch_cursor'])
            if indices is None or cursor<1 or sampler['loader_rng_before'] is None:
                raise ValueError('Saved recipe requires an actual completed original native batch')
            links=self.t.cat((self.native.positive.cpu(),sampler['train_negative'].cpu()),dim=0)
            with self.t.random.fork_rng(devices=[],enabled=True):
                self.t.set_rng_state(sampler['loader_rng_before'].cpu())
                iterator=iter(self.native.DataLoader(range(len(links)),1024,shuffle=True))
                for _ in range(cursor):rows=next(iterator)
                ids=indices.cpu()[rows]
            groups=(ids<len(self.native.positive)).long().tolist()
            planning=sys.modules[self.library.__name__+'.planning']
            keys=planning.query_keys(links[ids].tolist())
            if self._recipe_receipt(keys,groups,sampler_ids=ids)!=saved:
                raise ValueError('Saved source sampler/queries/contrast plan/responsibility law differs')
        self.base.load_state(value['native_integration']); self._last_recipe = copy.deepcopy(saved)
        self._last_VALID = None; self.assert_identity()

    def close(self):
        if self._active:
            raise ValueError('Cannot close during native task')
        if self._closed:
            return
        self.native.model.forward = self._original_forward
        self.native._next_batch,self.native._hydrate = self._original_next_batch,self._original_hydrate
        self.native._step = self._original_step
        self.native.core.private_parameters = self._original_private
        self.capture.close()
        for handle in self._handles:
            handle.remove()
        self._handles = []; self._last_VALID = None; self._closed = True; self.base.close()
