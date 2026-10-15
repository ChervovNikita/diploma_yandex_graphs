"""BUDDY task orchestration for frozen C6 and corrected responsibility v2.

The native source session still builds adjacency, samples negatives, chooses
the complete source minibatch, executes all four routes and supplies its losses.
This proxy observes the actual task calls and replaces only the declared risk
assignment/served readout. NCN/NCNC remain separate immutable packages.
"""
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
C6 = {'aggregation_mean_globalJ_v1': 'mean', 'aggregation_global_globalJ_v1': 'global',
      'aggregation_conditional_globalJ_v1': 'conditional'}
RESPONSIBILITY = {'link_unit_responsibility_private_v2': 'unit',
    'link_query_iid_responsibility_private_v2': 'query_iid',
    'link_endpoint_responsibility_private_v2': 'endpoint'}
METHODS = {**C6, **RESPONSIBILITY}
QUERY_SCOPE = 'ogbl-collab/official-time/source-node-ID/undirected/relation-0/v1'
STATE_SCHEMA = 'native-BUDDY-expanded-risk-wrapped-state-v1'
_CONTEXTS = {}


def receipt(path):
    path = Path(path).resolve(); raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def load(path, package=False):
    path = Path(path).resolve()
    name = '_new_BUDDY_port_'+hashlib.sha256(str(path).encode()+path.read_bytes()).hexdigest()
    if name not in sys.modules:
        options = {'submodule_search_locations': [str(path.parent)]} if package else {}
        spec = importlib.util.spec_from_file_location(name, path, **options)
        module = importlib.util.module_from_spec(spec); sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def bindings(root, expected):
    root = Path(root).resolve()
    assert HERE.is_relative_to(root)
    manifest_pin = receipt(HERE/'MANIFEST.json')
    if expected != manifest_pin['sha256']:
        raise ValueError('Pin the actual new link port manifest')
    pins = [manifest_pin]
    for row in json.loads((HERE/'MANIFEST.json').read_text())['files']:
        path = (HERE/row['path']).resolve()
        if not path.is_relative_to(HERE): raise ValueError('Own manifest escapes package')
        actual = receipt(path)
        if any(actual[key] != row[key] for key in ('bytes', 'sha256')):
            raise ValueError('New link port source changed: '+row['path'])
        pins.append(actual)
    for row in json.loads((HERE/'SOURCES.json').read_text())['source_files']:
        path = (PHASE/row['path_relative_to_phase']).resolve()
        if not path.is_relative_to(root): raise ValueError('Frozen dependency escapes repository')
        actual = receipt(path)
        if any(actual[key] != row[key] for key in ('bytes', 'sha256')):
            raise ValueError('Frozen dependency changed: '+row['path_relative_to_phase'])
        pins.append(actual)
    return pins


def tensor_binding(value):
    raw = value.detach().contiguous().cpu().numpy()
    return {'shape': list(value.shape), 'dtype': str(value.dtype),
            'sha256': hashlib.sha256(memoryview(raw).cast('B')).hexdigest()}


def native_context(native, base, job):
    settings = job['method_settings']; root = Path(job['recipe']['_repository_root']).resolve()
    path = Path(settings['context_manifest']).resolve()
    if not path.is_relative_to(root): raise ValueError('BUDDY materialized context escapes repository')
    pin = receipt(path)
    if pin['sha256'] != settings['context_manifest_sha256']: raise ValueError('BUDDY context manifest changed')
    manifest = json.loads(path.read_text())
    if manifest.get('schema') != 'native-BUDDY-year2007-materialized-context-v1' or manifest.get('status') != 'passed' \
            or manifest.get('optimizer_transitions') != 0 or manifest.get('TEST_accessed') is not False:
        raise ValueError('Successful authentic source context with zero updates and closed TEST required')
    if job['seed'] != 0 or native.args.year != 2007 or manifest['recipe']['native_fixed_config'] != native.recipe['native_fixed_config']:
        raise ValueError('This context binds the exact native seed0/year2007 recipe only')
    actual_data = base.native_binding['data_files']
    if actual_data != manifest['provider_data_files']: raise ValueError('Materialized context provider bytes/path differ from actual source')
    file_pin = manifest['context_file']; file_path = Path(file_pin['path']).resolve()
    if not file_path.is_relative_to(root): raise ValueError('BUDDY context tensor file escapes repository')
    key = file_pin['sha256']
    if key not in _CONTEXTS:
        if receipt(file_path) != file_pin: raise ValueError('BUDDY materialized context tensor bytes changed')
        _CONTEXTS[key] = native.t.load(file_path, map_location='cpu', weights_only=False)
    context = _CONTEXTS[key]
    if context['schema'] != 'native-BUDDY-year2007-materialized-inputs-v1': raise ValueError('Unknown materialized input schema')
    for name, actual in (('features',native.features),('degrees',native.degrees),('cards',native.cards),('source_positive_queries',native.positive)):
        if tensor_binding(actual) != manifest['static_tensor_bindings'][name]:
            raise ValueError('Actual native BUDDY static context differs: '+name)
    for hop, table in native.hash_tables.items():
        for name, value in table.items():
            if tensor_binding(value) != manifest['static_tensor_bindings']['hash_tables'][str(hop)][name]:
                raise ValueError('Actual native BUDDY hash bank differs from materialized context')
    return context, pin


class BUDDYExpandedSession:
    def __init__(self, base, job, pins):
        self.base, self.native, self.job = base, base.native, copy.deepcopy(job)
        n = self.native
        self.t = n.t
        self.method = job['procedure']
        if base.family != 'BUDDY':
            raise NotImplementedError('This separate port requires actual native BUDDY')
        if n.procedure != 'base' or n.members != 4 or n.updates != 0 or base.candidate_id is not None:
            raise ValueError('Attach only to a fresh standalone four-route native own-loss BUDDY session')
        if int(n.args.batch_size) != 1024 or int(n.positive.shape[1]) != 2:
            raise ValueError('Keep the complete native BUDDY minibatch and query interface')
        self.references = dict(n.source_parameter_references)
        self._active = False; self._call_depth = 0; self._captures = None; self._last_step_work = None
        self._original_next_batch = n._next_batch
        self._original_hydrate = n._hydrate
        self._batch_ids = self._links = self._groups = self._source_inputs = None
        self._original_step = n._step
        self.gate = None
        responsibility_package = load(EXPANSION/'structured_responsibility_control_v2/__init__.py', package=True)
        self.responsibility = import_module(responsibility_package.__name__+'.objectives')
        self.sampler = import_module(responsibility_package.__name__+'.sampler')
        self.c6 = load(EXPANSION/'conditional_aggregation_v1/core.py')
        self.steps = load(EXPANSION/'link/__init__.py', package=True)
        options = [{key: copy.deepcopy(value) for key,value in group.items() if key != 'params'} for group in n.optimizer.param_groups]
        if self.method in C6:
            head = n.model.lin
            self.gate = self.c6.make_aggregator(self.t, mode=C6[self.method], task_kind='binary', seed=job['seed'], reference=head.weight)
            n.model.add_module('_expanded_BUDDY_aggregation', self.gate)
            owners = [i for i, group in enumerate(n.optimizer.param_groups) if any(p is head.weight for p in group['params'])]
            if len(owners) != 1: raise ValueError('Native task head must have exactly one source optimizer owner')
            n.optimizer.param_groups[owners[0]]['params'].extend(self.gate.parameters())
            assignment = {'anchor': 'lin.weight', 'group': owners[0], 'new_groups': 0}
        else:
            assignment = {'shared': 'unchanged native Parameters', 'private': 'existing BE boundary banks', 'new_groups': 0}
        if options != [{key: value for key,value in group.items() if key != 'params'} for group in n.optimizer.param_groups]:
            raise ValueError('Native optimizer dictionaries/options changed')
        self.method_metadata = {'schema': 'native-BUDDY-expanded-risk-v1', 'method_id': self.method,
            'native_family': 'BUDDY', 'other_link_ports': {'NCN': 'separate finished package', 'NCNC': 'separate source/CPU package'},
            'source_receipts': pins, 'seed': int(job['seed']), 'query_scope': QUERY_SCOPE,
            'aggregation': self.gate.metadata() if self.gate is not None else None,
            'responsibility_mode': RESPONSIBILITY.get(self.method), 'raw_weight_floor': .25 if self.method in RESPONSIBILITY else None,
            'source_joint_batch_queries': 1024, 'source_positive_negative_counts': 'actual native mixed labels; no artificial class balancing',
            'native_reduction': 'one original joint BCE minibatch mean', 'native_sampler_adjacency_and_task_forwards': 'unchanged source session',
            'risk': 'all-block J=.5F+.5L_actual_served' if self.method in C6 else 'shared F; private within-source-group-normalized W',
            'optimizer_assignment': assignment, 'optimizer_options_changed': False,
            'native_parameters_replaced': 0, 'extra_backbones': 0, 'extra_task_forwards': 0,
            'source_RNG_policy': 'unchanged four private route streams; caller/checkpoint owns global RNG',
            'raw_query_control': 'two independent Exp halves for nonself; one Exp for self-loop',
            'normalized_marginal_matching_or_unbiased_risk_claim': False,
            'final_TEST_reconstruction': 'pending/unsupported', 'TEST_accessed': False,
            'qualification': 'separate exact BUDDY CPU engineering receipts required; no quality/convergence claim'}
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
                if self._links is None or not self.t.equal(links.cpu(), self._links):
                    raise ValueError('Actual BUDDY hydration endpoints differ from original sampler rows')
                self._source_inputs = values
                self._check_materialized_batch(values)
            elif hasattr(self,'context'):
                matched = False
                for expected in self.context['VALID_inputs'].values():
                    if links.shape == expected['links'].shape and self.t.equal(links.cpu(),expected['links']):
                        if any(not self.t.equal(value.detach().cpu(),bound) for value,bound in zip(values,expected['inputs'])):
                            raise ValueError('Actual source VALID hydration differs from materialized BUDDY context')
                        matched = True; self._VALID_context_checks += 1; break
                if not matched: raise ValueError('Unexpected BUDDY validation role/query context')
            return values
        def observe_model(module, arguments, output):
            if self._active:
                member = int(n.core.route)
                if member not in range(4) or self._captures[member] is not None:
                    raise ValueError('Expected one original BUDDY forward per route')
                if self._source_inputs is None or any(actual is not expected for actual,expected in zip(arguments[:4],self._source_inputs)):
                    raise ValueError('BUDDY route did not receive the actual source-hydrated inputs')
                self._captures[member] = output.reshape(-1)
            return None
        n._next_batch = types.MethodType(next_batch,n)
        n._hydrate = types.MethodType(hydrate,n)
        self._model_hook = n.model.register_forward_hook(observe_model)
        def step(instance, native_losses, native_pool):
            return self._step(native_losses, native_pool)
        n._step = types.MethodType(step, n)
        self._attach_context(job)
        self.assert_identity()

    def __getattr__(self, name): return getattr(self.base, name)

    def _attach_context(self, job):
        self.context, pin = native_context(self.native,self.base,job)
        self.method_metadata['materialized_native_context'] = pin
        self.method_metadata['context_scope'] = 'seed0 native year2007 static hash/features and first two original TRAIN batches; complete source VALID materialized'

    def _check_materialized_batch(self, values):
        index = int(self.native.updates)
        if index < 2:
            expected = self.context['TRAIN_batches'][index]
            if not self.t.equal(self._batch_ids,expected['sampler_ids']) or not self.t.equal(self._links,expected['links']) or not self.t.equal(self._groups,expected['labels']):
                raise ValueError('Actual BUDDY sampler rows differ from original materialized batch')
            if any(not self.t.equal(actual.detach().cpu(),bound) for actual,bound in zip(values,expected['inputs'])):
                raise ValueError('Actual BUDDY hydrated inputs differ from original materialized batch')

    def assert_identity(self):
        self.base.assert_identity()
        current = dict(self.native.model.named_parameters())
        if any(current.get(name) is not value for name,value in self.references.items()):
            raise ValueError('Original native Parameter identity changed')

    def _finite_gradients(self):
        for parameter in self.native.model.parameters():
            if parameter.grad is not None and not self.t.isfinite(parameter.grad).all().item():
                raise FloatingPointError('Nonfinite native/private/readout gradient before source optimizer transition')

    def _step(self, native_losses, native_pool):
        t, n = self.t, self.native
        if not self._active or any(row is None for row in self._captures) or self._groups is None:
            raise ValueError('Four complete original BUDDY mixed-query route captures required')
        raw = list(self._captures)
        logits = t.stack(raw)
        groups = self._groups.tolist()
        reconstructed = self.responsibility.reduce_native(t, self.responsibility.native_record_losses(t, logits, groups, 'BUDDY'), groups, 'BUDDY')
        if not t.allclose(reconstructed.detach(), t.stack(native_losses).detach(), atol=1e-7, rtol=1e-6):
            raise ValueError('Captured task logits do not recover the native scalar losses')
        if self.gate is not None:
            def native_loss(value):
                return n.native_loss(value, self._groups.to(value.device))
            terms = self.c6.native_objective(t, native_member_losses=native_losses, raw_logits=raw, aggregator=self.gate,
                served_native_loss=native_loss, native_regularizer=logits.new_zeros(()))
            terms['J'].backward(); self._finite_gradients(); n.optimizer.step()
            F, objective = terms['F'], terms['J']
            diagnostics = {'source_reduction_parity_checked': True, 'aggregation_mode': C6[self.method],
                'gate_parameter_count': sum(p.numel() for p in self.gate.parameters()),
                'served_loss': float(terms['L'].detach()), 'mean_route_weights': terms['weights'].detach().mean(1).cpu().tolist()}
            counts = {'reverse_collections': 1, 'optimizer_transitions': 1, 'all_gradients_from_old_state': True}
        else:
            signatures, batch = [], None
            for row in self._captures:
                endpoints = self._links.tolist()
                actual_batch = self.sampler.make_query_batch(endpoints, scope=QUERY_SCOPE, directed=False)
                signatures.append(self.sampler.authenticate(actual_batch, groups, role='TRAIN'))
                if batch is None: batch = actual_batch
            terms = self.responsibility.build_objectives(t, batch=batch, groups=groups, family='BUDDY', logits=logits,
                native_member_losses=native_losses, route_query_signatures=signatures, seed=n.seed,
                mode=RESPONSIBILITY[self.method], floor=.25, role='TRAIN')
            counts = self.steps.step_blocks(t, n.optimizer, tuple(self.references.values()), n.core.private_parameters(), terms, clip=self._finite_gradients)
            F, objective, diagnostics = terms['native_own_loss'], terms['private_objective'], terms['diagnostics']
            diagnostics.update(raw_weight_minimum=float(terms['raw_weights'].min()), raw_weight_maximum=float(terms['raw_weights'].max()),
                actual_four_route_query_signatures=signatures)
        n.updates += 1; self._last_step_work = counts
        return {'loss': float(objective.detach()), 'own_loss': float(F.detach()), 'native_own_loss': float(F.detach()),
            'pool_loss': float(native_pool.detach()), 'optimizer_updates': 1, 'lossfinite': True,
            'expanded_link_diagnostics': diagnostics, 'shared_criterion': float(F.detach()) if self.gate is None else float(objective.detach()),
            'private_criterion': float(objective.detach())}

    def train_update(self):
        if self._active: raise ValueError('Nested native update')
        self._captures = [None]*4; self._active = True
        self._batch_ids = self._links = self._groups = self._source_inputs = None
        try:
            result = self.base.train_update()
            result['work'].update(self._last_step_work, extra_source_forward_views=0, captured_native_task_query_records=len(self._links))
            self.assert_identity()
            return result
        finally:
            self._active = False; self._captures = None

    def train_epoch(self):
        # The frozen epoch driver calls base.train_update, so dispatch explicitly
        # through this proxy to keep the actual task observer active.
        before = self.base.epoch_progress(); rows = []
        for _ in range(before['maps_to_next_completed_epoch']): rows.append(self.train_update())
        after = self.base.epoch_progress()
        if not after['at_epoch_boundary'] or after['completed_epochs'] != before['completed_epochs']+1:
            raise ValueError('Native epoch boundary mismatch')
        return {'epoch': after['completed_epochs'], 'complete_native_TRAIN': True, 'optimizer_maps': len(rows),
            'native_epoch_optimizer_maps': self.base.layout['optimizer_maps_per_epoch'],
            'mean_update_loss': sum(row['loss'] for row in rows)/len(rows), 'epoch_progress': after,
            'native_work_totals': {key: sum(row['work'].get(key, 0) for row in rows) for key in ('optimizer_transitions','native_BUDDY_forwards','query_records','reverse_collections')},
            'source_loss_complete': before['resumed_prefix_maps'] == 0,
            'last_extension_diagnostics': rows[-1]['expanded_link_diagnostics']}

    def validate(self):
        self._VALID_context_checks = 0
        result = self.base.validate()
        if self._VALID_context_checks != 2: raise ValueError('Both complete materialized BUDDY VALID roles must be hydrated')
        result['materialized_VALID_context_verified'] = True
        if self.gate is not None:
            n, t = self.native, self.t
            mean, mean_loss, mean_metrics = result['outputs'], result['loss'], copy.deepcopy(result['metrics'])
            with t.no_grad():
                head = n.model.lin.weight
                readout = self.gate([row.to(head) for row in result['member_outputs']])
                served = readout['served_raw_logits'].cpu()
                from ogb.linkproppred import Evaluator
                evaluator = Evaluator(name='ogbl-collab'); evaluator.K=50
                metric = evaluator.eval({'y_pred_pos':served[:60084], 'y_pred_neg':served[60084:]})['hits@50']
                actual = {'outputs':served, 'loss':float(n.native_loss(served,result['labels'])), 'metrics':{'hits@50':float(metric)},
                    'diagnostic_loss':float(n.native_loss(served,result['labels'])), 'lossfinite':bool(t.isfinite(served).all()),
                    'source_split_group_diagnostic_loss':mean_loss, 'BUDDY_training_and_new_served_diagnostic_reduction':'joint native BCE mean'}
                mean_loss = float(n.native_loss(mean,result['labels']))
            result.update(actual, same_bank_mean_outputs=mean, same_bank_mean_loss=mean_loss,
                same_bank_mean_metrics=mean_metrics, selection_uses='actual declared served VALID Hits@50 only',
                mean_route_weights=readout['weights'].detach().mean(1).cpu().tolist())
        if self.gate is None:
            source_diagnostic=result['loss']
            joint=float(self.native.native_loss(result['outputs'],result['labels']))
            result.update(loss=joint,diagnostic_loss=joint,source_split_group_diagnostic_loss=source_diagnostic,
                BUDDY_training_and_new_served_diagnostic_reduction='joint native BCE mean')
        self.assert_identity()
        return result

    def metadata(self):
        return {**self.base.metadata(), 'expanded_BUDDY_port': copy.deepcopy(self.method_metadata)}

    def state(self):
        if self._active or self._call_depth: raise ValueError('Checkpoint only outside live native task calls')
        return {'schema': STATE_SCHEMA, 'method': copy.deepcopy(self.method_metadata), 'native_integration': self.base.state()}

    def load_state(self, value):
        if self._active or value.get('schema') != STATE_SCHEMA or value.get('method') != self.method_metadata:
            raise ValueError('Checkpoint method/source/query namespace differs')
        self.base.load_state(value['native_integration']); self.assert_identity()

    def close(self):
        if self._active: raise ValueError('Cannot close active task')
        self.native._next_batch = self._original_next_batch
        self.native._hydrate = self._original_hydrate
        self._model_hook.remove()
        self.native._step = self._original_step
        self.base.close()
