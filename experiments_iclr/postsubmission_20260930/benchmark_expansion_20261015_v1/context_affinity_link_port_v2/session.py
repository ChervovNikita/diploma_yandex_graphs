"""Native NCN own-loss update plus fixed common/split factual context loss."""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import types

HERE = Path(__file__).resolve().parent
EXPANSION = HERE.parent
PHASE = EXPANSION.parent
CORE = EXPANSION/'context_affinity_v1'
CORE_SHA256 = 'c510c0dbf02236c6345df99d2906534866cab9a65c8562557994e907303baa3b'
CORE_MANIFEST_SHA256 = 'fe6b26f875ece1173be06d8656b2d6442ac216e7afcef0e6d40fb7a5b36cf448'
TARGET_SPEC = EXPANSION/'expanded_study_design_v1/TASK_CONTEXT_SPEC.md'
TARGET_SPEC_SHA256 = '671173b3bcf5e2ba4e56c5ee9c1bfebd45f7494c596510bb3cabf182a73577ec'
METHODS = {'context_affinity_link_common_v2': 'common', 'context_affinity_link_split_v2': 'split'}
STATE_SCHEMA = 'native-NCN-context-affinity-state-v2'


def receipt(path):
    path = Path(path).resolve(); raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def load(path, package=False):
    path = Path(path).resolve()
    name = '_context_NCN_'+hashlib.sha256(str(path).encode()+path.read_bytes()).hexdigest()
    if name not in sys.modules:
        options = {'submodule_search_locations': [str(path.parent)]} if package else {}
        spec = importlib.util.spec_from_file_location(name, path, **options)
        module = importlib.util.module_from_spec(spec); sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def verify_package(directory, expected):
    directory = Path(directory).resolve(); pin = receipt(directory/'MANIFEST.json')
    if pin['sha256'] != expected:
        raise ValueError('Pin actual frozen package manifest: '+str(directory))
    pins = [pin]
    for row in json.loads((directory/'MANIFEST.json').read_text())['files']:
        path = (directory/row['path']).resolve()
        if not path.is_relative_to(directory):
            raise ValueError('Own manifest escapes package')
        actual = receipt(path)
        if any(actual[key] != row[key] for key in ('bytes', 'sha256')):
            raise ValueError('Frozen package changed: '+row['path'])
        pins.append(actual)
    return pins


def bindings(root, expected):
    root = Path(root).resolve()
    if not HERE.is_relative_to(root):
        raise ValueError('Actual repository root must contain this package')
    pins = verify_package(HERE, expected)+verify_package(CORE, CORE_MANIFEST_SHA256)
    if receipt(CORE/'core.py')['sha256'] != CORE_SHA256 or receipt(TARGET_SPEC)['sha256'] != TARGET_SPEC_SHA256:
        raise ValueError('Frozen shared objective or target specification changed')
    pins.append(receipt(TARGET_SPEC))
    for row in json.loads((HERE/'SOURCES.json').read_text())['source_files']:
        path = (PHASE/row['path_relative_to_phase']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Frozen dependency escapes repository')
        actual = receipt(path)
        if any(actual[key] != row[key] for key in ('bytes', 'sha256')):
            raise ValueError('Frozen native link dependency changed: '+row['path_relative_to_phase'])
        pins.append(actual)
    return pins


class NCNContextSession:
    def __init__(self, base, job, pins):
        self.base, self.native, self.job = base, base.native, copy.deepcopy(job)
        n = self.native; self.t = n.t; self.method = job['procedure']; self.mode = METHODS[self.method]
        if base.family != 'NCN' or n.procedure != 'base' or n.members != 4 or n.updates != 0 or base.candidate_id is not None:
            raise ValueError('Fresh actual native own-loss four-route NCN only; NCNC/BUDDY unsupported')
        if int(n.batch_size) != 65536 or tuple(n.positive.shape[:1]) != (2,) or n.predictor.lin[-1].weight.shape != (1, 64):
            raise ValueError('Keep full original NCN source minibatch and 64-wide task prehead')
        self.references = dict(n.source_parameter_references)
        self.context = load(HERE/'context.py')
        self.objective = load(CORE/'core.py')
        self.library = load(EXPANSION/'link/__init__.py', package=True)
        self.planning = sys.modules[self.library.__name__+'.planning']
        self.capture = self.library.TaskPreheadCapture(n.predictor.lin[-1], lambda: True)
        started = time.perf_counter()
        self.target_graph = self.context.TrainGraph(self.t, n.positive, int(n.x.shape[0]))
        self.preparation_seconds = time.perf_counter()-started
        self._active = False; self._depth = 0; self._last = None; self._work = None
        self._original_predictor = n.predictor.multidomainforward
        self._original_step = n._step
        self.method_metadata = {'schema': 'context-affinity-native-NCN-v1', 'method_id': self.method, 'mode': self.mode,
            'source_receipts': pins, 'fit_seed': int(n.seed), 'core_sha256': CORE_SHA256,
            'core_manifest_sha256': CORE_MANIFEST_SHA256, 'target_spec_sha256': TARGET_SPEC_SHA256,
            'support_addendum': receipt(HERE/'LINK_SUPPORT_ADDENDUM.md'),
            'target_graph': copy.deepcopy(self.target_graph.receipt),
            'A': 'sorted log1p endpoint degrees in full simple undirected loopless TRAIN graph excluding own pair',
            'B': ['log1p(CN)', 'log1p(AA_sum)', 'log1p(RA_sum)'],
            'target_graph_is_predictor_graph': False,
            'predictor_graph': 'unchanged native whole-positive-minibatch mask before symmetric sparse coalescing',
            'query_scope': 'ogbl-collab/official-time/source-node-ID/undirected/relation-0/v1',
            'selection': 'existing frozen link hash; unique nonself nonconflicting objects, earliest same-group duplicate; up to512pergroup',
            'native_F': 'mean of4route positive-mean-plus-negative-mean losses over all65536+65536originalqueries',
            'auxiliary_reduction': 'sum of supported positive and negative source group means, then4route mean',
            'assignment': 'allroutes.5qA+.5qB' if self.mode == 'common' else 'routes0/1qA;routes2/3qB',
            'update': 'F+.05A, ordinary all-block backward, one original Adamtransition',
            'temperature': .2, 'coefficient': .05, 'normalization_epsilon': 1e-12,
            'source_parameter_replacements': 0, 'new_parameters': 0, 'extra_backbone_or_predictor_forwards': 0,
            'optimizer_options_changed': False, 'serving': 'unchanged arithmetic mean raw logits; no context at inference',
            'TEST_accessed': False, 'novelty_or_accuracy_claim': False,
            'qualified_scope': 'separate bounded native NCN CPU engineering receipt required',
            'other_link_ports': {'NCNC': 'unsupported', 'BUDDY': 'unsupported'}, 'final_TEST_reconstruction': 'pending'}
        def predictor(instance, *args, **kwargs):
            if not self._active:
                return self._original_predictor(*args, **kwargs)
            if self._depth:
                raise ValueError('NCN-only proxy encountered recursive source predictor')
            member = int(n.core.route)
            if member not in range(4):
                raise ValueError('Missing actual native route identity')
            calls = self.capture.rows[member]
            if len(calls) >= 2:
                raise ValueError('Exactly one original positive and negative call per route required')
            edges = args[2] if len(args) > 2 else kwargs.get('tar_ei')
            perm = n.permutation[n.cursor-n.batch_size:n.cursor]
            expected = (n.positive if not calls else n.negative)[:, perm]
            if edges is None or not self.t.equal(edges, expected):
                raise ValueError('Actual task query endpoints/order changed from source sampler')
            groups = [1 if not calls else 0]*n.batch_size
            self._depth += 1
            try:
                with self.capture.task(member, edges.t().detach().cpu().tolist(), groups):
                    result = self._original_predictor(*args, **kwargs)
            finally:
                self._depth -= 1
            if result.numel() != n.batch_size:
                raise ValueError('Actual native task output row count changed')
            return result
        n.predictor.multidomainforward = types.MethodType(predictor, n.predictor)
        def step(instance, native_losses, native_pool):
            return self._step(native_losses, native_pool)
        n._step = types.MethodType(step, n)
        self.assert_identity()

    def __getattr__(self, name):
        return getattr(self.base, name)

    def assert_identity(self):
        self.base.assert_identity()
        current = dict(self.native.model.named_parameters())
        if any(current.get(name) is not value for name, value in self.references.items()):
            raise ValueError('Original native Parameter identity changed')

    def _make_targets(self, keys, groups, dtype):
        selected, ids, cohort = self.context.select(self.planning, keys, groups,
            seed=int(self.native.seed), nodes=self.target_graph.nodes)
        chosen_keys = [keys[i] for i in selected]
        a, b, target = self.target_graph.contexts(self.t, chosen_keys, dtype)
        labels = self.t.tensor([groups[i] for i in selected], dtype=self.t.long)
        identity = self.t.tensor(ids, dtype=self.t.long)
        relations = self.t.tensor([key[2] for key in chosen_keys], dtype=self.t.long)
        support = self.context.target_diagnostics(self.t, a, b, labels, identity, relations)
        return {'cohort': cohort, 'target': target, 'support': support,
            'A': a, 'B': b, 'labels': labels, 'object_ids': identity, 'relations': relations}

    def _step(self, native_losses, native_pool):
        t, n = self.t, self.native
        if not self._active or any(len(self.capture.rows[m]) != 2 for m in range(4)):
            raise ValueError('Four complete actual source task captures required')
        captured = self.capture.aligned(t)
        hidden = captured['preheads']
        if tuple(hidden.shape) != (4, 131072, 64) or not hidden.requires_grad:
            raise ValueError('Live original complete NCN preheads required')
        keys = self.planning.query_keys(captured['endpoints'], captured['relation_ids'])
        started = time.perf_counter()
        last = self._make_targets(keys, captured['groups'], hidden.dtype)
        self._last = last
        selection = last['cohort']['selected_rows']
        if selection:
            device = hidden.device
            extra = self.objective.context_auxiliary(t,
                preheads=hidden[:, t.tensor(selection, dtype=t.long, device=device)],
                labels=last['labels'].to(device), object_ids=last['object_ids'].to(device),
                relation_ids=last['relations'].to(device), context_a=last['A'].to(device), context_b=last['B'].to(device),
                mode=self.mode, reduction='binary_separate')
        else:
            zero = hidden.sum()*0
            extra = {'loss': zero, 'A': zero, 'diagnostics': {'input_objects': 0, 'eligible_anchors': 0,
                'omitted_anchors': 0, 'all_source_rows_preserved_in_F': True}}
        context_seconds = time.perf_counter()-started
        F = t.stack(native_losses).mean(); loss = F+extra['loss']
        if not t.isfinite(loss):
            raise FloatingPointError('Nonfinite original NCN risk plus context auxiliary')
        loss.backward()
        for parameter in n.model.parameters():
            if parameter.grad is not None and not t.isfinite(parameter.grad).all().item():
                raise FloatingPointError('Nonfinite source/private gradient before native Adam')
        n.optimizer.step(); n.updates += 1
        self._work = {'reverse_collections': 1, 'optimizer_transitions': 1, 'all_gradients_from_old_state': True,
            'extra_source_forward_views': 0, 'captured_native_task_query_records': 131072}
        diagnostics = {'auxiliary': extra['diagnostics'], 'cohort': copy.deepcopy(last['cohort']),
            'factual_context': copy.deepcopy(last['target']), 'support': copy.deepcopy(last['support']),
            'four_route_query_signatures': captured['route_query_signatures'],
            'context_preparation_host_seconds': self.preparation_seconds, 'context_per_update_host_seconds': context_seconds}
        return {'loss': float(loss.detach()), 'native_own_loss': float(F.detach()), 'own_loss': float(F.detach()),
            'auxiliary_loss': float(extra['loss'].detach()), 'context_A_loss': float(extra['A'].detach()),
            'pool_loss': float(native_pool.detach()), 'optimizer_updates': 1, 'lossfinite': True,
            'context_affinity_diagnostics': diagnostics}

    def train_update(self):
        if self._active:
            raise ValueError('No nested native source update')
        self.capture.clear(); self._active = True
        try:
            result = self.base.train_update()
            result['work'].update(self._work); self.assert_identity()
            return result
        finally:
            self._active = False; self.capture.clear()

    def train_epoch(self):
        before = self.base.epoch_progress(); rows = []
        for _ in range(before['maps_to_next_completed_epoch']):
            rows.append(self.train_update())
        after = self.base.epoch_progress()
        if not after['at_epoch_boundary'] or after['completed_epochs'] != before['completed_epochs']+1:
            raise ValueError('Native source epoch boundary changed')
        return {'epoch': after['completed_epochs'], 'complete_native_TRAIN': True, 'optimizer_maps': len(rows),
            'native_epoch_optimizer_maps': self.base.layout['optimizer_maps_per_epoch'], 'epoch_progress': after,
            'mean_update_loss': sum(row['loss'] for row in rows)/len(rows),
            'native_source_epoch_loss': self.base.epoch_accumulator['native_loss_sum']/len(rows)
                if before['resumed_prefix_maps'] == 0 else None,
            'native_source_epoch_loss_complete': before['resumed_prefix_maps'] == 0,
            'native_work_totals': {key: sum(row['work'].get(key, 0) for row in rows)
                for key in ('optimizer_transitions', 'encoder_forwards', 'reverse_collections')},
            'last_extension_diagnostics': rows[-1]['context_affinity_diagnostics']}

    def validate(self):
        result = self.base.validate(); self.assert_identity(); return result

    def metadata(self):
        return {**self.base.metadata(), 'context_affinity_NCN': copy.deepcopy(self.method_metadata)}

    def state(self):
        if self._active or self._depth:
            raise ValueError('Checkpoint only outside native task calls')
        return {'schema': STATE_SCHEMA, 'method': copy.deepcopy(self.method_metadata),
            'last_context': copy.deepcopy(self._last), 'native_integration': self.base.state()}

    def load_state(self, value):
        if self._active or value.get('schema') != STATE_SCHEMA or value.get('method') != self.method_metadata:
            raise ValueError('Checkpoint method/source/target graph differs')
        saved = value['last_context']; base_state = value['native_integration']
        if saved is not None:
            sampler = base_state['native']['sampler']; cursor = sampler['cursor']
            perm_cpu = sampler['permutation'][cursor-self.native.batch_size:cursor].detach().cpu()
            positive = self.native.positive
            negative = sampler['negative']
            # Checkpoint sampler tensors are CPU; each gather must use an
            # explicitly colocated int64 index before authentication on CPU.
            positive_rows = positive.index_select(1, perm_cpu.to(positive.device)).detach().cpu()
            negative_rows = negative.index_select(1, perm_cpu.to(negative.device)).detach().cpu()
            endpoints = self.t.cat((positive_rows, negative_rows), dim=1)
            keys = self.planning.query_keys(endpoints.t().detach().cpu().tolist())
            reconstructed = self._make_targets(keys, [1]*65536+[0]*65536, saved['A'].dtype)
            for key in ('cohort', 'target', 'support'):
                if reconstructed[key] != saved[key]:
                    raise ValueError('Saved actual context identity/statistics changed: '+key)
            for key in ('A', 'B', 'labels', 'object_ids', 'relations'):
                if not self.t.equal(reconstructed[key], saved[key].cpu()):
                    raise ValueError('Saved actual factual context tensor differs: '+key)
        self.base.load_state(base_state); self._last = copy.deepcopy(saved); self.assert_identity()

    def close(self):
        if self._active:
            raise ValueError('Cannot close during source update')
        self.native.predictor.multidomainforward = self._original_predictor
        self.native._step = self._original_step
        self.capture.close(); self.base.close()
