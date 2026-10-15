"""Native BUDDY joint BCE plus fixed common/split factual TRAIN contexts."""
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
CORE = EXPANSION / 'context_affinity_v1'
CORE_SHA256 = 'c510c0dbf02236c6345df99d2906534866cab9a65c8562557994e907303baa3b'
CORE_MANIFEST_SHA256 = 'fe6b26f875ece1173be06d8656b2d6442ac216e7afcef0e6d40fb7a5b36cf448'
TARGET_SPEC = EXPANSION / 'expanded_study_design_v1/TASK_CONTEXT_SPEC.md'
TARGET_SPEC_SHA256 = '671173b3bcf5e2ba4e56c5ee9c1bfebd45f7494c596510bb3cabf182a73577ec'
METHODS = {'context_affinity_BUDDY_common_v1': 'common', 'context_affinity_BUDDY_split_v1': 'split'}
QUERY_SCOPE = 'ogbl-collab/official-time/source-node-ID/undirected/relation-0/v1'
STATE_SCHEMA = 'native-BUDDY-factual-context-affinity-state-v1'


def receipt(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def load(path, package=False):
    path = Path(path).resolve()
    name = '_context_BUDDY_' + hashlib.sha256(str(path).encode() + path.read_bytes()).hexdigest()
    if name not in sys.modules:
        options = {'submodule_search_locations': [str(path.parent)]} if package else {}
        spec = importlib.util.spec_from_file_location(name, path, **options)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def verify_package(directory, expected):
    directory = Path(directory).resolve()
    pin = receipt(directory / 'MANIFEST.json')
    if pin['sha256'] != expected:
        raise ValueError('Frozen package manifest changed: ' + str(directory))
    pins = [pin]
    for row in json.loads((directory / 'MANIFEST.json').read_text())['files']:
        path = (directory / row['path']).resolve()
        if not path.is_relative_to(directory):
            raise ValueError('Manifest member escapes package')
        actual = receipt(path)
        if any(actual[key] != row[key] for key in ('bytes', 'sha256')):
            raise ValueError('Frozen package file changed: ' + str(path))
        pins.append(actual)
    return pins


def bindings(root, expected):
    root = Path(root).resolve()
    if not HERE.is_relative_to(root):
        raise ValueError('Actual repository root must contain this port')
    pins = verify_package(HERE, expected) + verify_package(CORE, CORE_MANIFEST_SHA256)
    if receipt(CORE / 'core.py')['sha256'] != CORE_SHA256 or receipt(TARGET_SPEC)['sha256'] != TARGET_SPEC_SHA256:
        raise ValueError('Frozen tensor objective or factual target specification changed')
    pins.append(receipt(TARGET_SPEC))
    for row in json.loads((HERE / 'SOURCES.json').read_text())['source_files']:
        path = (PHASE / row['path_relative_to_phase']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Native/context dependency escapes repository')
        actual = receipt(path)
        if any(actual[key] != row[key] for key in ('bytes', 'sha256')):
            raise ValueError('Frozen native/context dependency changed: ' + str(path))
        pins.append(actual)
    return pins


class BUDDYContextSession:
    def __init__(self, base, job, pins):
        self.base, self.native, self.job = base, base.native, copy.deepcopy(job)
        n = self.native
        self.t, self.method = n.t, job['procedure']
        self.mode = METHODS[self.method]
        if base.family != 'BUDDY' or n.procedure != 'base' or n.members != 4 or n.updates != 0 or base.candidate_id is not None:
            raise ValueError('Fresh four-route native own-loss BUDDY only')
        if n.args.year != 2007 or n.args.sign_k != 0 or n.args.batch_size != 1024 or n.positive.ndim != 2 or n.positive.shape[1] != 2:
            raise ValueError('Keep the literal native year2007 / SIGN-k0 / mixed1024 query interface')
        if n.model.lin.weight.shape != (1, 1040):
            raise ValueError('Original native 1040-wide BUDDY task prehead required')
        self.references = dict(n.source_parameter_references)
        self.context = load(EXPANSION / 'context_affinity_link_port_v2/context.py')
        self.objective = load(CORE / 'core.py')
        self.library = load(EXPANSION / 'link/__init__.py', package=True)
        self.planning = sys.modules[self.library.__name__ + '.planning']
        self.capture = self.library.TaskPreheadCapture(n.model.lin, lambda: True)
        started = time.perf_counter()
        self.target_graph = self.context.TrainGraph(self.t, n.positive.t().contiguous(), int(n.features.shape[0]))
        self.preparation_seconds = time.perf_counter() - started
        self._active = False
        self._depth = 0
        self._last = self._work = None
        self._batch_ids = self._links = self._groups = self._source_inputs = None
        self._logits = None
        self._VALID_context_checks = 0
        self._original_next_batch, self._original_hydrate = n._next_batch, n._hydrate
        self._original_forward, self._original_step = n.model.forward, n._step
        self.materialized_context = None
        context_pin = None
        if 'context_manifest' in job['method_settings']:
            auditor = load(EXPANSION / 'link_expansion_ports_BUDDY_v2/session.py')
            self.materialized_context, context_pin = auditor.native_context(n, base, job)
        self.method_metadata = {'schema': 'context-affinity-native-BUDDY-v1', 'method_id': self.method, 'mode': self.mode,
            'source_receipts': pins, 'fit_seed': int(n.seed), 'query_scope': QUERY_SCOPE,
            'core_sha256': CORE_SHA256, 'core_manifest_sha256': CORE_MANIFEST_SHA256,
            'target_spec_sha256': TARGET_SPEC_SHA256,
            'support_addendum': receipt(EXPANSION / 'context_affinity_link_port_v2/LINK_SUPPORT_ADDENDUM.md'),
            'target_graph': copy.deepcopy(self.target_graph.receipt),
            'target_population': 'all original native TRAIN positives after source year2007 filtering; simple/undirected/loopless only for auxiliary',
            'A': 'sorted log1p endpoint degrees after own-pair exclusion',
            'B': ['log1p(CN)', 'log1p(AA_sum)', 'log1p(RA_sum)'],
            'target_graph_is_predictor_graph': False,
            'native_predictor_inputs': 'unchanged source year2007 weighted graph/hash features and hydration',
            'native_F': 'four-route mean of original joint BCE over every original mixed source minibatch row',
            'auxiliary_reduction': 'joint supported-anchor mean, then four-route mean; no source-group balancing',
            'assignment': 'allroutes.5qA+.5qB' if self.mode == 'common' else 'routes0/1qA;routes2/3qB',
            'update': 'F+.05A, one ordinary all-block backward and one original Adam transition',
            'temperature': .2, 'coefficient': .05, 'normalization_epsilon': 1e-12,
            'query_selection': 'unchanged frozen link hash, at most512 distinct nonself nonconflicting queries per source group',
            'source_parameters_replaced': 0, 'new_parameters': 0, 'new_optimizer_groups': 0,
            'optimizer_options_changed': False, 'extra_source_forwards': 0,
            'native_BN': 'unchanged shared source bn_labels/bn_feats/bn_RA; no private BN',
            'serving': 'unchanged arithmetic mean native raw logits; factual contexts used only for TRAIN',
            'materialized_native_input_audit': context_pin,
            'materialized_audit_scope': None if context_pin is None else 'seed0/year2007: first two native1024 TRAIN batches and both complete VALID roles',
            'TEST_accessed': False, 'scientific_launch_enabled': False, 'native_CPU_qualification': 'pending',
            'other_link_families': 'separate immutable NCN/NCNC ports', 'final_TEST_reconstruction': 'pending'}

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
                if self._links is None or not self.t.equal(links.detach().cpu(), self._links):
                    raise ValueError('Native BUDDY hydration rows differ from actual sampler')
                self._source_inputs = values
                self._audit_inputs(links, values, training=True)
            elif self.materialized_context is not None:
                self._audit_inputs(links, values, training=False)
            return values

        def forward(instance, *args, **kwargs):
            if not self._active:
                return self._original_forward(*args, **kwargs)
            member = int(n.core.route)
            if self._depth or member not in range(4) or self._logits[member] is not None:
                raise ValueError('Exactly one original native BUDDY call per route required')
            if self._source_inputs is None or len(args) < 4 or any(actual is not expected for actual, expected in zip(args[:4], self._source_inputs)):
                raise ValueError('Native route input objects differ from actual source hydration')
            self._depth += 1
            try:
                with self.capture.task(member, self._links.tolist(), self._groups.tolist()):
                    value = self._original_forward(*args, **kwargs)
            finally:
                self._depth -= 1
            if value.numel() != len(self._batch_ids):
                raise ValueError('Original native task query population changed')
            self._logits[member] = value.reshape(-1)
            return value

        def step(instance, native_losses, native_pool):
            return self._step(native_losses, native_pool)

        n._next_batch, n._hydrate = types.MethodType(next_batch, n), types.MethodType(hydrate, n)
        n.model.forward, n._step = types.MethodType(forward, n.model), types.MethodType(step, n)
        self.assert_identity()

    def __getattr__(self, name):
        return getattr(self.base, name)

    def _audit_inputs(self, links, values, *, training):
        if self.materialized_context is None:
            return
        if training:
            if self.native.updates >= 2:
                return  # Audit receipt covers the first two original batches only; native sampler continues unchanged.
            expected = self.materialized_context['TRAIN_batches'][self.native.updates]
            if not self.t.equal(self._batch_ids, expected['sampler_ids']) or not self.t.equal(self._links, expected['links']) \
                    or not self.t.equal(self._groups, expected['labels']):
                raise ValueError('Actual source sampler differs from authenticated year2007 materialization')
        else:
            expected = next((row for row in self.materialized_context['VALID_inputs'].values()
                if links.shape == row['links'].shape and self.t.equal(links.detach().cpu(), row['links'])), None)
            if expected is None:
                raise ValueError('Unexpected source VALID role/query population')
            self._VALID_context_checks += 1
        if len(values) != len(expected['inputs']) or any(not self.t.equal(actual.detach().cpu(), saved) for actual, saved in zip(values, expected['inputs'])):
            raise ValueError('Actual source hydration differs from authenticated year2007 inputs')

    def assert_identity(self):
        self.base.assert_identity()
        current = dict(self.native.model.named_parameters())
        if any(current.get(name) is not value for name, value in self.references.items()):
            raise ValueError('Original native BUDDY Parameter identity changed')

    def _make_targets(self, keys, groups, dtype, sampler_ids):
        selected, ids, cohort = self.context.select(self.planning, keys, groups,
            seed=int(self.native.seed), nodes=self.target_graph.nodes)
        chosen = [keys[i] for i in selected]
        a, b, target = self.target_graph.contexts(self.t, chosen, dtype)
        labels = self.t.tensor([groups[i] for i in selected], dtype=self.t.long)
        identities = self.t.tensor(ids, dtype=self.t.long)
        relations = self.t.tensor([key[2] for key in chosen], dtype=self.t.long)
        support = self.context.target_diagnostics(self.t, a, b, labels, identities, relations)
        return {'cohort': cohort, 'target': target, 'support': support, 'A': a, 'B': b,
            'labels': labels, 'object_ids': identities, 'relations': relations,
            'source_sampler_ids': sampler_ids.detach().cpu().clone()}

    def _step(self, native_losses, native_pool):
        t, n = self.t, self.native
        if not self._active or any(row is None for row in self._logits) or any(len(self.capture.rows[m]) != 1 for m in range(4)):
            raise ValueError('Four complete original BUDDY mixed-query captures required')
        captured = self.capture.aligned(t)
        hidden = captured['preheads']
        if tuple(hidden.shape) != (4, len(self._batch_ids), 1040) or not hidden.requires_grad:
            raise ValueError('Live complete original native BUDDY preheads required')
        expected = t.stack([n.native_loss(row, self._groups.to(row.device)) for row in self._logits])
        if not t.allclose(expected.detach(), t.stack(native_losses).detach(), atol=1e-7, rtol=1e-6):
            raise ValueError('Actual native joint BCE was not recovered from complete source logits')
        keys = self.planning.query_keys(captured['endpoints'], captured['relation_ids'])
        started = time.perf_counter()
        last = self._make_targets(keys, captured['groups'], hidden.dtype, self._batch_ids)
        self._last = last
        selected = last['cohort']['selected_rows']
        if selected:
            device = hidden.device
            extra = self.objective.context_auxiliary(t,
                preheads=hidden[:, t.tensor(selected, dtype=t.long, device=device)],
                labels=last['labels'].to(device), object_ids=last['object_ids'].to(device),
                relation_ids=last['relations'].to(device), context_a=last['A'].to(device), context_b=last['B'].to(device),
                mode=self.mode, reduction='joint')
        else:
            zero = hidden.sum() * 0
            extra = {'loss': zero, 'A': zero, 'diagnostics': {'input_objects': 0, 'eligible_anchors': 0,
                'omitted_anchors': 0, 'native_task_population_changed': False}}
        context_seconds = time.perf_counter() - started
        F = t.stack(native_losses).mean()
        loss = F + extra['loss']
        if not t.isfinite(loss):
            raise FloatingPointError('Nonfinite native BUDDY BCE plus factual context auxiliary')
        loss.backward()
        for parameter in n.model.parameters():
            if parameter.grad is not None and not t.isfinite(parameter.grad).all().item():
                raise FloatingPointError('Nonfinite native/private gradient before source Adam')
        n.optimizer.step()
        n.updates += 1
        self._work = {'reverse_collections': 1, 'optimizer_transitions': 1, 'all_gradients_from_old_state': True,
            'extra_source_forward_views': 0, 'captured_native_task_query_records': len(self._batch_ids)}
        diagnostics = {'auxiliary': extra['diagnostics'], 'cohort': copy.deepcopy(last['cohort']),
            'factual_context': copy.deepcopy(last['target']), 'support': copy.deepcopy(last['support']),
            'four_route_query_signatures': captured['route_query_signatures'], 'native_joint_BCE_parity_checked': True,
            'native_source_group_counts': {'1': int((self._groups == 1).sum()), '0': int((self._groups == 0).sum())},
            'context_preparation_host_seconds': self.preparation_seconds, 'context_per_update_host_seconds': context_seconds}
        return {'loss': float(loss.detach()), 'native_own_loss': float(F.detach()), 'own_loss': float(F.detach()),
            'auxiliary_loss': float(extra['loss'].detach()), 'context_A_loss': float(extra['A'].detach()),
            'pool_loss': float(native_pool.detach()), 'optimizer_updates': 1, 'lossfinite': True,
            'context_affinity_diagnostics': diagnostics}

    def train_update(self):
        if self._active:
            raise ValueError('No nested native update')
        self.capture.clear()
        self._logits = [None] * 4
        self._batch_ids = self._links = self._groups = self._source_inputs = None
        self._active = True
        try:
            result = self.base.train_update()
            result['work'].update(self._work)
            self.assert_identity()
            return result
        finally:
            self._active = False
            self._logits = self._source_inputs = None
            self.capture.clear()

    def train_epoch(self):
        before = self.base.epoch_progress()
        rows = [self.train_update() for _ in range(before['maps_to_next_completed_epoch'])]
        after = self.base.epoch_progress()
        if not after['at_epoch_boundary'] or after['completed_epochs'] != before['completed_epochs'] + 1:
            raise ValueError('Native BUDDY epoch boundary changed')
        return {'epoch': after['completed_epochs'], 'complete_native_TRAIN': True, 'optimizer_maps': len(rows),
            'native_epoch_optimizer_maps': self.base.layout['optimizer_maps_per_epoch'], 'epoch_progress': after,
            'mean_update_loss': sum(row['loss'] for row in rows) / len(rows),
            'native_source_epoch_loss': self.base.epoch_accumulator['native_loss_sum'] * 1024 / self.base.layout['task_query_records_per_epoch']
                if before['resumed_prefix_maps'] == 0 else None,
            'native_source_epoch_loss_complete': before['resumed_prefix_maps'] == 0,
            'native_work_totals': {key: sum(row['work'].get(key, 0) for row in rows)
                for key in ('optimizer_transitions', 'native_BUDDY_forwards', 'query_records', 'reverse_collections')},
            'last_extension_diagnostics': rows[-1]['context_affinity_diagnostics']}

    def validate(self):
        self._VALID_context_checks = 0
        value = self.base.validate()
        if self.materialized_context is not None and self._VALID_context_checks != 2:
            raise ValueError('Both full official year2007 BUDDY VALID input roles must authenticate')
        self.assert_identity()
        value['materialized_VALID_context_verified'] = self.materialized_context is not None
        return value

    def metadata(self):
        return {**self.base.metadata(), 'context_affinity_BUDDY': copy.deepcopy(self.method_metadata)}

    def state(self):
        if self._active or self._depth:
            raise ValueError('Checkpoint only outside native task calls')
        return {'schema': STATE_SCHEMA, 'method': copy.deepcopy(self.method_metadata),
            'last_context': copy.deepcopy(self._last), 'native_integration': self.base.state()}

    def load_state(self, value):
        if self._active or value.get('schema') != STATE_SCHEMA or value.get('method') != self.method_metadata:
            raise ValueError('Checkpoint source/method/target graph differs')
        saved, native_state = value['last_context'], value['native_integration']['native']
        if saved is not None:
            sampler = native_state['sampler']
            indices, cursor = sampler['sample_indices'], int(sampler['batch_cursor'])
            if indices is None or cursor < 1 or sampler['loader_rng_before'] is None:
                raise ValueError('Saved factual targets require an actual completed native source minibatch')
            links = self.t.cat((self.native.positive.cpu(), sampler['train_negative'].cpu()), dim=0)
            with self.t.random.fork_rng(devices=[], enabled=True):
                self.t.set_rng_state(sampler['loader_rng_before'].cpu())
                iterator = iter(self.native.DataLoader(range(len(links)), 1024, shuffle=True))
                for _ in range(cursor):
                    rows = next(iterator)
                ids = indices.cpu()[rows]
            if not self.t.equal(ids, saved['source_sampler_ids'].cpu()):
                raise ValueError('Saved factual query IDs differ from the native saved sampler law')
            labels = (ids < len(self.native.positive)).long().tolist()
            keys = self.planning.query_keys(links[ids].tolist())
            rebuilt = self._make_targets(keys, labels, saved['A'].dtype, ids)
            for key in ('cohort', 'target', 'support'):
                if rebuilt[key] != saved[key]:
                    raise ValueError('Saved factual query identity/statistics changed: ' + key)
            for key in ('A', 'B', 'labels', 'object_ids', 'relations', 'source_sampler_ids'):
                if not self.t.equal(rebuilt[key], saved[key].cpu()):
                    raise ValueError('Saved factual target tensor changed: ' + key)
        self.base.load_state(value['native_integration'])
        self._last = copy.deepcopy(saved)
        self.assert_identity()

    def close(self):
        if self._active:
            raise ValueError('Cannot close during source update')
        n = self.native
        n._next_batch, n._hydrate = self._original_next_batch, self._original_hydrate
        n.model.forward, n._step = self._original_forward, self._original_step
        self.capture.close()
        self.base.close()
