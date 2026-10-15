"""Actual native-session integration of the frozen link query objectives.

The saved source models, frozen smoke files and finalized link helpers are
loaded and verified; no source file, class or Parameter is replaced. Only
task/update orchestration is extended in this new session proxy. No TEST path,
server operation, automatic fit loop, install or network access lives here.
"""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
EXPANSION = HERE.parent
PHASE = EXPANSION.parent


def _receipt(path):
    path = Path(path).resolve()
    digest, size = hashlib.sha256(), 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block); size += len(block)
    return {'path': str(path), 'bytes': size, 'sha256': digest.hexdigest()}


def _inside(path, root, label='path'):
    root = Path(root).resolve()
    path = Path(path)
    resolved = (path if path.is_absolute() else root/path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ValueError(label+' escapes the provided repository root: '+str(resolved)) from None
    return resolved


def _source_context(recipe):
    config = copy.deepcopy(recipe)
    root = Path(config.get('_repository_root', PHASE.parent)).resolve()
    redesign = _inside(config.get('_redesign_root', PHASE/'reference_wrapper_redesign_20261013_v1'), root, '_redesign_root')
    expected = _inside(redesign/'smoke_20261014/link_port/callbacks.py', root, 'native callback')
    callback = _inside(config.get('_native_callback', expected), root, '_native_callback')
    if callback != expected:
        raise ValueError('The native callback must be the frozen link port under _redesign_root')
    config.update(_repository_root=str(root), _redesign_root=str(redesign), _native_callback=str(callback))
    return config


def _prepare_recipe(recipe, data_root):
    config = _source_context(recipe)
    root = Path(config['_repository_root'])
    data_root = _inside(data_root, root, 'data_root')
    paths = config.get('data_files')
    if not isinstance(paths, dict) or set(paths) != {'graph', 'train', 'valid'}:
        raise ValueError('Explicit data_files with only graph/TRAIN/VALID roles are required; TEST is excluded')
    # Relative recipe paths use the repository root, independent of process cwd.
    config['data_files'] = {role: str(_inside(path, root, 'data_files.'+role)) for role, path in paths.items()}
    return config, data_root


def _check_hash(actual, expected, label):
    if expected is None:
        return
    if not isinstance(expected, str) or len(expected) != 64 or any(c not in '0123456789abcdefABCDEF' for c in expected):
        raise ValueError(label+' must be an explicit SHA256')
    if actual['sha256'] != expected.lower():
        raise ValueError(label+' differs from the supplied callback SHA256')


def _native_hash_bindings(recipe, callback_receipt):
    declared = {key: recipe[key] for key in ('callback_sha256', '_native_callback_sha256', 'native_callback_sha256') if key in recipe}
    for key, value in declared.items():
        _check_hash(callback_receipt, value, 'recipe.'+key)
    return declared


def _job_callback_binding(job, root, expected_path):
    path = _inside(job.get('callback_module', expected_path), root, 'job.callback_module')
    if path != Path(expected_path).resolve():
        raise ValueError('The supplied job callback path differs from the active entrypoint')
    actual = _receipt(path)
    _check_hash(actual, job.get('callback_sha256'), 'job.callback_sha256')
    return {'actual': actual, 'declared_sha256': job.get('callback_sha256')}


def _load(path, name, package=False):
    options = {'submodule_search_locations': [str(Path(path).parent)]} if package else {}
    spec = importlib.util.spec_from_file_location(name, path, **options)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _dependencies():
    root = EXPANSION / 'link'
    manifest = json.loads((root/'MANIFEST.json').read_text())
    rows = {row['path']: row for row in manifest['files']}
    receipts = [_receipt(root/'MANIFEST.json')]
    for name, row in rows.items():
        if name.endswith('.py') or name == 'CANDIDATES.json':
            actual = _receipt(root/name)
            if actual['bytes'] != row['bytes'] or actual['sha256'] != row['sha256']:
                raise ValueError('Finalized link dependency changed: '+name)
            receipts.append(actual)
    library = _load(root/'__init__.py', '_expansion_link_objectives_v1', package=True)
    roster = json.loads((root/'CANDIDATES.json').read_text())
    return library, {row['id']: row for row in roster['all_candidates']}, receipts


def _capture_rng(torch):
    import numpy as np
    ns = np.random.get_state()
    return {'python': random.getstate(), 'numpy': (ns[0], ns[1].tolist(), ns[2], ns[3], ns[4]),
            'torch_cpu': torch.get_rng_state().clone(),
            'torch_cuda': [value.clone() for value in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []}


def _restore_rng(torch, state):
    import numpy as np
    random.setstate(state['python'])
    ns = state['numpy']
    np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4]))
    torch.set_rng_state(state['torch_cpu'].cpu())
    if state['torch_cuda']:
        torch.cuda.set_rng_state_all([value.cpu() for value in state['torch_cuda']])


def _seed(torch, value):
    import numpy as np
    random.seed(value); np.random.seed(value % (2**32)); torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


class LinkSession:
    def __init__(self, native, candidate=None, expected_kwargs=None, recipe=None):
        self.native = native
        self.t = native.t
        self.family = str(native.metadata['family']).upper()
        config = _source_context(recipe or {})
        root = Path(config['_repository_root'])
        sources = copy.deepcopy(native.metadata['adapter_source_receipts'])
        for role, saved in sources.items():
            path = _inside(saved['path'], root, 'native source '+role)
            actual = _receipt(path)
            if any(actual[key] != saved[key] for key in ('path', 'bytes', 'sha256')):
                raise ValueError('Native source metadata differs from its actual source bytes: '+role)
        if sources['callbacks.py']['path'] != config['_native_callback']:
            raise ValueError('Native callback metadata differs from the supplied source binding')
        manifest = copy.deepcopy(native.metadata['adapter_manifest'])
        actual_manifest = _receipt(_inside(manifest['path'], root, 'native adapter manifest'))
        if any(actual_manifest[key] != manifest[key] for key in ('path', 'bytes', 'sha256')):
            raise ValueError('Native adapter manifest metadata differs from its actual source bytes')
        declared = _native_hash_bindings(config, sources['callbacks.py'])
        if set(native.data_files) != {'graph', 'train', 'valid'}:
            raise ValueError('Only actual GRAPH/TRAIN/VALID data may be bound; TEST is excluded')
        files = {role: _inside(path, root, 'native data_files.'+role) for role, path in native.data_files.items()}
        if 'data_files' in config and {role: str(path) for role, path in files.items()} != config['data_files']:
            raise ValueError('Native data files differ from the supplied source binding')
        self.native_binding = {'family': self.family, 'repository_root': str(root),
            'declared_callback_sha256': declared,
            'job_callback_binding': copy.deepcopy(config.get('_job_callback_binding')),
            'native_fixed_config': copy.deepcopy(native.metadata.get('declared_native_fixed_config')),
            'adapter_source_receipts': sources,
            'adapter_manifest_receipt': manifest,
            'source_proofs': copy.deepcopy(native.metadata['source_proofs']),
            'data_files': {role: _receipt(path) for role, path in files.items()}}
        self.library, roster, self.helper_receipts = _dependencies()
        self.epoch_module = _load(HERE/'epoch.py', '_expansion_link_epoch_v1')
        self.candidate_id = candidate
        self.kwargs = None
        self.capture = None
        self._in_update = False
        self.last_update_work = None
        self.epoch_accumulator = {'started_epoch': 0, 'observed_maps': 0, 'native_loss_sum': 0.,
                                  'criterion_sum': 0.}
        if candidate is not None:
            if candidate not in roster or roster[candidate]['kind'] != 'query_objective':
                raise ValueError('Only finalized query-objective candidates are integrated')
            if native.members != 4 or native.core is None or native.procedure != 'base':
                raise ValueError('Query extensions require the actual four-route native boundary own-loss session')
            self.kwargs = copy.deepcopy(roster[candidate]['kwargs'])
            if expected_kwargs is not None and expected_kwargs != self.kwargs:
                raise ValueError('Method kwargs differ from the finalized candidate definition')
            leaf = native.model.lin if self.family == 'BUDDY' else native.predictor.lin[-1]
            if self.family == 'NCNC' and native.task_call_scope is None:
                raise ValueError('Corrected explicit NCNC task-only output scope is required')
            gate = native.task_call_scope.task_output_active if self.family == 'NCNC' else lambda: True
            self.capture = self.library.TaskPreheadCapture(leaf, gate)
        self.layout = self.epoch_module.epoch_layout(self.family, len(native.positive) if self.family == 'BUDDY'
            else native.positive.shape[1], len(native.links) if self.family == 'BUDDY' else None)
        self.integration_receipts = [_receipt(__file__), _receipt(HERE/'epoch.py')]
        self.assert_identity()

    def __getattr__(self, name):
        return getattr(self.native, name)

    @property
    def source_metadata(self):
        return self.metadata()

    @property
    def completed_epochs(self):
        return self.epoch_progress()['completed_epochs']

    def metadata(self):
        return {**copy.deepcopy(self.native.source_metadata), 'native_integration': {
            'schema': 'actual-native-link-extension-v1', 'candidate_id': self.candidate_id,
            'method_kwargs': copy.deepcopy(self.kwargs), 'integration_sources': self.integration_receipts,
            'finalized_helper_sources': self.helper_receipts, 'epoch_layout': self.layout,
            'native_binding': copy.deepcopy(self.native_binding),
            'source_model_classes_or_parameters_replaced': False,
            'source_graph_and_negative_sampler_reused': True,
            'NCNC_existing_task_scope_reused': self.family == 'NCNC',
            'native_runtime_qualification': 'pending allocation receipts', 'TEST_opened': False}}

    def assert_identity(self):
        current = dict(self.native.model.named_parameters())
        if any(current.get(name) is not value for name, value in self.native.source_parameter_references.items()):
            raise ValueError('An original native Parameter identity changed')
        owned = [p for group in self.native.optimizer.param_groups for p in group['params']]
        if len(owned) != len({id(p) for p in owned}) or {id(p) for p in owned} != {id(p) for p in current.values()}:
            raise ValueError('Native optimizer ownership is incomplete or aliased')

    def epoch_progress(self):
        count = self.native.batch_cursor if self.family == 'BUDDY' else self.native.cursor//self.native.batch_size
        value = self.epoch_module.progress(self.layout, self.native.epoch, count)
        expected_updates = max(0, self.native.epoch-1)*self.layout['optimizer_maps_per_epoch']+count
        if self.native.updates != expected_updates:
            raise ValueError('Sampler cursor and completed optimizer maps differ; a failed map cannot be counted as trained')
        return value

    def _finish(self, logits, losses, pool):
        aligned = self.capture.aligned(self.t)
        bundle = self.library.build_objectives(self.t, family=self.family,
            logits=self.t.stack(logits).reshape(4, -1), native_member_losses=losses,
            native_pool_loss=pool, seed=self.native.seed, **aligned, **self.kwargs)
        if not all(bool(self.t.isfinite(bundle[key])) for key in ('shared_objective', 'private_objective')):
            raise FloatingPointError('Nonfinite native link extension criterion')
        counts = self.library.step_blocks(self.t, self.native.optimizer,
            tuple(self.native.source_parameter_references.values()), self.native.core.private_parameters(), bundle)
        self.native.updates += 1
        result = {'loss': float(bundle['private_objective'].detach()),
            'loss_semantics': 'private block criterion; shared criterion is separately retained',
            'shared_criterion': float(bundle['shared_objective'].detach()),
            'native_own_loss': float(bundle['native_own_loss'].detach()),
            'native_member_losses': [float(value.detach()) for value in losses],
            'pool_loss': float(pool.detach()), 'query_auxiliary_loss': float(bundle['query_auxiliary_loss'].detach()),
            'lossfinite': True, 'optimizer_updates': 1, 'extension_diagnostics': bundle['diagnostics'],
            'work': {**counts, 'extra_backbone_forwards': 0, 'native_task_query_records': len(aligned['groups'])}}
        return result

    def _ncn_update(self):
        n, t = self.native, self.t
        before = n.task_call_scope.capture() if n.task_call_scope else None
        n.model.train(); n.optimizer.zero_grad()
        perm = n._next_batch()
        mask = t.ones(n.positive.shape[1], dtype=t.bool, device=n.device); mask[perm] = False
        adj = n.SparseTensor.from_edge_index(n.positive[:, mask], sparse_sizes=(235868,235868)).to_device(
            n.device, non_blocking=True).to_symmetric()
        positive, negative = n.positive[:, perm], n.negative[:, perm]
        pos_ids, neg_ids = positive.t().detach().cpu().tolist(), negative.t().detach().cpu().tolist()
        positives, negatives, logits, losses = [], [], [], []
        for member in range(4):
            with n._route(member, True):
                h = n.encoder(n.x, adj)
                with self.capture.task(member, pos_ids, [1]*len(pos_ids)):
                    pos = n.predictor.multidomainforward(h, adj, positive, cndropprobs=[])
                with self.capture.task(member, neg_ids, [0]*len(neg_ids)):
                    neg = n.predictor.multidomainforward(h, adj, negative, cndropprobs=[])
                positives.append(pos); negatives.append(neg)
                logits.append(t.cat((pos.reshape(-1), neg.reshape(-1))))
                losses.append(-t.nn.functional.logsigmoid(pos).mean()-t.nn.functional.logsigmoid(-neg).mean())
        pool = -t.nn.functional.logsigmoid(t.stack(positives).mean(0)).mean() - \
               t.nn.functional.logsigmoid(-t.stack(negatives).mean(0)).mean()
        result = self._finish(logits, losses, pool)
        result['work'].update(encoder_forwards=4, top_level_positive_predictor_calls=4,
            top_level_negative_predictor_calls=4, positive_query_records=len(perm), negative_query_records=len(perm),
            source_record_mask_before_coalesce=True, complete_source_sized_minibatch=True)
        if n.task_call_scope:
            result['work']['NCNC_output_routing'] = n.task_call_scope.routing_since(before)
        return result

    def _buddy_update(self):
        n, t = self.native, self.t
        n.model.train(); ids = n._next_batch(); links = n.links[ids]; labels = n.labels[ids].to(n.device)
        inputs = n._hydrate(links)
        endpoints, groups = links.tolist(), n.labels[ids].tolist()
        n.optimizer.zero_grad(); logits, losses = [], []
        for member in range(4):
            with n._route(member, True):
                with self.capture.task(member, endpoints, groups):
                    output = n.model(*inputs, None, None)
                logits.append(output); losses.append(n.native_loss(output, labels))
        pool = n.native_loss(t.stack(logits).mean(0), labels)
        result = self._finish(logits, losses, pool)
        result['work'].update(native_BUDDY_forwards=4, query_records=len(ids),
            native_weighted_graph_and_hashes=True, native_shared_BN=True,
            full_source_minibatch=len(ids) == 1024)
        return result

    def train_update(self):
        if self._in_update:
            raise ValueError('Nested native updates are not supported')
        self._in_update = True
        try:
            if self.capture is None:
                result = self.native.train_update()
            else:
                self.capture.clear()
                result = self._buddy_update() if self.family == 'BUDDY' else self._ncn_update()
            self.assert_identity()
            self.last_update_work = copy.deepcopy(result.get('work', {}))
            result['epoch_progress'] = self.epoch_progress()
            started = result['epoch_progress']['started_native_epoch']
            if self.epoch_accumulator['started_epoch'] != started:
                self.epoch_accumulator = {'started_epoch': started, 'observed_maps': 0,
                                          'native_loss_sum': 0., 'criterion_sum': 0.}
            self.epoch_accumulator['observed_maps'] += 1
            self.epoch_accumulator['native_loss_sum'] += float(result.get('native_own_loss', result.get('own_loss', result['loss'])))
            self.epoch_accumulator['criterion_sum'] += float(result['loss'])
            return result
        finally:
            if self.capture is not None:
                self.capture.clear()
            self._in_update = False

    def train_epoch(self):
        before = self.epoch_progress()
        work, criterion_sum, last = {}, 0., None
        for _ in range(before['maps_to_next_completed_epoch']):
            last = self.train_update()
            criterion_sum += float(last['loss'])
            for key, value in last.get('work', {}).items():
                if type(value) in (int, float) and math.isfinite(value):
                    work[key] = work.get(key, 0)+value
        after = self.epoch_progress()
        if not after['at_epoch_boundary'] or after['completed_epochs'] != before['completed_epochs']+1:
            raise ValueError('Native link epoch failed to terminate at its exact source boundary')
        complete_loss = self.epoch_accumulator['observed_maps'] == self.layout['optimizer_maps_per_epoch']
        source_loss = (self.epoch_accumulator['native_loss_sum']*1024/self.layout['task_query_records_per_epoch']
                       if self.family == 'BUDDY' else
                       self.epoch_accumulator['native_loss_sum']/self.layout['optimizer_maps_per_epoch']) if complete_loss else None
        return {'epoch': after['completed_epochs'], 'complete_native_TRAIN': True,
            'optimizer_maps': before['maps_to_next_completed_epoch'], 'native_epoch_optimizer_maps': self.layout['optimizer_maps_per_epoch'],
            'resumed_prefix_optimizer_maps': before['resumed_prefix_maps'],
            'source_iterator_records_in_complete_epoch': self.layout['source_iterator_records_per_epoch'],
            'task_query_records_in_complete_epoch': self.layout['task_query_records_per_epoch'],
            'dropped_positive_tail': self.layout['dropped_positive_tail'],
            'included_partial_tail': self.layout['included_partial_tail'],
            'mean_update_loss': criterion_sum/before['maps_to_next_completed_epoch'],
            'native_source_epoch_loss': source_loss,
            'native_source_epoch_loss_complete': complete_loss,
            'native_source_epoch_loss_semantics': 'source sum(batch mean loss * 1024)/full query count, including partial tail' if self.family == 'BUDDY' else 'source mean of complete minibatch native task losses',
            'native_work_totals': work, 'epoch_progress': after,
            'last_extension_diagnostics': last.get('extension_diagnostics')}

    def validate(self):
        n, t = self.native, self.t
        n.model.eval()
        routing = n.task_call_scope.capture() if n.task_call_scope else None
        if n.members == 1:
            result = n.validate()
            member_outputs = result['outputs'].reshape(1, -1)
        else:
            by_member = [[] for _ in range(4)]
            with t.no_grad():
                if self.family == 'BUDDY':
                    for key in ('edge', 'edge_neg'):
                        pairs = n.valid[key]
                        for start in range(0, len(pairs), 1000000):
                            inputs = n._hydrate(pairs[start:start+1000000])
                            for member in range(4):
                                with n._route(member, False):
                                    by_member[member].append(n.model(*inputs, None, None).reshape(-1).cpu())
                else:
                    for member in range(4):
                        with n._route(member, False):
                            h = n.encoder(n.x, n.adj)
                            for key in ('edge', 'edge_neg'):
                                pairs = n.valid[key].to(n.device)
                                for perm in n.utils.PermIterator(n.device, len(pairs), n.test_batch_size, False):
                                    by_member[member].append(n.predictor(h, n.adj, pairs[perm].t()).reshape(-1).cpu())
            member_outputs = t.stack([t.cat(parts) for parts in by_member])
            pool = member_outputs.mean(0)
            work = {'all_official_VALID_records': True,
                    'native_BUDDY_forwards': 8 if self.family == 'BUDDY' else 0,
                    'encoder_forwards': 4 if self.family != 'BUDDY' else 0}
            if n.task_call_scope:
                work['NCNC_output_routing'] = n.task_call_scope.routing_since(routing)
            result = n._validate_result(pool[:60084], pool[60084:], work)
        groups = [1]*60084+[0]*100000
        result.update(member_outputs=member_outputs, labels=t.tensor(groups, dtype=t.long),
            ids=t.arange(160084), query_endpoints=t.cat((n.valid['edge'], n.valid['edge_neg'])).cpu(),
            complete_source_VALID=True, TEST_accessed=False)
        if n.members == 4:
            from ogb.linkproppred import Evaluator
            result['bank_diagnostics'] = self.library.link_bank_diagnostics(t, member_outputs, groups, Evaluator(name='ogbl-collab'))
        return result

    def state(self):
        if self._in_update:
            raise ValueError('A live forward/backward graph cannot be checkpointed')
        n = self.native
        return {'schema': 'actual-native-link-extension-state-v1', 'candidate_id': self.candidate_id,
            'method_kwargs': copy.deepcopy(self.kwargs), 'native': n.state(),
            'epoch_accumulator': copy.deepcopy(self.epoch_accumulator),
            'epoch_progress': self.epoch_progress(), 'global_rng': _capture_rng(self.t),
            'module_modes': {name: module.training for name, module in n.model.named_modules()},
            'requires_grad': {name: p.requires_grad for name, p in n.model.named_parameters()},
            'integration_sources': self.integration_receipts, 'helper_sources': self.helper_receipts,
            'native_binding': copy.deepcopy(self.native_binding)}

    def load_state(self, state):
        if self._in_update or state['candidate_id'] != self.candidate_id or state['method_kwargs'] != self.kwargs:
            raise ValueError('Checkpoint method differs or update is active')
        if state['integration_sources'] != self.integration_receipts or state['helper_sources'] != self.helper_receipts or state.get('native_binding') != self.native_binding:
            raise ValueError('Checkpoint source bytes/path changed; explicit migration is required')
        self.native.load_state(state['native'])
        self.epoch_accumulator = copy.deepcopy(state['epoch_accumulator'])
        modules = dict(self.native.model.named_modules()); parameters = dict(self.native.model.named_parameters())
        if set(modules) != set(state['module_modes']) or set(parameters) != set(state['requires_grad']):
            raise ValueError('Native checkpoint structure changed')
        for name, mode in state['module_modes'].items():
            modules[name].training = bool(mode)
        for name, flag in state['requires_grad'].items():
            parameters[name].requires_grad_(flag)
        if self.epoch_progress() != state['epoch_progress']:
            raise ValueError('Native sampler/epoch progress did not restore')
        _restore_rng(self.t, state['global_rng'])
        self.assert_identity()

    def comparison_state(self):
        n = self.native
        return {'sampler': n.sampler_state(), 'epoch_progress': self.epoch_progress(),
            'epoch_accumulator': copy.deepcopy(self.epoch_accumulator),
            'private_rng': n.private_rng.streams if n.private_rng else None,
            'NCNC_task_scope': n.task_call_scope.capture() if n.task_call_scope else None}

    def close(self):
        if self.capture is not None:
            self.capture.close()
        closer = getattr(self.native, 'close', None)
        if closer is not None:
            closer()


def transform(session, job):
    if job.get('TEST_access') is not False or job.get('family') != 'link':
        raise ValueError('Explicit family=link and TEST_access=false required')
    recipe, _ = _prepare_recipe(job.get('recipe', {}), job['data_root'])
    recipe['_job_callback_binding'] = _job_callback_binding(job, recipe['_repository_root'], recipe['_native_callback'])
    for key in ('_native_callback_sha256', 'native_callback_sha256'):
        if key in job:
            if key in recipe and recipe[key] != job[key]:
                raise ValueError('Conflicting job and recipe native callback SHA256')
            recipe[key] = job[key]
    settings = job.get('method_settings', {})
    candidate = settings.get('candidate_id', job['procedure'])
    return LinkSession(session, candidate, settings.get('kwargs'), recipe)


extend_session = transform


def _native_factory(recipe):
    config = _source_context(recipe)
    root, path = Path(config['_repository_root']), Path(config['_native_callback'])
    manifest = json.loads(_inside(path.parent/'MANIFEST.json', root, 'native adapter manifest').read_text())
    rows = {row['path']: row for row in manifest['files']}
    for name in ('callbacks.py', 'boundaries.py', 'source_loader.py'):
        actual = _receipt(_inside(path.parent/name, root, 'native source '+name))
        row = rows[name]
        if actual['bytes'] != row['bytes'] or actual['sha256'] != row['sha256']:
            raise ValueError('Frozen native link source changed: '+name)
        if name == 'callbacks.py':
            _native_hash_bindings(config, actual)
    for row in manifest['external_dependency_receipts']:
        actual = _receipt(_inside(path.parent/row['path_relative_to_link_port'], root, 'native external dependency'))
        if actual['bytes'] != row['bytes'] or actual['sha256'] != row['sha256']:
            raise ValueError('Frozen native link external dependency changed: '+actual['path'])
    return _load(path, '_expansion_frozen_native_link_v1')


def create_session(recipe, procedure, data_root, device):
    recipe, data_root = _prepare_recipe(recipe, data_root)
    if procedure == 'independent4':
        return IndependentLinkBank(recipe, data_root, device)
    library, roster, receipts = _dependencies()
    candidate = procedure if procedure in roster else recipe.get('link_candidate_id')
    if candidate is not None and roster[candidate]['kind'] != 'query_objective':
        raise ValueError('Optional adapter/gate integration is not implemented in this version')
    base = 'gnnm_boundary_native' if candidate is not None else procedure
    return LinkSession(_native_factory(recipe).create_session(recipe, base, data_root, device), candidate, recipe=recipe)


class LinkEpochAdapter:
    """One independently owned fit; selectors remain with root's full runner."""
    def __init__(self, session):
        if len(session.optimizers) != 1:
            raise ValueError('An I4 bank is four independent fit adapters, not one pooled selector')
        self.session = session
        self.model, self.optimizers = session.model, session.optimizers
    @property
    def completed_epochs(self): return self.session.completed_epochs
    def train_epoch(self): return self.session.train_epoch()
    def validate(self): return self.session.validate()
    def state(self): return self.session.state()
    def load_state(self, state): return self.session.load_state(state)
    def metadata(self): return self.session.metadata()
    def assert_identity(self): return self.session.assert_identity()
    def close(self): return self.session.close()


def create_adapter(torch, job, policy, root):
    if job.get('TEST_access') is not False or job.get('family') != 'link':
        raise ValueError('Explicit family=link and TEST_access=false required')
    if getattr(policy, 'family', None) != 'link':
        raise NotImplementedError('An explicitly link-specific policy is required; generic Policy uses an incompatible OGB source-reference fallback')
    if getattr(policy, 'metric', None) != 'hits@50' or getattr(policy, 'direction', None) != 'max' or getattr(policy, 'ties', None) != 'first':
        raise ValueError('A declared link policy must select full VALID Hits@50 with first ties')
    if job['procedure'] == 'independent4':
        raise ValueError('I4 full runs require four separately owned native fit jobs and selectors')
    root = Path(root).resolve()
    recipe = copy.deepcopy(job['recipe']); recipe['_seed'] = int(job['seed']); recipe['seed'] = int(job['seed'])
    recipe['_repository_root'] = str(root)
    recipe['_job_callback_binding'] = _job_callback_binding(job, root, __file__)
    for key in ('_native_callback_sha256', 'native_callback_sha256'):
        if key in job:
            if key in recipe and recipe[key] != job[key]:
                raise ValueError('Conflicting job and recipe native callback SHA256')
            recipe[key] = job[key]
    settings = job.get('method_settings', {})
    if settings.get('candidate_id'):
        recipe['link_candidate_id'] = settings['candidate_id']
    recipe, data_root = _prepare_recipe(recipe, job['data_root'])
    if getattr(policy, 'track', None) == 'source_reference' and (job['procedure'] not in ('native_single', 'native') or recipe.get('link_candidate_id')):
        raise ValueError('Source-reference fits must remain unextended native M1')
    session = create_session(recipe, job['procedure'], data_root, job.get('device', 'cuda:0'))
    if settings.get('kwargs') is not None and settings['kwargs'] != session.kwargs:
        raise ValueError('Job kwargs differ from the finalized candidate definition')
    return LinkEpochAdapter(session)


class IndependentLinkBank:
    """Genuine native I4 ownership for qualification, with per-member selection.

    Full-run scheduling should use four separate LinkEpochAdapter jobs. This
    optional bank does not invent a shared epoch/pooled checkpoint selector.
    """
    def __init__(self, recipe, data_root, device):
        recipe, data_root = _prepare_recipe(recipe, data_root)
        import torch
        self.t = torch
        self.recipe = copy.deepcopy(recipe)
        self.members, self.member_rng = [], []
        self.selected, self.stop_reasons = {}, {}
        offsets = recipe.get('native_seed_offsets')
        if not isinstance(offsets, (list, tuple)) or len(offsets) != 4 or any(type(value) is not int for value in offsets) or offsets[0] != 0 or len(set(offsets)) != 4:
            raise ValueError('Four explicit distinct native seed offsets with member zero offset0 required')
        outside = _capture_rng(torch)
        try:
            for offset in offsets:
                value = int(recipe.get('_seed', recipe.get('seed', 0)))+int(offset)
                config = copy.deepcopy(recipe); config['_seed'] = value; config['seed'] = value
                config.pop('link_candidate_id', None)
                _seed(torch, value)
                native = _native_factory(config).create_session(config, 'native_single', data_root, device)
                self.members.append(LinkSession(native, recipe=config))
                self.member_rng.append(_capture_rng(torch))
        finally:
            _restore_rng(torch, outside)
        self.model = torch.nn.ModuleList([member.model for member in self.members])
        self.optimizers = [member.optimizer for member in self.members]
        if len({id(parameter) for parameter in self.model.parameters()}) != sum(len(tuple(member.model.parameters())) for member in self.members):
            raise ValueError('Independent native models share a Parameter')
        self.source_parameter_references = {str(i)+'.'+name: parameter for i, member in enumerate(self.members)
                                            for name, parameter in member.source_parameter_references.items()}
        self.allow_missing_grad_names = [str(i)+'.'+name for i, member in enumerate(self.members)
                                         for name in member.allow_missing_grad_names]
        self.is_independent4 = True
        self.bank_rule = 'mean_logits'
        self.updates = 0

    def _own(self, member, method, *args):
        outside = _capture_rng(self.t)
        try:
            _restore_rng(self.t, self.member_rng[member])
            result = getattr(self.members[member], method)(*args)
            self.member_rng[member] = _capture_rng(self.t)
            return result
        finally:
            _restore_rng(self.t, outside)

    @property
    def source_metadata(self): return self.metadata()

    def metadata(self):
        return {'bank': 'four genuinely independent native link models', 'is_independent4': True,
            'separate_weights_optimizers_RNG_samplers_and_selectors': True,
            'native_seed_offsets': self.recipe['native_seed_offsets'],
            'members': [member.metadata() for member in self.members],
            'member_zero_reused_for_M1': True, 'pooled_epoch_selector': False, 'TEST_opened': False}

    def train_update(self):
        if self.stop_reasons:
            raise ValueError('After individual stopping use train_member_epoch; joint qualification updates are disabled')
        rows = [self._own(i, 'train_update') for i in range(4)]
        self.updates += 1
        return {'loss': sum(float(row['loss']) for row in rows)/4, 'optimizer_updates': 4,
                'member_updates': rows, 'independent_optimizers': True,
                'work': {'optimizer_transitions': 4, 'native_member_optimizer_maps': 4}}

    def train_member_epoch(self, member):
        if member in self.stop_reasons:
            raise ValueError('This independently selected/stopped member is no longer trainable')
        return self._own(member, 'train_epoch')

    def validate_member(self, member):
        return self._own(member, 'validate')

    def observe_member_validation(self, member, result):
        progress = self.members[member].epoch_progress()
        if not progress['at_epoch_boundary'] or progress['completed_epochs'] == 0:
            raise ValueError('Member selection requires a completed positive native epoch')
        if result.get('role') != 'VALID' or not result.get('complete_source_VALID'):
            raise ValueError('Only complete official VALID may select a member')
        score = float(result['metrics']['hits@50'])
        if not math.isfinite(score):
            raise ValueError('Nonfinite member selection metric')
        previous = self.selected.get(member)
        save = previous is None or score > previous['metric']
        if save:
            self.selected[member] = {'epoch': progress['completed_epochs'], 'metric': score,
                                      'source_state': self._own(member, 'state')}
        return {'save_selected': save, 'ties': 'first',
                'selected_epoch': self.selected[member]['epoch'], 'selected_metric': self.selected[member]['metric']}

    def stop_member(self, member, reason):
        if member not in self.selected or not reason:
            raise ValueError('A stopped member requires its own selected checkpoint and explicit reason')
        self.stop_reasons[member] = str(reason)

    def _aggregate(self, rows, selected):
        outputs = self.t.stack([row['outputs'] for row in rows])
        for row in rows[1:]:
            if not self.t.equal(rows[0]['query_endpoints'], row['query_endpoints']):
                raise ValueError('Independent VALID query identity/order differs')
        pool = outputs.mean(0)
        result = self.members[0].native._validate_result(pool[:60084], pool[60084:],
            {'independent_native_models': 4, 'all_official_VALID_records': True})
        from ogb.linkproppred import Evaluator
        library = self.members[0].library
        result.update(member_outputs=outputs, labels=rows[0]['labels'], ids=rows[0]['ids'],
            query_endpoints=rows[0]['query_endpoints'], complete_source_VALID=True,
            member_metrics=[row['metrics'] for row in rows], individually_selected_states=selected,
            bank_diagnostics=library.link_bank_diagnostics(self.t, outputs, rows[0]['labels'].tolist(), Evaluator(name='ogbl-collab')))
        return result

    def validate(self):
        return self._aggregate([self.validate_member(i) for i in range(4)], False)

    def validate_selected(self):
        if set(self.selected) != set(range(4)):
            raise ValueError('All four own selectors must produce a selected checkpoint')
        live = [self._own(i, 'state') for i in range(4)]
        member_rng = copy.deepcopy(self.member_rng)
        outside = _capture_rng(self.t)
        try:
            rows = []
            for i in range(4):
                self._own(i, 'load_state', self.selected[i]['source_state'])
                rows.append(self.validate_member(i))
            return self._aggregate(rows, True)
        finally:
            for i in range(4):
                self._own(i, 'load_state', live[i])
            self.member_rng = member_rng
            _restore_rng(self.t, outside)

    def state(self):
        return {'members': [self._own(i, 'state') for i in range(4)],
                'member_rng': copy.deepcopy(self.member_rng), 'updates': self.updates,
                'selected': copy.deepcopy(self.selected), 'stop_reasons': copy.deepcopy(self.stop_reasons),
                'global_rng': _capture_rng(self.t)}

    def load_state(self, state):
        for i in range(4):
            self.members[i].load_state(state['members'][i])
        self.member_rng = copy.deepcopy(state['member_rng']); self.updates = int(state['updates'])
        self.selected = copy.deepcopy(state['selected']); self.stop_reasons = copy.deepcopy(state['stop_reasons'])
        _restore_rng(self.t, state['global_rng'])

    def comparison_state(self):
        return {'members': [member.comparison_state() for member in self.members],
                'member_rng': self.member_rng, 'selected': self.selected, 'stop_reasons': self.stop_reasons}

    def close(self):
        for member in self.members:
            member.close()
