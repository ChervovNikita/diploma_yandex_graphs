"""Fixed/learned private V noise on the exact paid residual-only/private-BN arm.

No model, data or numerical library is loaded at import. No launch function.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent
E, PHASE = HERE.parents[1], HERE.parents[2]
PARENT = E / 'pilot_20261015_residual_only_private_bn_v1'
ARMS = ('fixed', 'learned')
SCHEMA = 'stochastic-private-residual-BN-v1'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    path = Path(path)
    name = '_stochastic_private_' + sha(path)
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def verify_sources(root, *, data=False):
    """Model-free source check; data integrity only in root-admitted qualifier/runtime."""
    root = Path(root).resolve()
    require(HERE.is_relative_to(root), 'Actual containing repository required')
    for row in json.loads((HERE / 'MANIFEST.json').read_text())['files']:
        path = (HERE / row['path']).resolve()
        require(path.is_relative_to(HERE) and sha(path) == row['sha256'], 'Frozen package changed: ' + row['path'])
    authority = json.loads((HERE / 'SOURCES.json').read_text())
    for row in authority['local_source_pins']:
        path = (PHASE / row['path_relative_to_phase']).resolve()
        require(path.is_relative_to(root) and sha(path) == row['sha256'], 'Retained source changed: ' + str(path))
    if data:
        for row in authority['allocation_only_pins']:
            path = Path(row['path']).resolve()
            require(path.is_relative_to(root) and sha(path) == row['sha256'], 'Frozen provider/role changed: ' + str(path))
        require(not any(Path(p).exists() for p in authority['absent_role_overrides']), 'Native role override appeared')
    return authority


def descriptor(seed, arm):
    require(type(seed) is int and seed in (101, 202, 303) and arm in ARMS, 'Only six frozen cells exist')
    job = json.loads((PARENT / ('JOB_seed' + str(seed) + '.json')).read_text())
    job.update(id='molbace_stochastic_private_v1/seed-' + str(seed) + '/' + arm,
               procedure='molbace_stochastic_private_' + arm + '_v1',
               method_settings={'arm': arm, 'METHOD_sha256': sha(HERE / 'METHOD.json'),
                                'noise_sha256': sha(HERE / 'noise.py'), 'SOURCES_sha256': sha(HERE / 'SOURCES.json')},
               session_transform={'path': str(Path(job['session_transform']['path']).parents[1] /
                    'research/stochastic_private_adapter_source_v1/extension.py'),
                    'function': 'transform', 'sha256': sha(__file__)})
    return job


def optimizer_contract(optimizer):
    return {'parameters': tuple(tuple(id(p) for p in g['params']) for g in optimizer.param_groups),
            'options': tuple({k: copy.deepcopy(v) for k, v in g.items() if k != 'params'} for g in optimizer.param_groups),
            'defaults': copy.deepcopy(optimizer.defaults)}


def transform(session, job):
    arm = job.get('method_settings', {}).get('arm')
    expected = descriptor(job.get('seed'), arm)
    if job.get('device') == 'cpu':
        expected['device'] = 'cpu'  # Explicit bounded engineering override only.
    require(job == expected, 'Exact paid parent recipe/policy plus frozen stochastic arm required')
    authority = verify_sources(Path(job['recipe']['_repository_root']), data=True)
    parent = json.loads((PARENT / ('JOB_seed' + str(job['seed']) + '.json')).read_text())
    load(PARENT / 'session_extension.py').transform(session, parent)
    torch, model, optimizer = session.torch, session.model, session.optimizer
    require(len(session.train_ids) == 1210 and len(session.valid_ids) == 151
            and session.full_responsibilities is None, 'Exact TRAIN1210/whole VALID151 and own-loss source required')
    native = tuple(model.native.parameters())
    old_private = tuple(model.private_parameters())
    old_contract = optimizer_contract(optimizer)
    noise = load(HERE / 'noise.py')
    streams, kl_routes, prior_identity = noise.attach(torch, model.input_residual, optimizer, arm=arm, seed=job['seed'])
    private = tuple(model.private_parameters())
    expected_count = 28800 if arm == 'learned' else 19200
    require(sum(p.numel() for p in private) == expected_count and
            all(a is b for a, b in zip(old_private, private)), 'Retain mu/U; add only learned rho when declared')
    contract = optimizer_contract(optimizer)
    require(contract['options'] == old_contract['options'] and contract['defaults'] == old_contract['defaults'],
            'Original Adam options/defaults must remain unchanged')
    binding = {'schema': SCHEMA, 'arm': arm, 'seed': job['seed'], 'settings': copy.deepcopy(job['method_settings']),
               'parent_JOB_sha256': sha(PARENT / ('JOB_seed' + str(job['seed']) + '.json')),
               'noise_seeds': list(streams.seeds), 'TRAIN_denominator': 4 * 1210,
               'verified_provider_and_role_pins': copy.deepcopy(authority['allocation_only_pins']),
               'serving': 'deterministic mu; original private BN; raw-logit mean', 'TEST_accessed': False}
    original_state, original_load, original_metadata = session.state, session.load_state, session.metadata
    busy = [False]

    def identity():
        require(session.model is model and session.optimizer is optimizer and session.full_responsibilities is None,
                'Retained runtime ownership required')
        require(optimizer_contract(optimizer) == contract, 'Native Adam ownership/options/defaults changed')
        owned = [p for group in optimizer.param_groups for p in group['params']]
        require(len(owned) == len({id(p) for p in owned}) and
                {id(p) for p in owned} == {id(p) for p in model.parameters()} and
                all(p.requires_grad for p in owned) and
                tuple(map(id, model.native.parameters())) == tuple(map(id, native)), 'Exact exhaustive trainable partition required')
        prior_identity()
        model.assert_source_identity()
        model.route_bn_state.assert_native_identity()

    def train_update(self):
        require(not busy[0], 'One nonnested native update required')
        identity()
        busy[0] = True
        noise_started = False
        complete = False
        try:
            self.model.train()
            batch, positions, skipped = self._next_train_batch()
            self.last_train_positions = positions.clone()
            require(torch.equal(batch.smoke_graph_id.reshape(-1).detach().cpu(), self.train_ids[positions].cpu()),
                    'Native TRAIN identities differ')
            self.optimizer.zero_grad()
            streams.begin()
            noise_started = True
            routes = [self.model.run_member(member, batch) for member in range(4)]
            losses = torch.stack([self._native_loss(route[0], batch) for route in routes])
            own, per_route_KL = losses.mean(), kl_routes()
            penalty = per_route_KL.sum() / (4 * 1210)
            loss = own + penalty
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite stochastic native objective; no clipping/retry')
            loss.backward()
            self.optimizer.step()
            self.updates += 1
            complete = True
            identity()
            return {'loss': float(loss.detach()), 'native_own_loss': float(own.detach()),
                    'native_member_losses': losses.detach().cpu().tolist(),
                    'KL_routes_coordinate_sums': per_route_KL.detach().cpu().tolist(), 'KL_penalty': float(penalty.detach()),
                    'optimizer_updates': 1, 'update': self.updates, 'TRAIN_graph_ids': self.train_ids[positions].tolist(),
                    'native_skipped_batches': skipped, 'native_batch_graphs': len(positions), 'native_route_forwards': 4,
                    'noise_draw_counts': list(streams.counts), 'noise_draw_sha256': streams.last_digest,
                    'work': {'optimizer_transitions': 1, 'reverse_collections': 1, 'extra_backbone_forwards': 0}}
        finally:
            try:
                if noise_started:
                    streams.finish(complete=complete)
            finally:
                busy[0] = False

    def state(self):
        require(not busy[0] and streams.counts == [self.updates] * 4, 'Save only a completed update boundary')
        identity()
        private_ids = {id(p) for p in private}
        all_bytes = private_bytes = 0
        for parameter, fields in optimizer.state.items():
            for tensor in fields.values():
                if hasattr(tensor, 'numel') and hasattr(tensor, 'element_size'):
                    size = tensor.numel() * tensor.element_size()
                    all_bytes += size
                    if id(parameter) in private_ids:
                        private_bytes += size
        return {'schema': SCHEMA, 'method': copy.deepcopy(binding), 'noise': streams.state(), 'parent_state': original_state(),
                'resource_accounting': {'actual_optimizer_state_tensor_bytes': all_bytes,
                    'actual_private_optimizer_state_tensor_bytes': private_bytes,
                    'private_trainable_parameters': expected_count,
                    'parent_accounting_scope': 'inherited point producer only; this outer accounting includes rho',
                    'serialization_overhead': 'separate actual checkpoint bytes'}}

    def load_state(self, value):
        require(not busy[0] and value.get('schema') == SCHEMA and value.get('method') == binding,
                'Exact method/seed/source/noise identity required')
        identity()
        source = value['parent_state']['source_state']
        require(source['responsibilities'] is None, 'No private responsibility substitution')
        # Prevent checkpoint replacement of the data-free prior center before
        # calling parent restore; parent restores original Parameter objects.
        saved_model = source['model']
        for key in ('prior_down', 'prior_log_sigma'):
            require(torch.equal(saved_model['input_residual.' + key].cpu(),
                                getattr(model.input_residual, key).detach().cpu()), 'Checkpoint prior changed')
        streams.load_state(copy.deepcopy(value['noise']), source['updates'])
        original_load(copy.deepcopy(value['parent_state']))
        identity()  # Semantic ownership check permits genuine Adam group-dict replacement.

    def metadata(self):
        value = original_metadata()
        value['inherited_point_producer_metadata'] = value.pop('candidate_extension', None)
        parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        value['candidate_extension'] = {**copy.deepcopy(binding),
            'private_trainable_parameters': expected_count, 'native_trainable_parameters': sum(p.numel() for p in native),
            'parameter_bytes': parameter_bytes, 'dense_Adam_two_moment_bytes': 2 * parameter_bytes,
            'prior_buffer_bytes': sum(getattr(model.input_residual, k).numel() * getattr(model.input_residual, k).element_size()
                                      for k in ('prior_down', 'prior_log_sigma')),
            'noise_CPU_draw_bytes_per_update': 4 * 8 * 300 * 4,
            'noise_RNG_state_bytes': sum(g.get_state().numel() * g.get_state().element_size() for g in streams.generators),
            'qualification': 'source prepared; no numerical/native execution claimed',
            'source_mean_field_not_full_network_posterior': True}
        return value

    session.train_update = types.MethodType(train_update, session)
    session.state, session.load_state = types.MethodType(state, session), types.MethodType(load_state, session)
    session.metadata = types.MethodType(metadata, session)
    session.stochastic_noise = streams  # Qualifier inspects only this explicit API.
    identity()
    return session
