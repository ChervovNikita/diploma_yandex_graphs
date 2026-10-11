"""Finite native Penn94 acquisition/qualification; numerical imports after root gate."""
import argparse
import copy
import math
from pathlib import Path
import random
import resource
import time
from types import SimpleNamespace
import traceback
from contracts import (ARMS, SEEDS, gate, scoped, receipt, describe, module, require, write, sha)


class Family:
    def __init__(self, cfg, output, custody):
        import numpy as np
        import torch
        import torch.nn.functional as F
        self.np, self.torch, self.F = np, torch, F
        self.cfg, self.output, self.custody = cfg, scoped(output), custody
        self.device = torch.device(cfg['device'])
        torch.cuda.set_device(self.device)
        require(torch.get_default_dtype() == torch.float32, 'Exact native float32 constructor provider')
        torch.set_num_threads(2)
        self.cuda = self.device.index or 0
        sources = cfg['source_receipts']
        self.models = module('Penn94_saved_native_models', receipt(sources['native_models']))
        self.common = module('Penn94_saved_native_seed', receipt(sources['native_common']))
        self.factors = module('Penn94_existing_factors', receipt(sources['factors']))
        self.routes = module('Penn94_existing_routes', receipt(sources['routes']))
        self.linkx = module('Penn94_pinned_author_LINKX', receipt(sources['LINKX_models']))
        here = Path(__file__).parent
        self.adapter = module('Penn94_safe_data_adapter', here / 'data_adapter.py')
        self.boundary = module('Penn94_matched_boundary', here / 'placement_control.py')
        self.original = module('Penn94_exact_TABM_training', here / 'native_tabm_training.py')
        self.original.F, self.original.TABM_INITS = F, 4
        for name, item in cfg['provider_modules'].items():
            import importlib
            loaded = importlib.import_module(name)
            require(Path(loaded.__file__).resolve() == receipt(item), 'Actual runtime provider module differs: ' + name)
        def load(key, expected):
            with np.load(receipt(cfg['inputs'][key]), allow_pickle=False) as a:
                require(set(a.files) == set(expected), 'Exact role-safe file schema: ' + key)
                return [a[k].copy() for k in expected]
        public_tensors = torch.load(receipt(cfg['inputs']['public']), map_location='cpu', weights_only=True)
        require(set(public_tensors) == {'x','edge_index'} and all(torch.is_tensor(v) for v in public_tensors.values()), 'Root-prepared tensor-only public.pt schema')
        x, edges = public_tensors['x'].numpy(), public_tensors['edge_index'].numpy()
        node_ids = np.arange(41554, dtype=np.int64)  # Complete original row order bound by root custodian receipt.
        train_ids, train_y = load('train', ('ids', 'y'))
        select_ids, select_y = load('select', ('ids', 'y'))
        cal_ids, = load('cal_ids', ('ids',))
        confirm_ids, = load('confirm_ids', ('ids',))
        n, d, e = cfg['node_count'], cfg['feature_dim'], cfg['edge_entries']
        require(x.dtype == np.float32 and x.shape == (41554, d) and np.isfinite(x).all()
                and edges.dtype == node_ids.dtype == np.int64 and edges.shape == (2, e)
                and np.array_equal(node_ids, np.arange(n, dtype=np.int64)), 'Complete canonical X/E/original node map required')
        require(edges[0].min() == 0 and edges.max() < n, 'LINKX row.min subtraction must be identity on complete graph')
        for name, ids in zip(('train', 'select', 'cal', 'confirm'), (train_ids, select_ids, cal_ids, confirm_ids)):
            require(ids.dtype == np.int64 and ids.ndim == 1 and len(ids) == cfg['role_counts'][name]
                    and len(np.unique(ids)) == len(ids) and np.array_equal(ids, np.sort(ids))
                    and ids.min() >= 0 and ids.max() < n, 'Exact sorted proper-role membership: ' + name)
        all_ids = np.concatenate((train_ids, select_ids, cal_ids, confirm_ids))
        require(len(np.unique(all_ids)) == len(all_ids), 'TRAIN/SELECT/CAL/CONFIRM disjointness')
        for ids, y in ((train_ids, train_y), (select_ids, select_y)):
            require(y.dtype == np.int64 and y.shape == ids.shape and set(np.unique(y)) == {0, 1}, 'Binary allowed-role targets')
        self.public = x, edges, node_ids
        self.graph_versions = None
        self.data, self.batch, self.train_ids = self.adapter.native_train_view(torch, SimpleNamespace, torch.from_numpy(x),
            torch.from_numpy(edges), train_ids, train_y, 'two_logit_CE', self.device)
        self.bce_data, _, _ = self.adapter.native_train_view(torch, SimpleNamespace, self.data.x,
            self.data.edge_index, train_ids, train_y, 'one_logit_BCE', self.device)
        self.select_ids = torch.from_numpy(select_ids).to(self.device)
        self.select_y = torch.from_numpy(select_y).to(self.device)
        self.link_data = SimpleNamespace(graph={'node_feat': self.data.x, 'edge_index': self.data.edge_index, 'num_nodes': n})
        self.train_y = self.data.y[self.train_ids]
        self.native_initial_hashes = {}
        self.seen_I4_storage = {}
        self.transfer_scope = {'public_and_two_TRAIN_views': True, 'SELECT_truth_separate': True,
            'CAL_and_CONFIRM_ids_only': True, 'CONFIRM_truth_access': False}

    def synchronize(self):
        self.torch.cuda.synchronize(self.device)

    def fingerprint(self, value):
        import hashlib
        h = hashlib.sha256()
        def visit(v):
            if self.torch.is_tensor(v):
                a = v.detach().cpu().contiguous().numpy()
                h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
            elif isinstance(v, dict):
                for k in sorted(v, key=str):
                    h.update(str(k).encode()); visit(v[k])
            elif isinstance(v, (list, tuple)):
                for item in v: visit(item)
            else: h.update(repr(v).encode())
        visit(value)
        return h.hexdigest()

    def global_rng(self):
        torch, np = self.torch, self.np
        state = np.random.get_state()
        return {'python': random.getstate(), 'numpy': (state[0], state[1].tolist(), state[2], state[3], state[4]),
                'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(self.device)}

    def restore_rng(self, value):
        state = value['numpy']
        random.setstate(value['python'])
        self.np.random.set_state((state[0], self.np.asarray(state[1], dtype=self.np.uint32), state[2], state[3], state[4]))
        self.torch.set_rng_state(value['cpu'].cpu()); self.torch.cuda.set_rng_state(value['cuda'].cpu(), self.device)

    def parameter_count(self, arm):
        d = self.cfg['feature_dim']
        return {ARMS[0]: 132*d+176010, ARMS[1]: 132*d+167048, ARMS[2]: 132*d+166794,
                ARMS[3]: 128*d+165762, ARMS[4]: 129*d+168324, ARMS[5]: 129*d+168324,
                ARMS[6]: 128*d+5369090}[arm]

    def make(self, arm, seed, member):
        self.synchronize(); creation_start = time.perf_counter()
        torch = self.torch
        base_seed = 10000 * seed
        native_seed = base_seed + 1000003 * member
        self.common.set_seed(native_seed if arm != ARMS[1] else base_seed)
        streams, initial_native = [], None
        if arm == ARMS[1]:
            model = self.models.TABMModel('SAGE', 2, self.cfg['feature_dim'], 128, 1, 1, 8,
                                         'LayerNorm', .2, tabm_inits=4, device=self.device)
            require(all(p.requires_grad for p in model.parameters()) and model.input_be_block.W.bias is None
                    and model.output_be_block.W.bias is None
                    and bool(((model.input_be_block.R == -1) | (model.input_be_block.R == 1)).all())
                    and bool((model.output_be_block.R == 1).all())
                    and all(bool((b.S == 1).all()) and bool((b.B == 0).all()) for b in (model.input_be_block, model.output_be_block)),
                    'Exact original BEBlock binary boundaries and native initializer invariants')
        elif arm == ARMS[6]:
            model = self.linkx.LINKX(self.cfg['feature_dim'], 128, 2, 2, 41554, dropout=.5,
                cache=False, inner_activation=False, inner_dropout=False, init_layers_A=1, init_layers_X=1)
            model.reset_parameters()  # Explicit prospectively bound constructor + native reset law.
        else:
            body = self.models.Model('SAGE', 2, self.cfg['feature_dim'], 128, 2, 1, 8, 'LayerNorm', .2)
            native_params = list(body.named_parameters())
            initial_native = self.fingerprint({n: p for n, p in native_params})
            adapter = self.routes.NativeModelAdapter(torch, body)
            if arm == ARMS[2]:
                model = self.boundary.wrap_native_boundary(torch, self.factors, body, adapter,
                    [base_seed + 1000003*m + 2000003 for m in range(4)])
            elif arm in (ARMS[0], ARMS[4], ARMS[5]):
                routes = 4 if arm == ARMS[0] else 1
                wrapper = self.routes.wrap_shared(torch, self.factors, body, adapter,
                    members=routes, kind='baseline', rank=16, block_seed=base_seed+4000037)
                with torch.no_grad():
                    for m in range(routes):
                        generator = torch.Generator(device='cpu').manual_seed(base_seed+1000003*(member+m)+2000003)
                        signs = torch.randint(0, 2, (self.cfg['feature_dim'],), generator=generator)*2-1
                        body.input_linear.r[m].copy_(signs.to(body.input_linear.r))
                model = wrapper if routes == 4 else body
            else:
                model = body
            require({id(p) for _, p in native_params} <= {id(p) for p in model.parameters()},
                    'Native W/b/norm parameter objects retained by factor installation')
            self.native_initial_hashes[seed, arm, member] = initial_native
        model.to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01 if arm == ARMS[6] else .001,
                                      weight_decay=.001 if arm == ARMS[6] else 0)
        ids = [id(p) for g in optimizer.param_groups for p in g['params']]
        require(len(ids) == len(set(ids)) and set(ids) == {id(p) for p in model.parameters()}, 'Exactly one optimizer owner per parameter')
        require(sum(p.numel() for p in model.parameters()) == self.parameter_count(arm), 'Actual source formula parameter count')
        routes = 4 if arm in ARMS[:3] else 1
        if arm not in (ARMS[1], ARMS[6]):
            for m in range(routes):
                self.common.set_seed(base_seed+1000003*(member+m)+3000007)
                streams.append({'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(self.device)})
        self.synchronize()
        return model, optimizer, streams, {'creation_seconds': time.perf_counter()-creation_start, 'native_seed': native_seed if arm != ARMS[1] else base_seed,
            'block_label': seed, 'actual_base_seed': base_seed,
            'native_initial_sha256': initial_native, 'parameters': self.parameter_count(arm),
            'routes': routes, 'representation': 'BCE1' if arm == ARMS[1] else 'CE2',
            'dropout': 'original_global_sequential' if arm == ARMS[1] else 'LINKX_global' if arm == ARMS[6] else 'existing_persistent_member_scopes'}

    def logits(self, model, streams, arm, train=False):
        torch = self.torch
        before = (self.data.x._version, self.data.edge_index._version)
        if arm == ARMS[1]:
            value = torch.stack([model(self.bce_data, self.bce_data.x, tabm_seed=m) for m in range(4)])
        elif arm == ARMS[6]:
            value = model(self.link_data).unsqueeze(0)
        elif arm in (ARMS[0], ARMS[2]):
            value = model(self.batch, route_scope=lambda m: self.routes.existing_member_rng_scope(torch, streams, m, self.cuda))[0]
        else:
            with self.routes.existing_member_rng_scope(torch, streams, 0, self.cuda):
                value = model(self.data, self.data.x).unsqueeze(0)
        require(before == (self.data.x._version, self.data.edge_index._version), 'Native forward mutated public X/E')
        require(value.dtype == torch.float32 and bool(torch.isfinite(value).all()), 'Finite native float32 logits')
        require(value.shape == ((4, 41554) if arm == ARMS[1] else (4 if arm in (ARMS[0], ARMS[2]) else 1, 41554, 2)),
                'Complete node identity/output shape')
        return value

    def raw_cache(self, raw, representation):
        """Cache native-device float32 decision authority; no truth or native calls."""
        torch = self.torch
        raw = raw.to(self.device)
        with torch.no_grad():
            probability = self.adapter.raw_binary_probability_pool(torch, raw,
                'one_logit_BCE' if representation == 'BCE1' else 'two_logit_CE', 'probability_mean')
            logit = self.adapter.raw_binary_probability_pool(torch, raw,
                'one_logit_BCE' if representation == 'BCE1' else 'two_logit_CE', 'logit_mean')
            member = (raw > 0).long() if representation == 'BCE1' else raw.argmax(-1)
            logit_decision = (raw.mean(0) > 0).long() if representation == 'BCE1' else logit.argmax(-1)
        return {'node_ids': self.public[2], 'raw_logits': raw.cpu().numpy(),
            'raw_probability_pool': probability.cpu().numpy(), 'raw_logit_pool': logit.cpu().numpy(),
            'raw_probability_decision': probability.argmax(-1).cpu().numpy(),
            'raw_logit_decision': logit_decision.cpu().numpy(), 'member_decision': member.cpu().numpy()}

    def gradient_record(self, model, arm):
        torch = self.torch
        groups = {'stem': [], 'message': [], 'ffn': [], 'head': [], 'private': [], 'LINKX': []}
        for name, p in model.named_parameters():
            if p.grad is None: continue
            require(bool(torch.isfinite(p.grad).all()), 'Finite complete TRAIN gradients: ' + name)
            if name.endswith(('.r', '.s', '.R', '.S', '.B')): key = 'private'
            elif arm == ARMS[6]: key = 'LINKX'
            elif 'input_' in name: key = 'stem'
            elif 'conv.' in name: key = 'message'
            elif 'feed_forward_module' in name: key = 'ffn'
            elif 'output_' in name: key = 'head'
            else: continue
            groups[key].append(float(p.grad.abs().sum()))
        totals = {k: sum(v) for k, v in groups.items()}
        required = ('LINKX',) if arm == ARMS[6] else ('stem', 'message', 'ffn', 'head')
        require(all(totals[k] > 0 for k in required), 'Full graph TRAIN gradient reaches every native region')
        if arm not in (ARMS[3], ARMS[6]): require(totals['private'] > 0, 'Applicable private factors live')
        return totals

    def fit(self, arm, seed, member, qualification=False, created=None):
        torch, F = self.torch, self.F
        folder = self.output / f'{arm}_seed{seed}' / f'member{member}'
        folder.mkdir(parents=True, exist_ok=False)
        before, selected, step, saves = time.perf_counter(), 0, 0, 0
        creation = {'creation_seconds': 0.}
        trace, hooks = [], []
        phase = {'name': 'TRAIN_native_routes'}
        timings = {'train_seconds': 0., 'select_seconds': 0., 'checkpoint_save_seconds': 0.}
        ops = {'optimizer_steps': 0, 'backward_calls': 0, 'TRAIN_native_routes': 0, 'SELECT_native_routes': 0,
               'selected_native_routes': 0, 'checkpoint_writes': 0}
        record = {'arm': arm, 'seed': seed, 'member': member, 'status': 'failed_retained'}
        try:
            torch.cuda.reset_peak_memory_stats(self.device)
            model, optimizer, streams, creation = created if created is not None else self.make(arm, seed, member)
            record['construction'] = creation
            native = model.body if hasattr(model, 'body') else model
            def native_call(_module, _args): ops[phase['name']] += 1
            def backward_called(gradient):
                ops['backward_calls'] += 1
                return gradient
            head_weight = (model.output_be_block.W.weight if arm == ARMS[1] else model.mlp_final.lins[-1].weight if arm == ARMS[6] else native.output_linear.weight)
            hooks = [native.register_forward_pre_hook(native_call), head_weight.register_hook(backward_called),
                optimizer.register_step_post_hook(lambda *_args: ops.__setitem__('optimizer_steps', ops['optimizer_steps']+1))]
            bound = 10 if qualification and arm == ARMS[1] else 1 if qualification else 5000 if arm == ARMS[1] else 500 if arm == ARMS[6] else 1000
            patience = None if arm == ARMS[6] else 300
            best, stale, trace = -math.inf, 0, []
            for step in range(1, bound+1):
                self.synchronize(); start = time.perf_counter()
                phase['name'] = 'TRAIN_native_routes'
                if arm == ARMS[1]:
                    versions = self.bce_data.x._version, self.bce_data.edge_index._version
                    loss_value = self.original.train_step_tabm(model, self.bce_data, optimizer, True)
                    require(versions == (self.bce_data.x._version, self.bce_data.edge_index._version), 'Original TRAIN public mutation guard')
                else:
                    model.train(); optimizer.zero_grad()
                    logits = self.logits(model, streams, arm, True)
                    if arm == ARMS[6]: loss = F.nll_loss(logits[0, self.train_ids].log_softmax(-1), self.train_y)
                    else:
                        logits = logits[:, self.train_ids]
                        loss = F.cross_entropy(logits.reshape(-1, 2), self.train_y.repeat(len(logits)))
                    require(bool(torch.isfinite(loss)), 'Finite full TRAIN own loss')
                    loss.backward(); optimizer.step(); loss_value = float(loss.detach())
                require(math.isfinite(loss_value), 'Finite original/full TRAIN scalar')
                gradients = self.gradient_record(model, arm) if qualification else None
                self.synchronize(); timings['train_seconds'] += time.perf_counter()-start
                phase['name'] = 'SELECT_native_routes'
                model.eval(); self.synchronize(); start = time.perf_counter()
                with torch.no_grad(): logits = self.logits(model, streams, arm)[:, self.select_ids]
                decision = (logits.mean(0) > 0).long() if arm == ARMS[1] else logits.softmax(-1).mean(0).argmax(-1)
                correct = int(decision.eq(self.select_y).sum()); accuracy = correct / len(self.select_y)
                self.synchronize(); timings['select_seconds'] += time.perf_counter()-start
                improving = accuracy > best
                if improving: stale = 0
                else: stale += 1
                eligible_save = improving and (arm != ARMS[1] or step % 10 == 0)
                if eligible_save:
                    best, selected = accuracy, step
                    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'streams': copy.deepcopy(streams),
                        'global_rng': self.global_rng(), 'arm': arm, 'seed': seed, 'member': member, 'selected_step': step,
                        'selection_correct': correct, 'selection_nodes': len(self.select_y), 'custody': self.custody,
                        'model_sha256': self.fingerprint(model.state_dict()), 'optimizer_sha256': self.fingerprint(optimizer.state_dict())}
                    self.synchronize(); start = time.perf_counter(); torch.save(state, folder/'selected.pt')
                    self.synchronize(); timings['checkpoint_save_seconds'] += time.perf_counter()-start
                    saves += 1; ops['checkpoint_writes'] += 1
                trace.append({'update': step, 'TRAIN_own_loss': loss_value, 'SELECT_correct': correct, 'SELECT_nodes': len(self.select_y),
                    'strictly_above_saved_best_before_update': improving, 'divisible_by_ten': step%10 == 0,
                    'checkpoint_written': eligible_save, 'selected_step_after_update': selected, 'stale': stale})
                write(folder/'PROGRESS.json', {'arm': arm, 'seed': seed, 'member': member, 'update': step, 'selected_step': selected})
                if patience is not None and stale >= patience: break
            require(selected > 0, 'Native cadence produced no selected checkpoint; retain failure')
            self.synchronize(); restore_start = time.perf_counter()
            state = torch.load(folder/'selected.pt', map_location=self.device, weights_only=True)
            require(state['custody'] == self.custody and (state['arm'], state['seed'], state['member']) == (arm, seed, member), 'Selected source/config/role identity')
            if qualification:
                with torch.no_grad():
                    for p in model.parameters(): p.zero_()
                    for b in model.buffers(): b.zero_()
                optimizer.state.clear(); streams.clear()
            model.load_state_dict(state['model'], strict=True); optimizer.load_state_dict(state['optimizer'])
            streams = copy.deepcopy(state['streams'])
            for stream in streams:
                stream['cpu'] = stream['cpu'].cpu(); stream['cuda'] = stream['cuda'].cpu()
            self.restore_rng(state['global_rng'])
            require(self.fingerprint(model.state_dict()) == state['model_sha256']
                    and self.fingerprint(optimizer.state_dict()) == state['optimizer_sha256']
                    and self.fingerprint(streams) == self.fingerprint(state['streams'])
                    and self.fingerprint(self.global_rng()) == self.fingerprint(state['global_rng']), 'Exact full selected native/optimizer/stream/RNG restoration')
            self.synchronize(); timings['restoration_seconds'] = time.perf_counter()-restore_start
            phase['name'] = 'selected_native_routes'
            model.eval(); self.synchronize(); serving_start = time.perf_counter()
            with torch.no_grad(): final = self.logits(model, streams, arm).detach().cpu().numpy()
            self.synchronize(); timings['selected_fullgraph_serving_seconds'] = time.perf_counter()-serving_start
            self.synchronize(); pool_start = time.perf_counter()
            cache = self.raw_cache(torch.from_numpy(final), creation['representation'])
            self.synchronize(); timings['selected_cached_two_pool_seconds'] = time.perf_counter()-pool_start
            authority = cache['raw_logit_decision'] if arm == ARMS[1] else cache['raw_probability_decision']
            require(int((authority[self.select_ids.cpu().numpy()] == self.select_y.cpu().numpy()).sum())
                    == state['selection_correct'], 'Restored native-device raw selected decisions equal saved SELECT criterion')
            self.np.savez(folder/'selected_RAW.npz', **cache)
            record.update(status='selected_finite', selected_step=selected, completed_updates=step,
                checkpoint=describe(folder/'selected.pt'), selected_logits=describe(folder/'selected_RAW.npz'),
                qualification_gradient_regions=gradients if qualification else None,
                selected_restore_verified=True, original_ten_step_save_exercised=(arm != ARMS[1] or selected%10==0),
                actual_parameters=creation['parameters'], actual_parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
                optimizer_tensor_state_bytes=sum(v.numel()*v.element_size() for x in optimizer.state.values() for v in x.values() if torch.is_tensor(v)),
                native_factor_dropout_seed_scope=creation['dropout'], trace=trace)
            del model, optimizer, streams, state, final
        except Exception as error:
            record.update(status='failed_retained', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(), completed_updates=step,
                selected_step=selected, checkpoint_writes=saves, failure_does_not_promote_remaining_seeds=True)
        for hook in hooks: hook.remove()
        record['trace'] = trace
        record.update(operation_counts=ops, operation_scope='Native routes counted at actual forward entry; backwards at head gradient; optimizer steps at post-step hook. Failed calls may be incomplete; costs remain retained.', costs={**timings, 'fit_wall_seconds': time.perf_counter()-before,
            'constructor_seconds_outside_fit_wall': creation['creation_seconds'] if created is not None else 0.,
            'CUDA_peak_allocated_bytes': torch.cuda.max_memory_allocated(self.device),
            'CUDA_peak_reserved_bytes': torch.cuda.max_memory_reserved(self.device),
            'process_lifetime_peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024},
            full_graph_nodes=41554, CONFIRM_truth_access=False,
            scope='Full native fit with TRAIN loss and SELECT only; CAL/CONFIRM memberships are public and their truth is excluded.')
        write(folder/'RESULT.json', record)
        torch.cuda.empty_cache()
        return record

    def run(self, qualification=False):
        started, rows = time.perf_counter(), []
        seeds = (SEEDS[0],) if qualification else SEEDS
        for seed in seeds:
            for arm in ARMS:
                created = None
                if arm == ARMS[5]:
                    try:
                        created = [self.make(arm, seed, m) for m in range(4)]
                        parameters = [p for model, _, _, _ in created for p in model.parameters()]
                        optimizers = [id(optimizer) for _, optimizer, _, _ in created]
                        require(len({id(p) for p in parameters}) == len(parameters)
                            and len({p.untyped_storage().data_ptr() for p in parameters}) == len(parameters)
                            and len(set(optimizers)) == 4, 'Four actual complete I4 bodies/storage/optimizers are disjoint')
                    except Exception as error:
                        created = None
                        I4_error = {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()}
                if arm == ARMS[5] and created is None:
                    fits = []
                    for member in range(4):
                        folder = self.output / f'{arm}_seed{seed}' / f'member{member}'
                        folder.mkdir(parents=True, exist_ok=False)
                        failure = {'arm': arm, 'seed': seed, 'member': member, 'status': 'failed_retained',
                            'completed_updates': 0, 'operation_counts': {}, 'costs': None, 'I4_setup_failure': I4_error,
                            'CONFIRM_truth_access': False, 'no_sequential_or_seed_rescue_after_I4_setup_failure': True}
                        write(folder/'RESULT.json', failure); fits.append(failure)
                else:
                    fits = [self.fit(arm, seed, m, qualification, created[m] if created is not None else None)
                        for m in range(4 if arm == ARMS[5] else 1)]
                if arm == ARMS[5]:
                    for fit in fits:
                        fit['simultaneous_I4_disjoint_storage_verified'] = created is not None
                        if created is None and qualification:
                            fit['status'] = 'failed_retained'
                            fit['I4_qualification_error'] = 'Simultaneous four-body storage/optimizer ownership was not qualified'
                    for fit in fits:
                        write(self.output / f'{arm}_seed{seed}' / f'member{fit["member"]}' / 'RESULT.json', fit)
                    del created
                finite = all(r['status'] == 'selected_finite' for r in fits)
                row = {'arm': arm, 'seed': seed, 'fits': fits, 'status': 'selected_finite' if finite else 'failed_retained',
                    'primary_pool': 'logit_mean' if arm == ARMS[1] else 'probability_mean',
                    'representation': 'BCE1' if arm == ARMS[1] else 'CE2', 'CONFIRM_truth_access': False}
                if finite:
                    if arm == ARMS[5]:
                        values = []
                        for fit in fits:
                            with self.np.load(receipt(fit['selected_logits']), allow_pickle=False) as a:
                                require(self.np.array_equal(a['node_ids'], self.public[2]), 'Independent selected body row identity')
                                values.append(a['raw_logits'].copy())
                        raw = self.np.concatenate(values, axis=0)
                        name = self.output/f'{arm}_seed{seed}'/'selected_RAW.npz'
                        pool_start = time.perf_counter()
                        self.np.savez(name, **self.raw_cache(self.torch.from_numpy(raw), 'CE2'))
                        self.synchronize()
                        row['independent_selected_four_body_cached_pool_seconds'] = time.perf_counter()-pool_start
                        row['selected_logits'] = describe(name)
                    else: row['selected_logits'] = fits[0]['selected_logits']
                rows.append(row)
        native_pairs = [(self.native_initial_hashes.get((s, ARMS[0], 0)), self.native_initial_hashes.get((s, ARMS[2], 0))) for s in seeds]
        placement_match = all(a is not None and a == b for a,b in native_pairs)
        # Distinct fit/body/checkpoint records attest independent acquisition. Runtime parameter ownership is checked per constructor.
        require(len(rows) == len(seeds)*7 and sum(len(r['fits']) for r in rows) == len(seeds)*10, 'Complete terminal roster')
        value = {'terminal': True, 'all_selected_finite': all(r['status']=='selected_finite' for r in rows),
            'banks': len(rows), 'body_fit_units': sum(len(r['fits']) for r in rows), 'records': rows,
            'placement_native_initial_bytes_equal': placement_match, 'custody': self.custody,
            'config_sha256': self.custody['config_sha256'], 'wall_seconds': time.perf_counter()-started,
            'CONFIRM_truth_access': False, 'calibration_endpoints_required_before_confirmation': 90,
            'comparisons': [], 'predictive_utility_not_inferred_from_qualification': True}
        if qualification:
            value['passed'] = value['all_selected_finite'] and placement_match
            value['scientific_fits'] = 0
            value['scope'] = 'Disposable actual complete41554-node depth2/width128 native qualification; original10 steps, every other body1 step; no CONFIRM truth.'
            write(self.output/'QUALIFICATION.json', value)
        else:
            value['interpretation_eligible'] = False
            write(self.output/'COMPLETE_NATIVE.json', value)
        return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True); parser.add_argument('--admission', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--stage', choices=('qualify','acquire'), required=True)
    args = parser.parse_args()
    cfg, admission, custody = gate(args.config, args.admission, args.stage)
    output = scoped(args.output); require(not output.exists(), 'Fresh output only')
    output.mkdir(); start = time.perf_counter()
    family = Family(cfg, output, custody)
    setup_seconds = time.perf_counter()-start
    write(output/'SETUP_COSTS.json', {'source_import_and_full_graph_role_transfer_seconds': setup_seconds,
        'scope': 'Nested in owner stage wall; source/provider imports, public arrays and two TRAIN-only views included. No CONFIRM truth.'})
    value = family.run(args.stage == 'qualify')
    write(output/'STAGE_END.json', {'terminal': True, 'success': value.get('passed', value['all_selected_finite']) and value['placement_native_initial_bytes_equal'],
        'stage': args.stage, 'wall_seconds': time.perf_counter()-start, 'CONFIRM_truth_access': False})
    print('{"terminal":true,"scientific_interpretation_eligible":false}')


if __name__ == '__main__':
    main()
