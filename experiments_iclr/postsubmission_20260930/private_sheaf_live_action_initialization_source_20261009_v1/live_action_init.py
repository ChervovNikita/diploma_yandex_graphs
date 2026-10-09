"""Inactive hooks/screens around the exact V2 NSD bank; no numerical imports.

Torch/NumPy/native modules are supplied by the existing qualified V2 caller.
This module has no CLI or launcher. Every public execution entry defaults off.
"""
import hashlib
import itertools
import json
from pathlib import Path
import resource
import time

from support import require, sha

HERE = Path(__file__).resolve().parent


def policy():
    return json.loads((HERE / 'INITIALIZATION_TEMPLATE_DISABLED.json').read_text())


def authorize(execute=False, authorization=None):
    if not execute:
        return False
    value = authorization or {}
    require(value.get('enabled') is True and value.get('release_owner') == 'root'
            and value.get('action') == 'live_action_initialization_study', 'Explicit root initialization release')
    require(value.get('source_manifest_sha256') == sha(HERE / 'MANIFEST.json')
            and value.get('template_sha256') == sha(HERE / 'INITIALIZATION_TEMPLATE_DISABLED.json'), 'Exact source/template release')
    require(value.get('core_V2_admission_and_source_reviewed') is True
            and value.get('operator_capture_numerically_qualified') is True
            and value.get('runtime_and_exact_roles_bound') is True
            and value.get('execute_context_or_engineering_assessor') is False, 'Separate qualified initializer custody')
    return True


def _factors(bank):
    return [value for member in bank.members for learner in member.sheaf_learners
            for value in (learner.linear1.r, learner.linear1.s)]


def _identity(torch, bank, frozen=False):
    bank.assert_ownership()
    require(all(torch.all(value == 1).item() for value in _factors(bank)), 'Every incidence r/s exactly one')
    if frozen:
        require(all(not value.requires_grad and value.grad is None for value in _factors(bank)), 'Frozen factors have no gradients')


def _bytes(torch, tree):
    if torch.is_tensor(tree):
        return tree.numel() * tree.element_size()
    if isinstance(tree, dict):
        return sum(_bytes(torch, value) for value in tree.values())
    if isinstance(tree, (tuple, list)):
        return sum(_bytes(torch, value) for value in tree)
    return 0


def _cost(torch, helpers, common, device, started, usage0):
    helpers.synchronize(torch, device)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return dict(wall_seconds=time.perf_counter()-started, CPU_user_seconds=usage.ru_utime-usage0.ru_utime,
                CPU_system_seconds=usage.ru_stime-usage0.ru_stime,
                process_RSS_high_water_cumulative_bytes=common.process_peak_rss_bytes(),
                peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_CUDA_reserved_bytes=torch.cuda.max_memory_reserved(device))


def warm_once(np, torch, roc_auc_score, core_fit, old, common, helpers, fresh_bank,
              cfg, data, seed, identity, folder, execute=False, authorization=None):
    """One exact100 warmup; reuse core train_update/evaluate/optimizer unmodified."""
    if not authorize(execute, authorization):
        return dict(status='inactive', model_or_data_access=False)
    p = policy()
    require(seed in p['seeds'] and cfg == p['optimizer'], 'Frozen V2 seed/optimizer')
    folder.mkdir(parents=True, exist_ok=False)
    device = data['x'].device
    torch.cuda.reset_peak_memory_stats(device)
    started, usage0 = time.perf_counter(), resource.getrusage(resource.RUSAGE_SELF)
    counters = {name: 0 for name in ('train_forward_attempts', 'train_forwards_completed', 'backward_attempts',
        'backwards_completed', 'Adam_attempts', 'Adam_steps_completed', 'validation_forward_attempts',
        'validation_forwards_completed', 'fresh_bank_attempts', 'fresh_banks_completed')}
    record = dict(status='started', seed=seed, identity=identity, epochs_required=100,
                  factor_initialization='all ones frozen; no private optimizer state', TEST_truth=False)
    def persist(): common.write_json(folder / 'WARMUP.json', dict(record, counters=counters))
    model = opt = warm = None
    stage, selected, best_key = 'construction', None, None
    persist()
    try:
        helpers.seed_all(np, torch, seed)
        counters['fresh_bank_attempts'] += 1
        model = fresh_bank()
        counters['fresh_banks_completed'] += 1
        _identity(torch, model)
        for value in _factors(model): value.requires_grad_(False)
        opt = core_fit.optimizer(torch, model, cfg)
        for epoch in range(1, p['warmup_epochs']+1):
            stage = 'warm_train_'+str(epoch)
            own_nll = core_fit.train_update(torch, model, opt, data, counters)
            _identity(torch, model, frozen=True)
            require(all(value not in opt.state for value in _factors(model)), 'No private Adam moments or coupled decay during warmup')
            stage = 'warm_validate_'+str(epoch)
            scores, pooled, member_scores, member_logp = core_fit.evaluate(np, torch, roc_auc_score, helpers, old.exact,
                model, data, seed, epoch, counters, 'validation')
            key = old.selector(scores, epoch)
            if best_key is None or key > best_key:
                best_key = key
                selected = dict(schema='owned-geometry-only-shared-selected-state-v1', identity=identity, seed=seed,
                    epoch=epoch, model_state=old.cpu_tree(torch, model.state_dict()),
                    optimizer_state=old.cpu_tree(torch, opt.state_dict()), training_rng=helpers.capture_rng(np, torch),
                    scores=scores, role_logp=pooled, member_scores=member_scores, member_role_logp=member_logp,
                    context_regularizer=0.0, TEST_truth_saved=False)
            common.append_jsonl(folder / 'WARM_HISTORY.jsonl', dict(epoch=epoch, own_TRAIN_nll=own_nll,
                pooled_scores=scores, member_scores=member_scores, selected=key == best_key))
            record['epochs_completed'] = epoch
            persist()
        opt.zero_grad(set_to_none=True)
        _identity(torch, model, frozen=True)
        warm = dict(status='complete', schema='one100-shared-identity-warm-state-v1', seed=seed, identity=identity, epoch=100,
                    model_state=old.cpu_tree(torch, model.state_dict()), optimizer_state=old.cpu_tree(torch, opt.state_dict()),
                    training_rng=helpers.capture_rng(np, torch), selected_before_init=selected,
                    private_optimizer_state_present=False, counters=counters)
        warm['CPU_tensor_payload_bytes'] = _bytes(torch, warm)
        stage = 'save_common_warm_state'
        path = folder / 'WARM_STATE.pt'
        temporary = path.with_suffix('.tmp')
        torch.save(warm, temporary); temporary.replace(path)
        warm['checkpoint_sha256'] = sha(path)
        warm['checkpoint_bytes'] = path.stat().st_size
        record.update(status='complete', warm_state_sha256=sha(path), warm_state_bytes=path.stat().st_size,
                      selected_before_init_epoch=selected['epoch'], slow_Adam_moments_retained=True,
                      original_constructors_inferred_from_completed_banks=counters['fresh_banks_completed'])
        persist()
        return warm
    except Exception as error:
        record.update(status='failed', failure=common.failure_record(error, stage)); persist()
        raise
    finally:
        cost = _cost(torch, helpers, common, device, started, usage0)
        record.update(cost=cost); persist()
        if warm is not None:
            warm['cost'] = cost
            warm['record_counters'] = dict(counters)
        model = opt = None


class FirstActionCapture:
    """Observe exact author return values; hook callbacks always return None."""
    def __init__(self, torch, member, train_index, counters, common_value=None):
        self.torch, self.member, self.train_index, self.counters = torch, member, train_index, counters
        self.common_value = common_value
        self.first_sparse = None
        self.action = self.baseline_value = None
        self.value_materiality = None
        self.builder_calls = self.value_calls = 0
        self.handles = []

    def __enter__(self):
        m = self.member
        require(m.right_weights and len(m.lin_right_weights) == m.layers, 'Pinned first right-feature module exists')
        require(m.laplacian_builder.normalised and not m.laplacian_builder.training and not m.training,
                'Original normalized builder in eval; no guessed normalizer or TRAIN jitter')
        provider = type(m).forward.__globals__['torch_sparse']
        self.native_spmm = provider.spmm
        try:
            self.handles.append(m.laplacian_builder.register_forward_hook(self._builder))
            self.handles.append(m.lin_right_weights[0].register_forward_hook(self._value))
        except BaseException:
            for handle in reversed(self.handles): handle.remove()
            self.handles.clear()
            raise
        return self

    def _builder(self, module, inputs, output):
        self.builder_calls += 1
        self.counters['screen_native_builder_returns'] += 1
        if self.builder_calls == 1:
            require(isinstance(output, tuple) and len(output) == 2, 'Original builder returns live sparse pair and detached diagnostic')
            self.first_sparse = output[0]
        return None

    def _value(self, module, inputs, output):
        t, m = self.torch, self.member
        self.value_calls += 1
        require(self.value_calls == 1 and self.first_sparse is not None and not t.is_grad_enabled(), 'One first-layer eval value after builder')
        indices, values = self.first_sparse
        require(indices.shape[0] == 2 and values.numel() == indices.shape[1]
                and output.shape[0] == m.graph_size*m.final_d, 'Actual sparse/value block shapes')
        require(t.isfinite(values).all().item() and t.isfinite(output).all().item(), 'Finite native normalized weights and transformed value')
        if self.common_value is None:
            self.baseline_value = output.detach().clone()
            value = self.baseline_value
            self.value_materiality = dict(bitwise_equal=True, max_abs=0.0)
        else:
            value = self.common_value
            require(value.shape == output.shape and value.device == output.device and value.dtype == output.dtype,
                    'Same actual baseline value coordinates/dtype/device')
            self.value_materiality = dict(bitwise_equal=t.equal(value, output),
                                         max_abs=float((value-output).abs().max().item()))
        off = (indices[0] // m.final_d) != (indices[1] // m.final_d)
        off_indices, off_values = indices[:, off], values[off]
        self.counters['extra_offnode_spmm_attempts'] += 1
        acted = self.native_spmm(off_indices, off_values, output.shape[0], output.shape[0], value)
        self.counters['extra_offnode_spmm_completed'] += 1
        require(t.isfinite(acted).all().item(), 'Finite deployed off-node action on common actual V')
        self.action = acted.reshape(m.graph_size, m.final_d, -1)[self.train_index].reshape(-1).detach().cpu().double()
        self.counters['offnode_sparse_copy_bytes_total'] += off_indices.numel()*off_indices.element_size()+off_values.numel()*off_values.element_size()
        self.counters['screen_action_CPU_bytes_total'] += self.action.numel()*self.action.element_size()
        self.first_sparse = None
        return None

    def __exit__(self, exc_type, exc, tb):
        for handle in reversed(self.handles): handle.remove()
        self.handles.clear(); self.first_sparse = None
        if exc_type is None:
            require(self.builder_calls == self.member.layers and self.value_calls == 1 and self.action is not None,
                    'Every original layer completed and exactly one first action captured')
        return False


def _hash(domain, seed, *parts):
    value = '|'.join(['live-action-init-v1', domain, str(seed)] + [str(part) for part in parts])
    return hashlib.sha256(value.encode('utf-8')).digest()


def _directions(torch, factor, seed, count):
    result = []
    for index in range(count):
        signs = [1.0 if _hash('direction', seed, index, coordinate)[0] & 1 else -1.0
                 for coordinate in range(factor.numel())]
        value = torch.tensor(signs, device=factor.device, dtype=factor.dtype)
        value = value / torch.linalg.vector_norm(value)
        result.append(value)
    return result


def _norm(torch, value):
    return float(torch.linalg.vector_norm(value).item())


def _pair_choice(torch, seed, candidates, vector_key):
    scores = []
    for j, k in itertools.combinations(sorted(candidates), 2):
        a, b = candidates[j][vector_key], candidates[k][vector_key]
        cosine = max(-1.0, min(1.0, float(torch.dot(a, b).item()) / (_norm(torch, a)*_norm(torch, b))))
        scores.append((1.0-cosine*cosine, _hash('pair-tie', seed, j, k), j, k))
    require(scores, 'Two common admissible directions required')
    chosen = sorted(scores, key=lambda value: (-value[0], value[1], value[2], value[3]))[0]
    return [chosen[2], chosen[3]], [dict(pair=[j,k], score=score) for score, _, j, k in scores]


def candidate_screen(np, torch, old, common, helpers, fresh_bank, warm, data, seed, folder,
                     execute=False, authorization=None):
    """One common17-path screen. Reads x/TRAIN tensors only, never VALID."""
    if not authorize(execute, authorization):
        return dict(status='inactive', model_or_data_access=False)
    p = policy()
    require(seed == warm['seed'] and seed in p['seeds'] and warm['epoch'] == 100, 'Same fixed common warm state')
    folder.mkdir(parents=True, exist_ok=False)
    device = data['x'].device
    torch.cuda.reset_peak_memory_stats(device)
    started, usage0 = time.perf_counter(), resource.getrusage(resource.RUSAGE_SELF)
    before = helpers.capture_rng(np, torch)
    counters = {name: 0 for name in ('screen_native_forward_attempts','screen_native_forwards_completed',
        'screen_native_builder_returns','extra_offnode_spmm_attempts','extra_offnode_spmm_completed',
        'offnode_sparse_copy_bytes_total','screen_action_CPU_bytes_total','fresh_bank_attempts','fresh_banks_completed')}
    record = dict(status='started', seed=seed, candidate_direction_count=8, candidate_trial_native_forwards=16,
                  baseline_native_forwards=1, all_TRAIN_rows=True, VALID_read=False, TEST_truth=False, trials=[])
    def persist(): common.write_json(folder / 'SCREEN.json', dict(record, counters=counters))
    model = result = None
    stage = 'construction'
    persist()
    try:
        counters['fresh_bank_attempts'] += 1
        model = fresh_bank()
        counters['fresh_banks_completed'] += 1
        model.load_state_dict(warm['model_state'], strict=True); model.eval()
        require(old.exact(torch, old.cpu_tree(torch, model.state_dict()), warm['model_state']), 'Exact warm parameters before screen')
        _identity(torch, model)
        helpers.restore_rng(np, torch, warm['training_rng'])
        member = model.members[0]
        factor = member.sheaf_learners[0].linear1.r
        directions = _directions(torch, factor, seed, p['candidate_direction_count'])
        common_value = None
        def observe():
            nonlocal common_value
            counters['screen_native_forward_attempts'] += 1
            with torch.no_grad(), FirstActionCapture(torch, member, data['train_index'], counters, common_value) as capture:
                logp = member(data['x'])
                require(logp.shape == (data['x'].shape[0],2) and torch.isfinite(logp).all().item(), 'Finite full native screen output')
                nll = torch.nn.functional.nll_loss(logp[data['train_index']], data['train_y'])
                require(torch.isfinite(nll).item(), 'Finite all-TRAIN screen NLL')
                margin = (logp[data['train_index'],1]-logp[data['train_index'],0]).detach().cpu().double()
                if common_value is None: common_value = capture.baseline_value
            counters['screen_native_forwards_completed'] += 1
            return dict(nll=float(nll.item()), action=capture.action, margin=margin,
                        common_value_materiality=capture.value_materiality)
        stage = 'baseline'
        baseline = observe()
        action_scale = max(_norm(torch, baseline['action']), p['vector_reference_floor'])
        margin_scale = max(_norm(torch, baseline['margin']), p['vector_reference_floor'])
        nll_limit = baseline['nll']+p['relative_nll_budget']*max(baseline['nll'],p['nll_reference_floor'])
        record.update(baseline_TRAIN_nll=baseline['nll'], nll_limit=nll_limit,
                      all_TRAIN_count=int(data['train_index'].numel()), common_native_value_bytes=common_value.numel()*common_value.element_size())
        candidates = {}
        for index, direction in enumerate(directions):
            observed = []
            for sign in (1,-1):
                stage = 'direction_'+str(index)+'_sign_'+str(sign)
                with torch.no_grad(): factor.copy_(1+sign*p['step']*direction)
                value = observe()
                da, dm = value['action']-baseline['action'], value['margin']-baseline['margin']
                action_relative, margin_relative = _norm(torch,da)/action_scale, _norm(torch,dm)/margin_scale
                passed = value['nll'] <= nll_limit and action_relative > p['visibility_floor'] and margin_relative > p['visibility_floor']
                observed.append(dict(value=value, passed=passed))
                record['trials'].append(dict(direction=index, sign=sign, TRAIN_nll=value['nll'],
                    action_relative_norm=action_relative, classifier_relative_norm=margin_relative,
                    common_value_materiality=value['common_value_materiality'], sign_admissible=passed))
                persist()
            a = (observed[0]['value']['action']-observed[1]['value']['action'])*0.5
            b = (observed[0]['value']['margin']-observed[1]['value']['margin'])*0.5
            if all(value['passed'] for value in observed) and _norm(torch,a)/action_scale > p['visibility_floor'] and _norm(torch,b)/margin_scale > p['visibility_floor']:
                candidates[index] = dict(action=a, classifier=b)
        with torch.no_grad(): factor.fill_(1)
        _identity(torch, model)
        require(old.exact(torch, old.cpu_tree(torch, model.state_dict()), warm['model_state']), 'Screen never changed final slow/factor state')
        record['admissible_direction_ids'] = sorted(candidates)
        require(len(candidates) >= 2, 'Whole four-arm comparison unqualified: fewer than two common admissible directions')
        random_pair = sorted(sorted(candidates, key=lambda index: (_hash('random-order',seed,index),index))[:2])
        action_pair, action_scores = _pair_choice(torch,seed,candidates,'action')
        classifier_pair, classifier_scores = _pair_choice(torch,seed,candidates,'classifier')
        result = dict(status='complete', seed=seed, warm_identity=warm['identity'],
                      warm_checkpoint_sha256=warm['checkpoint_sha256'],
                      choices=dict(identity=None, random_admissible=random_pair, live_action_rank=action_pair,
                                   classifier_response_rank=classifier_pair), directions_CPU=[value.detach().cpu() for value in directions])
        coincident = [list(pair) for pair in itertools.combinations(p['arms'],2)
                      if result['choices'][pair[0]] == result['choices'][pair[1]]]
        result['coincident_choices'] = coincident
        record.update(status='complete', choices=result['choices'], action_pair_scores=action_scores,
                      classifier_pair_scores=classifier_scores, coincident_choices=coincident, standalone_screen_native_forwards=17,
                      equal_allocated_screen_native_forwards_per_arm=17/4,
                      source_utility_established=False, numerical_qualification_established=False)
        persist()
        return result
    except Exception as error:
        record.update(status='failed', failure=common.failure_record(error,stage)); persist()
        raise
    finally:
        helpers.restore_rng(np, torch, before)
        require(old.exact(torch,helpers.capture_rng(np,torch),before),'Screen restores caller RNG exactly')
        cost = _cost(torch,helpers,common,device,started,usage0)
        record.update(cost=cost); persist()
        if result is not None:
            result['cost'] = cost
            result['record_counters'] = dict(counters)
        model = None


def restore_branch(np, torch, old, helpers, model, opt, initialization, seed, execute=False, authorization=None):
    if not authorize(execute, authorization):
        return dict(status='inactive', model_or_data_access=False)
    p = policy()
    warm, screen, arm = initialization['warm'], initialization['screen'], initialization['arm']
    require(warm['seed'] == screen['seed'] == seed and warm['epoch'] == 100
            and screen['status'] == 'complete' and arm in p['arms'], 'One exact completed warm/screen for every arm')
    require(screen['warm_identity'] == warm['identity']
            and screen['warm_checkpoint_sha256'] == warm['checkpoint_sha256'], 'Exact screen-to-epoch100 checkpoint binding')
    model.load_state_dict(warm['model_state'],strict=True)
    opt.load_state_dict(warm['optimizer_state'])
    require(old.exact(torch,old.cpu_tree(torch,model.state_dict()),warm['model_state'])
            and old.exact(torch,old.cpu_tree(torch,opt.state_dict()),warm['optimizer_state']), 'Exact shared warm model and slow Adam moments')
    _identity(torch,model)
    require(all(value not in opt.state for value in _factors(model)), 'No warm private moment to erase/replace')
    for value in _factors(model): value.requires_grad_(True)
    pair = screen['choices'][arm]
    if pair is not None:
        for member_index, (direction_index,sign) in enumerate(((pair[0],1),(pair[0],-1),(pair[1],1),(pair[1],-1))):
            factor = model.members[member_index].sheaf_learners[0].linear1.r
            direction = screen['directions_CPU'][direction_index].to(device=factor.device,dtype=factor.dtype)
            with torch.no_grad(): factor.copy_(1+sign*p['step']*direction)
    for member in model.members:
        require(torch.all(member.sheaf_learners[0].linear1.s == 1).item(), 'First output factor stays one')
        for learner in member.sheaf_learners[1:]:
            require(torch.all(learner.linear1.r == 1).item() and torch.all(learner.linear1.s == 1).item(), 'Later factors stay one initially')
    model.assert_ownership()
    helpers.restore_rng(np,torch,warm['training_rng'])
    require(old.exact(torch,helpers.capture_rng(np,torch),warm['training_rng']), 'Exact epoch100 TRAIN stream for branch')
    return dict(status='ready', arm=arm, pair=pair, start_epoch=101, final_epoch=500,
                slow_Adam_moments_retained=True, only_first_input_factor_values_changed=True,
                common_screen_native_forwards=17, equal_allocated_screen_native_forwards=17/4,
                standalone_screen_native_forwards=17, source_utility_established=False)
