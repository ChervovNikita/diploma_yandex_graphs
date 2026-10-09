"""Inactive concrete SeHGNN M4 own training and source-supply integration.

Only stdlib is imported here. Runtime, exact engine/adapter/helper, real rebuilt
views, an already-placed native prototype and caller master scaler are supplied.
There is no provider loader, CLI, release launcher or external orchestration.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import gc
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAMILIES = ('director', 'actor', 'keyword')
TRIAL_SCALES = (1.0, 0.5, 0.25, 0.125)


class BankContractError(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise BankContractError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def plain_metadata(value):
    """Strip str/numeric/container subclasses from checkpoint identity metadata.

    TorchVersion is a str subclass accepted by JSON but rejected by weights_only
    Torch loading. Metadata never needs arbitrary objects or scientific tensors.
    Native tensor/model/optimizer/scaler state is handled separately by cpu_tree.
    """
    if value is None or type(value) is bool:
        return value
    if isinstance(value, str):
        return str.__str__(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        require(math.isfinite(value), 'Finite plain checkpoint metadata number')
        return float(value)
    if isinstance(value, dict):
        require(all(isinstance(key, str) for key in value), 'Checkpoint metadata keys must be strings')
        result = {}
        for key, item in value.items():
            key = str.__str__(key)
            require(key not in result, 'Checkpoint metadata key normalization collision')
            result[key] = plain_metadata(item)
        return result
    if isinstance(value, tuple):
        return tuple(plain_metadata(item) for item in value)
    if isinstance(value, list):
        return [plain_metadata(item) for item in value]
    raise BankContractError('Checkpoint metadata must contain only plain primitives/containers')


def checkpoint_tree(torch, value):
    """Already-owned CPU tensors plus exact plain metadata/containers only."""
    if type(value) is torch.Tensor:
        require(value.device.type == 'cpu' and not value.requires_grad, 'Checkpoint tensors must be owned detached CPU state')
        return value
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            require(isinstance(key, (str, int)) and type(key) is not bool, 'Plain string/int checkpoint state key')
            key = str.__str__(key) if isinstance(key, str) else int(key)
            require(key not in result, 'Checkpoint key normalization collision')
            result[key] = checkpoint_tree(torch, item)
        return result
    if isinstance(value, tuple):
        return tuple(checkpoint_tree(torch, item) for item in value)
    if isinstance(value, list):
        return [checkpoint_tree(torch, item) for item in value]
    return plain_metadata(value)


def source_gate():
    seal = json.loads((HERE/'SEAL.json').read_text())
    require(seal['runtime_disabled'] is True and sha(HERE/'MANIFEST.json') == seal['manifest_sha256'], 'Exact inactive bank source seal')
    for row in json.loads((HERE/'MANIFEST.json').read_text())['files']:
        path = (HERE/row['path']).resolve(strict=True)
        require(path.is_relative_to(HERE) and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Changed sealed bank source')
    sources = json.loads((HERE/'SOURCE_BINDINGS.json').read_text())
    for row in sources['files']:
        path = (HERE.parent/row['path']).resolve(strict=True)
        require(path.is_relative_to(HERE.parent) and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Changed bank dependency')
    return sources


@dataclass(frozen=True)
class BankConfig:
    enabled: bool = False
    root_source_review_approved: bool = False
    source_seal_sha256: str | None = None
    native_qualification_binding: str | None = None
    full_view_binding: str | None = None
    role_binding: str | None = None
    member_rng_seeds: tuple[int, ...] = ()
    assignments: tuple[str | None, ...] = ()
    source_corrections: bool = False

    def require_enabled(self):
        require(self.enabled is True and self.root_source_review_approved is True, 'Inactive bank: explicit root-qualified enablement required')
        for value in (self.source_seal_sha256, self.native_qualification_binding, self.full_view_binding, self.role_binding):
            require(type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value), 'Exact literal bank/qualification/full/role SHA256 bindings')
        require(self.source_seal_sha256 == sha(HERE/'SEAL.json'), 'Exact reviewed bank seal release')
        require(len(self.member_rng_seeds) == 4 and len(set(self.member_rng_seeds)) == 4
                and all(type(seed) is int and 0 <= seed < 2**32 for seed in self.member_rng_seeds), 'Four explicit distinct prospectively frozen member RNG seeds')
        require(len(self.assignments) == 4 and self.assignments.count(None) == 1
                and sorted(a for a in self.assignments if a is not None) == sorted(FAMILIES), 'Fixed M4: three family recipients and one unassigned member')
        require(type(self.source_corrections) is bool, 'Explicit source-correction policy')


@dataclass(frozen=True)
class ReplayToken:
    member: int
    source: str | None
    mode: str
    role: str
    row_binding: str
    view_binding: str
    buffer_stamp: tuple
    rng: object


def parameter_stamp(parameters):
    return tuple((id(p), int(p._version), bool(p.requires_grad)) for p in parameters)


def buffer_stamp(member):
    return tuple((name, id(value), int(value._version)) for name, value in member.named_buffers(remove_duplicate=False))


@contextmanager
def scratch_native_state(rt, engine, member, mode, rng):
    """Restore original buffer objects, modes and RNG; never copy parameters.

    Scratch BN buffers remain with the returned autograd tape. Copying saved
    buffer values back into that tape's tensors would invalidate its versions.
    """
    torch, np = rt['torch'], rt['numpy']
    caller_rng = engine.capture_rng(np, torch)
    modules = tuple(member.modules())
    modes = tuple(module.training for module in modules)
    original = tuple((module, dict(module._buffers), set(module._non_persistent_buffers_set)) for module in modules)
    try:
        for module, buffers, _ in original:
            module._buffers.clear()
            module._buffers.update({name: None if value is None else value.detach().clone() for name, value in buffers.items()})
        member.train(mode == 'train')
        engine.restore_rng(np, torch, rng)
        yield
    finally:
        for (module, buffers, nonpersistent), training in zip(original, modes):
            module._buffers.clear()
            module._buffers.update(buffers)
            module._non_persistent_buffers_set = nonpersistent
            module.training = training
        engine.restore_rng(np, torch, caller_rng)
        require(engine.exact(torch, engine.capture_rng(np, torch), caller_rng), 'Native transaction preserves caller streams')
        require(all(module.training == mode0 for module, mode0 in zip(modules, modes)), 'Native transaction restores every module mode')
        require(all(set(module._buffers) == set(buffers) and all(module._buffers[k] is value for k, value in buffers.items())
                    for module, buffers, _ in original), 'Native transaction restores original buffer objects')


class BankSession:
    """Actual streamed Adam, native callback, complete-role eval and restore."""
    def __init__(self, rt, engine, adapter, helper, full_ctx, views, prototype, scalar, costs, config=BankConfig()):
        config.require_enabled()
        sources = source_gate()
        self.rt, self.engine, self.adapter, self.helper = rt, engine, adapter, helper
        self.ctx, self.views, self.scalar, self.costs, self.config = full_ctx, dict(views), scalar, costs, config
        self.counters = dict(own_epochs=0, own_member_forwards=0, own_backwards=0, actual_Adam_steps=0,
                             source_reference_forwards=0, source_replay_forwards=0, source_trial_forwards=0, eval_forwards=0)
        self._phase, self._active_cache = None, None
        torch = rt['torch']
        require(rt['device'].type == 'cuda' and rt['model_class'] is rt['model_module'].SeHGNN
                and rt['model_module'].torch is torch, 'Exact native CUDA/AMP runtime')
        require(isinstance(scalar, torch.cuda.amp.GradScaler) and scalar.is_enabled(), 'Caller-owned enabled native CUDA AMP master scaler')
        for module, key in ((engine, 'native_engine_sha256'), (adapter, 'adapter_source_sha256'), (helper, 'source_helper_sha256')):
            require(sha(module.__file__) == sources[key], 'Exact supplied source module: '+key)
        require(sha(engine.capture_rng.__code__.co_filename) == sources['native_state_sha256'], 'Exact native state helper')
        require(sha(rt['model_module'].__file__) == sources['native_model_sha256']
                and sha(rt['native'].__file__) == sources['native_helpers_sha256'], 'Exact native architecture and helpers')
        require(set(views) == set(FAMILIES), 'All three concrete real-family caches required')
        require(len(full_ctx.train_index) == full_ctx.train_count and 0 < full_ctx.train_count <= 10000
                and len(full_ctx.valid_index) == full_ctx.valid_count, 'Complete native role counts')
        self.rows = {'TRAIN': full_ctx.train_index.detach().cpu().clone(),
                     'KNOWN': torch.cat((full_ctx.train_index, full_ctx.valid_index)).detach().cpu()}
        require(self.rows['TRAIN'].tolist() == list(full_ctx.data.train_ids)
                and full_ctx.valid_index.tolist() == list(full_ctx.data.valid_ids), 'Fixed complete native role order')
        self.row_bindings = {role: digest(dict(role=role, seed=full_ctx.seed, ids=ids.tolist(), role_binding=config.role_binding))
                             for role, ids in self.rows.items()}
        self._row_guard = tuple((role, id(ids), int(ids._version)) for role, ids in self.rows.items())
        self.targets = full_ctx.targets_cuda[self.rows['TRAIN']].to(dtype=torch.float32)
        require(self.targets.shape == (full_ctx.train_count, 5) and torch.isfinite(self.targets).all().item()
                and ((self.targets == 0) | (self.targets == 1)).all().item(), 'All and only complete TRAIN Bernoulli targets')
        self.descriptors = {}
        for family, view in self.views.items():
            require(sha(view.source_supply_descriptor.__func__.__code__.co_filename) == sources['view_source_sha256'], 'Actual exact source-view class implementation')
            require(view.source == family and view.metadata['construction_complete'] is True
                    and view.metadata['full_view_binding'] == config.full_view_binding
                    and view.metadata['role_binding'] == config.role_binding
                    and view.metadata['native_qualification_binding'] == config.native_qualification_binding
                    and view.metadata['source_seal_sha256'] == sources['view_seal_sha256'], 'Actual reviewed source-view binding')
            for name in ('data', 'targets', 'targets_cuda', 'train_index', 'valid_index'):
                require(getattr(view, name) is getattr(full_ctx, name), 'View preserves native role/target identity: '+name)
            require(view.seed == full_ctx.seed and view.train_count == full_ctx.train_count and view.valid_count == full_ctx.valid_count
                    and view.data_size == full_ctx.data_size, 'View preserves full native context')
            for name in ('feats', 'label_feats'):
                factual, probe = getattr(full_ctx, name), getattr(view, name)
                require(tuple(probe) == tuple(factual) and all(probe[k].shape == factual[k].shape for k in factual), 'Complete source namespace/key/shape preservation')
            self.descriptors[family] = view.source_supply_descriptor(helper)
        self._cache_guard = self._cache_stamp()
        with costs.measure('bank_install_and_deduplicated_Adam', gpu=True):
            self.bank = adapter.install_member_bank(rt['model_module'], prototype, adapter.AdapterConfig(enabled=True, members=4))
            self.ownership = helper.verify_ownership(self.bank.members, self.bank.slow_names, self.bank.private_names)
            require(all(p.dtype == torch.float32 and p.device == rt['device'] for p in self.ownership.all_parameters), 'Ordinary native FP32 placement before AMP; no half parameter retyping')
            slow = self.bank.slow_parameters()
            private = tuple(p for m in range(4) for p in self.bank.private_parameters(m))
            require(len({id(p) for p in (*slow, *private)}) == len(slow)+len(private), 'Own Adam deduplicates every slow/fast object')
            self.optimizer = torch.optim.Adam([{'params': slow, 'lr': .001, 'weight_decay': 0},
                                               {'params': private, 'lr': .001, 'weight_decay': 0}])
            self.verify_optimizer()
        self.helper_config = helper.Config(enabled=True, score_mode='bernoulli_marginal_logits')
        self.member_rng = []
        caller = engine.capture_rng(rt['numpy'], torch)
        try:
            for seed in config.member_rng_seeds:
                rt['native'].set_random_seed(seed)
                self.member_rng.append(engine.capture_rng(rt['numpy'], torch))
        finally:
            engine.restore_rng(rt['numpy'], torch, caller)
            require(engine.exact(torch, engine.capture_rng(rt['numpy'], torch), caller), 'Member RNG initialization preserves native loader/master streams')

    def _cache_stamp(self):
        result = []
        for source, ctx in [(None, self.ctx), *self.views.items()]:
            for namespace in ('feats', 'label_feats'):
                mapping = getattr(ctx, namespace)
                result.append((source, namespace, id(mapping), tuple((k, id(v), int(v._version), tuple(v.shape), str(v.dtype), str(v.device)) for k, v in mapping.items())))
        return tuple(result)

    def verify_optimizer(self):
        actual = tuple(p for group in self.optimizer.param_groups for p in group['params'])
        require(len(actual) == len({id(p) for p in actual}) and {id(p) for p in actual} == {id(p) for p in self.ownership.all_parameters}, 'One own Adam state per genuinely owned object')
        require(len(self.optimizer.param_groups) == 2 and all(group['lr'] == .001 and group['weight_decay'] == 0 for group in self.optimizer.param_groups), 'Native Adam lr/zero-decay own policy')
        self.bank.verify_ownership()

    def token(self, member, source, mode, role='TRAIN'):
        require(member in range(4) and source in (None, *FAMILIES) and mode in ('train', 'eval') and role in self.rows, 'Explicit native callback path')
        require(role == 'TRAIN' or (source is None and mode == 'eval'), 'Complete known-role serving is factual eval only')
        binding = self.config.full_view_binding if source is None else self.views[source].binding
        return ReplayToken(member, source, mode, role, self.row_bindings[role], binding,
                           buffer_stamp(self.bank.members[member]), self.engine.cpu_tree(self.rt['torch'], self.member_rng[member]))

    def native_forward(self, member, source, mode, token):
        """Complete fixed row IDs, isolated caches, scratch BN, matched owned RNG.

        TRAIN reference/replay/trial and eval all use FP32, autocast disabled.
        The unchanged helper's unscaled VJPs therefore never traverse half AMP
        layers. The explicit differentiable FP32 score cast is still required.
        Ordinary own updates alone use native CUDA AMP. Neither callback touches
        Adam, its moments or the caller's GradScaler.
        """
        torch, engine = self.rt['torch'], self.engine
        require(member in range(4) and source in (None, *FAMILIES) and mode in ('train', 'eval'), 'Declared native callback path')
        require(type(token) is ReplayToken and (token.member, token.source, token.mode) == (member, source, mode), 'Exact callback replay token')
        require(token.role in self.rows and token.row_binding == self.row_bindings[token.role]
                and tuple((role, id(ids), int(ids._version)) for role, ids in self.rows.items()) == self._row_guard, 'Fixed complete TRAIN/known row binding')
        require(token.role == 'TRAIN' or (source is None and mode == 'eval'), 'Known-role calls are factual eval only')
        ctx = self.ctx if source is None else self.views[source]
        binding = self.config.full_view_binding if source is None else ctx.binding
        require(token.view_binding == binding and self._cache_stamp() == self._cache_guard, 'Unchanged canonical factual/source caches')
        model = self.bank.members[member]
        require(token.buffer_stamp == buffer_stamp(model), 'No hidden native buffer advance between reference/replay/trial')
        parameters, gradients = self.ownership.stamp(), self.ownership.grad_stamp()
        before_cache = self._active_cache
        try:
            ids = self.rows[token.role]
            # Advanced indexing creates detached per-call copies; canonical CPU
            # caches are never exposed to the native forward by reference.
            features = {k: value[ids].to(self.rt['device']) for k, value in ctx.feats.items()}
            labels = {k: value[ids].to(self.rt['device']) for k, value in ctx.label_feats.items()}
            self._active_cache = (source, token.row_binding, features, labels)
            with scratch_native_state(self.rt, engine, model, mode, token.rng), torch.cuda.amp.autocast(enabled=False):
                result = model(ids.to(self.rt['device']), features, labels, None).float()
            require(result.shape == (len(ids), 5) and result.dtype == torch.float32, 'Explicit complete-row FP32 Bernoulli-marginal logits')
            self.counters[self._phase or 'eval_forwards'] += 1
            return result
        finally:
            self._active_cache = before_cache
            require(self._cache_stamp() == self._cache_guard and self._active_cache is before_cache, 'Canonical/staged cache state restored')
            require(self.ownership.stamp() == parameters and self.ownership.grad_stamp() == gradients, 'Native callback preserves current parameters and gradients, including trial factors')
            require(buffer_stamp(model) == token.buffer_stamp, 'Native callback restores prediction-relevant buffer identities/versions')

    def train_epoch(self):
        """Same complete native shuffled batch, streamed mean own BCE, one Adam.

        Each native member BN evolves only here. Every member has its own RNG
        state; the native generator=None loader advances the separate master RNG.
        """
        torch, engine = self.rt['torch'], self.engine
        displacements, losses = None, []
        require(self._cache_stamp() == self._cache_guard, 'Own step uses unchanged full input')
        with self.costs.measure('streamed_M4_mean_native_BCE_own_epoch', gpu=True):
            for batch_number, batch in enumerate(self.ctx.train_loader):
                require(batch_number == 0 and len(batch) == self.ctx.train_count
                        and torch.equal(batch.sort().values.cpu(), self.rows['TRAIN']), 'Exactly one complete native TRAIN minibatch')
                before = [[p.detach().clone() for p in block] for block in self.ownership.private]
                features = {k: value[batch].to(self.rt['device']) for k, value in self.ctx.feats.items()}
                labels = {k: value[batch].to(self.rt['device']) for k, value in self.ctx.label_feats.items()}
                targets = self.ctx.targets_cuda[batch]
                self.optimizer.zero_grad()
                for m, model in enumerate(self.bank.members):
                    caller_rng = engine.capture_rng(self.rt['numpy'], torch)
                    modes = tuple((module, module.training) for module in model.modules())
                    try:
                        engine.restore_rng(self.rt['numpy'], torch, self.member_rng[m])
                        model.train()
                        with torch.cuda.amp.autocast():
                            output = model(batch, features, labels, None)
                            loss = torch.nn.functional.binary_cross_entropy_with_logits(output, targets)
                        require(torch.isfinite(output).all().item() and math.isfinite(float(loss.detach().item())), 'Finite native own five-logit/BCE')
                        self.scalar.scale(loss/4).backward()
                        losses.append(float(loss.detach().item()))
                        self.counters['own_member_forwards'] += 1
                        self.counters['own_backwards'] += 1
                        self.member_rng[m] = engine.capture_rng(self.rt['numpy'], torch)
                    finally:
                        for module, training in modes:
                            module.training = training
                        engine.restore_rng(self.rt['numpy'], torch, caller_rng)
                        require(engine.exact(torch, engine.capture_rng(self.rt['numpy'], torch), caller_rng), 'Owned member forward preserves loader/master RNG')
                    del output, loss
                # GradScaler owns finite overflow/skip behavior. Observe actual
                # optimizer.step, rather than counting every scaler.step request.
                self.scalar.unscale_(self.optimizer)
                step = self.optimizer.step
                def observed_step(*args, **kwargs):
                    require(all(p.grad is None or torch.isfinite(p.grad).all().item() for p in self.ownership.all_parameters), 'Finite actual unscaled Adam gradients')
                    result = step(*args, **kwargs)
                    self.counters['actual_Adam_steps'] += 1
                    return result
                self.optimizer.step = observed_step
                try:
                    self.scalar.step(self.optimizer)
                    self.scalar.update()
                finally:
                    self.optimizer.step = step
                displacements = {m: tuple(p.detach().clone()-old for p, old in zip(block, before[m]))
                                 for m, block in enumerate(self.ownership.private)}
                del before, features, labels, targets
        require(displacements is not None and len(losses) == 4, 'One complete native own epoch')
        self.counters['own_epochs'] += 1
        self.verify_optimizer()
        return dict(mean_native_own_BCE=sum(losses)/4, member_native_own_BCE=losses, own_displacements=displacements)

    def correct(self, own_displacements):
        """Concrete unchanged-helper reference, streamed VJPs and finite trials."""
        require(self.config.source_corrections is True, 'Source correction explicitly enabled in reviewed policy')
        torch, engine, helper = self.rt['torch'], self.engine, self.helper
        with self.costs.measure('complete_TRAIN_source_reference_replay_cone_trials_and_guards', gpu=True), torch.cuda.amp.autocast(enabled=False):
            adam_before = engine.cpu_tree(torch, self.optimizer.state_dict())
            scalar_before = engine.cpu_tree(torch, self.scalar.state_dict())
            slow_before = parameter_stamp(self.ownership.shared)
            grads_before = self.ownership.grad_stamp()
            rng_before = engine.cpu_tree(torch, self.member_rng)
            global_before = engine.capture_rng(self.rt['numpy'], torch)
            try:
                keys = [helper.OutputKey('train_full', m) for m in range(4)]
                keys += [helper.OutputKey('train_probe', m, a) for a in FAMILIES for m in range(4)]
                keys += [helper.OutputKey('eval_full', m) for m in range(4)]
                tokens = {key: self.token(key.member, key.source, 'eval' if key.kind == 'eval_full' else 'train') for key in keys}
                self._phase = 'source_reference_forwards'
                with torch.no_grad():
                    outputs = {key: self.native_forward(key.member, key.source, tokens[key].mode, tokens[key]).detach().clone() for key in keys}
                plan = helper.make_credit_plan(self.helper_config, self.ownership, self.config.assignments,
                    self.descriptors, self.config.full_view_binding, outputs, self.targets, tokens, target_role='TRAIN')
                self._phase = 'source_replay_forwards'
                with torch.enable_grad():
                    for key in plan.replay_keys():
                        output = self.native_forward(key.member, key.source, tokens[key].mode, tokens[key])
                        helper.accumulate_replay(plan, key, output)
                        del output
                direction = helper.build_private_direction(plan, own_displacements)
                decisions, accepted_scale = [], None
                self._phase = 'source_trial_forwards'
                if not direction.zero:
                    for scale in TRIAL_SCALES:
                        decision = helper.try_private_step(plan, direction, scale, self.native_forward)
                        decisions.append(dict(scale=scale, accepted=decision.accepted, violations=list(decision.violations), values=decision.values))
                        if decision.accepted:
                            accepted_scale = scale
                            break
                status = 'zero' if direction.zero else ('accepted' if accepted_scale is not None else 'rejected')
                dose = math.sqrt(sum(float((p.detach().double()-old.double()).square().sum().item())
                                     for m in plan.active() for p, old in zip(self.ownership.private[m], plan.origin_private[m])))
                require(math.isfinite(dose) and (status == 'accepted' or dose == 0), 'Exact zero/rejected private dose')
                plan.consumed = True
                return dict(status=status, accepted_scale=accepted_scale, private_L2_dose=dose,
                            g_dot_d=direction.g_dot_d, reference_values=plan.values, trials=decisions,
                            complete_TRAIN_rows=self.ctx.train_count, row_binding=self.row_bindings['TRAIN'],
                            source_score_precision='native_FP32_TRAIN_reference_replay_trial_autocast_disabled', strict_guard_atol=0.0)
            finally:
                self._phase = None
                require(engine.exact(torch, self.optimizer.state_dict(), adam_before), 'Private correction preserves own Adam moments and groups exactly')
                require(engine.exact(torch, self.scalar.state_dict(), scalar_before), 'Private correction preserves caller master scaler')
                require(parameter_stamp(self.ownership.shared) == slow_before and self.ownership.grad_stamp() == grads_before, 'Private correction preserves shared parameters and all grad fields')
                require(engine.exact(torch, self.member_rng, rng_before)
                        and engine.exact(torch, engine.capture_rng(self.rt['numpy'], torch), global_before), 'Source calls do not advance any own/master stream')
                self.bank.verify_ownership()

    def evaluate(self):
        """Complete TRAIN/VALID factual-only FP32 marginal serving and selector."""
        torch, helper = self.rt['torch'], self.helper
        with self.costs.measure('full_input_complete_TRAIN_VALID_FP32_pool_evaluation', gpu=True), torch.cuda.amp.autocast(enabled=False):
            tokens = [self.token(m, None, 'eval', 'KNOWN') for m in range(4)]
            probabilities, scores = helper.serve_full_input(self.helper_config, self.ownership, self.native_forward, tokens,
                                                           full_view_binding=self.config.full_view_binding)
            result, role_outputs = {}, {}
            for role, index, start, end in [('TRAIN', self.ctx.train_index, 0, self.ctx.train_count),
                                           ('VALID', self.ctx.valid_index, self.ctx.train_count, self.ctx.train_count+self.ctx.valid_count)]:
                truth = self.ctx.targets_cuda[index].float()
                logits = tuple(value[start:end] for value in scores)
                current = probabilities[start:end]
                # Stable marginal BCE of the probability mean, one observed
                # label entry at a time, matching the frozen helper score.
                observed = [torch.where(truth.bool(), torch.nn.functional.logsigmoid(value),
                                         torch.nn.functional.logsigmoid(-value)) for value in logits]
                marginal_bce = -(torch.logsumexp(torch.stack(observed), dim=0)-math.log(4)).mean()
                member_bce = [float(torch.nn.functional.binary_cross_entropy_with_logits(value, truth).item()) for value in logits]
                micro, macro = self.rt['native'].evaluator(truth, (current > .5).int())
                require(math.isfinite(float(marginal_bce)) and all(math.isfinite(v) for v in member_bce), 'Finite complete factual selector/own risk')
                result[role] = dict(BCE=float(marginal_bce), member_BCE=member_bce, micro_F1=float(micro), macro_F1=float(macro))
                role_outputs[role] = dict(probabilities=current.detach().cpu().clone(), member_logits=tuple(v.detach().cpu().clone() for v in logits))
            return result, role_outputs

    def snapshot(self, epoch, scores, outputs, identity):
        torch, engine = self.rt['torch'], self.engine
        identity = plain_metadata(identity)
        with self.costs.measure('selected_bank_CPU_state_copy', gpu=True):
            saved = dict(schema='owned-SeHGNN-M4-bank-selected-state-v1', identity=identity, seed=self.ctx.seed, epoch=epoch,
                        role_binding=self.config.role_binding, full_view_binding=self.config.full_view_binding,
                        source_seal_sha256=self.config.source_seal_sha256, native_qualification_binding=self.config.native_qualification_binding,
                        view_bindings={a: view.binding for a, view in self.views.items()},
                        TRAIN_row_binding=self.row_bindings['TRAIN'], known_row_binding=self.row_bindings['KNOWN'],
                        member_rng_seeds=list(self.config.member_rng_seeds), assignments=list(self.config.assignments),
                        source_corrections=self.config.source_corrections,
                        bank_state=engine.cpu_tree(torch, self.bank.state_dict()), optimizer_state=engine.cpu_tree(torch, self.optimizer.state_dict()),
                        scaler_state=engine.cpu_tree(torch, self.scalar.state_dict()), member_rng=engine.cpu_tree(torch, self.member_rng),
                        master_rng=engine.capture_rng(self.rt['numpy'], torch), scores=scores, outputs=outputs, known_roles_only=True)
            return checkpoint_tree(torch, saved)

    def restore_selected(self, saved, identity):
        torch, engine = self.rt['torch'], self.engine
        identity = plain_metadata(identity)
        with self.costs.measure('selected_bank_shared_alias_preflight_and_restore', gpu=True):
            require(saved['schema'] == 'owned-SeHGNN-M4-bank-selected-state-v1' and saved['identity'] == identity
                    and saved['seed'] == self.ctx.seed and saved['role_binding'] == self.config.role_binding
                    and saved['full_view_binding'] == self.config.full_view_binding
                    and saved['source_seal_sha256'] == self.config.source_seal_sha256
                    and saved['native_qualification_binding'] == self.config.native_qualification_binding
                    and saved['view_bindings'] == {a: view.binding for a, view in self.views.items()}
                    and saved['TRAIN_row_binding'] == self.row_bindings['TRAIN'] and saved['known_row_binding'] == self.row_bindings['KNOWN']
                    and saved['member_rng_seeds'] == list(self.config.member_rng_seeds) and saved['assignments'] == list(self.config.assignments)
                    and saved['source_corrections'] == self.config.source_corrections, 'Exact selected owned bank identity/roles/policy')
            self.bank.verify_shared_state_dict(saved['bank_state'])
            self.bank.load_state_dict(saved['bank_state'], strict=True)
            self.optimizer.load_state_dict(saved['optimizer_state'])
            self.scalar.load_state_dict(saved['scaler_state'])
            self.member_rng = engine.cpu_tree(torch, saved['member_rng'])
            require(engine.exact(torch, engine.cpu_tree(torch, self.bank.state_dict()), saved['bank_state'])
                    and engine.exact(torch, engine.cpu_tree(torch, self.optimizer.state_dict()), saved['optimizer_state'])
                    and engine.exact(torch, self.scalar.state_dict(), saved['scaler_state'])
                    and engine.exact(torch, self.member_rng, saved['member_rng']), 'Exact selected bank/buffers/Adam/scaler/member streams')
            engine.restore_rng(self.rt['numpy'], torch, saved['master_rng'])
            require(engine.exact(torch, engine.capture_rng(self.rt['numpy'], torch), saved['master_rng']), 'Exact selected master RNG')
            self.verify_optimizer()


def fit_bank(session, identity):
    """Concrete native 200-epoch/strict complete VALID BCE/50-patience fit.

    Uses the bank's factual marginal pool for selection. No adaptive objective,
    seed schedule, data load, output writer, launch or scientific admission.
    """
    require(isinstance(identity, dict) and bool(identity), 'Explicit owned fit identity')
    identity = plain_metadata(identity)
    best_epoch, best_loss, selected, history = -1, 1000000.0, None, []
    for epoch in range(200):
        with session.costs.measure('native_pre_epoch_GC'):
            gc.collect()
        own = session.train_epoch()
        correction = session.correct(own.pop('own_displacements')) if session.config.source_corrections else dict(status='disabled', private_L2_dose=0.0)
        own.pop('own_displacements', None)
        with session.costs.measure('native_post_TRAIN_cache_cleanup'):
            session.rt['torch'].cuda.empty_cache()
        scores, outputs = session.evaluate()
        improved = scores['VALID']['BCE'] < best_loss
        if improved:
            best_epoch, best_loss = epoch, scores['VALID']['BCE']
            selected = session.snapshot(epoch, scores, outputs, identity)
        history.append(dict(epoch=epoch, own=own, correction=correction, scores=scores, selected=improved, counters=dict(session.counters)))
        if epoch-best_epoch > 50:
            break
    require(selected is not None, 'Complete native bank selected state exists')
    return dict(selected_state=selected, history=history, selected_epoch=best_epoch, selected_VALID_BCE=best_loss,
                counters=dict(session.counters), selection='strict complete factual VALID marginal BCE; earliest ties; epoch-best_epoch>50',
                launch_or_pilot_authorized=False)


def reconstruct_selected(rt, engine, adapter, helper, full_ctx, views, costs, saved, identity, config=BankConfig()):
    """Fresh exact native constructor/install/Adam/scaler, factual-only serving.

    The caller's master scaler is never received. Caller global RNG is restored.
    """
    config.require_enabled()
    caller = engine.capture_rng(rt['numpy'], rt['torch'])
    session = prototype = None
    try:
        prototype = engine.make_model(rt, full_ctx, costs)
        fresh_scalar = rt['torch'].cuda.amp.GradScaler()
        session = BankSession(rt, engine, adapter, helper, full_ctx, views, prototype, fresh_scalar, costs, config)
        prototype = None
        session.restore_selected(saved, identity)
        scores, outputs = session.evaluate()
        diagnostics = {}
        for role in ('TRAIN', 'VALID'):
            current, previous = outputs[role], saved['outputs'][role]
            diagnostics[role] = dict(max_abs_probability=float((current['probabilities']-previous['probabilities']).abs().max().item()),
                max_abs_member_logit=max(float((x-y).abs().max().item()) for x, y in zip(current['member_logits'], previous['member_logits'])),
                prediction_changes=int(((current['probabilities']>.5)!=(previous['probabilities']>.5)).sum().item()))
        return dict(epoch=saved['epoch'], scores=scores, recorded_scores=saved['scores'], replay_diagnostics=diagnostics,
                    exact_state_restore=True, output_bitwise_gate=False, known_roles_only=True)
    finally:
        session = prototype = None
        gc.collect()
        rt['torch'].cuda.empty_cache()
        engine.restore_rng(rt['numpy'], rt['torch'], caller)
        require(engine.exact(rt['torch'], engine.capture_rng(rt['numpy'], rt['torch']), caller), 'Fresh selected reconstruction preserves caller streams')
