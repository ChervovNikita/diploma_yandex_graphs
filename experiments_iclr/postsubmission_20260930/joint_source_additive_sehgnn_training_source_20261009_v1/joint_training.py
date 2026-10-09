"""Disabled ten-condition native joint P/PS source. No provider imports/launcher."""
from dataclasses import dataclass
import gc
import json
from pathlib import Path
import resource
import sys
import time

from joint_contracts import (ASSIGNMENTS, Config, FAMILIES, FAST_RANK1_PER_MEMBER,
    NATIVE_SCALARS, SPECS, VIEWS, read, require, sha, source_gate, validate_modules)
from joint_additive_adapter import install, shared_additive


@dataclass(frozen=True)
class Token:
    member: int
    source: str | None
    mode: str
    role: str
    rng: object


def unique(values):
    result, seen = [], set()
    for value in values:
        if id(value) not in seen:
            seen.add(id(value)); result.append(value)
    return tuple(result)


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def prepare_context(rt, modules, input_root, role_file, costs, config=Config()):
    """Optional exact native seam; setup is charged, no alternate input names."""
    config.require_enabled()
    sources = source_gate(); validate_modules(rt, modules, sources)
    engine, loader, producer = (modules[key] for key in ('engine', 'role_loader', 'source_views'))
    role_file = Path(role_file)
    require(role_file.is_absolute() and not role_file.is_symlink() and sha(role_file) == config.role_binding, 'Exact original frozen role file')
    role = read(role_file); require(role['seed'] == config.base_seed, 'Body seeds never resplit the outer role')
    loader.source_gate()
    with costs.measure('exact_role_loader_and_complete_native_setup_and_all_three_raw_views', gpu=True):
        data = loader.load(Path(input_root), role_file, read(Path(loader.__file__).parent / 'SOURCE_EXPECTATIONS.json'))
        require(data.input_bindings == role['input_files'], 'Exact full input hash bindings')
        static = engine.prepare_static(rt, data, costs)
        ctx = engine.prepare_seed(rt, static, role, role['seed'], costs)
        binding = modules['metadata'].digest(dict(study=config.study_binding, role_binding=config.role_binding,
            seed=ctx.seed, inputs=data.input_bindings, feature_shapes=static.feature_shapes,
            label_shapes={key: list(value.shape) for key, value in ctx.label_feats.items()}))
        require(binding == config.full_view_binding, 'Prospective full native context digest')
        producer.source_gate()
        views = producer.build_family_views(rt, static, ctx, costs, producer.ViewConfig(enabled=True,
            root_source_review_approved=True, native_qualification_binding=config.native_qualification_binding,
            full_view_binding=binding, role_binding=config.role_binding, source_seal_sha256=sources['source_view_seal_sha256']))
    return ctx, views


class Session:
    def __init__(self, rt, modules, ctx, views, costs, config=Config()):
        config.require_enabled()
        self.rt = rt
        self.group, self.models, self.ownership, self.optimizers = None, (), None, []
        self._allocation_started = False
        try:
            self._initialize(rt, modules, ctx, views, costs, config)
        except BaseException:
            if self._allocation_started:
                self.release_models()
            raise

    def _initialize(self, rt, modules, ctx, views, costs, config):
        sources = source_gate(); validate_modules(rt, modules, sources)
        self.rt, self.modules, self.ctx, self.views, self.costs, self.config = rt, modules, ctx, dict(views), costs, config
        self.engine, self.adapter, self.metadata, self.helper = (modules[key] for key in ('engine', 'adapter', 'metadata', 'helper'))
        self.spec = SPECS[config.condition]
        self.kind, self.parameterization, self.rank, self.credit, self.copied = self.spec
        self.count = 1 if self.kind == 'single' else 4
        torch = rt['torch']
        require(rt['device'].type == 'cuda' and ctx.seed == config.base_seed, 'Qualified complete native CUDA role')
        require(set(views) == set(FAMILIES) and len(ctx.feats) == 25 and len(ctx.label_feats) == 12, 'All original channels and three rebuilt raw-family views')
        require(0 < ctx.train_count <= 10000 and ctx.train_index.tolist() == list(ctx.data.train_ids)
                and ctx.valid_index.tolist() == list(ctx.data.valid_ids), 'Complete original TRAIN/VALID role rows')
        for family, view in self.views.items():
            require(sha(view.source_supply_descriptor.__func__.__code__.co_filename) == sources['modules']['source_views']['sha256'], 'Exact real source-view implementation')
            require(view.source == family and view.metadata['construction_complete'] is True
                    and view.metadata['full_view_binding'] == config.full_view_binding
                    and view.metadata['role_binding'] == config.role_binding
                    and view.metadata['native_qualification_binding'] == config.native_qualification_binding
                    and view.metadata['source_seal_sha256'] == sources['source_view_seal_sha256'], 'Exact native source-view provenance')
            for name in ('data', 'targets', 'targets_cuda', 'train_index', 'valid_index'):
                require(getattr(view, name) is getattr(ctx, name), 'Source view preserves actual role/target object')
            require(view.seed == ctx.seed and view.train_count == ctx.train_count
                    and view.valid_count == ctx.valid_count and view.data_size == ctx.data_size, 'Source view preserves complete native context')
            for name in ('feats', 'label_feats'):
                require(tuple(getattr(view, name)) == tuple(getattr(ctx, name))
                        and all(getattr(view, name)[key].shape == value.shape for key, value in getattr(ctx, name).items()), 'Full namespace and shape preservation')
        self.rows = {'TRAIN': ctx.train_index.detach().cpu().clone(),
                     'KNOWN': torch.cat((ctx.train_index, ctx.valid_index)).detach().cpu()}
        self.targets = ctx.targets_cuda[self.rows['TRAIN']].float()
        require(self.targets.shape == (ctx.train_count, 5) and torch.isfinite(self.targets).all().item()
                and ((self.targets == 0) | (self.targets == 1)).all().item(), 'Only complete hard TRAIN Bernoulli targets')
        self.cache_guard = self.cache_stamp()
        self.row_guard = tuple((name, id(ids), int(ids._version)) for name, ids in self.rows.items())
        self.counters = dict(epochs_completed=0, reference_forwards=0, gradient_replays=0, P_backwards=0,
            S_private_VJPs=0, factual_eval_forwards=0, stopped_peer_forwards=0, actual_Adam_steps=0)
        self.body_counters = [dict(actual_Adam_steps=0, supervised_epochs=0, peer_only_epochs=0) for _ in range(self.count)]
        caller = self.engine.capture_rng(rt['numpy'], torch)
        self._allocation_started = True
        try:
            with costs.measure('fresh_complete_native_condition_models_adapters_and_Adam', gpu=True):
                if self.kind == 'shared':
                    rt['native'].set_random_seed(config.body_seeds[0])
                    prototype = self.engine.make_model(rt, ctx, costs)
                    if self.parameterization == 'be':
                        self.group = self.adapter.install_member_bank(rt['model_module'], prototype, self.adapter.AdapterConfig(enabled=True, members=4))
                    else:
                        self.group = shared_additive(rt, self.adapter, prototype, 1, config.adapter_seeds, self.copied)
                    self.models = tuple(self.group.members)
                    self.slow_names, self.private_names = self.group.slow_names, self.group.private_names
                    self.ownership = self.helper.verify_ownership(self.models, self.slow_names, self.private_names)
                else:
                    models = []
                    for member in range(self.count):
                        rt['native'].set_random_seed(config.body_seeds[member])
                        model = self.engine.make_model(rt, ctx, costs)
                        if self.parameterization == 'add':
                            seeds = config.adapter_seeds if self.rank == 4 else (config.adapter_seeds[member],)
                            model, slow_names, private_names = install(rt['model_module'], self.adapter, model, self.rank, seeds)
                        else:
                            slow_names, private_names = tuple(dict(model.named_parameters(remove_duplicate=False))), ()
                        models.append(model)
                    self.models = tuple(models)
                    self.slow_names, self.private_names = slow_names, private_names
                    self.group = torch.nn.ModuleList(models)
                    if self.kind == 'independent':
                        self.ownership = self.helper.verify_ownership(self.models, (), private_names, slow_names, require_shared=False)
                    else:
                        named = dict(models[0].named_parameters(remove_duplicate=False))
                        slow, private = tuple(named[name] for name in slow_names), tuple(named[name] for name in private_names)
                        require(len(unique((*slow, *private))) == len(slow) + len(private), 'Single owns every native/adapter tensor once')
                        self.ownership = self.helper.Ownership(slow, (private,), (*slow, *private), ((),))
                require(all(not tuple(model.named_buffers()) for model in self.models), 'Pinned IMDB uses stateless normalization; mutable-buffer peer behavior is not guessed')
                require(all(parameter.dtype == torch.float32 and parameter.device == rt['device'] for parameter in self.ownership.all_parameters), 'Original full native FP32 parameters on qualified device')
                self.optimizers = []
                owners = [tuple(self.ownership.all_parameters)] if self.kind != 'independent' else [tuple(model.parameters()) for model in self.models]
                for owner in owners:
                    private_ids = {id(value) for block in self.ownership.private for value in block}
                    slow = tuple(value for value in owner if id(value) not in private_ids)
                    private = tuple(value for value in owner if id(value) in private_ids)
                    groups = [dict(params=slow, lr=.001, weight_decay=0, ownership='native')]
                    if private:
                        groups.append(dict(params=private, lr=.001, weight_decay=0, ownership='private'))
                    self.optimizers.append(torch.optim.Adam(groups, lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0, amsgrad=False))
        finally:
            self.engine.restore_rng(rt['numpy'], torch, caller)
        self.member_rng = []
        caller = self.engine.capture_rng(rt['numpy'], torch)
        try:
            for seed in config.member_rng_seeds:
                rt['native'].set_random_seed(seed)
                self.member_rng.append(self.engine.capture_rng(rt['numpy'], torch))
        finally:
            self.engine.restore_rng(rt['numpy'], torch, caller)
        self.score_config = self.helper.Config(enabled=True, score_mode='bernoulli_marginal_logits')
        self.active = [True] * self.count
        self.counts = self.parameter_report()
        self.verify_optimizers()

    def parameter_report(self):
        private = [sum(value.numel() for value in block) for block in self.ownership.private]
        expected = 0 if self.parameterization == 'native' else FAST_RANK1_PER_MEMBER * self.rank
        require(private == [expected] * self.count, 'Exact private scalar count in every complete path')
        native = [sum(value.numel() for name, value in model.named_parameters() if name in self.slow_names) for model in self.models]
        require(native == [NATIVE_SCALARS] * self.count, 'Every body retains all complete native scalars')
        total = sum(value.numel() for value in self.ownership.all_parameters)
        require(total == NATIVE_SCALARS * (self.count if self.kind == 'independent' else 1) + sum(private), 'Actual deduplicated parameter budget')
        return dict(native_scalars_per_body=native, private_scalars_per_body=private, actual_owned_total_scalars=total,
            parameter_objects=len(self.ownership.all_parameters), ownership=self.kind, rank=self.rank,
            private_penalty='none', native_optimizer_decay=0, private_optimizer_decay=0,
            all_six_sites=list(self.adapter.SITES) if self.parameterization != 'native' else [],
            per_site_private_scalars=[{site: sum(value.numel() for name, value in model.named_parameters() if name in (site + '.v', site + '.u', site + '.input_factor', site + '.output_factor')) for site in self.adapter.SITES} for model in self.models])

    def verify_optimizers(self):
        owned = tuple(value for optimizer in self.optimizers for group in optimizer.param_groups for value in group['params'])
        require(len(owned) == len(unique(owned)) and {id(value) for value in owned} == {id(value) for value in self.ownership.all_parameters}, 'Exactly one Adam/moment owner per deduplicated tensor')
        require(all(group['lr'] == .001 and group['weight_decay'] == 0 and group['eps'] == 1e-8 and group['betas'] == (.9, .999) and not group['amsgrad'] for optimizer in self.optimizers for group in optimizer.param_groups), 'Explicit native Adam and zero private penalty; no inherited identity prior')

    def cache_stamp(self):
        return tuple((source, name, id(mapping), tuple((key, id(value), int(value._version), tuple(value.shape), str(value.dtype), str(value.device)) for key, value in mapping.items()))
            for source, context in [(None, self.ctx), *self.views.items()] for name in ('feats', 'label_feats') for mapping in (getattr(context, name),))

    def token(self, member, source, mode, role='TRAIN'):
        return Token(member, source, mode, role, self.engine.cpu_tree(self.rt['torch'], self.member_rng[member]))

    def forward(self, token, ids=None):
        torch = self.rt['torch']
        require(token.member in range(self.count) and token.source in VIEWS and token.mode in ('train', 'eval'), 'Only complete native factual/raw-source paths')
        require(token.role in self.rows and (token.role == 'TRAIN' or token.source is None), 'Unknown/TEST role access refused')
        require(self.cache_stamp() == self.cache_guard and tuple((name, id(rows), int(rows._version)) for name, rows in self.rows.items()) == self.row_guard, 'Fixed exact input caches/role IDs')
        ids = self.rows[token.role] if ids is None else ids
        context = self.ctx if token.source is None else self.views[token.source]
        model = self.models[token.member]
        stamp, gradients = self.ownership.stamp(), self.ownership.grad_stamp()
        features = {key: value[ids].to(self.rt['device']) for key, value in context.feats.items()}
        labels = {key: value[ids].to(self.rt['device']) for key, value in context.label_feats.items()}
        with self.metadata.scratch_native_state(self.rt, self.engine, model, token.mode, token.rng), torch.cuda.amp.autocast(enabled=False):
            output = model(ids.to(self.rt['device']), features, labels, None).float()
            require(not tuple(model.named_buffers()), 'Unexpected mutable native buffers are refused')
            after_rng = self.engine.capture_rng(self.rt['numpy'], torch)
        require(output.shape == (len(ids), 5) and output.dtype == torch.float32 and torch.isfinite(output).all().item(), 'Finite complete native five-logit FP32 output')
        require(self.ownership.stamp() == stamp and self.ownership.grad_stamp() == gradients and self.cache_stamp() == self.cache_guard, 'Forward changes no parameters/gradients/canonical caches')
        return output, after_rng

    def train_epoch(self):
        """One old-state joint step; all refs then one live replay graph at a time."""
        self.config.require_enabled(); self.verify_optimizers()
        torch = self.rt['torch']
        require(any(self.active), 'No step after every own body stopped')
        reference, tokens, pending = {}, {}, list(self.member_rng)
        reference_losses, losses, credit_losses, drift = {}, {}, {}, {}
        stamp = self.ownership.stamp()
        with self.costs.measure('complete_four_view_references_and_same_state_gradient_replays', gpu=True):
            batches = 0
            for batch in self.ctx.train_loader:
                batches += 1
                require(batches == 1 and len(batch) == self.ctx.train_count and torch.equal(batch.sort().values.cpu(), self.rows['TRAIN']), 'One complete native shuffled TRAIN minibatch')
                truth = self.ctx.targets_cuda[batch].float()
                for owner, optimizer in enumerate(self.optimizers):
                    if self.kind != 'independent' or self.active[owner]:
                        optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    for source in VIEWS:
                        for member in range(self.count):
                            token = Token(member, source, 'train', 'TRAIN', pending[member])
                            output, after = self.forward(token, batch)
                            tokens[member, source] = token
                            reference[member, source] = output.detach().clone()
                            reference_losses[str((member, source))] = float(-self.helper._observed_logp(output, truth, self.score_config).mean().item())
                            pending[member] = after
                            self.counters['reference_forwards'] += 1
                            if not self.active[member]:
                                self.counters['stopped_peer_forwards'] += 1
                require(self.ownership.stamp() == stamp, 'Every reference precedes every optimizer step at one old state')
                scale = 4 if self.kind == 'independent' else 1
                for source in VIEWS:
                    for member in range(self.count):
                        if not self.active[member]:
                            continue
                        output, _ = self.forward(tokens[member, source], batch)
                        self.counters['gradient_replays'] += 1
                        drift[str((member, source))] = float((output.detach() - reference[member, source]).abs().max().item())
                        own = -self.helper._observed_logp(output, truth, self.score_config).mean()
                        losses[str((member, source))] = float(own.detach().item())
                        extra_grad = None
                        if source is None and self.credit != 'none' and member < 3:
                            families = (ASSIGNMENTS[member],) if self.credit == 'assigned' else FAMILIES
                            terms = []
                            for family in families:
                                observed = [self.helper._observed_logp(output, truth, self.score_config)]
                                observed += [self.helper._observed_logp(reference[peer, family], truth, self.score_config).detach() for peer in range(4) if peer != member]
                                term = self.helper._mean_pool_nll(observed, self.score_config)
                                terms.append(term)
                                credit_losses[str((member, family))] = float(term.detach().item())
                            credit = sum(terms) / len(terms)
                            block = self.ownership.private[member]
                            extra_grad = torch.autograd.grad(credit * scale / 3, block, retain_graph=True, allow_unused=True)
                            self.counters['S_private_VJPs'] += 1
                        (own * scale / (4 * self.count)).backward()
                        self.counters['P_backwards'] += 1
                        if extra_grad is not None:
                            for parameter, gradient in zip(self.ownership.private[member], extra_grad):
                                if gradient is not None:
                                    require(torch.isfinite(gradient).all().item(), 'Finite positive contextual private gradient')
                                    if parameter.grad is None:
                                        parameter.grad = gradient.detach().clone()
                                    else:
                                        parameter.grad.add_(gradient)
                            del credit, terms, observed
                        del output, own, extra_grad
                require(self.ownership.stamp() == stamp, 'All P/S gradient contributions at unchanged old parameters')
                require(all(parameter.grad is None or torch.isfinite(parameter.grad).all().item() for parameter in self.ownership.all_parameters), 'Finite actual unscaled FP32 Adam gradients')
                for owner, optimizer in enumerate(self.optimizers):
                    if self.kind == 'independent' and not self.active[owner]:
                        require(all(parameter.grad is None for group in optimizer.param_groups for parameter in group['params']), 'Stopped bodies receive no gradient or optimizer update')
                        continue
                    optimizer.step()
                    self.counters['actual_Adam_steps'] += 1
                    if self.kind == 'independent':
                        self.body_counters[owner]['actual_Adam_steps'] += 1
                self.member_rng = [self.engine.cpu_tree(torch, rng) for rng in pending]
            require(batches == 1, 'Complete literal one-batch native epoch')
        self.counters['epochs_completed'] += 1
        for member in range(self.count):
            self.body_counters[member]['supervised_epochs' if self.active[member] else 'peer_only_epochs'] += 1
        self.verify_optimizers()
        return dict(P_reference_own_BCE_by_member_view=reference_losses,
            P_reference_complete_mean=sum(reference_losses.values()) / (4 * self.count),
            P_replay_active_own_BCE_by_member_view=losses, S_BCE_by_recipient_source=credit_losses,
            replay_max_abs_logit_drift=drift, active_members=list(self.active), gradients=dict(shared='P only',
            private='P+(1/3)S; independent complete body gradients multiplied by4'),
            replay_drift_is_observation_not_acceptance_gate=True,
            old_guards_or_negative_absence_BCE=False, all_source_views_paid=True)

    def native_forward(self, member, source, mode, token):
        require((token.member, token.source, token.mode) == (member, source, mode), 'Exact serving token')
        return self.forward(token)[0]

    def evaluate(self):
        require(self.config.intent == 'scientific_training', 'Qualification does not calculate VALID quality or select outcomes')
        torch = self.rt['torch']
        with self.costs.measure('complete_factual_FP32_TRAIN_VALID_serving', gpu=True):
            tokens = [self.token(member, None, 'eval', 'KNOWN') for member in range(self.count)]
            probabilities, logits = self.helper.serve_full_input(self.score_config, self.ownership, self.native_forward, tokens, full_view_binding=self.config.full_view_binding)
            self.counters['factual_eval_forwards'] += self.count
            scores, outputs = {}, {}
            for role, ids, start, end in (('TRAIN', self.ctx.train_index, 0, self.ctx.train_count),
                                       ('VALID', self.ctx.valid_index, self.ctx.train_count, self.ctx.train_count + self.ctx.valid_count)):
                truth = self.ctx.targets_cuda[ids].float()
                values = tuple(value[start:end] for value in logits)
                observed = [self.helper._observed_logp(value, truth, self.score_config) for value in values]
                metrics = self.modules['diagnostics'].f1_scores(torch, probabilities[start:end], truth)
                scores[role] = dict(BCE=float(self.helper._mean_pool_nll(observed, self.score_config).item()),
                    member_BCE=[float(-value.mean().item()) for value in observed], **metrics)
                outputs[role] = dict(probabilities=probabilities[start:end].cpu().clone(), member_logits=tuple(value.cpu().clone() for value in values))
            return scores, outputs

    def snapshot(self, epoch, identity, member=None):
        torch = self.rt['torch']
        with self.costs.measure('owned_selected_CPU_state_copy', gpu=True):
            state = dict(schema='joint-proper-source-owned-selected-v1', identity=self.metadata.plain_metadata(identity),
                condition=self.config.condition, epoch=epoch, study_binding=self.config.study_binding,
                role_binding=self.config.role_binding, full_view_binding=self.config.full_view_binding,
                source_seal_sha256=self.config.source_seal_sha256, counts=self.counts,
                initialization=dict(body_seeds=self.config.body_seeds, adapter_seeds=self.config.adapter_seeds,
                    member_rng_seeds=self.config.member_rng_seeds),
                normalization='body times4' if self.kind == 'independent' else 'literal committee or single P',
                member=member, private_decay=0, native_decay=0, buffers_stateless=True)
            if member is None:
                state.update(models_state=self.engine.cpu_tree(torch, self.group.state_dict()),
                    optimizer_states=self.engine.cpu_tree(torch, [optimizer.state_dict() for optimizer in self.optimizers]),
                    rng=self.engine.cpu_tree(torch, self.member_rng))
            else:
                state.update(model_state=self.engine.cpu_tree(torch, self.models[member].state_dict()),
                    optimizer_state=self.engine.cpu_tree(torch, self.optimizers[member].state_dict()),
                    rng=self.engine.cpu_tree(torch, self.member_rng[member]))
            return self.metadata.checkpoint_tree(torch, state)

    def restore(self, selected, identity, member=None):
        require(selected['identity'] == self.metadata.plain_metadata(identity) and selected['condition'] == self.config.condition
                and selected['study_binding'] == self.config.study_binding and selected['role_binding'] == self.config.role_binding
                and selected['full_view_binding'] == self.config.full_view_binding and selected['source_seal_sha256'] == self.config.source_seal_sha256
                and selected['member'] == member and selected['counts'] == self.counts
                and selected['initialization'] == dict(body_seeds=self.config.body_seeds, adapter_seeds=self.config.adapter_seeds,
                    member_rng_seeds=self.config.member_rng_seeds), 'Exact own selected checkpoint identity/budget/initialization')
        torch = self.rt['torch']
        with self.costs.measure('owned_selected_state_restore', gpu=True):
            if member is None:
                require(len(selected['optimizer_states']) == len(self.optimizers)
                        and len(selected['rng']) == self.count, 'Complete selected optimizer/RNG owners')
                if self.kind == 'shared':
                    self.group.verify_shared_state_dict(selected['models_state'])
                self.group.load_state_dict(selected['models_state'], strict=True)
                for optimizer, saved in zip(self.optimizers, selected['optimizer_states']):
                    optimizer.load_state_dict(saved)
                self.member_rng = self.engine.cpu_tree(torch, selected['rng'])
                require(self.engine.exact(torch, self.engine.cpu_tree(torch, self.group.state_dict()), selected['models_state']), 'Exact native/adapter/buffer selected state')
                require(self.engine.exact(torch, self.engine.cpu_tree(torch, [optimizer.state_dict() for optimizer in self.optimizers]), selected['optimizer_states'])
                        and self.engine.exact(torch, self.member_rng, selected['rng']), 'Exact selected Adam and owned RNG state')
            else:
                self.models[member].load_state_dict(selected['model_state'], strict=True)
                self.optimizers[member].load_state_dict(selected['optimizer_state'])
                self.member_rng[member] = self.engine.cpu_tree(torch, selected['rng'])
                require(self.engine.exact(torch, self.engine.cpu_tree(torch, self.models[member].state_dict()), selected['model_state']), 'Exact independently own-selected body state')
                require(self.engine.exact(torch, self.engine.cpu_tree(torch, self.optimizers[member].state_dict()), selected['optimizer_state'])
                        and self.engine.exact(torch, self.member_rng[member], selected['rng']), 'Exact independently own-selected Adam/RNG state')
            self.verify_optimizers()

    def release_models(self):
        """Release terminal training objects before a fresh selected constructor."""
        self.group, self.models, self.ownership, self.optimizers = None, (), None, []
        gc.collect()
        self.rt['torch'].cuda.empty_cache()

    def selected_diagnostics(self, checkpoint_hashes, folder):
        """Reuse the sealed current4x3 assay only on complete fresh own states."""
        require(self.config.intent == 'scientific_training' and self.count == 4, 'Selected committee-only diagnostic, no four-copy single surrogate')
        torch, diagnostic = self.rt['torch'], self.modules['diagnostics']
        values = diagnostic.collect_selected_logits(self.rt, self.engine, self.metadata, self.models,
            self.member_rng, self.ctx, self.views, self.costs)
        result = diagnostic.analyze(torch, values, self.ctx.targets[self.ctx.valid_index], self.ctx.data.valid_ids, self.costs)
        result['paired_identity'] = self.metadata.plain_metadata(dict(study_binding=self.config.study_binding,
            role_seed=self.config.base_seed, base_seed=self.config.base_seed, role_binding=self.config.role_binding,
            full_view_binding=self.config.full_view_binding, input_files=self.ctx.data.input_bindings))
        result['condition_identity'] = dict(condition=self.config.condition, source_seal_sha256=self.config.source_seal_sha256,
            selected_checkpoint_sha256s=list(checkpoint_hashes), actual_ownership=self.kind, private_penalty='none')
        with self.costs.measure('complete_selected_current_VALID_diagnostic_write', gpu=True):
            torch.save(self.metadata.checkpoint_tree(torch, self.engine.cpu_tree(torch, result)), folder / 'SELECTED_VALID_DIAGNOSTICS.pt')
        return dict(binding=sha(folder / 'SELECTED_VALID_DIAGNOSTICS.pt'), all4x3=True, numerical_success_assessed=False)


def make_condition(rt, modules, ctx, views, costs, config=Config()):
    return Session(rt, modules, ctx, views, costs, config)


def fit_condition(session, identity, folder):
    """Fixed native patience/selector only; no scheduler, retries or launch."""
    session.config.require_enabled()
    require(session.config.intent == 'scientific_training', 'No complete fits during engineering qualification')
    folder = Path(folder); require(folder.is_absolute() and not folder.exists(), 'Fresh root-owned resident output')
    cost_file = Path(session.costs.folder) / 'COST_EVENTS.jsonl'
    require(cost_file.is_absolute() and cost_file.is_file() and not cost_file.is_symlink()
            and folder != cost_file.parent, 'Separate existing root entry cost folder includes charged construction')
    folder.mkdir(parents=True, exist_ok=False)
    torch, engine = session.rt['torch'], session.engine
    started, usage = time.perf_counter(), resource.getrusage(resource.RUSAGE_SELF)
    record = dict(status='started', complete=False, condition=session.config.condition, counts=session.counts,
        study_binding=session.config.study_binding, role_binding=session.config.role_binding,
        source_seal_sha256=session.config.source_seal_sha256, no_teacher=True, TEST_access=False,
        root_entry_cost_log=dict(path=str(cost_file), prefix_bytes=cost_file.stat().st_size,
            prefix_sha256=sha(cost_file), prefix_events=len(session.costs.rows)),
        scalar_fit_timing_includes_prior_setup_or_constructor=False,
        setup_costs_must_be_retained_even_if_in_a_separate_root_log=True,
        P='four-view own supervised BCE', S='positive proper restored-pool BCE, recipient-only; none for P',
        independent_losses_coupled=session.kind == 'independent' and session.credit != 'none')
    write(folder / 'RESULT.json', record)
    slots = session.count if session.kind == 'independent' else 1
    best, best_epoch = [float('inf')] * slots, [-1] * slots
    checkpoint_paths = [folder / ('BODY' + str(member) + '_SELECTED.pt') for member in range(slots)]
    fresh = None
    try:
        torch.cuda.reset_peak_memory_stats(session.rt['device'])
        for epoch in range(200):
            with session.costs.measure('native_pre_epoch_GC'):
                gc.collect()
            training = session.train_epoch()
            scores, outputs = session.evaluate()
            current = scores['VALID']['member_BCE'] if session.kind == 'independent' else [scores['VALID']['BCE']]
            for slot in range(slots):
                if session.kind == 'independent' and not session.active[slot]:
                    continue
                if current[slot] < best[slot]:
                    best[slot], best_epoch[slot] = current[slot], epoch
                    with session.costs.measure('own_selected_checkpoint_write', gpu=True):
                        torch.save(session.snapshot(epoch, identity, slot if session.kind == 'independent' else None), checkpoint_paths[slot])
                if epoch - best_epoch[slot] > 50:
                    if session.kind == 'independent':
                        session.active[slot] = False
                        session.optimizers[slot].zero_grad(set_to_none=True)
                    else:
                        session.active = [False] * session.count
            with (folder / 'HISTORY.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(epoch=epoch, training=training, scores=scores, best_epoch=list(best_epoch),
                    active_members=list(session.active), counters=dict(session.counters)), sort_keys=True, allow_nan=False) + '\n')
            if not any(session.active):
                break
        require(all(path.exists() for path in checkpoint_paths), 'Every own complete selected state retained')
        record.update(selected_epochs=list(best_epoch), selected_VALID_BCE=list(best),
            selected_checkpoint_sha256s=[sha(path) for path in checkpoint_paths],
            stopped_bodies_retained_as_terminal_detached_peers=session.kind == 'independent',
            counters=dict(session.counters), body_counters=session.body_counters)
        rt, modules, ctx, views, costs, config = session.rt, session.modules, session.ctx, session.views, session.costs, session.config
        with session.costs.measure('release_terminal_models_before_fresh_constructor'):
            session.release_models()
        fresh = Session(rt, modules, ctx, views, costs, config)
        with session.costs.measure('safe_selected_checkpoint_restore_for_final_serving', gpu=True):
            for slot, path in enumerate(checkpoint_paths):
                saved = torch.load(path, map_location='cpu', weights_only=True)
                fresh.restore(saved, identity, slot if session.kind == 'independent' else None)
        scores, outputs = fresh.evaluate()
        with session.costs.measure('fresh_restored_known_role_full_logits_write', gpu=True):
            torch.save(engine.cpu_tree(torch, outputs), folder / 'FRESH_SELECTED_KNOWN_OUTPUTS.pt')
        if fresh.count == 4:
            record['selected_current_diagnostics'] = fresh.selected_diagnostics(record['selected_checkpoint_sha256s'], folder)
        record.update(fresh_selected_scores=scores, status='complete', complete=True,
            fresh_constructor_selected_restore=True, final_serving_counts=dict(fresh.counters),
            selector='strict factual VALID BCE; independent own bodies or coherent shared pool; earliest ties; epoch-best>50',
            serving='fresh FP32 all-full probability mean; strict>0.5', quality_success_assessed=False,
            source_innovation_assessed=False, no_partial_or_surviving_subset=True)
    except BaseException as error:
        record.update(status='failed', complete=False, failure=dict(type=type(error).__name__, message=str(error)),
            counters=dict(session.counters), body_counters=session.body_counters)
        raise
    finally:
        cleanup_failures = []
        for name, owned in (('fresh_selected_models', fresh), ('terminal_training_models', session)):
            if owned is not None:
                try:
                    with session.costs.measure('release_' + name):
                        owned.release_models()
                except BaseException as error:
                    cleanup_failures.append(dict(scope=name, type=type(error).__name__, message=str(error)))
        try:
            torch.cuda.synchronize(session.rt['device'])
            record.update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(session.rt['device']),
                peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(session.rt['device']))
        except BaseException as error:
            cleanup_failures.append(dict(scope='final_CUDA_observation', type=type(error).__name__, message=str(error)))
        if cleanup_failures:
            record.update(status='failed', complete=False, cleanup_failures=cleanup_failures)
        end = resource.getrusage(resource.RUSAGE_SELF)
        record.update(seconds=time.perf_counter() - started, CPU_user_seconds=end.ru_utime - usage.ru_utime,
            CPU_system_seconds=end.ru_stime - usage.ru_stime, cumulative_process_RSS_peak_bytes=int(end.ru_maxrss * (1 if sys.platform == 'darwin' else 1024)),
            RSS_is_process_lifetime_highwater=True,
            root_entry_cost_log_final=dict(path=str(cost_file), bytes=cost_file.stat().st_size,
                sha256=sha(cost_file), events=len(session.costs.rows)),
            retained_payload_files=[dict(path=path.name, bytes=path.stat().st_size, sha256=sha(path))
                for path in sorted(folder.iterdir()) if path.suffix == '.pt' or path.name == 'HISTORY.jsonl'],
            CUDA_peaks_include_prior_setup_or_constructor=False)
        write(folder / 'RESULT.json', record)
        write(folder / 'COMPLETE.json', dict(status=record['status'], complete=record['complete'], no_TEST_access=True,
            study_binding=session.config.study_binding, result_sha256=sha(folder / 'RESULT.json')))
    require(record['complete'], 'Actual final cleanup status is authoritative')
    return record


def BE_P(rt, modules, ctx, views, costs, config=Config(condition='BE_P')):
    require(config.condition == 'BE_P', 'Named constructor condition')
    return make_condition(rt, modules, ctx, views, costs, config)


def BE_PS(rt, modules, ctx, views, costs, config=Config(condition='BE_PS')):
    require(config.condition == 'BE_PS', 'Named constructor condition')
    return make_condition(rt, modules, ctx, views, costs, config)


def ADD_P(rt, modules, ctx, views, costs, config=Config(condition='ADD_P')):
    require(config.condition == 'ADD_P', 'Named constructor condition')
    return make_condition(rt, modules, ctx, views, costs, config)


def ADD_PS(rt, modules, ctx, views, costs, config=Config(condition='ADD_PS')):
    require(config.condition == 'ADD_PS', 'Named constructor condition')
    return make_condition(rt, modules, ctx, views, costs, config)


def ADD_P_copied_dictionary(rt, modules, ctx, views, costs, config=Config(condition='ADD_P_copied_dictionary')):
    require(config.condition == 'ADD_P_copied_dictionary', 'Named constructor condition')
    return make_condition(rt, modules, ctx, views, costs, config)


def ADD_PS_neutral_all_source_average(rt, modules, ctx, views, costs, config=Config(condition='ADD_PS_neutral_all_source_average')):
    require(config.condition == 'ADD_PS_neutral_all_source_average', 'Named constructor condition')
    return make_condition(rt, modules, ctx, views, costs, config)


def independent_ADD_P(rt, modules, ctx, views, costs, config=Config(condition='independent_ADD_P')):
    require(config.condition == 'independent_ADD_P', 'Named constructor condition')
    return make_condition(rt, modules, ctx, views, costs, config)


def independent_ADD_PS(rt, modules, ctx, views, costs, config=Config(condition='independent_ADD_PS')):
    require(config.condition == 'independent_ADD_PS', 'Named constructor condition')
    return make_condition(rt, modules, ctx, views, costs, config)


def native_single_P(rt, modules, ctx, views, costs, config=Config(condition='native_single_P')):
    require(config.condition == 'native_single_P', 'Named constructor condition')
    return make_condition(rt, modules, ctx, views, costs, config)


def rank4_single_P(rt, modules, ctx, views, costs, config=Config(condition='rank4_single_P')):
    require(config.condition == 'rank4_single_P', 'Named constructor condition')
    return make_condition(rt, modules, ctx, views, costs, config)
