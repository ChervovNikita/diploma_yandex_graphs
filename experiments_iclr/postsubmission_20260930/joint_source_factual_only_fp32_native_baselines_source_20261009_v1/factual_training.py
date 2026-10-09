"""Thin native factual controls, using injected immutable V2 Session contracts."""
import gc
import json
import resource
import sys
import time
from pathlib import Path

from factual_contracts import CONDITIONS, POLICY, Config, dependencies, read, require, sha


def prepare_factual_context(rt, modules, input_root, role_file, costs, config=Config()):
    """Only full native preparation; no source-view producer call or teacher."""
    config.require_enabled()
    joint, native = dependencies(rt, modules)
    engine, loader, metadata = (native[key] for key in ('engine', 'role_loader', 'metadata'))
    role_file = Path(role_file)
    require(role_file.is_absolute() and not role_file.is_symlink() and sha(role_file) == config.role_binding, 'Exact original role file')
    role = read(role_file); require(role['seed'] == config.base_seed, 'Original frozen outer role')
    loader.source_gate()
    with costs.measure('factual_only_role_loader_and_complete_native_setup', gpu=True):
        data = loader.load(Path(input_root), role_file, read(Path(loader.__file__).parent / 'SOURCE_EXPECTATIONS.json'))
        require(data.input_bindings == role['input_files'], 'Exact original full input bindings')
        static = engine.prepare_static(rt, data, costs)
        ctx = engine.prepare_seed(rt, static, role, role['seed'], costs)
        binding = metadata.digest(dict(study=config.study_binding, role_binding=config.role_binding,
            seed=ctx.seed, inputs=data.input_bindings, feature_shapes=static.feature_shapes,
            label_shapes={key: list(value.shape) for key, value in ctx.label_feats.items()}))
        require(binding == config.full_view_binding, 'Same prospective factual context digest')
    return ctx


def make_condition(rt, modules, ctx, costs, config=Config()):
    config.require_enabled()
    joint, native = dependencies(rt, modules)

    class FactualSession(joint.Session):
        """Reuse native role/eval/BN/optimizer/state contracts, no view machinery."""
        def forward(self, token, ids=None, *, persist_buffers=False):
            require(token.source is None, 'Factual controls expose full native inputs only')
            return super().forward(token, ids, persist_buffers=persist_buffers)

        def _initialize(self, rt, modules, ctx, views, costs, config):
            config.require_enabled()
            dependencies(rt, modules)
            require(not views, 'Factual controls have no source-absent views')
            self.rt, self.modules, self.integration_modules = rt, native, modules
            self.ctx, self.views, self.costs, self.config = ctx, {}, costs, config
            self.engine, self.adapter, self.metadata, self.helper = (native[key] for key in ('engine', 'adapter', 'metadata', 'helper'))
            self.kind = 'single' if config.condition == CONDITIONS[0] else 'independent'
            self.parameterization, self.rank, self.credit, self.copied = 'native', 0, 'none', False
            self.count = 1 if self.kind == 'single' else 4
            torch = rt['torch']
            require(type(ctx) is self.engine.SeedContext and rt['device'].type == 'cuda' and ctx.seed == config.base_seed
                    and len(ctx.feats) == 25 and len(ctx.label_feats) == 12, 'Exact complete native full-context CUDA seam')
            require(0 < ctx.train_count <= 10000 and ctx.train_index.tolist() == list(ctx.data.train_ids)
                    and ctx.valid_index.tolist() == list(ctx.data.valid_ids), 'Complete frozen TRAIN/VALID populations')
            self.rows = {'TRAIN': ctx.train_index.detach().cpu().clone(),
                         'KNOWN': torch.cat((ctx.train_index, ctx.valid_index)).detach().cpu()}
            require(all(ids.dtype == torch.int64 and ids.device.type == 'cpu' and ids.unique().numel() == ids.numel()
                        for ids in self.rows.values()), 'Unique native canonical role IDs')
            self.targets = ctx.targets_cuda[self.rows['TRAIN']].float()
            require(self.targets.shape == (ctx.train_count, 5) and torch.isfinite(self.targets).all().item()
                    and ((self.targets == 0) | (self.targets == 1)).all().item(), 'Complete hard TRAIN Bernoulli targets')
            self.cache_guard = self.cache_stamp()
            self.row_guard = tuple((name, id(ids), int(ids._version)) for name, ids in self.rows.items())
            self.counters = dict(epochs_completed=0, factual_train_forwards=0, P_backwards=0,
                actual_Adam_steps=0, factual_eval_forwards=0, source_absent_forwards=0, source_credit_VJPs=0)
            self.body_counters = [dict(actual_Adam_steps=0, supervised_epochs=0, stopped_epochs=0) for _ in range(self.count)]
            caller = self.engine.capture_rng(rt['numpy'], torch)
            self._allocation_started = True
            try:
                with costs.measure('fresh_factual_native_bodies_and_own_Adam', gpu=True):
                    models = []
                    for seed in config.body_seeds:
                        rt['native'].set_random_seed(seed)
                        models.append(self.engine.make_model(rt, ctx, costs))
                    self.models, self.group = tuple(models), torch.nn.ModuleList(models)
                    named = [dict(model.named_parameters(remove_duplicate=False)) for model in models]
                    self.slow_names, self.private_names = tuple(named[0]), ()
                    require(all(tuple(mapping) == self.slow_names for mapping in named), 'Same complete native parameter namespace')
                    blocks = tuple(tuple(mapping.values()) for mapping in named)
                    owned = tuple(value for block in blocks for value in block)
                    require(len({id(value) for value in owned}) == len(owned)
                            and len({self.adapter._storage(value) for value in owned}) == len(owned), 'Genuine native bodies have no object/storage aliases')
                    self.ownership = self.helper.Ownership(blocks[0] if self.kind == 'single' else (),
                        tuple(() for _ in models), owned, blocks if self.kind == 'independent' else ((),))
                    self.verify_buffers()
                    require(all(value.dtype == torch.float32 and value.device == rt['device'] for value in owned), 'Full native FP32 parameters')
                    self.optimizers = [torch.optim.Adam([dict(params=block, lr=.001, weight_decay=0, ownership='native')],
                        lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0, amsgrad=False) for block in blocks]
                self.member_rng = []
                for seed in config.member_rng_seeds:
                    rt['native'].set_random_seed(seed)
                    self.member_rng.append(self.engine.capture_rng(rt['numpy'], torch))
            finally:
                self.engine.restore_rng(rt['numpy'], torch, caller)
            self.score_config = self.helper.Config(enabled=True, score_mode='bernoulli_marginal_logits')
            self.active = [True] * self.count
            self.counts = self.parameter_report()
            self.verify_optimizers()

        def train_epoch(self):
            """One full live native P backward/BN advance/own Adam per active body."""
            self.config.require_enabled(); self.verify_optimizers(); self.verify_buffers()
            torch = self.rt['torch']
            require(any(self.active), 'No factual update after all own bodies stopped')
            losses, stamp = {}, self.ownership.stamp()
            with self.costs.measure('factual_only_complete_native_live_TRAIN_and_Adam', gpu=True):
                batches = 0
                for batch in self.ctx.train_loader:
                    batches += 1
                    require(batches == 1 and type(batch) is torch.Tensor and batch.device.type == 'cpu'
                            and batch.dtype == torch.int64 and batch.ndim == 1 and batch.numel() == self.ctx.train_count
                            and batch.unique().numel() == batch.numel()
                            and torch.equal(batch.sort().values, self.rows['TRAIN'].sort().values), 'Complete factual TRAIN permutation only')
                    require(self.cache_stamp() == self.cache_guard
                            and tuple((name, id(ids), int(ids._version)) for name, ids in self.rows.items()) == self.row_guard, 'Original full caches and role IDs')
                    truth = self.ctx.targets_cuda[batch].float()
                    for member, model in enumerate(self.models):
                        if not self.active[member]:
                            continue
                        self.optimizers[member].zero_grad(set_to_none=True)
                        token = self.token(member, None, 'train')
                        features = {key: value[batch].to(rt['device']) for key, value in self.ctx.feats.items()}
                        labels = {key: value[batch].to(rt['device']) for key, value in self.ctx.label_feats.items()}
                        with self.metadata.scratch_native_state(rt, self.engine, model, 'train', token.rng), torch.cuda.amp.autocast(enabled=False):
                            self._copy_buffers(model, token.buffers_before)
                            output = model(batch.to(rt['device']), features, labels, None).float()
                            self.verify_buffers()
                            after_buffers = self.buffer_state(member)
                            after_rng = self.engine.capture_rng(rt['numpy'], torch)
                        require(self.engine.exact(torch, self.buffer_state(member), token.buffers_before), 'Scratch restores original canonical BN state')
                        require(output.shape == (self.ctx.train_count, 5) and torch.isfinite(output).all().item(), 'Finite complete factual TRAIN logits')
                        require(all(after_buffers[name].item() == token.buffers_before[name].item() + 1
                                    for name in joint.BN_BUFFER_NAMES if name.endswith('num_batches_tracked')), 'One native live BN advance per factual epoch')
                        self._copy_buffers(model, after_buffers)
                        own = -self.helper._observed_logp(output, truth, self.score_config).mean()
                        losses[str(member)] = float(own.detach().item())
                        own.backward()  # Each complete body receives B(full,y), with no /4.
                        self.member_rng[member] = self.engine.cpu_tree(torch, after_rng)
                        self.counters['factual_train_forwards'] += 1
                        self.counters['P_backwards'] += 1
                        del output, own, features, labels
                    require(self.ownership.stamp() == stamp and self.cache_stamp() == self.cache_guard, 'All full-loss contributions before any own Adam step')
                    require(all(value.grad is None or torch.isfinite(value.grad).all().item() for value in self.ownership.all_parameters), 'Finite actual FP32 factual gradients')
                    for member, optimizer in enumerate(self.optimizers):
                        if not self.active[member]:
                            require(all(value.grad is None for group in optimizer.param_groups for value in group['params']), 'Stopped factual bodies receive no gradient/Adam')
                            continue
                        optimizer.step()
                        self.counters['actual_Adam_steps'] += 1
                        self.body_counters[member]['actual_Adam_steps'] += 1
                require(batches == 1, 'Exactly one complete native TRAIN batch')
            self.counters['epochs_completed'] += 1
            for member in range(self.count):
                self.body_counters[member]['supervised_epochs' if self.active[member] else 'stopped_epochs'] += 1
            self.verify_optimizers(); self.verify_buffers()
            return dict(training_policy=POLICY, P_full_BCE_by_active_body=losses,
                body_loss_reduction='mean complete TRAIN rows and five labels; coefficient1; no /4',
                active_members=list(self.active), actual_factual_TRAIN_forwards=len(losses),
                source_absent_views_or_credit=False, stopped_TRAIN_or_peer_calls=0)

        def snapshot(self, epoch, identity, member=None):
            state = super().snapshot(epoch, identity, member)
            state.update(training_policy=POLICY, factual_view_count=1,
                normalization='one complete factual BCE per body, coefficient1', source_credit=False,
                factual_buffer_policy='one-live-full-TRAIN-advance-terminal-eval-no-replay',
                reused_native_buffer_restore_contract=joint.BUFFER_POLICY,
                joint_v2_source_review_binding=self.config.joint_v2_source_review_binding)
            return state

        def restore(self, selected, identity, member=None):
            require(selected['training_policy'] == POLICY and selected['factual_view_count'] == 1
                    and selected['normalization'] == 'one complete factual BCE per body, coefficient1'
                    and selected['source_credit'] is False
                    and selected['factual_buffer_policy'] == 'one-live-full-TRAIN-advance-terminal-eval-no-replay'
                    and selected['reused_native_buffer_restore_contract'] == joint.BUFFER_POLICY
                    and selected['joint_v2_source_review_binding'] == self.config.joint_v2_source_review_binding, 'Exact own factual training policy')
            return super().restore(selected, identity, member)

        def selected_diagnostics(self, checkpoint_hashes, folder):
            raise RuntimeError('Factual controls provide full factual outputs; source-ablated diagnostics are not constructed')

    return FactualSession(rt, modules, ctx, {}, costs, config)


def factual_native_single_P(rt, modules, ctx, costs, config=Config(condition=CONDITIONS[0])):
    require(config.condition == CONDITIONS[0], 'Named factual single constructor')
    return make_condition(rt, modules, ctx, costs, config)


def factual_independent_native4_P(rt, modules, ctx, costs, config=Config(condition=CONDITIONS[1])):
    require(config.condition == CONDITIONS[1], 'Named genuine factual independent4 constructor')
    return make_condition(rt, modules, ctx, costs, config)


def fit_factual_condition(session, identity, folder):
    """Fixed native patience/selector only; no scheduler, retries or launch."""
    session.config.require_enabled()
    joint, native = dependencies(session.rt, session.integration_modules)
    write = joint.write
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
        P='one full factual BCE per complete body; coefficient1', S='none',
        training_policy=POLICY, independent_losses_coupled=False, source_views_constructed=False)
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
            stopped_bodies_keep_terminal_weights_buffers_rng=session.kind == 'independent',
            counters=dict(session.counters), body_counters=session.body_counters)
        rt, modules, ctx, costs, config = session.rt, session.integration_modules, session.ctx, session.costs, session.config
        with session.costs.measure('release_terminal_models_before_fresh_constructor'):
            session.release_models()
        fresh = make_condition(rt, modules, ctx, costs, config)
        with session.costs.measure('safe_selected_checkpoint_restore_for_final_serving', gpu=True):
            for slot, path in enumerate(checkpoint_paths):
                saved = torch.load(path, map_location='cpu', weights_only=True)
                fresh.restore(saved, identity, slot if session.kind == 'independent' else None)
        scores, outputs = fresh.evaluate()
        with session.costs.measure('fresh_restored_known_role_full_logits_write', gpu=True):
            torch.save(engine.cpu_tree(torch, outputs), folder / 'FRESH_SELECTED_KNOWN_OUTPUTS.pt')
        record.update(fresh_selected_scores=scores, status='complete', complete=True,
            fresh_constructor_selected_restore=True, final_serving_counts=dict(fresh.counters),
            selector='strict full factual VALID BCE; each native body owns its selector; earliest ties; epoch-best>50',
            serving='fresh FP32 all-full probability mean; strict>0.5', quality_success_assessed=False,
            source_innovation_assessed=False, source_absent_diagnostics_constructed=False, no_partial_or_surviving_subset=True)
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
