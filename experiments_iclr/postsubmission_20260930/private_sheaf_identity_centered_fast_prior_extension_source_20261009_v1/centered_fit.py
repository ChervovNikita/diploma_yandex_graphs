"""Prospective centered-fast-prior variant; V2 full shared-fit structure retained."""
import gc
import resource
import time
from bank import make_bank
from support import replay_materiality, require, sha, topology_bytes


def evaluate(np, torch, roc_auc_score, helpers, exact, bank, data, seed, epoch, counters, phase):
    bank.eval(); bank.assert_ownership()
    before = helpers.capture_rng(np, torch)
    try:
        helpers.seed_all(np, torch, seed+2000003+epoch)
        members = []
        with torch.no_grad():
            for member in bank.members:
                counters[phase + '_forward_attempts'] += 1
                logp = member(data['x'])
                require(logp.shape == (data['x'].shape[0], 2) and torch.isfinite(logp).all().item(), 'Finite complete native member')
                members.append({role: logp[data[role+'_index']].detach().cpu() for role in ('train', 'valid')})
                counters[phase + '_forwards_completed'] += 1
                del logp
            pooled = {role: torch.log(torch.stack([value[role].exp() for value in members]).mean(0)) for role in ('train', 'valid')}
            scores = {role: helpers.metrics(torch, roc_auc_score, pooled[role], data[role+'_y'].cpu()) for role in ('train', 'valid')}
            member_scores = [{role: helpers.metrics(torch, roc_auc_score, value[role], data[role+'_y'].cpu()) for role in ('train', 'valid')} for value in members]
        return scores, pooled, member_scores, members
    finally:
        helpers.restore_rng(np, torch, before)
        require(exact(torch, helpers.capture_rng(np, torch), before), 'Exact owned evaluation stream restoration')


def parameter_groups(bank, cfg):
    private_names = {path+suffix for path in bank.factor_paths for suffix in ('.r','.s')}
    slow_incidence, slow_other, fast = [], [], []
    for name,value in bank.named_parameters():
        local = name.split('.',2)[2]
        if local in private_names: fast.append(value)
        elif 'sheaf_learners' in name: slow_incidence.append(value)
        else: slow_other.append(value)
    values = slow_incidence+slow_other+fast
    require(len(values)==len({id(value) for value in values})==len(list(bank.parameters())), 'One centered optimizer entry per deduplicated parameter')
    return [dict(params=slow_incidence,weight_decay=cfg['sheaf_decay']),
            dict(params=slow_other,weight_decay=cfg['weight_decay']),dict(params=fast,weight_decay=0.0)]


def optimizer(torch, bank, cfg):
    return torch.optim.Adam(parameter_groups(bank,cfg), lr=cfg['lr'])


def member_prior(torch, model, member, coefficient):
    private_names = {path+suffix for path in model.factor_paths for suffix in ('.r','.s')}
    factors = [value for name,value in member.named_parameters() if name in private_names]
    require(len(factors)==2*member.layers and coefficient==0.0005, 'Exact incidence r/s centered prior coefficient and coverage')
    return coefficient/(2*4)*sum((value-1).square().sum() for value in factors)


def train_update(torch, model, opt, data, counters, coefficient, observer=None):
    """V2 streamed update with explicit identity-centered mean-objective factor prior."""
    model.train(); model.assert_ownership(); opt.zero_grad(set_to_none=True)
    versions = [(value, value._version) for value in model.parameters()]
    own_nll, own_prior = [], []
    for index, member in enumerate(model.members):
        counters['train_forward_attempts'] += 1
        logp = member(data['x'])
        require(logp.shape == (data['x'].shape[0], 2) and torch.isfinite(logp).all().item(), 'Finite native full member TRAIN output')
        counters['train_forwards_completed'] += 1
        nll = torch.nn.functional.nll_loss(logp[data['train_index']], data['train_y'])
        require(torch.isfinite(nll).item(), 'Finite own all-TRAIN NLL')
        own_nll.append(float(nll.item()))
        prior = member_prior(torch,model,member,coefficient)
        require(torch.isfinite(prior).item(), 'Finite explicit identity-centered member prior')
        own_prior.append(float(prior.item()))
        counters['backward_attempts'] += 1
        (nll / 4 + prior).backward()
        counters['backwards_completed'] += 1
        del logp, nll, prior
        if observer is not None: observer(index, model)
    member = None
    require(all(value._version == version for value, version in versions), 'All four backwards used unchanged old parameters')
    require(all(value.grad is None or torch.isfinite(value.grad).all().item() for value in model.parameters()), 'Finite accumulated native gradients')
    model.assert_ownership()
    counters['Adam_attempts'] += 1
    opt.step()
    counters['Adam_steps_completed'] += 1
    return own_nll, own_prior


def fit(np, torch, roc_auc_score, old, common, helpers, placement, adapter, cfg, config, data, role_meta, identity, output, seed):
    folder = output / 'shared' / ('seed'+str(seed))
    folder.mkdir(parents=True, exist_ok=False)
    started, usage0 = time.perf_counter(), resource.getrusage(resource.RUSAGE_SELF)
    device, record = data['x'].device, dict(kind='centered_shared_fit', base_seed=seed, seed=seed, status='started', family='geometry_only_shared_M4_identity_centered_fast_prior', identity=identity,
        context_regularizer=0.0, training_objective='mean four own TRAIN NLL plus lambda/(2M) sum private squared distance from one', serving='mean four class probabilities',
        shared_native_stem_feature_head_epsilon_incidence_weights=True, only_incidence_r_s_private=True,
        private_likelihood_gradient_scale=0.25, private_coupled_Adam_weight_decay=0.0, identity_centered_prior_coefficient=cfg['sheaf_decay'], mean_prior_gradient_scale=0.25,
        independent_likelihood_gradient_scale=1.0, private_regularization_mismatch_disclosed=True,
        full_graph=True, all_TRAIN_and_VALID_rows=True, TEST_truth_present=False, automatic_retry=False)
    counters = {key: 0 for key in ('train_forward_attempts', 'train_forwards_completed', 'backward_attempts', 'backwards_completed',
        'Adam_attempts', 'Adam_steps_completed', 'validation_forward_attempts', 'validation_forwards_completed',
        'restore_forward_attempts', 'restore_forwards_completed', 'original_constructor_attempts', 'original_constructors_completed')}
    def persist(): common.write_json(folder / 'RESULT.json', dict(record, counters=counters))
    persist()
    model = opt = restored = restored_opt = None
    stage, best_key, best_epoch, stale, checkpoint_seconds = 'construction', None, None, 0, 0.0
    checkpoint = folder / 'SELECTED_STATE.pt'
    try:
        torch.cuda.reset_peak_memory_stats(device)
        helpers.seed_all(np, torch, seed)
        args = dict(config['native_args'], graph_size=data['x'].shape[0], input_dim=data['x'].shape[1], output_dim=2, device=str(device))
        native_factory = placement.make_native_placed_factory(torch, adapter, data['cpu_edge_index'], data['edge_index'], args)
        def observed_factory():
            counters['original_constructor_attempts'] += 1
            value = native_factory()
            counters['original_constructors_completed'] += 1
            return value
        def fresh_bank(): return make_bank(torch, adapter, observed_factory)
        model = fresh_bank(); opt = optimizer(torch, model, cfg)
        helpers.synchronize(torch, device)
        record.update(native_args=args, ownership=model.assert_ownership(), parameters=helpers.parameter_counts(model),
            construction_seconds=time.perf_counter()-started, static_topology_unique_bytes=topology_bytes(torch, model.members),
            native_saved_transport_payload_bytes=4*args['layers']*role_meta['support_counts']['canonical_undirected']*args['d']**2*data['x'].element_size(),
            actual_directed_support=role_meta['support_counts']['canonical_directed'], original_constructors_per_fresh_bank=1,
            factor_weight_decay=0.0, identity_centered_prior_coefficient=cfg['sheaf_decay'], shared_map_weight_decay=cfg['sheaf_decay'], shared_other_weight_decay=cfg['weight_decay'])
        for epoch in range(1, cfg['max_epochs']+1):
            stage = 'train_epoch_'+str(epoch)
            record['last_attempted_epoch'] = epoch; persist()
            tick = time.perf_counter()
            own_nll, own_prior = train_update(torch, model, opt, data, counters, cfg['sheaf_decay'])
            helpers.synchronize(torch, device)
            train_seconds = time.perf_counter()-tick
            stage = 'validation_epoch_'+str(epoch)
            tick = time.perf_counter()
            scores, pooled, member_scores, member_logp = evaluate(np, torch, roc_auc_score, helpers, old.exact, model, data, seed, epoch, counters, 'validation')
            helpers.synchronize(torch, device)
            validation_seconds = time.perf_counter()-tick
            key = old.selector(scores, epoch)
            selected = best_key is None or key > best_key
            if selected:
                best_key, best_epoch, stale = key, epoch, 0
                tick = time.perf_counter()
                saved = dict(schema='owned-identity-centered-fast-prior-selected-state-v1', identity=identity, seed=seed, epoch=epoch,
                    model_state=old.cpu_tree(torch, model.state_dict()), optimizer_state=old.cpu_tree(torch, opt.state_dict()),
                    training_rng=helpers.capture_rng(np, torch), scores=scores, role_logp=pooled, member_scores=member_scores,
                    member_role_logp=member_logp, context_regularizer=0.0, TEST_truth_saved=False)
                temporary = checkpoint.with_suffix('.tmp')
                torch.save(saved, temporary); temporary.replace(checkpoint)
                checkpoint_seconds += time.perf_counter()-tick
                del saved
            else: stale += 1
            common.append_jsonl(folder / 'HISTORY.jsonl', dict(epoch=epoch, own_TRAIN_nll=own_nll,
                mean_own_TRAIN_nll=sum(own_nll)/4, member_weighted_centered_prior=own_prior,
                total_weighted_centered_prior=sum(own_prior), total_training_objective=sum(own_nll)/4+sum(own_prior), pooled_scores=scores, member_scores=member_scores, selected=selected,
                stale_epochs=stale, train_seconds=train_seconds, validation_seconds=validation_seconds, context_regularizer=0.0))
            record.update(epochs_completed=epoch, selected_epoch=best_epoch, checkpoint_write_seconds=checkpoint_seconds)
            persist(); del pooled, member_logp
            if stale >= cfg['patience']: break
        stage = 'fresh_selected_state_reconstruction'
        del model, opt
        model = opt = None
        gc.collect(); torch.cuda.empty_cache()
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        require(saved['identity'] == identity and saved['seed'] == seed and saved['epoch'] == best_epoch and saved['context_regularizer'] == 0,
                'Exact owned selected checkpoint identity')
        restored = fresh_bank()
        restored.load_state_dict(saved['model_state'], strict=True)
        require(old.exact(torch, old.cpu_tree(torch, restored.state_dict()), saved['model_state']), 'Exact restored bank parameter/buffer state')
        restored.assert_ownership()
        restored_opt = optimizer(torch, restored, cfg); restored_opt.load_state_dict(saved['optimizer_state'])
        require(old.exact(torch, old.cpu_tree(torch, restored_opt.state_dict()), saved['optimizer_state']), 'Exact selected shared optimizer state')
        helpers.restore_rng(np, torch, saved['training_rng'])
        require(old.exact(torch, helpers.capture_rng(np, torch), saved['training_rng']), 'Exact selected bank training streams')
        stage = 'four_path_selected_serving'
        scores, pooled, member_scores, member_logp = evaluate(np, torch, roc_auc_score, helpers, old.exact, restored, data, seed, best_epoch, counters, 'restore')
        record.update(scores=scores, member_scores=member_scores, selected_scores=saved['scores'],
            score_source='fresh_reconstructed_selected_serving', replay_policy='exact_state_finite_serving_recorded_materiality')
        persist()
        diagnostics = replay_materiality(torch, scores, pooled, saved['scores'], saved['role_logp'])
        member_diagnostics = [replay_materiality(torch, member_scores[index], member_logp[index], saved['member_scores'][index], saved['member_role_logp'][index]) for index in range(4)]
        stage = 'persist_fresh_selected_serving_reference'
        tick = time.perf_counter()
        reference = dict(schema='fresh-identity-centered-fast-prior-selected-serving-reference-v1', identity=identity,
            base_seed=seed, selected_epoch=best_epoch, checkpoint_identity=dict(identity=saved['identity'], seed=saved['seed'], epoch=saved['epoch']),
            checkpoint_sha256=sha(checkpoint), pooled_role_logp=pooled, member_role_logp=member_logp,
            scores=scores, member_scores=member_scores, reload_diagnostics=diagnostics, member_reload_diagnostics=member_diagnostics,
            score_source='fresh_reconstructed_selected_serving', TEST_truth=False, server_only=True)
        reference_path = folder / 'SELECTED_SERVING_REFERENCE.pt'
        temporary = reference_path.with_suffix('.tmp')
        torch.save(reference, temporary); temporary.replace(reference_path)
        record.update(serving_reference_write_seconds=time.perf_counter()-tick, serving_reference_bytes=reference_path.stat().st_size,
            serving_reference_sha256=sha(reference_path), server_only_serving_reference=True, extra_reference_prediction_calls=0)
        del reference
        record.update(status='complete', selected_epoch=best_epoch, scores=scores, member_scores=member_scores,
            selected_scores=saved['scores'], reload_diagnostics=diagnostics, member_reload_diagnostics=member_diagnostics,
            model_parameter_buffer_state_exact=True, optimizer_state_exact=True, selected_training_streams_exact=True,
            checkpoint_sha256=sha(checkpoint), checkpoint_bytes=checkpoint.stat().st_size,
            reported_state='fresh original four-path bank, selected state, probability-mean serving',
            output_bitwise_requirement=False, original_constructors_total=2)
    except Exception as error:
        record.update(status='failed', failure=common.failure_record(error, stage))
        common.append_jsonl(output / 'FAILURES.jsonl', record)
    except BaseException as error:
        record.update(status='interrupted', failure=common.failure_record(error, stage))
        raise
    finally:
        try:
            helpers.synchronize(torch, device)
            record.update(peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(device), peak_CUDA_reserved_bytes=torch.cuda.max_memory_reserved(device))
        except Exception as error:
            record.update(status='failed', cost_failure=common.failure_record(error, 'CUDA_finalize'))
        usage = resource.getrusage(resource.RUSAGE_SELF)
        record.update(complete_attempt_seconds_before_final_cleanup=time.perf_counter()-started, CPU_user_seconds=usage.ru_utime-usage0.ru_utime,
            CPU_system_seconds=usage.ru_stime-usage0.ru_stime, process_RSS_high_water_cumulative_bytes=common.process_peak_rss_bytes(),
            cost_includes_constructor_clone_all_paths_failures_checkpoint_and_restore=True, streamed_old_parameter_backwards=True,
            native_completed_map_evaluations=config['native_args']['layers']*sum(counters[key] for key in
                ('train_forwards_completed', 'validation_forwards_completed', 'restore_forwards_completed')),
            map_evaluation_count_inferred_from_completed_full_calls=True)
        persist()
        model = opt = restored = restored_opt = None
        gc.collect()
        try: torch.cuda.empty_cache()
        except Exception as error:
            record.update(status='failed', cleanup_failure=common.failure_record(error, 'CUDA_empty_cache')); persist()
        usage = resource.getrusage(resource.RUSAGE_SELF)
        record.update(complete_attempt_seconds=time.perf_counter()-started, CPU_user_seconds=usage.ru_utime-usage0.ru_utime,
            CPU_system_seconds=usage.ru_stime-usage0.ru_stime, process_RSS_high_water_cumulative_bytes=common.process_peak_rss_bytes())
        persist()
    return record
