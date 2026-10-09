"""Complete independent native fits, individual best states, then probability pool."""
import gc
from pathlib import Path
import resource
import time
from support import gross_diagnostics, read, require, sha


def adopted_member0(common, receipt, config, base, expected_runtime):
    artifacts = receipt['member0_artifacts'][str(base)]
    for artifact in artifacts.values():
        require(sha(artifact['path']) == artifact['sha256'], 'Frozen member0 origin unchanged at actual reuse')
    native = read(artifacts['result']['path'])
    identity = native['identity']
    require(native['status'] == 'complete' and native['family'] == 'native_single'
            and native['config_id'] == config['id'] and native['seed'] == base,
            'Exact complete native single for frozen member0')
    require(identity['source_seal_sha256'] == receipt['NSD_V2_source_seal_sha256']
            and identity['native_commit'] == '11e21b561d884713ab1a18a521a7dc2fb26b9361'
            and identity['role_archive_sha256'] == receipt['roles_sha256']
            and identity['role_metadata_sha256'] == receipt['role_metadata_sha256'], 'Frozen native source/roles match')
    require(all(native['native_args'][name] == value for name, value in config['native_args'].items()), 'Exact original frozen native args')
    runtime = read(artifacts['runtime_metadata']['path'])
    require(runtime['versions'] and all(expected_runtime.get(name) == value for name, value in runtime['versions'].items()), 'Original native runtime matches recorded runtime')
    require(runtime['deterministic_algorithms'] == receipt['deterministic_algorithms'] and runtime['allow_tf32'] is False
            and runtime['device'] == receipt['device'], 'Original native deterministic/device policy matches freeze')
    require(native['checkpoint_sha256'] == artifacts['checkpoint']['sha256'] and native['test_truth_present'] is False,
            'Original owned selected checkpoint and no TEST truth')
    return dict(kind='independent_member', member=0, base_seed=base, seed=base, status='complete', reused=True,
                checkpoint_path=artifacts['checkpoint']['path'], checkpoint_sha256=artifacts['checkpoint']['sha256'],
                native_result=native, artifact_receipt=artifacts, original_complete_attempt_seconds=native['complete_attempt_seconds'],
                original_training_cost_charged=True, original_histories_and_failures_retained_by_receipt=True,
                complete_native_encoder_head_and_optimizer_owned=True, no_learned_parameter_sharing=True)


def run(np, torch, roc_auc_score, old, common, helpers, placement, adapter, cfg, config, data, identity, receipt, release, output, base, retain):
    folder = output / 'independent' / ('base'+str(base))
    folder.mkdir(parents=True, exist_ok=False)
    components = []
    for member in range(4):
        actual_seed = base + 1000003*member
        if member == 0 and release['reuse_native_member0']:
            component = adopted_member0(common, receipt, config, base, release['expected_runtime_versions'])
        else:
            usage0 = resource.getrusage(resource.RUSAGE_SELF)
            # Exact V2 loop: complete fresh encoder/head, own unscaled NLL, own optimizer and checkpoint.
            native = helpers.fit_one(np, torch, roc_auc_score, adapter, dict(optimizer=cfg, checkpoint_rule=receipt['checkpoint_rule']), config, actual_seed, 'native_single',
                                     data, read(output / 'ROLE_METADATA.json'), identity, folder, data['x'].device)
            usage = resource.getrusage(resource.RUSAGE_SELF)
            checkpoint = folder / ('native_single__'+config['id']+'__seed'+str(actual_seed)) / 'SELECTED_STATE.pt'
            if any(native.get(name) for name in ('cost_query_failure', 'cleanup_failure', 'cost_finalization_failure')): native['status'] = 'failed'
            component = dict(kind='independent_member', member=member, base_seed=base, seed=actual_seed, status=native['status'], reused=False,
                checkpoint_path=str(checkpoint), checkpoint_sha256=native.get('checkpoint_sha256'), native_result=native,
                CPU_user_seconds=usage.ru_utime-usage0.ru_utime, CPU_system_seconds=usage.ru_stime-usage0.ru_stime,
                complete_native_encoder_head_and_optimizer_owned=True, no_learned_parameter_sharing=True, original_training_cost_charged=True)
        common.write_json(folder / ('MEMBER'+str(member)+'.json'), component)
        components.append(component); retain(component)
    started, usage0 = time.perf_counter(), resource.getrusage(resource.RUSAGE_SELF)
    record = dict(kind='independent_pool', base_seed=base, seed=base, status='started', components=components,
        objective='four separately fitted unscaled own NLLs with separate optimizers', serving='mean own-selected member probabilities',
        individual_best_checkpoints=True, simultaneous_learned_parameter_sharing=False, selected_serving_model_residency='streamed one complete body at a time',
        automatic_retry=False, TEST_truth_present=False, root_freeze_receipt_required=True)
    def persist(): common.write_json(folder / 'POOL_RESULT.json', record)
    model = None
    stage, forward_attempts, forwards_completed, constructor_attempts, constructors_completed = 'component_completeness', 0, 0, 0, 0
    serving_scope_started = False
    persist()
    try:
        require(all(value['status'] == 'complete' for value in components), 'All four independently selected complete bodies required')
        torch.cuda.reset_peak_memory_stats(data['x'].device)
        serving_scope_started = True
        selected_outputs, member_scores, bindings, native_parameter_counts = [], [], [], []
        for component in components:
            stage = 'own_selected_reconstruction_member_'+str(component['member'])
            checkpoint = Path(component['checkpoint_path'])
            require(sha(checkpoint) == component['checkpoint_sha256'], 'Exact individual selected checkpoint')
            saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
            require(saved['family'] == 'native_single' and saved['config_id'] == config['id'] and saved['seed'] == component['seed']
                    and saved['epoch'] == component['native_result']['selected_epoch'] and saved['test_truth_saved'] is False
                    and saved['identity'] == component['native_result']['identity'], 'Individual best state identity')
            helpers.seed_all(np, torch, component['seed'])
            args = dict(config['native_args'], graph_size=data['x'].shape[0], input_dim=data['x'].shape[1], output_dim=2, device=str(data['x'].device))
            native_factory = placement.make_native_placed_factory(torch, adapter, data['cpu_edge_index'], data['edge_index'], args)
            constructor_attempts += 1
            model = native_factory()
            constructors_completed += 1
            model.load_state_dict(saved['state_dict'], strict=True)
            require(old.exact(torch, old.cpu_tree(torch, model.state_dict()), saved['state_dict']), 'Exact full independent selected parameters/buffers')
            native_parameter_counts.append(helpers.parameter_counts(model))
            forward_attempts += 1
            scores, logp = helpers.evaluate(np, torch, roc_auc_score, model, data, component['seed'], saved['epoch'])
            forwards_completed += 1
            diagnostics = gross_diagnostics(torch, scores, logp, saved['scores'], saved['role_logp'])
            selected_outputs.append(logp); member_scores.append(scores)
            bindings.append(dict(member=component['member'], seed=component['seed'], own_selected_epoch=saved['epoch'], checkpoint_sha256=sha(checkpoint),
                                 model_parameter_buffer_state_exact=True, reload_diagnostics=diagnostics))
            record.update(member_bindings=bindings, full_native_selected_serving_forwards=forwards_completed)
            persist()
            model = None; del saved
            gc.collect(); torch.cuda.empty_cache()
        pooled = {role: torch.log(torch.stack([value[role].exp() for value in selected_outputs]).mean(0)) for role in ('train', 'valid')}
        scores = {role: helpers.metrics(torch, roc_auc_score, pooled[role], data[role+'_y'].cpu()) for role in ('train', 'valid')}
        reference = dict(schema='four_individual_best_native_states-v1', identity=identity, base_seed=base, member_bindings=bindings,
                         pooled_role_logp=pooled, scores=scores, member_scores=member_scores, TEST_truth=False)
        reference_path = folder / 'SELECTED_SERVING_REFERENCE.pt'
        temporary = reference_path.with_suffix('.tmp'); torch.save(reference, temporary); temporary.replace(reference_path)
        record.update(status='complete', scores=scores, member_scores=member_scores, member_bindings=bindings,
            complete_independent_active_parameters=sum(value['active'] for value in native_parameter_counts),
            complete_independent_parameter_bytes=sum(value['parameter_bytes'] for value in native_parameter_counts),
            serving_reference_sha256=sha(reference_path), no_ensemble_checkpoint_selection=True,
            per_member_native_constructor_cost_charged=True, trained_independent_bodies=4, native_serving_full_paths=4)
    except Exception as error:
        record.update(status='failed', failure=common.failure_record(error, stage))
        common.append_jsonl(output / 'FAILURES.jsonl', record)
    except BaseException as error:
        record.update(status='interrupted', failure=common.failure_record(error, stage)); raise
    finally:
        try:
            helpers.synchronize(torch, data['x'].device)
            record.update(selected_serving_peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(data['x'].device) if serving_scope_started else None,
                selected_serving_peak_CUDA_reserved_bytes=torch.cuda.max_memory_reserved(data['x'].device) if serving_scope_started else None)
            model = None; gc.collect(); torch.cuda.empty_cache()
        except Exception as error:
            record.update(status='failed', selected_serving_cost_or_cleanup_failure=common.failure_record(error, 'finalize'))
        usage = resource.getrusage(resource.RUSAGE_SELF)
        record.update(selected_serving_seconds=time.perf_counter()-started, selected_serving_CPU_user_seconds=usage.ru_utime-usage0.ru_utime,
            selected_serving_CPU_system_seconds=usage.ru_stime-usage0.ru_stime, process_RSS_high_water_cumulative_bytes=common.process_peak_rss_bytes(),
            serving_constructor_attempts=constructor_attempts, serving_constructors_completed=constructors_completed,
            selected_serving_scope_started=serving_scope_started,
            serving_forward_attempts=forward_attempts, serving_forwards_completed=forwards_completed,
            cost_includes_individual_native_fit_records_reuse_origins_and_added_serving=True)
        persist()
    retain(record)
    return record
