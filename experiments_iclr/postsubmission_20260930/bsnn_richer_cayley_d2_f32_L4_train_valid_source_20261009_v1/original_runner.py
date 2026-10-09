"""Inactive original-author Cayley d2 full-graph TRAIN/VALID, three paired seeds."""
import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import resource
import time
from support import (HERE, PHASE, borrowed, capture, cpu_tree, exact, factory, native_class,
                     read, require, restore, seed_scipy, seed_streams, selector, sha, source_checks)


def evaluate(np, torch, roc_auc_score, helpers, model, data, seed, epoch, counters, phase):
    model.eval()
    before = capture(np, torch, helpers, model)
    try:
        seed_streams(np, torch, helpers, model, seed + 2000003 + epoch)
        with torch.no_grad():
            probabilities = []
            for draw in range(4):
                counters[phase + '_forward_attempts'] += 1
                logp, kl = model(data['x'])
                require(logp.shape == (data['x'].shape[0], 2) and torch.isfinite(logp).all().item()
                        and torch.isfinite(kl).all().item(), 'Finite native full-graph draw')
                probabilities.append(torch.exp(logp))
                counters[phase + '_forwards_completed'] += 1
            pooled = torch.log(torch.mean(torch.stack(probabilities), 0))
            role_logp = {role: pooled[data[role + '_index']].detach().cpu() for role in ('train', 'valid')}
            scores = {role: helpers.metrics(torch, roc_auc_score, role_logp[role], data[role + '_y'].cpu()) for role in ('train', 'valid')}
        return scores, role_logp
    finally:
        restore(np, torch, helpers, model, before)
        require(exact(torch, capture(np, torch, helpers, model), before), 'Evaluation restored all owned streams')


def optimizer(torch, model, cfg):
    sheaf, other = model.grouped_parameters()
    params = sheaf + other
    require(len({id(p) for p in params}) == len(params) == len(list(model.parameters()))
            and {id(p) for p in params} == {id(p) for p in model.parameters()}, 'Complete original parameter partition')
    return torch.optim.Adam([{'params': sheaf, 'weight_decay': cfg['sheaf_decay']},
                             {'params': other, 'weight_decay': cfg['weight_decay']}], lr=cfg['lr'])


def fit(np, torch, roc_auc_score, common, helpers, cls, protocol, data, role_meta, identity, output, seed):
    folder = output / ('seed' + str(seed))
    folder.mkdir(exist_ok=False)
    started, usage0 = time.perf_counter(), resource.getrusage(resource.RUSAGE_SELF)
    device = data['x'].device
    counters = {k: 0 for k in ('train_forward_attempts', 'train_forwards_completed', 'backward_attempts',
        'backwards_completed', 'Adam_attempts', 'Adam_steps_completed', 'validation_forward_attempts',
        'validation_forwards_completed', 'restore_forward_attempts', 'restore_forwards_completed')}
    record = dict(seed=seed, status='started', identity=identity, config_id=protocol['config_id'],
                  all_TRAIN_rows_used_each_epoch=True, full_graph_used=True, TEST_truth_present=False,
                  TRAIN_count=role_meta['role_counts']['train'], VALID_count=role_meta['role_counts']['valid'],
                  actual_directed_support=role_meta['support_counts']['canonical_directed'],
                  evaluation_draws=4, training_draws=1, automatic_retry=False, float_reload_threshold=None)
    common.write_json(folder / 'RESULT.json', record)
    model = opt = restored = restored_opt = None
    stage, best_epoch, best_key, stale = 'construct', None, None, 0
    checkpoint = folder / 'SELECTED_STATE.pt'
    try:
        if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
        helpers.seed_all(np, torch, seed)
        args = dict(protocol['native_args'], graph_size=data['x'].shape[0], input_dim=data['x'].shape[1], output_dim=2, device=str(device))
        make = factory(torch, cls, data['cpu_edge_index'], data['edge_index'], args)
        model = make()
        seed_scipy(np, model, seed)  # Keep Torch training stream after native initialization.
        opt = optimizer(torch, model, protocol['optimizer'])
        record.update(native_args=args, parameters=helpers.parameter_counts(model), construction_seconds=time.perf_counter()-started)
        topo = helpers.static_topology_bytes(model)
        # The borrowed topology counter covers the model/builder; include the private edge-weight index storage only if distinct.
        seen = {v.untyped_storage().data_ptr() for owner in (model, model.laplacian_builder)
                for v in vars(owner).values() if isinstance(v, torch.Tensor)}
        weight_index = model.weight_learner.full_left_right_idx
        if weight_index.untyped_storage().data_ptr() not in seen: topo += weight_index.untyped_storage().nbytes()
        record['static_topology_bytes'] = topo
        for epoch in range(1, protocol['max_epochs'] + 1):
            stage = 'train_epoch_' + str(epoch)
            record['last_attempted_epoch'] = epoch
            tick = time.perf_counter()
            model.train()
            opt.zero_grad(set_to_none=True)
            counters['train_forward_attempts'] += 1
            logp, kl = model(data['x'])
            counters['train_forwards_completed'] += 1
            beta = torch.sigmoid(torch.tensor(((epoch - 1) % 40) / 2 - 10))
            nll = torch.nn.functional.nll_loss(logp[data['train_index']], data['train_y'])
            loss = nll + beta * kl
            require(torch.isfinite(loss).item() and torch.isfinite(kl).item(), 'Finite native TRAIN NLL plus annealed KL')
            last_loss = dict(nll=float(nll.item()), kl=float(kl.item()), beta=float(beta.item()), objective=float(loss.item()))
            counters['backward_attempts'] += 1
            loss.backward()
            counters['backwards_completed'] += 1
            require(all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters()), 'Finite native gradients')
            counters['Adam_attempts'] += 1
            opt.step()
            counters['Adam_steps_completed'] += 1
            del logp, kl, nll, loss
            helpers.synchronize(torch, device)
            train_seconds = time.perf_counter()-tick
            stage = 'validation_epoch_' + str(epoch)
            tick = time.perf_counter()
            scores, role_logp = evaluate(np, torch, roc_auc_score, helpers, model, data, seed, epoch, counters, 'validation')
            helpers.synchronize(torch, device)
            valid_seconds = time.perf_counter()-tick
            key = selector(scores, epoch)
            selected = best_key is None or key > best_key
            if selected:
                best_key, best_epoch, stale = key, epoch, 0
                saved = dict(schema='owned-original-BSNN-selected-state-v1', identity=identity, seed=seed, epoch=epoch,
                    model_state=cpu_tree(torch, model.state_dict()), optimizer_state=cpu_tree(torch, opt.state_dict()),
                    training_rng=capture(np, torch, helpers, model), scores=scores, role_logp=role_logp,
                    eval_seed=seed + 2000003 + epoch, eval_draws=4, TEST_truth_saved=False)
                temp = checkpoint.with_suffix('.tmp')
                torch.save(saved, temp)
                temp.replace(checkpoint)
                del saved
            else: stale += 1
            common.append_jsonl(folder / 'HISTORY.jsonl', dict(epoch=epoch, TRAIN=scores['train'], VALID=scores['valid'],
                selected=selected, stale_epochs=stale, native_training=last_loss, train_seconds=train_seconds, validation_seconds=valid_seconds))
            record['epochs_completed'] = epoch
            record['selected_epoch'] = best_epoch
            common.write_json(folder / 'RESULT.json', dict(record, counters=counters))
            del role_logp
            if stale >= protocol['patience']: break
        stage = 'restore_selected_model_optimizer_and_streams'
        del model, opt
        model = opt = None
        gc.collect()
        if device.type == 'cuda':
            try: torch.cuda.empty_cache()
            except Exception as error:
                record['cleanup_failure'] = common.failure_record(error, 'empty_cache')
                common.write_json(folder / 'RESULT.json', record)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        require(saved['identity'] == identity and saved['seed'] == seed and saved['epoch'] == best_epoch
                and saved['eval_draws'] == 4, 'Owned selected checkpoint identity')
        restored = make()
        restored.load_state_dict(saved['model_state'], strict=True)
        require(exact(torch, cpu_tree(torch, restored.state_dict()), saved['model_state']), 'Exact selected parameter/buffer state')
        restored_opt = optimizer(torch, restored, protocol['optimizer'])
        restored_opt.load_state_dict(saved['optimizer_state'])
        require(exact(torch, cpu_tree(torch, restored_opt.state_dict()), saved['optimizer_state']), 'Exact selected optimizer state')
        restore(np, torch, helpers, restored, saved['training_rng'])
        require(exact(torch, capture(np, torch, helpers, restored), saved['training_rng']), 'Exact selected Python/NumPy/SciPy/Torch streams')
        stage = 'four_draw_selected_serving_reconstruction'
        scores, logp = evaluate(np, torch, roc_auc_score, helpers, restored, data, seed, best_epoch, counters, 'restore')
        differences = {role: dict(max_abs_logp=float((logp[role]-saved['role_logp'][role]).abs().max().item()),
            max_abs_probability=float((logp[role].exp()-saved['role_logp'][role].exp()).abs().max().item()),
            prediction_changes=int((logp[role].argmax(-1) != saved['role_logp'][role].argmax(-1)).sum().item()),
            metric_signed_differences={key: float(scores[role][key]-saved['scores'][role][key]) for key in scores[role]},
            metric_absolute_differences={key: abs(float(scores[role][key]-saved['scores'][role][key])) for key in scores[role]}) for role in ('train', 'valid')}
        record.update(status='complete', selected_epoch=best_epoch, scores=scores, selected_scores=saved['scores'],
            reload_diagnostics=differences, model_parameter_buffer_state_exact=True, optimizer_state_exact=True,
            selected_streams_restored=True, reported_state='fresh native constructor, selected model/optimizer, four-draw replay',
            checkpoint_sha256=sha(checkpoint), reload_output_bitwise_required=False, reload_float_threshold=None)
    except Exception as error:
        record.update(status='failed', failure=common.failure_record(error, stage))
        common.append_jsonl(output / 'FAILURES.jsonl', record)
    except BaseException as error:
        record.update(status='interrupted', failure=common.failure_record(error, stage))
        raise
    finally:
        try: helpers.synchronize(torch, device)
        except Exception as error:
            record.update(status='failed', cost_finalization_failure=common.failure_record(error, 'synchronize'))
        usage = resource.getrusage(resource.RUSAGE_SELF)
        record.update(counters=counters, complete_attempt_seconds=time.perf_counter()-started,
            CPU_user_seconds=usage.ru_utime-usage0.ru_utime, CPU_system_seconds=usage.ru_stime-usage0.ru_stime,
            process_RSS_high_water_cumulative_bytes=common.process_peak_rss_bytes(),
            cost_includes_construction_all_draws_failures_checkpoint_restore=True,
            completed_native_sheaf_samples=protocol['native_args']['layers'] * sum(counters[k] for k in
                ('train_forwards_completed', 'validation_forwards_completed', 'restore_forwards_completed')))
        if device.type == 'cuda':
            try:
                record.update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device), peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device))
            except Exception as error:
                record.update(status='failed', cuda_cost_failure=common.failure_record(error, 'cuda_peak_cost'))
        common.write_json(folder / 'RESULT.json', record)
        del model, opt, restored, restored_opt
        gc.collect()
        if device.type == 'cuda':
            try: torch.cuda.empty_cache()
            except Exception as error:
                record.update(status='failed', cleanup_failure=common.failure_record(error, 'empty_cache'))
                common.write_json(folder / 'RESULT.json', record)
    return record


def run(args):
    pins = source_checks()
    protocol = read(HERE / 'PROTOCOL.json')
    release = read(args.release)
    require(release['enabled'] is True and release['source_seal_sha256'] == sha(HERE / 'SEAL.json')
            and release['runtime_qualification_passed'] is True and release['native_bsnn_work_qualification_passed'] is True
            and release['baseline_configuration_reviewed'] is True, 'Explicit root-qualified and reviewed release')
    require(type(release['deterministic_algorithms']) is bool and release['expected_runtime_versions'], 'Qualified runtime/deterministic policy')
    require(set(release['expected_runtime_versions']) == {'torch', 'numpy', 'scipy', 'torch-geometric', 'torch-sparse', 'torch-scatter', 'torch-householder', 'scikit-learn'}, 'Complete root-qualified native dependency versions')
    require(len(release['execution_source_commit']) == 40 and all(c in '0123456789abcdef' for c in release['execution_source_commit']), 'Pinned committed wrapper identity')
    require(sha(args.roles) == release['roles_sha256'] and sha(args.roles.parent / 'ROLE.json') == release['role_metadata_sha256'], 'Exact same official role archive/metadata')
    output = args.output.resolve()
    require(str(output) == release['output_directory'] and not output.exists()
            and output.is_relative_to(PHASE.resolve()) and not output.is_relative_to(HERE), 'Fresh bound normal phase output')
    panel_started = time.perf_counter()
    panel_usage = resource.getrusage(resource.RUSAGE_SELF)
    common, helpers = borrowed(pins)
    output.mkdir(parents=True, exist_ok=False)
    records = []
    failure = None
    try:
        common.write_json(output / 'SCHEDULE.json', dict(protocol=protocol, release=release, required_seeds=protocol['seeds'], automatic_retry=False))
        import numpy as np
        import torch
        from sklearn.metrics import roc_auc_score
        require(str(torch.__version__) == '2.1.2+cu118' and np.__version__ == '1.26.4', 'Qualified existing float32 runtime')
        versions = {key: importlib.metadata.version(key) for key in ('torch', 'numpy', 'scipy', 'torch-geometric', 'torch-sparse', 'torch-scatter', 'torch-householder', 'scikit-learn')}
        require(all(versions.get(k) == v for k, v in release['expected_runtime_versions'].items()), 'Root-qualified provider versions')
        device = torch.device(release['device'])
        require(device == torch.device('cuda:0') and torch.cuda.device_count() == 1, 'One root-owned visible physical GPU')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(release['deterministic_algorithms'])
        arrays, role_meta, role_sha = common.read_roles(np, args.roles)
        require(role_meta.get('exposure_classification') == 'original_paper_benchmark_exploratory', 'Same corrected official Tolokers role classification')
        data = {key: torch.from_numpy(value).to(device) for key, value in arrays.items()}
        data['cpu_edge_index'] = torch.from_numpy(arrays['edge_index'])
        cls = native_class(pins)
        identity = dict(wrapper_manifest_sha256=read(HERE / 'SEAL.json')['manifest_sha256'], source_commit=release['execution_source_commit'],
            author_commit=pins['author_commit'], author_model_program_sha256=pins['author_model_program_sha256'],
            roles_sha256=release['roles_sha256'], role_metadata_sha256=role_sha, runtime_versions=versions,
            config=protocol, TEST_truth_present=False, exploratory_original_benchmark=True)
        common.write_json(output / 'RUN_IDENTITY.json', identity)
        for seed in protocol['seeds']:
            records.append(fit(np, torch, roc_auc_score, common, helpers, cls, protocol, data, role_meta, identity, output, seed))
            common.append_jsonl(output / 'ALL_FITS.jsonl', records[-1])
    except BaseException as error:
        failure = common.failure_record(error, 'panel')
        raise
    finally:
        seeds_without_returned_records = [s for s in protocol['seeds'] if s not in {r['seed'] for r in records}]
        for seed in protocol['seeds']:
            if seed not in {r['seed'] for r in records}:
                partial = output / ('seed' + str(seed)) / 'RESULT.json'
                record = read(partial) if partial.exists() else dict(seed=seed, status='failed_before_fit', failure=failure,
                    actual_fit_started=False, recorded_fit_cost_seconds=0, TEST_truth_present=False, automatic_retry=False)
                records.append(record)
        common.write_json(output / 'ALL_FITS.json', records)
        complete = len(records) == 3 and {r['seed'] for r in records} == set(protocol['seeds']) and all(r['status'] == 'complete' for r in records)
        panel_final_usage = resource.getrusage(resource.RUSAGE_SELF)
        common.write_json(output / 'COMPLETE.json', dict(complete=complete, required_seeds=protocol['seeds'], records=records,
            seeds_without_returned_records=seeds_without_returned_records, failure=failure,
            complete_panel_seconds=time.perf_counter()-panel_started,
            CPU_user_seconds=panel_final_usage.ru_utime-panel_usage.ru_utime,
            CPU_system_seconds=panel_final_usage.ru_stime-panel_usage.ru_stime,
            process_RSS_high_water_cumulative_bytes=common.process_peak_rss_bytes(),
            includes_setup_role_loading_imports_and_all_fit_costs=True,
            all3_required_before_comparison=True, comparative_opening_authorized=False, TEST_truth_present=False, automatic_retry=False))
    require(complete, 'All three paired attempts required; failures retained, no survivor summary')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--release', type=Path)
    parser.add_argument('--roles', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(inactive=True, protocol=read(HERE / 'PROTOCOL.json'), model_or_role_import=False)))
        return
    require(args.release and args.roles and args.output, 'Explicit release/roles/output required')
    run(args)


if __name__ == '__main__': main()
