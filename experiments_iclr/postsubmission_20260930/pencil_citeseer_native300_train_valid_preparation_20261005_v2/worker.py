#!/usr/bin/env python3
"""One fresh complete native300 PENCIL fit with TRAIN/VALID-only interfaces."""
import argparse
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from common import (ADAPTER, HERE, NCN, PHASE, PREP, binary, file_row, gate,
                    load_source, output_bytes, require, sha, utc, write)


class QuietProgress:
    def __init__(self, *args, **kwargs): pass
    def update(self, *args, **kwargs): pass
    def set_description(self, *args, **kwargs): pass
    def close(self): pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--release-sha256', required=True)
    parser.add_argument('--seed', type=int, choices=(0, 1, 2), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    plan, release, dependency, bindings, available = gate(
            args.release, args.release_sha256, args.seed, args.output)
    args.output.mkdir()
    progress = dict(status='RUNNING', stage='imports', UTC=utc(), seed=args.seed,
            epoch_index=None, TRAIN_batches=0, TRAIN_queries=0, VALID_batches=0,
            VALID_queries=0, optimizer_updates=0, TEST_access=False,
            source_manifest_sha256=sha(HERE / 'MANIFEST.json'), release_sha256=args.release_sha256)
    write(args.output / 'PROGRESS.json', progress)
    dist = None
    input_hook = output_hook = None
    try:
        import torch
        import numpy as np
        import psutil
        import yaml
        import torch.distributed as dist
        from transformers import BertConfig
        from importlib.metadata import version as package_version
        for distribution, version in dependency['runtime_distribution_versions'].items():
            require(package_version(distribution) == version, 'Selected distribution version changed')
        require(torch.__version__ == '2.1.2+cu118' and np.__version__ == '1.26.4'
                and torch.cuda.device_count() == 1, 'Protected one-GPU core runtime differs')
        sys.path.insert(0, str(PREP / 'native'))
        old_cwd = Path.cwd()
        try:
            os.chdir(PREP / 'native')
            import run_lp
            import utils
            from datasets.dataset_map import ShaDowKHopSeqFromEdgesMapDataset
            from datasets.utils import get_unique_edges_with_mapping
            from models.transformers import lp_model
        finally:
            os.chdir(old_cwd)
        for row in dependency['import_source_pins']:
            module = sys.modules.get(row['module'])
            require(module is not None and Path(module.__file__).resolve() == Path(row['path']).resolve(),
                    'Admitted ancillary import source differs')
        for row in bindings['native_module_files']:
            module = sys.modules.get(row['module'])
            require(module is not None and Path(module.__file__).resolve() == PHASE / row['phase_relative'],
                    'Native module resolved outside pinned closure')
        torch.cuda.set_device(0)
        torch.cuda.init()
        require(not torch.are_deterministic_algorithms_enabled(), 'Native runtime profile differs')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('medium')
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        parent = psutil.Process()
        memory = dict(max_parent_plus_live_descendant_RSS_bytes=0,
                      max_cuda_allocated_bytes=0, max_cuda_reserved_bytes=0)

        def check_resources():
            total = parent.memory_info().rss
            for descendant in parent.children(recursive=True):
                try:
                    total += descendant.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            memory['max_parent_plus_live_descendant_RSS_bytes'] = max(
                    memory['max_parent_plus_live_descendant_RSS_bytes'], total)
            memory['max_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
            memory['max_cuda_reserved_bytes'] = torch.cuda.max_memory_reserved()
            caps = release['caps']
            require(time.monotonic() - started <= caps['per_fit_wall_seconds'], 'Per-fit wall ceiling exceeded')
            for key in memory:
                require(memory[key] <= caps[key], 'Resource ceiling exceeded: ' + key)
            require(output_bytes(args.output) <= caps['max_fit_retained_and_temporary_output_bytes'],
                    'Selected/temporary output retention ceiling exceeded')

        rendezvous = args.output / 'RANK0_RENDEZVOUS'
        dist.init_process_group('nccl', init_method=rendezvous.as_uri(), rank=0,
                                world_size=1, timeout=timedelta(minutes=30))
        run_lp.tqdm = QuietProgress
        config_dict = yaml.safe_load((ADAPTER / 'FEATURE_ENABLED_TRAIN_VALID_PROPOSAL.yaml').read_text())
        config_dict['seed'] = args.seed
        configs = utils.Config(config_dict)
        require(configs.num_epochs == 300 and configs.eval_every == 2 and configs.bf16 is False
                and configs.num_workers == 8 and configs.batch_size_training == 256
                and configs.gradient_accumulation_steps == 8 and configs.max_num_samples == -1
                and configs.debug is False and configs.load_model_path == 'None' and configs.resume == 0
                and configs.use_features is True and configs.feature_fusion == 'early'
                and configs.lr == .0001 and configs.weight_decay == .01,
                'Prospectively frozen native recipe differs')
        data_source = load_source('pencil_native300_selective_inputs', NCN / 'run.py')
        x, train, valid, pool, identities = data_source.load_available(dict(
                available_manifest_relative=str(available.relative_to(PHASE)),
                available_manifest_sha256=sha(available), qualified_feature_shape=[3327, 3703]))
        adapter = load_source('pencil_native300_selective_adapter', ADAPTER / 'train_valid_adapter.py')
        train_raw, valid_raw, coverage = adapter.build_train_valid(x, train, valid, pool, configs,
                dataset_class=ShaDowKHopSeqFromEdgesMapDataset, unique_edge_function=get_unique_edges_with_mapping)
        dist.barrier(device_ids=[0])
        utils.set_seed(configs.seed)
        feature_dim, is_gnn = utils.get_feature_dim(configs, train_raw, encoding_scheme='adjacency_row')
        require(feature_dim == 3703 and is_gnn is False, 'Real RNG-consuming feature discovery differs')
        tokenizer = run_lp.STokenizer(num_nodes=842)

        class LocalAutoConfig:
            @staticmethod
            def from_pretrained(name):
                require(name == 'bert-base-uncased', 'Unexpected non-weight config alias')
                return BertConfig.from_dict(json.loads((PREP / 'metadata/config.json').read_text()))

        lp_model.AutoConfig = LocalAutoConfig
        scratch = args.output / 'empty_model_factory_directory'
        scratch.mkdir()
        model = run_lp.get_model(configs, tokenizer, 0, 1, 0, None, str(scratch),
                use_features=True, feature_dim=feature_dim, encoding_scheme='adjacency_row',
                is_gnn=False, use_bf16=False)
        require(not list(scratch.iterdir()), 'Unexpected automatic checkpoint/resume output')
        scratch.rmdir()
        parameters = list(model.parameters())
        trainable_elements = sum(p.numel() for p in parameters if p.requires_grad)
        require(trainable_elements == 22693889, 'Qualified trainable model size differs')
        optimizer = torch.optim.AdamW(model.parameters(), lr=configs.lr, weight_decay=configs.weight_decay)
        require(not optimizer.state, 'Fresh native AdamW must have no donor state')
        native_step = optimizer.step
        first_adam = None

        def counted_native_step(*step_args, **step_kwargs):
            nonlocal first_adam
            active = [parameter for parameter in parameters if parameter.grad is not None]
            require(active and all(bool(torch.isfinite(p.grad).all()) for p in active),
                    'Missing/nonfinite native accumulated gradients')
            # Exactly one unchanged ordinary native AdamW step; no audit update.
            returned = native_step(*step_args, **step_kwargs)
            progress['optimizer_updates'] += 1
            check_resources()
            if first_adam is None:
                require(all(bool(torch.isfinite(p).all()) for p in parameters),
                        'Nonfinite parameter after first ordinary AdamW step')
                moment_bytes = 0
                for parameter in active:
                    state = optimizer.state[parameter]
                    require(set(state) == {'step', 'exp_avg', 'exp_avg_sq'}, 'Unexpected native AdamW state')
                    require(bool(torch.isfinite(state['step']).all()) and float(state['step']) == 1,
                            'Native first-step state counter differs')
                    for key in ('exp_avg', 'exp_avg_sq'):
                        value = state[key]
                        require(value.shape == parameter.shape and value.dtype == parameter.dtype
                                and value.device == parameter.device and bool(torch.isfinite(value).all()),
                                'AdamW moment shape/dtype/device/finiteness differs')
                        moment_bytes += value.numel() * value.element_size()
                first_adam = dict(status='PASS_FIRST_ORDINARY_NATIVE_ADAMW_UPDATE', UTC=utc(),
                        optimizer_update_index=1, extra_audit_updates=0,
                        active_gradient_parameter_elements=sum(p.numel() for p in active),
                        total_trainable_parameter_elements=trainable_elements,
                        active_parameter_state_count=len(active), actual_FP32_moment_bytes=moment_bytes,
                        all_active_moments_allocated_and_finite=True, all_model_parameters_finite=True,
                        **memory, full_later_epoch_readiness=False, TEST_access=False)
                write(args.output / 'FIRST_ADAM_UPDATE.json', first_adam)
            return returned

        optimizer.step = counted_native_step
        collator = run_lp.Collator(tokenizer)
        evaluator = run_lp.Evaluator(metric='heart-citeseer')
        shapes = dict(max_sequence_positions=0, min_sequence_positions=844,
                      max_batch_queries=0, raw_feature_width=3703, structural_width=1686)
        phase = ['TRAIN']

        def observe_input(module, positional, kwargs):
            structural, features = kwargs['input_embeds'], kwargs['feature_embeds']
            batch, length, width = structural.shape
            require(width == 1686 and 2 <= length <= 844 and batch <= 256
                    and tuple(features.shape) == (batch, length, 3703)
                    and structural.dtype == features.dtype == torch.float32, 'Native batch shape/precision differs')
            shapes['max_sequence_positions'] = max(shapes['max_sequence_positions'], length)
            shapes['min_sequence_positions'] = min(shapes['min_sequence_positions'], length)
            shapes['max_batch_queries'] = max(shapes['max_batch_queries'], batch)
            progress[phase[0] + '_batches'] += 1
            progress[phase[0] + '_queries'] += batch
            check_resources()

        def observe_output(module, positional, kwargs, output):
            require(output.logits.shape == (kwargs['input_embeds'].shape[0],)
                    and bool(torch.isfinite(output.logits).all()), 'Nonfinite native output or wrong shape')
            if phase[0] == 'TRAIN':
                require(output.loss is not None and bool(torch.isfinite(output.loss)), 'Nonfinite native TRAIN loss')
            check_resources()
            write(args.output / 'PROGRESS.json', {**progress, **memory,
                                                'elapsed_seconds': time.monotonic() - started})

        input_hook = model.module.register_forward_pre_hook(observe_input, with_kwargs=True)
        output_hook = model.module.register_forward_hook(observe_output, with_kwargs=True)
        setup_seconds = time.monotonic() - started

        def tensor_sha(tensor):
            return __import__('hashlib').sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

        def save_selected(epoch, score, evaluator_score, output, train_loader, valid_loader):
            complete = output['y_pred_all'].detach().cpu()
            require(complete.dtype == torch.float32 and complete.shape == (113727,), 'Native restored VALID score dtype/shape differs')
            positive = complete[:227].numpy().copy()
            negative = complete[227:].reshape(227, 500).numpy().copy()
            require(np.isfinite(positive).all() and np.isfinite(negative).all(), 'Nonfinite selected VALID scores')
            numpy_rng = np.random.get_state()
            state = dict(schema='pencil_citeseer_selected_native300_full_state_v1',
                    source_manifest_sha256=sha(HERE / 'MANIFEST.json'), release_sha256=args.release_sha256,
                    seed=args.seed, epoch_index=epoch, epoch_number=epoch + 1,
                    selection_metric='VALID_MRR', native_broadcast_selection_score=score,
                    prebroadcast_evaluator_MRR=evaluator_score, tie='first epoch',
                    model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                    native_total_steps=total_steps, native_total_trained_samples=total_samples,
                    optimizer_updates=progress['optimizer_updates'], config=config_dict, scheduler=None,
                    input_identities=identities,
                    rng_state=dict(python=random.getstate(), numpy=dict(kind=numpy_rng[0], keys=numpy_rng[1].tolist(),
                        position=int(numpy_rng[2]), has_gaussian=int(numpy_rng[3]), cached_gaussian=float(numpy_rng[4])),
                        torch_cpu=torch.get_rng_state(), torch_cuda=torch.cuda.get_rng_state_all(),
                        train_loader_generator=train_loader.generator.get_state(),
                        valid_loader_generator=valid_loader.generator.get_state(),
                        distributed_sampler_epoch=train_loader.sampler.epoch),
                    initialization='fresh native scratch', resource_donor_loaded=False,
                    TEST_access=False, other_fit_donor=False)
            binary(args.output / 'SELECTED_FULL_STATE.pt', lambda stream: torch.save(state, stream), check_resources)
            binary(args.output / 'VALID_POSITIVE_SCORES.npy', lambda stream: np.save(stream, positive, allow_pickle=False), check_resources)
            binary(args.output / 'VALID_NEGATIVE_SCORES.npy', lambda stream: np.save(stream, negative, allow_pickle=False), check_resources)
            selected = dict(status='SELECTED_TRAIN_VALID_NATIVE300_STATE', seed=args.seed,
                    epoch_index=epoch, epoch_number=epoch + 1, metric='VALID_MRR',
                    native_broadcast_selection_score=score, prebroadcast_evaluator_MRR=evaluator_score,
                    first_tie=True, restored_occurrences=113727, score_dtype='float32',
                    positive_shape=[227], negative_shape=[227, 500],
                    files=[file_row(args.output / name) for name in
                        ('SELECTED_FULL_STATE.pt', 'VALID_POSITIVE_SCORES.npy', 'VALID_NEGATIVE_SCORES.npy')],
                    selected_state_server_only=True, TEST_access=False, other_fit_donor=False)
            write(args.output / 'SELECTION.json', selected)
            check_resources()
            return selected

        history = []
        selected = None
        best = -float('inf')
        total_steps = total_samples = 0
        for epoch in range(300):
            progress.update(epoch_index=epoch, stage='complete_native_TRAIN_epoch')
            phase[0] = 'TRAIN'
            before = dict(progress)
            train_loader, valid_loader, absent = run_lp.build_loaders(epoch=epoch, tokenizer=tokenizer,
                    configs=configs, collator=collator, train_dataset_raw=train_raw, valid_dataset_raw=valid_raw,
                    test_dataset_raw=None, use_features=True, encoding_scheme='adjacency_row', is_gnn=False)
            require(absent is None and train_loader.pin_memory is True and valid_loader.pin_memory is True
                    and train_loader.num_workers == valid_loader.num_workers == 8
                    and not train_loader.drop_last and not valid_loader.drop_last, 'Native loader policy differs')
            queries = len(train_raw)
            positives = int((train_raw.all_edges_with_y[:, 2] == 1).sum())
            negatives = int((train_raw.all_edges_with_y[:, 2] == 0).sum())
            require(positives == 3870 and 0 < negatives <= 3870 and positives + negatives == queries,
                    'Native ratio-one sampling population differs; no redraw/truncation allowed')
            batches = len(train_loader)
            stream = dict(epoch_index=epoch, actual_TRAIN_positives=positives, actual_TRAIN_negatives=negatives,
                    requested_TRAIN_negatives=3870, native_mixed_queries_sha256=tensor_sha(train_raw.all_edges_with_y),
                    distributed_sampler_order_sha256=tensor_sha(torch.tensor(list(train_loader.sampler))),
                    loader_seed=args.seed * 100 + epoch, native_negative_underfill_preserved=True)
            old_steps, old_samples = total_steps, total_samples
            torch.cuda.synchronize()
            train_start = time.monotonic()
            # Native source and all original optimizer-step sites remain unchanged.
            total_steps, total_samples = run_lp.train_loop(model, train_loader, optimizer,
                    epoch=epoch, num_epochs=300, gradient_accumulation_steps=8, max_num_samples=-1,
                    total_train_steps=total_steps, total_trained_samples=total_samples, rank=0, local_rank=0,
                    configs=configs, wandb_run=None, world_size=1, is_gnn=False, use_bf16=False)
            torch.cuda.synchronize()
            train_seconds = time.monotonic() - train_start
            updates = progress['optimizer_updates'] - before['optimizer_updates']
            require(total_steps - old_steps == progress['TRAIN_batches'] - before['TRAIN_batches'] == batches
                    and total_samples - old_samples == progress['TRAIN_queries'] - before['TRAIN_queries'] == queries
                    and updates == math.ceil(batches / 8) and all(p.grad is None for p in parameters),
                    'Complete native TRAIN/update/partial-window clearing differs')
            require(all(bool(torch.isfinite(p).all()) for p in parameters), 'Nonfinite native trained parameter')
            row = dict(epoch_index=epoch, TRAIN_batches=batches, TRAIN_queries=queries,
                       optimizer_updates=updates, stream=stream, TRAIN_seconds=train_seconds,
                       VALID_performed=False, selection_publication_seconds=0)
            if (epoch + 1) % 2 == 0:
                phase[0] = 'VALID'
                progress['stage'] = 'complete_native_VALID_and_restoration'
                valid_start = time.monotonic()
                output = run_lp.evaluate_loop(model, valid_loader, rank=0, world_size=1,
                        evaluator=evaluator, dataset_len=len(valid_raw), split_name='valid', show_progress=False,
                        compute_loss=False, check_sequential_indices=True, is_gnn=False, use_bf16=False)
                torch.cuda.synchronize()
                valid_seconds = time.monotonic() - valid_start
                require(output['y_pred_all'].shape == output['y_true_all'].shape == (113727,)
                        and torch.equal(output['idx_all'], torch.arange(113727, device=0))
                        and bool((output['y_true_all'][:227] == 1).all())
                        and bool((output['y_true_all'][227:] == 0).all())
                        and progress['VALID_queries'] - before['VALID_queries'] == len(valid_raw)
                        and progress['VALID_batches'] - before['VALID_batches'] == len(valid_loader),
                        'Complete native unique VALID traversal and occurrence restoration differs')
                evaluator_score = float(output['metrics']['mrr'])
                require(math.isfinite(evaluator_score) and 0 <= evaluator_score <= 1, 'Native VALID MRR invalid')
                # Preserve run_lp.main's native float32 score broadcast before selection.
                raw_score_tensor = torch.tensor(evaluator_score, device=0)
                dist.broadcast(raw_score_tensor, src=0)
                score = utils.normalize_score(raw_score_tensor.item(), 'mrr')
                improved = score > best
                save_start = time.monotonic()
                if improved:
                    selected = save_selected(epoch, score, evaluator_score, output, train_loader, valid_loader)
                    best = score
                row.update(VALID_performed=True, VALID_batches=len(valid_loader),
                        VALID_unique_queries=len(valid_raw), VALID_restored_occurrences=113727,
                        prebroadcast_VALID_MRR=evaluator_score, native_selection_VALID_MRR=score,
                        strict_improvement=improved, first_tie=True, VALID_seconds=valid_seconds,
                        selection_publication_seconds=time.monotonic() - save_start)
                del output
            history.append(row)
            write(args.output / 'EPOCH_HISTORY.json', dict(seed=args.seed, epochs=history, TEST_access=False))
            del train_loader, valid_loader
            check_resources()
        evaluated = [row for row in history if row['VALID_performed']]
        require(len(history) == 300 and len(evaluated) == 150 and first_adam is not None and selected is not None,
                'Complete native300 cohort member, first Adam state and selection required')
        first_best = max(evaluated, key=lambda row: row['native_selection_VALID_MRR'])
        require(selected['epoch_index'] == first_best['epoch_index']
                and selected['native_broadcast_selection_score'] == first_best['native_selection_VALID_MRR'],
                'Native first-tie selection differs')
        require(progress['TRAIN_batches'] == sum(row['TRAIN_batches'] for row in history)
                and progress['TRAIN_queries'] == sum(row['TRAIN_queries'] for row in history)
                and progress['optimizer_updates'] == sum(row['optimizer_updates'] for row in history)
                and progress['VALID_queries'] == 150 * len(valid_raw), 'Full native fit counters differ')
        check_resources()
        progress.update(status='COMPLETE_NATIVE300_TRAIN_VALID_FIT', stage='complete')
        result = dict(**progress, **memory, completed_epochs=300, complete_VALID_passes=150,
                native300_selection=selected, first_Adam_update=first_adam, shapes=shapes,
                authenticated_input_identities=identities, coverage=coverage,
                setup_seconds=setup_seconds, inclusive_worker_seconds=time.monotonic() - started,
                source_changes=False, native_optimizer_steps_preserved=True, checkpoint_server_only=True,
                resource_donor_loaded=False, automatic_retry=False, exact_official_SOTA_reproduction=False)
        write(args.output / 'FIT.json', result)
        check_resources()
        names = ['FIRST_ADAM_UPDATE.json', 'EPOCH_HISTORY.json', 'SELECTION.json',
                 'SELECTED_FULL_STATE.pt', 'VALID_POSITIVE_SCORES.npy', 'VALID_NEGATIVE_SCORES.npy', 'FIT.json']
        write(args.output / 'FINAL_CUSTODY.json', dict(complete=True, seed=args.seed,
                files=[file_row(args.output / name) for name in names],
                inclusive_worker_seconds=time.monotonic() - started, TEST_access=False))
        write(args.output / 'PROGRESS.json', {**progress, **memory,
                                            'elapsed_seconds': time.monotonic() - started})
    except BaseException as error:
        progress.update(status='FAIL_PRESERVED_NO_RETRY', error=type(error).__name__ + ': ' + str(error),
                        inclusive_worker_seconds=time.monotonic() - started)
        write(args.output / 'FAILURE.json', progress)
        raise
    finally:
        if input_hook is not None:
            input_hook.remove()
        if output_hook is not None:
            output_hook.remove()
        if dist is not None and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
