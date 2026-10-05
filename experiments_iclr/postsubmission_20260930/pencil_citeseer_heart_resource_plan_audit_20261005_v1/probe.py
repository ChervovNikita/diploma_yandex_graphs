#!/usr/bin/env python3
"""Source-only preparation: full native PENCIL TRAIN/VALID, zero updates.

Launching requires an explicit root-reviewed release. Numerical imports occur
only inside main after the source, data authority, host and release gates. No
scientific score, checkpoint, optimizer state or TEST interface is provided.
"""
import argparse
import ast
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
GPU_UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
PREP = PHASE / 'pencil_collab_paired_predictive_preparation_20261004_v3'
ADAPTER = PHASE / 'pencil_citeseer_heart_competitor_source_plan_20261005_v1'
NCN = PHASE / 'citeseer_heart_ncn_trainval_runner_source_20261005_v1'
AVAILABLE = PHASE / 'citeseer_heart_official_acquisition_server_20261005_v1/AVAILABLE_MANIFEST.json'
FEATURE = PHASE / 'citeseer_feature_qualification_20261005_v2/RESULT.json'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, path)


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class QuietProgress:
    def __init__(self, *args, **kwargs): pass
    def update(self, *args, **kwargs): pass
    def set_description(self, *args, **kwargs): pass
    def close(self): pass


class ZeroUpdateTransform(ast.NodeTransformer):
    """The only native-loop change: replace all three optimizer step sites."""
    def __init__(self):
        self.replacements = 0

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if (isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == 'optimizer' and node.func.attr == 'step'):
            require(not node.args and not node.keywords, 'Native step signature changed')
            node.func.attr = 'resource_boundary'
            self.replacements += 1
        return node


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--release-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(platform.system() == 'Linux' and Path.cwd().resolve() == REPO,
            'Stage and launch from the authorized Linux repository')
    require(HERE.is_relative_to(PHASE) and args.release.resolve().is_relative_to(PHASE),
            'Prepared source/release leaves the authorized phase')
    require(sha(args.release) == args.release_sha256, 'Exact release differs')
    release = json.loads(args.release.read_text())
    require(release.get('root_source_review_approved') is True
            and release.get('full_zero_update_resource_probe_authorized') is True
            and release.get('preserve_active_ncn_cohort') is True,
            'Explicit reviewed qualification release absent')
    require(socket.gethostname() == 'anogena-2-0', 'Authorized one-GPU hostname differs')
    devices = subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                             check=True, capture_output=True, text=True, timeout=30)
    require(devices.stdout.split() == [GPU_UUID], 'Authorized singleton GPU UUID differs')
    require(os.environ.get('RANK') == os.environ.get('LOCAL_RANK') == '0'
            and os.environ.get('WORLD_SIZE') == '1', 'Exactly one directly supervised rank')
    require(sha(HERE / 'MANIFEST.json') == release['source_manifest_sha256'],
            'Reviewed source manifest differs')
    bindings = json.loads((HERE / 'MANIFEST.json').read_text())
    for row in bindings['files']:
        require(sha(HERE / row['path']) == row['sha256'], 'Prepared packet changed')
    for row in bindings['external_source_files']:
        path = PHASE / row['phase_relative']
        require(path.resolve().is_relative_to(PHASE)
                and sha(path) == row['sha256'], 'Pinned external source/receipt changed')
    require(sha(AVAILABLE) == '1b9c8bb57278d91b0f6212136225afcfd6b067c6b17e0dfed7ee36dd6316efdc'
            and sha(FEATURE) == 'cf1f3c2d66735838b8acb860e466fe459ed8e35b15c559878233e9ff7bb62a68',
            'Reusable authenticated data receipts differ')
    dependency = PHASE / release['dependency_binding_phase_relative']
    require(dependency.resolve().is_relative_to(PHASE)
            and sha(dependency) == release['dependency_binding_sha256'],
            'Reviewed one-GPU dependency admission absent')
    dependency_record = json.loads(dependency.read_text())
    require(dependency_record.get('host') == 'anogena-2-0'
            and dependency_record.get('one_gpu_native_pencil_import_pass') is True,
            '18.77 installation is insufficient; qualify exact one-GPU native imports')
    require(dependency_record.get('PYTHONPATH') == os.environ.get('PYTHONPATH')
            and dependency_record.get('installed_file_inventory_complete') is True,
            'Exact selected one-GPU dependency path/inventory admission required')
    for row in dependency_record['dependency_files']:
        path = Path(row['path'])
        require(path.resolve().is_relative_to(REPO) and sha(path) == row['sha256'],
                'An admitted repo-owned dependency file changed')
    require(args.output.resolve().is_relative_to(PHASE) and not args.output.exists(),
            'Output must be fresh inside the authorized phase')
    args.output.mkdir()
    started = time.monotonic()
    progress = dict(stage='imports', status='RUNNING', UTC=datetime.now(timezone.utc).isoformat(),
                    fits=0, optimizer_updates=0, would_be_optimizer_updates=0,
                    TRAIN_batches=0, TRAIN_queries=0, VALID_batches=0, VALID_queries=0,
                    TEST_access=False, ranking_metrics_computed=False,
                    scores_saved=False, checkpoints_saved=False, release_sha256=args.release_sha256)
    write(args.output / 'PROGRESS.json', progress)
    dist = None
    try:
        import torch
        import numpy as np
        import psutil
        import yaml
        import torch.distributed as dist
        from transformers import BertConfig
        for distribution, version in dependency_record['runtime_distribution_versions'].items():
            from importlib.metadata import version as package_version
            require(package_version(distribution) == version, 'Admitted dependency version changed: ' + distribution)
        require(torch.__version__ == '2.1.2+cu118' and np.__version__ == '1.26.4'
                and torch.cuda.device_count() == 1, 'NCN core runtime must remain untouched')
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
        for row in dependency_record['import_source_pins']:
            module = sys.modules.get(row['module'])
            require(module is not None and Path(module.__file__).resolve() == Path(row['path']).resolve(),
                    'One-GPU dependency import source differs: ' + row['module'])
        # Custody is checked after importing, including imports with generic names.
        for row in bindings['native_module_files']:
            module = sys.modules.get(row['module'])
            require(module is not None and Path(module.__file__).resolve() == PHASE / row['phase_relative'],
                    'Native module resolved outside the pinned source closure: ' + row['module'])
        torch.cuda.set_device(0)
        torch.cuda.init()
        require(not torch.are_deterministic_algorithms_enabled(), 'Native runtime profile differs')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('medium')
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        memory = dict(max_parent_plus_live_descendant_RSS_bytes=0,
                      max_parent_RSS_bytes=0, max_cuda_allocated_bytes=0, max_cuda_reserved_bytes=0)
        parent = psutil.Process()
        def check_resources():
            own = parent.memory_info().rss
            total = own
            for child in parent.children(recursive=True):
                try: total += child.memory_info().rss
                except psutil.NoSuchProcess: pass
            memory['max_parent_RSS_bytes'] = max(memory['max_parent_RSS_bytes'], own)
            memory['max_parent_plus_live_descendant_RSS_bytes'] = max(memory['max_parent_plus_live_descendant_RSS_bytes'], total)
            memory['max_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
            memory['max_cuda_reserved_bytes'] = torch.cuda.max_memory_reserved()
            require(time.monotonic() - started <= release['caps']['wall_seconds'], 'Reviewed wall ceiling exceeded')
            for key in ('max_parent_plus_live_descendant_RSS_bytes', 'max_cuda_allocated_bytes', 'max_cuda_reserved_bytes'):
                require(memory[key] <= release['caps'][key], 'Reviewed resource ceiling exceeded: ' + key)
            write(args.output / 'PROGRESS.json', {**progress, 'elapsed_seconds': time.monotonic() - started, **memory})
        dist.init_process_group('nccl', init_method=(args.output / 'RANK0_RENDEZVOUS').as_uri(),
                                rank=0, world_size=1, timeout=timedelta(minutes=30))
        run_lp.tqdm = QuietProgress
        config_dict = yaml.safe_load((ADAPTER / 'FEATURE_ENABLED_TRAIN_VALID_PROPOSAL.yaml').read_text())
        configs = utils.Config(config_dict)
        require(configs.num_epochs == 300 and configs.bf16 is False and configs.num_workers == 8
                and configs.batch_size_training == 256 and configs.gradient_accumulation_steps == 8
                and configs.max_num_samples == -1 and configs.debug is False
                and configs.load_model_path == 'None' and configs.resume == 0,
                'Full native proposed recipe changed')
        source = load_source('qualified_ncn_selective_data_source', NCN / 'run.py')
        data_job = dict(available_manifest_relative=str(AVAILABLE.relative_to(PHASE)),
                        available_manifest_sha256=sha(AVAILABLE), qualified_feature_shape=[3327, 3703])
        progress['stage'] = 'authenticated_inputs_native_constructors'
        x, train, valid, pool, identities = source.load_available(data_job)
        adapter = load_source('pencil_citeseer_selective_adapter', ADAPTER / 'train_valid_adapter.py')
        train_raw, valid_raw, coverage = adapter.build_train_valid(x, train, valid, pool, configs,
            dataset_class=ShaDowKHopSeqFromEdgesMapDataset, unique_edge_function=get_unique_edges_with_mapping)
        dist.barrier(device_ids=[0])
        utils.set_seed(configs.seed)
        feature_dim, is_gnn = utils.get_feature_dim(configs, train_raw, encoding_scheme='adjacency_row')
        require(feature_dim == 3703 and is_gnn is False, 'Native RNG-consuming feature discovery differs')
        tokenizer = run_lp.STokenizer(num_nodes=842)
        class LocalAutoConfig:
            @staticmethod
            def from_pretrained(name):
                require(name == 'bert-base-uncased', 'Unexpected non-weight configuration alias')
                return BertConfig.from_dict(json.loads((PREP / 'metadata/config.json').read_text()))
        lp_model.AutoConfig = LocalAutoConfig
        scratch = args.output / 'empty_model_factory_directory'
        scratch.mkdir()
        model = run_lp.get_model(configs, tokenizer, 0, 1, 0, None, str(scratch),
            use_features=True, feature_dim=feature_dim, encoding_scheme='adjacency_row',
            is_gnn=False, use_bf16=False)
        require(not list(scratch.iterdir()), 'Automatic resume or unrequested model output')
        scratch.rmdir()
        parameters = list(model.parameters())
        initial_parameters = {name: sha_tensor(value, torch) for name, value in model.named_parameters()}
        trainable_elements = sum(p.numel() for p in parameters if p.requires_grad)
        shapes = dict(max_sequence_positions=0, min_sequence_positions=844,
                      max_batch_queries=0, raw_feature_width=3703, structural_width=1686)
        phase = ['TRAIN']
        def observe_input(module, positional, kwargs):
            structural, features = kwargs['input_embeds'], kwargs['feature_embeds']
            batch, length, width = structural.shape
            require(width == 1686 and 2 <= length <= 844 and batch <= 256
                    and tuple(features.shape) == (batch, length, 3703)
                    and structural.dtype == features.dtype == torch.float32,
                    'Native structural/feature shape or precision differs')
            shapes['max_sequence_positions'] = max(shapes['max_sequence_positions'], length)
            shapes['min_sequence_positions'] = min(shapes['min_sequence_positions'], length)
            shapes['max_batch_queries'] = max(shapes['max_batch_queries'], batch)
            progress[phase[0] + '_batches'] += 1
            progress[phase[0] + '_queries'] += batch
            check_resources()
        def observe_output(module, positional, kwargs, output):
            require(output.logits.shape == (kwargs['input_embeds'].shape[0],)
                    and bool(torch.isfinite(output.logits).all()), 'Native output not finite/complete')
            if phase[0] == 'TRAIN':
                require(output.loss is not None and bool(torch.isfinite(output.loss)), 'Nonfinite native training loss')
            check_resources()
        pre_hook = model.module.register_forward_pre_hook(observe_input, with_kwargs=True)
        post_hook = model.module.register_forward_hook(observe_output, with_kwargs=True)
        class ZeroUpdateOptimizer:
            """No torch Optimizer, Adam state or step method is available."""
            def zero_grad(self):
                for parameter in parameters:
                    parameter.grad = None
            def resource_boundary(self):
                gradients = [p.grad for p in parameters if p.grad is not None]
                require(gradients and all(bool(torch.isfinite(g).all()) for g in gradients),
                        'Missing/nonfinite native accumulated gradients')
                progress['would_be_optimizer_updates'] += 1
                check_resources()
        transformed = ast.parse(inspect.getsource(run_lp.train_loop))
        rewrite = ZeroUpdateTransform()
        transformed = rewrite.visit(transformed)
        require(rewrite.replacements == 3, 'Expected precisely three native step call sites')
        ast.fix_missing_locations(transformed)
        namespace = dict(run_lp.__dict__)
        exec(compile(transformed, '<pinned-native-train-loop-only-step-sites-suppressed>', 'exec'), namespace)
        zero_train_loop = namespace['train_loop']
        collator = run_lp.Collator(tokenizer)
        train_loader, valid_loader, absent = run_lp.build_loaders(epoch=0, tokenizer=tokenizer,
            configs=configs, collator=collator, train_dataset_raw=train_raw, valid_dataset_raw=valid_raw,
            test_dataset_raw=None, use_features=True, encoding_scheme='adjacency_row', is_gnn=False)
        require(absent is None and train_loader.pin_memory is True and valid_loader.pin_memory is True,
                'Native loader transfer profile or TEST absence differs')
        coverage.update(TRAIN_mixed_queries_epoch0=len(train_raw), TRAIN_microbatches=len(train_loader),
                        TRAIN_negative_queries_epoch0=len(train_raw) - 3870,
                        VALID_microbatches=len(valid_loader), loader_pin_memory=True,
                        TRAIN_input_order_sha256=sha_tensor(train_raw.all_edges_with_y, torch),
                        VALID_unique_order_sha256=sha_tensor(valid_raw.all_edges_with_y, torch),
                        VALID_inverse_map_sha256=sha_tensor(valid_raw.orig_to_unique, torch))
        timings = dict(imports_runtime_data_model_seconds=time.monotonic() - started)
        progress['stage'] = 'full_TRAIN_backward_zero_update'
        train_start = time.monotonic()
        steps, samples = zero_train_loop(model, train_loader, ZeroUpdateOptimizer(),
            epoch=0, num_epochs=300, gradient_accumulation_steps=8, max_num_samples=-1,
            total_train_steps=0, total_trained_samples=0, rank=0, local_rank=0,
            configs=configs, wandb_run=None, world_size=1, is_gnn=False, use_bf16=False)
        torch.cuda.synchronize()
        timings['full_TRAIN_backward_zero_update_seconds'] = time.monotonic() - train_start
        require(steps == progress['TRAIN_batches'] == len(train_loader)
                and samples == progress['TRAIN_queries'] == len(train_raw)
                and progress['would_be_optimizer_updates'] == math.ceil(len(train_loader) / 8),
                'Full native TRAIN coverage/accumulation differs')
        require(all(p.grad is None for p in parameters), 'Native boundary clearing differs')
        progress['stage'] = 'full_VALID_forward_and_complete_inverse_restoration'
        phase[0] = 'VALID'
        valid_start = time.monotonic()
        output = run_lp.evaluate_loop(model, valid_loader, rank=0, world_size=1, evaluator=None,
            dataset_len=len(valid_raw), split_name='valid', show_progress=False,
            compute_loss=False, check_sequential_indices=True, is_gnn=False, use_bf16=False)
        torch.cuda.synchronize()
        timings['full_VALID_forward_restore_seconds'] = time.monotonic() - valid_start
        require(progress['VALID_queries'] == len(valid_raw) and progress['VALID_batches'] == len(valid_loader)
                and output['y_pred_all'].shape == output['y_true_all'].shape == (113727,)
                and bool(torch.isfinite(output['y_pred_all']).all())
                and output['metrics'] is None and 'avg_loss' not in output,
                'Complete VALID traversal/restoration without metrics differs')
        del output
        require({name: sha_tensor(value, torch) for name, value in model.named_parameters()} == initial_parameters,
                'A model parameter changed despite zero-step qualification')
        pre_hook.remove(); post_hook.remove()
        check_resources()
        progress.update(stage='complete', status='PASS_ZERO_UPDATE_RESOURCE_ONLY')
        # Adam has two persistent FP32 moment tensors per trainable parameter.
        # This is missing from this probe, and step temporaries remain unmeasured.
        result = {**progress, **memory, 'input_identities': identities, 'coverage': coverage,
                  'shapes': shapes, 'timings': timings, 'trainable_parameter_elements': trainable_elements,
                  'additional_Adam_FP32_moment_bytes_not_measured': trainable_elements * 8,
                  'full_fit_memory_readiness': False, 'predictive_competence': False,
                  'inclusive_seconds': time.monotonic() - started,
                  'elapsed_ETA_excludes_Adam_step_state_and_scientific_save_cost': True,
                  'parameter_hashes_equal_before_after': True}
        write(args.output / 'RESULT.json', result)
        print(json.dumps(result, sort_keys=True))
    except BaseException as error:
        progress.update(status='FAIL', error=type(error).__name__ + ': ' + str(error),
                        inclusive_seconds=time.monotonic() - started)
        write(args.output / 'FAILURE.json', progress)
        raise
    finally:
        if dist is not None and dist.is_initialized():
            dist.destroy_process_group()


def sha_tensor(tensor, torch):
    # Input/parameter custody only; this helper is never called on predictions.
    contiguous = tensor.detach().cpu().contiguous()
    return hashlib.sha256(contiguous.numpy().tobytes(order='C')).hexdigest()


if __name__ == '__main__':
    main()
