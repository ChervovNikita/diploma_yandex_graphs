"""Exactly one complete native PENCIL epoch and full VALID; no retained scores/state."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import threading
import time
from common import DATA_FILES, HERE, final_file_custody, gate, native_module_custody, require, sha, write


class QuietProgress:
    """Suppress native progress text containing training loss; no numerical behavior changes."""
    def __init__(self, *args, **kwargs): pass
    def update(self, *args, **kwargs): pass
    def set_description(self, *args, **kwargs): pass
    def close(self): pass


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--release-sha256', required=True)
    args = parser.parse_args()
    plan, runtime, execution = gate(args.release, args.release_sha256)
    require(os.environ.get('RANK') == os.environ.get('LOCAL_RANK') == '0'
            and os.environ.get('WORLD_SIZE') == '1', 'Exactly one torchrun rank required')
    child = execution/'run01'; child.mkdir(exist_ok=False)
    progress = dict(stage='imports', TRAIN_batches=0, TRAIN_queries=0, VALID_batches=0, VALID_queries=0,
                    optimizer_updates=0, TEST_reads=False, scores_saved=False, status='RUNNING')
    write(child/'PROGRESS.json', progress)
    sys.path.insert(0, str(HERE/'native'))
    import torch
    import torch.distributed as dist
    import torch_sparse
    import torch_scatter
    import yaml
    from transformers import BertConfig
    from torch_geometric.data import Data
    from data_adapter import load_data, tensor_sha
    import_cwd = Path.cwd()
    try:
        # Native definitions.find_root() can start at cwd. Its owned marker is
        # present in this source closure, avoiding any repository-root mutation.
        os.chdir(HERE/'native')
        import run_lp
        import utils
        from datasets.ogbl.dataset import filter_by_year
        from datasets.dataset_map import ShaDowKHopSeqFromEdgesMapDataset
        from models.transformers import lp_model
    finally:
        os.chdir(import_cwd)
    native_module_custody()

    require(not torch.are_deterministic_algorithms_enabled(), 'Native default deterministic-algorithms profile differs')
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'One isolated CUDA device required')
    torch.cuda.set_device(0)
    torch.cuda.init()  # Explicitly initialize before allocator reset (the proven v4 startup repair).
    properties = torch.cuda.get_device_properties(0)
    require(properties.name == runtime['gpu_name'] and properties.total_memory == runtime['gpu_total_memory_bytes'], 'GPU differs')
    # Preserve run_lp.main's native throughput/precision profile, which differs from NCNC's float32 profile.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('medium')
    torch.set_num_threads(2)
    torch.cuda.synchronize(0); torch.cuda.reset_peak_memory_stats(0)
    for row in runtime['runtime_source_pins']:
        if row['module'] in ('torch_sparse', 'torch_scatter'):
            module = sys.modules[row['module']]
            require(Path(module.__file__).resolve() == Path(row['path']).resolve(), 'Sparse/scatter import path differs')
    release = json.loads(args.release.read_text())
    from common import dependency_custody
    dependency = dependency_custody(plan, release)
    for row in dependency['added_package_source_pins']:
        if row['module'] in sys.modules:
            require(Path(sys.modules[row['module']].__file__).resolve() == Path(row['path']).resolve(), 'Added package import differs')
    known_binary_paths = {Path(row['path']).resolve() for row in runtime['runtime_binary_files']}
    extension_dirs = {Path(row['path']).resolve().parent for row in runtime['runtime_source_pins']
                      if row['module'] in ('torch_sparse', 'torch_scatter')}
    loaded = {Path(x).resolve() for x in torch.ops.loaded_libraries if Path(x).resolve().parent in extension_dirs}
    require(loaded and loaded <= known_binary_paths
            and all(any(x.parent == d for x in loaded) for d in extension_dirs), 'Unadmitted sparse/scatter binary loaded')

    ended = threading.Event(); monitor_errors = []
    def check_caps():
        require(time.monotonic()-started <= plan['caps']['wall_seconds'], 'Child wall cap')
        require(torch.cuda.max_memory_allocated(0) <= plan['caps']['cuda_allocated_bytes'], 'CUDA allocated cap')
        require(torch.cuda.max_memory_reserved(0) <= plan['caps']['cuda_reserved_bytes'], 'CUDA reserved cap')
        require(not monitor_errors, 'Resource monitor failed: ' + repr(monitor_errors))
    def monitor():
        while not ended.wait(.25):
            try:
                check_caps()
            except BaseException as error:
                monitor_errors.append(type(error).__name__ + ': ' + str(error))
                try:
                    write(child/'MONITOR_FAILURE.json', dict(status='RESOURCE_MONITOR_FAILED',
                          conditions=list(monitor_errors),stage=progress['stage'],
                          elapsed_seconds=time.monotonic()-started,TEST_reads=False,scores_saved=False))
                finally:
                    # A separate failure receipt avoids racing the ordinary progress file.
                    # Its presence can never satisfy the complete-success output envelope.
                    # Parent cleanup closes this exact worker's owned descendants.
                    os._exit(2)
    watcher = threading.Thread(target=monitor, daemon=True); watcher.start()
    timings = {}; phase_started = started
    dist.init_process_group('nccl', init_method='env://', world_size=1, rank=0, timeout=timedelta(minutes=30))
    run_lp.tqdm = QuietProgress
    config_dict = yaml.safe_load((HERE/'OFFICIAL_CONFIG_REFERENCE.yaml').read_text())
    config_dict.update(use_features=True, feature_fusion='early', seed=0, max_sequence_length=154)
    configs = utils.Config(config_dict)
    require(configs.num_epochs == 20 and configs.max_num_samples == -1 and configs.debug is False
            and configs.bf16 is True and configs.use_ddp is True and configs.load_model_path == 'None'
            and configs.resume == 0, 'Frozen native recipe differs')
    authority = json.loads((HERE/'metadata/DATA_AUTHORITY.json').read_text())
    progress['stage'] = 'authenticated_data_and_native_constructors'; write(child/'PROGRESS.json', progress)
    data = load_data(authority)
    graph = Data(num_nodes=235868, x=data['x'], edge_index=data['raw_edge_index'], id=torch.arange(235868, dtype=torch.long))
    graph, split_edge = filter_by_year(graph, data['split_edge'])
    # Native _read_ogbl_collab casts TRAIN weights before constructing TRAIN/VALID.
    graph.edge_weight = graph.edge_weight.to(torch.float)
    del data['raw_edge_index']
    require(set(split_edge) == {'train', 'valid'}, 'Selective adapter opened another split')
    filtered_records = len(split_edge['train']['edge'])
    train_raw = ShaDowKHopSeqFromEdgesMapDataset(graph, configs.sampling_config, pretrain_mode=False,
                                                split_edge=split_edge, data_split='train')
    valid_raw = ShaDowKHopSeqFromEdgesMapDataset(graph, configs.sampling_config, pretrain_mode=False,
                                                split_edge=split_edge, data_split='valid', labels=None, orig_to_unique=None)
    require(len(valid_raw) == 160084, 'Full official VALID query population differs')
    dist.barrier(device_ids=[0])
    utils.set_seed(configs.seed)
    feature_dim, is_gnn = utils.get_feature_dim(configs, train_raw, encoding_scheme='adjacency_row')
    require(feature_dim == 128 and is_gnn is False, 'Native feature discovery differs')
    depth, neighbors = configs.sampling_config.edge_ego.depth_neighbors[-1]
    max_nodes = 2*sum(neighbors**d for d in range(depth+1)); require(max_nodes == 152, 'Native node bound differs')
    tokenizer = run_lp.STokenizer(num_nodes=max_nodes)

    class PinnedLocalAutoConfig:
        @staticmethod
        def from_pretrained(name):
            require(name == 'bert-base-uncased', 'Unexpected model-config alias')
            # Only the author-default configuration is supplied; no weights/network/cache lookup.
            return BertConfig.from_dict(json.loads((HERE/'metadata/config.json').read_text()))
    lp_model.AutoConfig = PinnedLocalAutoConfig
    scratch = child/'empty_model_directory'; scratch.mkdir()
    model = run_lp.get_model(configs, tokenizer, 0, 1, 0, None, str(scratch), use_features=True,
                             feature_dim=feature_dim, encoding_scheme='adjacency_row', is_gnn=False, use_bf16=True)
    require(not list(scratch.iterdir()), 'Unexpected automatic checkpoint state'); scratch.rmdir()
    optimizer = torch.optim.AdamW(model.parameters(), lr=configs.lr, weight_decay=configs.weight_decay)
    original_step = optimizer.step
    def counted_step(*step_args, **step_kwargs):
        value = original_step(*step_args, **step_kwargs)
        progress['optimizer_updates'] += 1; check_caps(); return value
    optimizer.step = counted_step
    collator = run_lp.Collator(tokenizer)
    train_loader, valid_loader, absent_loader = run_lp.build_loaders(
        epoch=0, tokenizer=tokenizer, configs=configs, collator=collator,
        train_dataset_raw=train_raw, valid_dataset_raw=valid_raw, test_dataset_raw=None,
        use_features=True, encoding_scheme='adjacency_row', is_gnn=False)
    require(absent_loader is None and train_loader.drop_last is False and valid_loader.drop_last is False
            and train_loader.num_workers == valid_loader.num_workers == 12, 'Native complete loader policy differs')
    train_queries = len(train_raw)
    train_positive = int((train_raw.all_edges_with_y[:, 2] == 1).sum())
    train_negative = int((train_raw.all_edges_with_y[:, 2] == 0).sum())
    require(train_positive + train_negative == train_queries and train_positive == int(round(filtered_records*.5))
            and 0 < train_negative <= train_positive, 'Native epoch0 query count differs; no redraw or hidden truncation')
    stream = dict(filtered_TRAIN_records=filtered_records, filtered_TRAIN_records_sha256=tensor_sha(split_edge['train']['edge']),
                  filtered_graph_sha256=tensor_sha(graph.edge_index), filtered_graph_edges=graph.edge_index.shape[1],
                  filtered_graph_weights_sha256=tensor_sha(graph.edge_weight), authenticated_data_digests=data['digests'],
                  TRAIN_positive_queries=train_positive, TRAIN_negative_queries=train_negative,
                  requested_native_negative_queries=train_positive,
                  VALID_positive_queries=60084, VALID_negative_queries=100000,
                  native_mixed_queries_sha256=tensor_sha(train_raw.all_edges_with_y),
                  native_sample_indices_sha256=tensor_sha(train_raw.sample_idx), sampler_order_sha256=tensor_sha(torch.tensor(train_raw.sampler)),
                  actual_distributed_sampler_order_sha256=tensor_sha(torch.tensor(list(train_loader.sampler))),
                  native_50_percent_policy=True, future_positive_rejection=False, seed=0, epoch=0)
    torch.cuda.synchronize(0); timings['imports_runtime_data_model_loader_seconds'] = time.monotonic()-phase_started
    shapes = dict(structural_width=306, raw_feature_width=128, max_sequence_positions=0, min_sequence_positions=2048,
                  tokenizer_max_nodes=152, model_max_position_embeddings=2048, full_batch_size=1024)
    phase = ['TRAIN']
    def observe_input(module, positional, kwargs):
        tensor = kwargs['input_embeds']; feature = kwargs['feature_embeds']; count = tensor.shape[0]
        require(tensor.ndim == 3 and tensor.shape[2] == 306 and 2 <= tensor.shape[1] <= 154
                and feature.shape == (count, tensor.shape[1], 128), 'Actual native batch shape differs')
        require(kwargs['num_nodes'].shape == (count,) and bool((kwargs['num_nodes'] <= 152).all()), 'Native sampled node shape differs')
        shapes['max_sequence_positions'] = max(shapes['max_sequence_positions'], tensor.shape[1])
        shapes['min_sequence_positions'] = min(shapes['min_sequence_positions'], tensor.shape[1])
        progress[phase[0]+'_batches'] += 1; progress[phase[0]+'_queries'] += count
        check_caps()
    def observe_output(module, positional, kwargs, output):
        require(output.logits.shape == (kwargs['input_embeds'].shape[0],)
                and bool(torch.isfinite(output.logits).all()), 'Native output shape/nonfinite failure')
        require(output.loss is not None and bool(torch.isfinite(output.loss).all()), 'Native loss nonfinite')
        # No logits, losses, histograms or score summaries leave this observation.
        write(child/'PROGRESS.json', {**progress, 'elapsed_seconds':time.monotonic()-started})
        check_caps()
    input_hook = model.register_forward_pre_hook(observe_input, with_kwargs=True)
    output_hook = model.register_forward_hook(observe_output, with_kwargs=True)
    try:
        progress['stage'] = 'native_full_epoch'; torch.cuda.synchronize(0); phase_started = time.monotonic()
        total_steps, total_samples = run_lp.train_loop(
            model, train_loader, optimizer, epoch=0, num_epochs=configs.num_epochs,
            gradient_accumulation_steps=8, max_num_samples=-1, total_train_steps=0, total_trained_samples=0,
            rank=0, local_rank=0, configs=configs, wandb_run=None, world_size=1, is_gnn=False, use_bf16=True)
        torch.cuda.synchronize(0); timings['complete_native_epoch_seconds'] = time.monotonic()-phase_started
        require(total_steps == progress['TRAIN_batches'] == len(train_loader)
                and total_samples == progress['TRAIN_queries'] == train_queries
                and progress['optimizer_updates'] == (len(train_loader)+7)//8, 'Incomplete native epoch')
        phase[0] = 'VALID'; progress['stage'] = 'native_full_VALID'; phase_started = time.monotonic()
        output = run_lp.evaluate_loop(model, valid_loader, rank=0, world_size=1, evaluator=None,
                                     dataset_len=len(valid_raw), split_name='valid', show_progress=False,
                                     compute_loss=False, check_sequential_indices=True, is_gnn=False, use_bf16=True)
        torch.cuda.synchronize(0); timings['complete_native_VALID_seconds'] = time.monotonic()-phase_started
        require(output['metrics'] is None and 'avg_loss' not in output
                and output['y_pred_all'].shape == (160084,) and bool(torch.isfinite(output['y_pred_all']).all())
                and torch.equal(output['idx_all'], torch.arange(160084, device=0))
                and progress['VALID_batches'] == len(valid_loader) == 157
                and progress['VALID_queries'] == 160084, 'Full finite sequential VALID coverage failed')
        del output
        phase_started = time.monotonic()
        for parameter in model.parameters():
            require(bool(torch.isfinite(parameter).all()), 'Nonfinite final parameter')
        for state in optimizer.state.values():
            for value in state.values():
                if torch.is_tensor(value): require(bool(torch.isfinite(value).all()), 'Nonfinite Adam state')
        torch.cuda.synchronize(0); timings['final_state_finite_guard_seconds'] = time.monotonic()-phase_started
        check_caps()
        peak_allocated, peak_reserved = torch.cuda.max_memory_allocated(0), torch.cuda.max_memory_reserved(0)
        phase_started = time.monotonic()
        input_hook.remove(); output_hook.remove()
        del optimizer, model, train_loader, valid_loader
        dist.destroy_process_group()
        timings['state_discard_and_process_group_close_seconds'] = time.monotonic()-phase_started
        progress.update(stage='complete_resource_only', status='COMPLETE_RESOURCE_ONLY')
        write(child/'PROGRESS.json', progress)
        custody_started = time.monotonic()
        native_module_custody()
        files = final_file_custody(plan,runtime,args.release,args.release_sha256,Path(authority['dataset_root']).resolve())
        write(child/'FILE_CUSTODY.json', files)
        timings['final_file_custody_seconds'] = time.monotonic()-custody_started
        check_caps()
        result = dict(schema='pencil-collab-resource-v1',status='COMPLETE_RESOURCE_ONLY',
                      source_manifest_sha256=sha(HERE/'MANIFEST.json'),release_sha256=args.release_sha256,
                      workload=plan['workload'],data_files_opened=list(DATA_FILES),completed_native_epochs=1,
                      TRAIN_queries=train_queries,TRAIN_batches=progress['TRAIN_batches'],VALID_queries=progress['VALID_queries'],
                      VALID_batches=progress['VALID_batches'],optimizer_updates=progress['optimizer_updates'],
                      VALID_sequential_indices=True,all_observed_outputs_finite=True,shapes=shapes,stream=stream,
                      timings=timings,inclusive_child_wall_seconds=time.monotonic()-started,
                      cuda_peak_allocated_bytes=peak_allocated,cuda_peak_reserved_bytes=peak_reserved,
                      runtime=dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
                                   GPU_UUID=plan['GPU_UUID'],logical_device=0,world_size=1,DDP=True,
                                   bf16_autocast=True,TF32=True,float32_matmul_precision='medium',
                                   deterministic_algorithms=False,loaded_sparse_scatter_libraries=sorted(str(x) for x in loaded),
                                   distributions=plan['distribution_versions']),
                      TEST_reads=False,predictive_metrics_computed=False,scores_saved=False,checkpoint_saved=False,
                      state_donor=False,automatic_retry=False,scientific_20_epoch_fit_completed=False,
                      protocol_differences=plan['protocol_differences'])
        write(child/'RESOURCE.json', result)
        final_rows = [dict(path=name,bytes=(child/name).stat().st_size,sha256=sha(child/name))
                      for name in ('PROGRESS.json','RESOURCE.json','FILE_CUSTODY.json')]
        check_caps()
        write(child/'FINAL_CUSTODY.json', dict(completed=True,file_custody_sha256=sha(child/'FILE_CUSTODY.json'),
              files=final_rows,inclusive_child_wall_seconds_through_custody=time.monotonic()-started,
              finite_final_write_and_fsync_tail_unmeasured=True))
    finally:
        ended.set(); watcher.join(timeout=1)


if __name__ == '__main__':
    main()
