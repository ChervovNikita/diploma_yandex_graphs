"""One fresh native 20-epoch PENCIL fit; VALID-selected full artifacts; no TEST."""
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
    parser.add_argument('--seed',type=int,choices=(0,1,2),required=True)
    args = parser.parse_args()
    plan, runtime, execution = gate(args.release, args.release_sha256, args.seed)
    require(os.environ.get('RANK') == os.environ.get('LOCAL_RANK') == '0'
            and os.environ.get('WORLD_SIZE') == '1', 'Exactly one directly supervised rank required')
    child = execution/'run01'; child.mkdir(exist_ok=False)
    progress = dict(stage='imports', TRAIN_batches=0, TRAIN_queries=0, VALID_batches=0, VALID_queries=0,
                    optimizer_updates=0, seed=args.seed, epoch_index=None, TEST_reads=False, scores_saved=False, status='RUNNING')
    write(child/'PROGRESS.json', progress)
    sys.path.insert(0, str(HERE/'native'))
    import torch
    import numpy as np
    import random
    import math
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
                          elapsed_seconds=time.monotonic()-started,TEST_reads=False,scores_saved=progress['scores_saved']))
                finally:
                    # A separate failure receipt avoids racing the ordinary progress file.
                    # Its presence can never satisfy the complete-success output envelope.
                    # Parent cleanup closes this exact worker's owned descendants.
                    os._exit(2)
    watcher = threading.Thread(target=monitor, daemon=True); watcher.start()
    timings = {}; phase_started = started
    rendezvous = child/'RANK0_RENDEZVOUS'
    dist.init_process_group('nccl', init_method=rendezvous.as_uri(), world_size=1, rank=0, timeout=timedelta(minutes=30))
    run_lp.tqdm = QuietProgress
    config_dict = yaml.safe_load((HERE/'OFFICIAL_CONFIG_REFERENCE.yaml').read_text())
    config_dict.update(use_features=True, feature_fusion='early', seed=args.seed, max_sequence_length=154,
                       eval_test=False,only_eval_test=False,eval_every=1,save_only_improve=True)
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
    evaluator = run_lp.Evaluator(metric=configs.dataset)
    metric_pin = authority['ogb_evaluator']
    require(Path(sys.modules['ogb.linkproppred.evaluate'].__file__).resolve()==Path(metric_pin['path']).resolve(),
            'Existing metric-source origin differs')
    require(torch.equal(valid_raw.all_edges_with_y[:60084,:2],split_edge['valid']['edge'])
            and torch.equal(valid_raw.all_edges_with_y[60084:,:2],split_edge['valid']['edge_neg']), 'Official VALID ordering differs')
    torch.cuda.synchronize(0);timings['imports_runtime_data_model_seconds']=time.monotonic()-phase_started
    shapes=dict(structural_width=306,raw_feature_width=128,max_sequence_positions=0,min_sequence_positions=2048,
                tokenizer_max_nodes=152,model_max_position_embeddings=2048,full_batch_size=1024)
    phase=['TRAIN']
    def observe_input(module,positional,kwargs):
        tensor=kwargs['input_embeds'];feature=kwargs['feature_embeds'];count=tensor.shape[0]
        require(tensor.ndim==3 and tensor.shape[2]==306 and 2<=tensor.shape[1]<=154
                and feature.shape==(count,tensor.shape[1],128),'Native scientific batch shape differs')
        require(kwargs['num_nodes'].shape==(count,) and bool((kwargs['num_nodes']<=152).all()),'Native sampled node shape differs')
        shapes['max_sequence_positions']=max(shapes['max_sequence_positions'],tensor.shape[1])
        shapes['min_sequence_positions']=min(shapes['min_sequence_positions'],tensor.shape[1])
        progress[phase[0]+'_batches']+=1;progress[phase[0]+'_queries']+=count;check_caps()
    def observe_output(module,positional,kwargs,output):
        require(output.logits.shape==(kwargs['input_embeds'].shape[0],) and bool(torch.isfinite(output.logits).all()),'Nonfinite/native output shape')
        require(output.loss is not None and bool(torch.isfinite(output.loss).all()),'Native nonfinite loss')
        write(child/'PROGRESS.json',{**progress,'elapsed_seconds':time.monotonic()-started});check_caps()
    def binary(path,writer):
        temporary=path.with_suffix(path.suffix+'.tmp')
        with temporary.open('xb') as stream:
            writer(stream);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
        descriptor=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)
        check_caps()
    def file_row(name):
        path=child/name;return dict(path=name,bytes=path.stat().st_size,sha256=sha(path))
    def save_selected(epoch,score,output,total_steps,total_samples,train_loader,valid_loader):
        native_scores=output['y_pred_all'].detach().cpu()
        require(native_scores.dtype in (torch.bfloat16,torch.float32),'Unexpected native score dtype')
        float_scores=native_scores.float()
        require(torch.equal(float_scores.to(native_scores.dtype),native_scores),'Score conversion changed native values')
        positive=float_scores[:60084].numpy().copy()
        negative=float_scores[60084:].numpy().copy()
        require(positive.shape==(60084,) and negative.shape==(100000,) and positive.dtype==negative.dtype==np.float32
                and np.isfinite(positive).all() and np.isfinite(negative).all(),'Selected VALID score archive contract differs')
        numpy_rng=np.random.get_state()
        state=dict(schema='pencil-selected-full-scientific-state-v1',source_manifest_sha256=sha(HERE/'MANIFEST.json'),
                   release_sha256=args.release_sha256,seed=args.seed,epoch=epoch+1,epoch_index=epoch,
                   selection_metric='VALID_hits@50',selection_score=score,selection_tie='first_epoch',
                   model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),
                   step=total_steps,total_trained_samples=total_samples,optimizer_updates=progress['optimizer_updates'],
                   config=config_dict,initialization_source='fresh_native_scratch_per_seed',scheduler=None,
                   rng_state=dict(python=random.getstate(),numpy=dict(kind=numpy_rng[0],keys=numpy_rng[1].tolist(),
                       position=int(numpy_rng[2]),has_gaussian=int(numpy_rng[3]),cached_gaussian=float(numpy_rng[4])),
                       torch_cpu=torch.get_rng_state(),torch_cuda=torch.cuda.get_rng_state_all(),
                       train_loader_generator=train_loader.generator.get_state(),valid_loader_generator=valid_loader.generator.get_state(),
                       distributed_sampler_epoch=train_loader.sampler.epoch),TEST_reads=False,heldout_release=False,other_fit_donor=False)
        binary(child/'SELECTED_FULL_STATE.pt',lambda stream:torch.save(state,stream))
        binary(child/'VALID_POSITIVE_SCORES.npy',lambda stream:np.save(stream,positive,allow_pickle=False))
        binary(child/'VALID_NEGATIVE_SCORES.npy',lambda stream:np.save(stream,negative,allow_pickle=False))
        selected=dict(status='SELECTED_SCIENTIFIC_STATE_VALID_ONLY',seed=args.seed,epoch_index=epoch,epoch_number=epoch+1,
                      metric='VALID_hits@50',hits50=score,first_tie=True,positive_queries=60084,negative_queries=100000,
                      score_dtype='float32',native_score_dtype=str(native_scores.dtype),native_values_preserved=True,
                      score_semantics='native logits; lossless bf16/float32 conversion after native metric computation',
                      score_order='official VALID positive/negative order',source_manifest_sha256=sha(HERE/'MANIFEST.json'),
                      release_sha256=args.release_sha256,files=[file_row(n) for n in
                        ('SELECTED_FULL_STATE.pt','VALID_POSITIVE_SCORES.npy','VALID_NEGATIVE_SCORES.npy')],
                      optimizer_updates_at_selection=progress['optimizer_updates'],TEST_reads=False,heldout_release=False,
                      historical_Collab_TEST_consumed=True,resource_donor_loaded=False,other_fit_donor=False)
        write(child/'SELECTION.json',selected)
        progress['scores_saved']=True;write(child/'PROGRESS.json',progress)
        check_caps();return selected
    input_hook=model.register_forward_pre_hook(observe_input,with_kwargs=True)
    output_hook=model.register_forward_hook(observe_output,with_kwargs=True)
    history=[];selected=None;best=-float('inf');total_steps=total_samples=0
    try:
        for epoch in range(configs.num_epochs):
            progress.update(epoch_index=epoch,stage='native_full_epoch');phase[0]='TRAIN';before=dict(progress)
            train_loader,valid_loader,absent_loader=run_lp.build_loaders(
                epoch=epoch,tokenizer=tokenizer,configs=configs,collator=collator,
                train_dataset_raw=train_raw,valid_dataset_raw=valid_raw,test_dataset_raw=None,
                use_features=True,encoding_scheme='adjacency_row',is_gnn=False)
            require(absent_loader is None and train_loader.drop_last is False and valid_loader.drop_last is False
                    and train_loader.num_workers==valid_loader.num_workers==12,'Native complete loader policy differs')
            require(train_loader.pin_memory is True and valid_loader.pin_memory is True,'Native loader pin-memory default differs')
            train_loader.pin_memory=False;valid_loader.pin_memory=False
            require(train_loader.pin_memory is False and valid_loader.pin_memory is False,'Admitted unpinned loader policy required')
            queries=len(train_raw);positive=int((train_raw.all_edges_with_y[:,2]==1).sum());negative=int((train_raw.all_edges_with_y[:,2]==0).sum())
            require(positive==int(round(filtered_records*.5)) and positive+negative==queries
                    and 0<negative<=positive,'Native exposure policy differs; no redraw/truncation permitted')
            batches=len(train_loader);expected_updates=(batches+7)//8
            stream=dict(epoch_index=epoch,seed=args.seed,TRAIN_positive_queries=positive,TRAIN_negative_queries=negative,
                        requested_negative_queries=positive,native_mixed_queries_sha256=tensor_sha(train_raw.all_edges_with_y),
                        native_sample_indices_sha256=tensor_sha(train_raw.sample_idx),
                        native_internal_sampler_sha256=tensor_sha(torch.tensor(train_raw.sampler)),
                        distributed_sampler_order_sha256=tensor_sha(torch.tensor(list(train_loader.sampler))),
                        loader_seed=args.seed*100+epoch,positive_cycle_seed=args.seed*100+50*epoch//100,
                        native_50_percent_policy=True,future_positive_rejection=False)
            torch.cuda.synchronize(0);phase_started=time.monotonic()
            old_steps,old_samples=total_steps,total_samples
            total_steps,total_samples=run_lp.train_loop(
                model,train_loader,optimizer,epoch=epoch,num_epochs=configs.num_epochs,
                gradient_accumulation_steps=8,max_num_samples=-1,total_train_steps=total_steps,total_trained_samples=total_samples,
                rank=0,local_rank=0,configs=configs,wandb_run=None,world_size=1,is_gnn=False,use_bf16=True)
            torch.cuda.synchronize(0);train_seconds=time.monotonic()-phase_started
            observed_batches=progress['TRAIN_batches']-before['TRAIN_batches'];observed_queries=progress['TRAIN_queries']-before['TRAIN_queries']
            updates=progress['optimizer_updates']-before['optimizer_updates']
            require(total_steps-old_steps==observed_batches==batches and total_samples-old_samples==observed_queries==queries
                    and updates==expected_updates,'Complete native epoch/update count failed')
            phase[0]='VALID';progress['stage']='native_full_VALID';phase_started=time.monotonic()
            output=run_lp.evaluate_loop(model,valid_loader,rank=0,world_size=1,evaluator=evaluator,
                    dataset_len=len(valid_raw),split_name='valid',show_progress=False,compute_loss=False,
                    check_sequential_indices=True,is_gnn=False,use_bf16=True)
            torch.cuda.synchronize(0);valid_seconds=time.monotonic()-phase_started
            require(output['y_pred_all'].shape==output['y_true_all'].shape==(160084,)
                    and bool(torch.isfinite(output['y_pred_all']).all()) and torch.equal(output['idx_all'],torch.arange(160084,device=0))
                    and bool((output['y_true_all'][:60084]==1).all()) and bool((output['y_true_all'][60084:]==0).all())
                    and progress['VALID_batches']-before['VALID_batches']==len(valid_loader)==157
                    and progress['VALID_queries']-before['VALID_queries']==160084,'Full VALID coverage/order/labels failed')
            score=float(output['metrics']['hits@50']);require(math.isfinite(score) and 0<=score<=1,'Invalid native VALID Hits50')
            save_started=time.monotonic();improved=score>best
            if improved:
                selected=save_selected(epoch,score,output,total_steps,total_samples,train_loader,valid_loader);best=score
            row=dict(epoch_index=epoch,TRAIN_batches=batches,TRAIN_queries=queries,optimizer_updates=updates,
                     cumulative_optimizer_updates=progress['optimizer_updates'],native_total_steps=total_steps,
                     native_total_trained_samples=total_samples,VALID_batches=157,VALID_queries=160084,VALID_hits50=score,
                     selected_on_strict_improvement=improved,first_tie=True,stream=stream,
                     times_seconds=dict(TRAIN=train_seconds,VALID=valid_seconds,selection_publication=time.monotonic()-save_started))
            history.append(row);write(child/'EPOCH_HISTORY.json',dict(seed=args.seed,epochs=history,TEST_reads=False))
            del output,train_loader,valid_loader;check_caps()
        require(len(history)==20 and selected is not None,'All 20 scientific epochs/selection required')
        require(progress['optimizer_updates']==sum(r['optimizer_updates'] for r in history)
                and progress['TRAIN_batches']==sum(r['TRAIN_batches'] for r in history)
                and progress['VALID_batches']==20*157 and progress['VALID_queries']==20*160084,'Full-fit counters differ')
        first_best=max(range(20),key=lambda index:history[index]['VALID_hits50'])
        require(selected['epoch_index']==first_best and selected['hits50']==history[first_best]['VALID_hits50'],'First-tie selection differs')
        for parameter in model.parameters():require(bool(torch.isfinite(parameter).all()),'Nonfinite final fit parameter')
        peak_allocated=torch.cuda.max_memory_allocated(0);peak_reserved=torch.cuda.max_memory_reserved(0);check_caps()
        input_hook.remove();output_hook.remove();input_hook=output_hook=None
        del model,optimizer,original_step,counted_step
        torch.cuda.empty_cache();torch.cuda.synchronize(0)
        dist.destroy_process_group();rendezvous.unlink(missing_ok=True)
        progress.update(stage='complete_predictive_fit',status='COMPLETE_PREDICTIVE_FIT',scores_saved=True)
        write(child/'PROGRESS.json',progress);native_module_custody()
        files=final_file_custody(plan,runtime,args.release,args.release_sha256,Path(authority['dataset_root']).resolve())
        write(child/'FILE_CUSTODY.json',files);check_caps()
        result=dict(schema='pencil-paired-predictive-fit-v1',status='COMPLETE_PREDICTIVE_FIT',seed=args.seed,
                    source_manifest_sha256=sha(HERE/'MANIFEST.json'),release_sha256=args.release_sha256,workload=plan['workload'],
                    completed_native_epochs=20,optimizer_updates=progress['optimizer_updates'],TRAIN_batches=progress['TRAIN_batches'],
                    TRAIN_queries=progress['TRAIN_queries'],VALID_batches=progress['VALID_batches'],VALID_queries=progress['VALID_queries'],
                    every_epoch_full_VALID=True,VALID_sequential_indices=True,all_observed_outputs_finite=True,shapes=shapes,
                    selected_epoch_index=selected['epoch_index'],selected_VALID_hits50=best,selection_metric='VALID_hits@50',first_tie=True,
                    expected2000_updates_observed=(progress['optimizer_updates']==2000),all_epochs793_batches=all(r['TRAIN_batches']==793 for r in history),
                    selected_artifacts=selected['files'],loader_pin_memory_policy=plan['loader_pin_memory_policy'],data_files_opened=list(DATA_FILES),
                    filtered_TRAIN_records=filtered_records,filtered_graph_sha256=tensor_sha(graph.edge_index),authenticated_data_digests=data['digests'],
                    inclusive_child_wall_seconds=time.monotonic()-started,cuda_peak_allocated_bytes=peak_allocated,cuda_peak_reserved_bytes=peak_reserved,
                    setup_timings=timings,protocol_differences=plan['protocol_differences'],TEST_reads=False,historical_Collab_TEST_consumed=True,
                    result_class='adapted_native_PENCIL_scientific_baseline',exact_author_reproduction=False,
                    selected_state_server_side=True,native_score_dtype=selected['native_score_dtype'],native_score_values_preserved=True,
                    heldout_release=False,scores_saved=True,checkpoint_saved=True,resource_fit_state_loaded=False,fresh_model_optimizer=True,
                    other_fit_donor=False,state_donor=False,automatic_retry=False)
        write(child/'FIT.json',result);check_caps()
        names=['PROGRESS.json','EPOCH_HISTORY.json','SELECTION.json','SELECTED_FULL_STATE.pt',
               'VALID_POSITIVE_SCORES.npy','VALID_NEGATIVE_SCORES.npy','FILE_CUSTODY.json','FIT.json']
        final_rows=[file_row(name) for name in names];check_caps()
        write(child/'FINAL_CUSTODY.json',dict(completed=True,seed=args.seed,files=final_rows,
              file_custody_sha256=sha(child/'FILE_CUSTODY.json'),inclusive_child_wall_seconds_through_custody=time.monotonic()-started,
              finite_final_write_and_fsync_tail_unmeasured=True))
    finally:
        ended.set();watcher.join(timeout=1)
        if input_hook is not None:input_hook.remove()
        if output_hook is not None:output_hook.remove()


if __name__=='__main__':
    main()
