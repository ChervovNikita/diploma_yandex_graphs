"""Separate inactive memory-bounded full-input engineering assessment of V2 core."""
import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import resource
import time
import reference
from support import HERE, PHASE, admission, load_core, read, require, sha, source_checks


def run(args):
    pins, protocol = source_checks(), read(HERE/'PROTOCOL.json')
    core, bank, fit, q = load_core(pins)
    release, receipt, config, output = admission(args,pins,core)
    require(set(release['expected_runtime_versions']) == set(q.VERSIONS), 'Complete existing runtime versions')
    started, usage0 = time.perf_counter(), resource.getrusage(resource.RUSAGE_SELF)
    old, common, helpers, placement = core.borrowed(core.source_checks())
    output.mkdir(parents=True,exist_ok=False)
    counters = {name:0 for name in ('train_forward_attempts','train_forwards_completed','backward_attempts','backwards_completed',
        'Adam_attempts','Adam_steps_completed','reference_train_forward_attempts','reference_train_forwards_completed',
        'reference_backward_attempts','reference_backwards_completed','reference_Adam_attempts','reference_Adam_steps_completed',
        'original_serving_forward_attempts','original_serving_forwards_completed','restored_serving_forward_attempts',
        'restored_serving_forwards_completed','original_constructor_attempts','original_constructors_completed')}
    prior = read(PHASE/pins['prior_failure']['path'])
    record = dict(status='started',qualification_passed=False,source_only_preparation=False,
        qualification_kind=protocol['assessment_method'],new_post_failure_resource_plan=True,
        failed_V2_joint_attempt_preserved=True,prior_failure_receipt=pins['prior_failure'],
        prior_failed_attempt_cost=prior['metadata']['TERMINAL.json'],prior_partial_progress=prior['qualification_progress'],
        prior_partial_qualification_binding=release['prior_partial_qualification'],
        prior_partially_completed_proof=read(release['prior_partial_qualification']['path']),
        reference_bodies_live_at_once=1,one_live_autograd_graph=True,repeat_failed_joint_method=False,
        full_graph=True,all_TRAIN_rows=True,validation_metric_access=False,scientific_metric_function_called=False,
        TEST_truth_present=False,automatic_retry=False,tiny_float_gate=False,context_regularizer=0.0,
        scientific_full_fits_or_VALID_selected_checkpoints_created=False,
        private_likelihood_gradient_scale=0.25,private_coupled_Adam_weight_decay=receipt['optimizer']['sheaf_decay'],
        new_cost_does_not_replace_failed_attempt_cost=True,reference_members=[],phase_memory=[],
        identity=dict(source_seal_sha256=pins['scientific_core_seal_sha256'],
            manifest_sha256=read(PHASE/pins['scientific_core_directory']/'SEAL.json')['manifest_sha256'],
            assessor_source_seal_sha256=sha(HERE/'SEAL.json'),assessor_source_sha256=sha(HERE/'assessor.py'),
            reference_source_sha256=sha(HERE/'reference.py'),assessor_manifest_sha256=read(HERE/'SEAL.json')['manifest_sha256'],
            roles_sha256=release['roles_sha256'],role_metadata_sha256=release['role_metadata_sha256'],configuration_id=config['id'],
            configuration=config,optimizer=receipt['optimizer'],admission_receipt_sha256=sha(args.admission),
            root_release_sha256=sha(args.release),execution_source_commit=release['execution_source_commit'],base_seed=7409))
    observer = None
    def persist():
        common.write_json(output/'QUALIFICATION.json',dict(record,counters=counters,
            per_backward_gradient_observations=[] if observer is None else observer.records))
    persist()
    stage,model,opt,device,torch = 'runtime',None,None,None,None
    try:
        import numpy as np
        import torch
        versions = {name:importlib.metadata.version(name) for name in q.VERSIONS}
        require(versions == release['expected_runtime_versions'] and str(torch.__version__) == '2.1.2+cu118'
                and np.__version__ == '1.26.4','Exact root-qualified existing runtime')
        device = torch.device(release['device'])
        require(device == torch.device('cuda:0') and torch.cuda.device_count() == 1,'One root-owned visible cuda:0')
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; torch.backends.cudnn.benchmark=False
        torch.use_deterministic_algorithms(release['deterministic_algorithms'])
        torch.cuda.init(); torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device)
        record['identity'].update(runtime_versions=versions,device=str(device),deterministic_algorithms=release['deterministic_algorithms'],allow_tf32=False)
        def phase_memory(name):
            helpers.synchronize(torch,device)
            row = dict(phase=name,allocated_bytes=torch.cuda.memory_allocated(device),reserved_bytes=torch.cuda.memory_reserved(device),
                       cumulative_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),cumulative_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
            record['phase_memory'].append(row); persist()
            require(row['cumulative_peak_reserved_bytes'] <= release['owned_GPU_cap_bytes'],'Declared assessor CUDA reservation cap exceeded')
        stage='full_input_roles'
        arrays,role_meta,role_sha=common.read_roles(np,args.roles)
        require(role_meta.get('exposure_classification')=='original_paper_benchmark_exploratory' and role_sha==release['role_metadata_sha256'],
                'Same complete original benchmark roles')
        del arrays['valid_y'],arrays['valid_index']
        data={name:torch.from_numpy(value).to(device) for name,value in arrays.items()}
        data['cpu_edge_index']=torch.from_numpy(arrays['edge_index'])
        common.write_json(output/'ROLE_METADATA.json',role_meta)
        record.update(validation_arrays_dropped_before_model_path=True,input_role_counts=role_meta['role_counts'],actual_support_counts=role_meta['support_counts'])
        adapter=common.load_adapter()
        cfg,seed=receipt['optimizer'],release['qualifier_seed']
        native_args=dict(config['native_args'],graph_size=data['x'].shape[0],input_dim=data['x'].shape[1],output_dim=2,device=str(device))
        factory=placement.make_native_placed_factory(torch,adapter,data['cpu_edge_index'],data['edge_index'],native_args)
        def observed_factory():
            counters['original_constructor_attempts']+=1
            value=factory()
            counters['original_constructors_completed']+=1
            return value
        def fresh_bank(): return bank.make_bank(torch,adapter,observed_factory)
        stage='actual_V2_streamed_bank_construction'
        helpers.seed_all(np,torch,seed)
        tick=time.perf_counter()
        model=fresh_bank(); opt=fit.optimizer(torch,model,cfg)
        helpers.synchronize(torch,device)
        record.update(native_args=native_args,streamed_construction_seconds=time.perf_counter()-tick,
                      initial_structure=q.structural(torch,model,opt,data,role_meta,initial=True))
        groups=q.parameter_groups(model)
        factor_paths=model.factor_paths
        native_objects,native_count=model.native_parameter_objects,model.native_parameter_count
        initial_state=old.cpu_tree(torch,model.state_dict())
        initialization_end_rng=helpers.capture_rng(np,torch)
        original_topology=q.topology_snapshot(torch,model)
        forward_rng,backward_end_rng={},{}
        def before_forward(index):
            def capture(_member,_inputs):
                require(index==len(forward_rng),'Four actual forward RNG captures in fixed order')
                forward_rng[index]=helpers.capture_rng(np,torch)
            return capture
        hooks=[member.register_forward_pre_hook(before_forward(index)) for index,member in enumerate(model.members)]
        observer=q.GradientObserver(torch,model,persist)
        def observe(index,live_model):
            observer(index,live_model)
            backward_end_rng[index]=helpers.capture_rng(np,torch)
            persist()
        stage='exact_V2_streamed_four_backward_one_Adam_step'
        tick=time.perf_counter()
        try:
            record['streamed_own_TRAIN_NLL']=fit.train_update(torch,model,opt,data,counters,observe)
        finally:
            for handle in hooks: handle.remove()
        helpers.synchronize(torch,device)
        require(set(forward_rng)==set(backward_end_rng)==set(range(4)) and old.exact(torch,forward_rng[0],initialization_end_rng),
                'Each actual member forward/backward RNG captured, initial stream exact')
        record.update(streamed_update_seconds=time.perf_counter()-tick,after_streamed_structure=q.structural(torch,model,opt,data,role_meta),
                      streamed_cache_ownership=q.cache_ownership(torch,model,data,role_meta),streamed_topology=q.verify_topology(torch,model,original_topology),
                      forward_RNG_capture='read-only pre-forward hooks on original complete member paths; removed before serving')
        streamed_gradients=q.snapshot_gradients(torch,model)
        streamed_state=old.cpu_tree(torch,model.state_dict())
        streamed_optimizer=old.cpu_tree(torch,opt.state_dict())
        streamed_end_rng=helpers.capture_rng(np,torch)
        phase_memory('actual_streamed_bank')
        stage='original_label_free_four_path_serving'
        tick=time.perf_counter()
        original_serving=q.label_free_serving(np,torch,helpers,old,model,data,seed,counters,'original_serving')
        helpers.synchronize(torch,device)
        record.update(original_serving_seconds=time.perf_counter()-tick,original_serving_topology=q.verify_topology(torch,model,original_topology),
                      original_serving_cache_ownership=q.cache_ownership(torch,model,data,role_meta))
        phase_memory('original_label_free_serving')
        model=opt=None
        gc.collect(); torch.cuda.empty_cache()
        stage='four_sequential_full_input_M1_differential_references'
        tick=time.perf_counter()
        accumulator,contribution_counts=reference.accumulate(np,torch,adapter,observed_factory,helpers,old,q,data,role_meta,initial_state,
            factor_paths,native_objects,native_count,original_topology,forward_rng,backward_end_rng,streamed_gradients,
            counters,record['reference_members'],persist,phase_memory)
        record.update(sequential_reference_seconds=time.perf_counter()-tick,
            gradient_differences=q.differences(torch,streamed_gradients,accumulator,groups),CPU_gradient_contribution_counts=contribution_counts,
            own_NLL_signed_differences=[row['own_TRAIN_NLL']-value for row,value in zip(record['reference_members'],record['streamed_own_TRAIN_NLL'])],
            reference_completed=True,reference_numeric_difference_threshold=None)
        persist()
        stage='tape_free_accumulated_gradient_same_V2_Adam_reference'
        tick=time.perf_counter()
        model=fresh_bank(); model.load_state_dict(initial_state,strict=True)
        require(old.exact(torch,old.cpu_tree(torch,model.state_dict()),initial_state),'Tape-free update starts at exact initial bank state')
        record['update_reference_topology_exactly_matches_original']=q.equivalent_topology(torch,model,original_topology)
        update_topology=q.topology_snapshot(torch,model)
        opt=fit.optimizer(torch,model,cfg)
        record['update_reference_initial_structure']=q.structural(torch,model,opt,data,role_meta,initial=True)
        reference.apply_accumulated_update(torch,model,opt,initial_state,accumulator,old,counters)
        helpers.synchronize(torch,device)
        record.update(accumulated_reference_update_seconds=time.perf_counter()-tick,
            update_reference_after_structure=q.structural(torch,model,opt,data,role_meta),
            update_reference_topology=q.verify_topology(torch,model,update_topology),
            parameter_update_differences=q.differences(torch,{name:streamed_state[name].double()-initial_state[name].double() for name in groups},
                {name:value.detach().cpu().double()-initial_state[name].double() for name,value in model.named_parameters()},groups),
            update_reference_forward_calls=0,update_reference_same_device_and_exact_V2_Adam=True)
        phase_memory('tape_free_reference_Adam')
        model=opt=None
        gc.collect(); torch.cuda.empty_cache()
        stage='serialize_owned_streamed_engineering_state'
        tick=time.perf_counter()
        state_path=output/'ENGINEERING_STATE.pt'
        state=dict(schema='V2-core-sequential-M1-differential-engineering-state-v3',identity=record['identity'],
            model_state=streamed_state,optimizer_state=streamed_optimizer,training_rng=streamed_end_rng,label_free_serving=original_serving,
            actual_member_forward_rng=forward_rng,actual_member_backward_end_rng=backward_end_rng,
            CPU_reference_gradients=accumulator,actual_streamed_gradients=streamed_gradients,
            TEST_truth_saved=False,VALID_truth_saved=False,scientific_checkpoint=False)
        temporary=state_path.with_suffix('.tmp'); torch.save(state,temporary); temporary.replace(state_path)
        record.update(engineering_state_write_seconds=time.perf_counter()-tick,engineering_state_sha256=sha(state_path),
                      engineering_state_bytes=state_path.stat().st_size,server_only_engineering_state=True)
        del state
        stage='fresh_exact_streamed_state_reconstruction'
        tick=time.perf_counter()
        saved=torch.load(state_path,map_location='cpu',weights_only=True)
        require(saved['identity']==record['identity'] and saved['TEST_truth_saved'] is False and saved['VALID_truth_saved'] is False,'Exact engineering state identity')
        model=fresh_bank(); model.load_state_dict(saved['model_state'],strict=True)
        require(old.exact(torch,old.cpu_tree(torch,model.state_dict()),saved['model_state']),'Exact reconstructed engineering model parameter/buffer state')
        record['reconstructed_topology_exactly_matches_original']=q.equivalent_topology(torch,model,original_topology)
        opt=fit.optimizer(torch,model,cfg); opt.load_state_dict(saved['optimizer_state'])
        require(old.exact(torch,old.cpu_tree(torch,opt.state_dict()),saved['optimizer_state']),'Exact reconstructed engineering Adam state')
        helpers.restore_rng(np,torch,saved['training_rng'])
        require(old.exact(torch,helpers.capture_rng(np,torch),saved['training_rng']),'Exact reconstructed engineering training RNG')
        topology=q.topology_snapshot(torch,model)
        record.update(restored_structure=q.structural(torch,model,opt,data,role_meta),model_parameter_buffer_restore_exact=True,
                      optimizer_restore_exact=True,training_RNG_restore_exact=True,reconstruction_seconds=time.perf_counter()-tick)
        stage='restored_label_free_four_path_serving'
        tick=time.perf_counter()
        serving=q.label_free_serving(np,torch,helpers,old,model,data,seed,counters,'restored_serving')
        helpers.synchronize(torch,device)
        record.update(restored_serving_seconds=time.perf_counter()-tick,label_free_serving_materiality=q.serving_materiality(torch,saved['label_free_serving'],serving),
                      restored_cache_ownership=q.cache_ownership(torch,model,data,role_meta),restored_topology=q.verify_topology(torch,model,topology),
                      finite_reconstructed_serving=True)
        phase_memory('restored_label_free_serving')
        require(counters['train_forwards_completed']==counters['backwards_completed']==counters['reference_train_forwards_completed']
                ==counters['reference_backwards_completed']==4 and counters['Adam_steps_completed']==counters['reference_Adam_steps_completed']==1,
                'Exact four actual and four sequential reference full TRAIN forwards/backwards, two complete Adam steps')
        require(counters['original_serving_forwards_completed']==counters['restored_serving_forwards_completed']==4
                and counters['original_constructors_completed']==7,'Exact four original/restored serving paths and seven fresh original constructors')
        record.update(status='complete',qualification_passed=True)
    except Exception as error:
        record.update(status='failed',qualification_passed=False,failure=common.failure_record(error,stage))
    except BaseException as error:
        record.update(status='interrupted',qualification_passed=False,failure=common.failure_record(error,stage))
        raise
    finally:
        try:
            if torch is not None and device is not None:
                helpers.synchronize(torch,device)
                record.update(peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_CUDA_reserved_bytes=torch.cuda.max_memory_reserved(device))
                model=opt=None
                gc.collect(); torch.cuda.empty_cache()
        except Exception as error:
            record.update(status='failed',qualification_passed=False,cost_or_cleanup_failure=common.failure_record(error,'finalize'))
        usage=resource.getrusage(resource.RUSAGE_SELF)
        record.update(complete_attempt_seconds=time.perf_counter()-started,CPU_user_seconds=usage.ru_utime-usage0.ru_utime,
            CPU_system_seconds=usage.ru_stime-usage0.ru_stime,process_RSS_high_water_cumulative_bytes=common.process_peak_rss_bytes(),
            actual_TRAIN_forwards=counters['train_forwards_completed']+counters['reference_train_forwards_completed'],
            all_native_forwards=sum(counters[key] for key in ('train_forwards_completed','reference_train_forwards_completed',
                'original_serving_forwards_completed','restored_serving_forwards_completed')),
            native_completed_map_evaluations=config['native_args']['layers']*sum(counters[key] for key in ('train_forwards_completed',
                'reference_train_forwards_completed','original_serving_forwards_completed','restored_serving_forwards_completed')),
            cost_includes_all_constructors_clones_hooks_CPU_snapshots_accumulation_forwards_backwards_steps_serialization_and_cleanup=True,
            actual_bank_has_four_persistent_native_caches=True,sequential_reference_has_one_complete_native_body_at_a_time=True,
            comparative_opening_authorized=False)
        persist()
    require(record['qualification_passed'],'Separate bounded differential engineering assessment failed; retained attempt/cost, no retry')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    for name in ('release','admission','roles','output'): parser.add_argument('--'+name,type=Path)
    args=parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(inactive=True,numeric_model_or_role_import=False,protocol=read(HERE/'PROTOCOL.json'))))
        return
    require(args.release and args.admission and args.roles and args.output,'Explicit separate root release/admission/roles/fresh output required')
    run(args)


if __name__=='__main__': main()
