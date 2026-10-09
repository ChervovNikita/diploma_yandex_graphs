"""Inactive minimal full-input qualification of the sealed centered-prior change."""
import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time

HERE=Path(__file__).resolve().parent
PHASE=HERE.parent


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):digest.update(block)
    return digest.hexdigest()


def require(value,message):
    if not value:raise ValueError(message)


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def run(args):
    release,admit=read(args.release),read(args.admission)
    require(release['enabled'] is True and release['release_owner']=='root' and release['action']=='centered_prior_minimal_full_input_qualify',
            'Explicit root centered engineering release')
    require(isinstance(release['execution_source_commit'],str) and len(release['execution_source_commit'])==40
            and all(value in '0123456789abcdef' for value in release['execution_source_commit']),'Exact published centered scientific source anchor')
    centered=PHASE/'private_sheaf_identity_centered_fast_prior_extension_source_20261009_v1'
    require(sha(centered/'SEAL.json')==release['source_seal_sha256']=='b151054b833b059998376baff89a7ac55ee664024f47d5d704c9fbd6a0fc8ef0'
            and sha(__file__)==release['qualifier_source_sha256'] and sha(args.admission)==release['admission_receipt_sha256']
            and sha(args.roles)==release['roles_sha256'] and sha(args.roles.parent/'ROLE.json')==release['role_metadata_sha256'],
            'Exact centered source/admission/roles')
    original_protocol=read(PHASE/'private_sheaf_train_valid_runner_20261009_v2/RUNNER_PROTOCOL.json')
    config=next(value for value in original_protocol['configs'] if value['id']=='d4_f16_L4')
    require(admit['native_args']==config['native_args'] and admit['optimizer']==original_protocol['optimizer']
            and admit['runtime_versions']==release['expected_runtime_versions'] and admit['deterministic_algorithms']==release['deterministic_algorithms']
            and release['qualifier_seed']==7409 and release['allow_tf32'] is False and release['device']=='cuda:0',
            'Fixed full-input native architecture/runtime/seed/policy')
    scope=release['two_recipe_scientific_scope']
    require(sha(scope['path'])==scope['sha256']=='24111e6cc00ba233ed3f1d801dba4ab0166254dde0fbd49120b582c9787032f3','Exact frozen two-recipe scope')
    cs=load(centered/'support.py','_minimal_centered_support');pins=cs.source_checks();fit=cs.load_centered_fit(pins)
    v3=load(PHASE/'private_sheaf_geometry_only_full_input_assessor_source_20261009_v3/support.py','_minimal_existing_assessor_support')
    core,bank,_vanilla_fit,q=v3.load_core(v3.source_checks())
    old,common,helpers,placement=cs.borrowed(pins)
    absent=object();previous=sys.modules.get('support',absent)
    try:
        sys.modules['support']=cs
        probe=load(centered/'prior_probe.py','_minimal_centered_prior_probe')
    finally:
        if previous is absent:sys.modules.pop('support',None)
        else:sys.modules['support']=previous
    output=args.output.resolve()
    require(str(output)==release['output_directory'] and not output.exists() and output.is_relative_to(PHASE)
            and not output.is_relative_to(HERE) and not any(output.is_relative_to(PHASE/path) for path in pins['protected_directories']),
            'Fresh exact centered engineering output')
    output.mkdir(parents=True,exist_ok=False)
    started,usage0=time.perf_counter(),resource.getrusage(resource.RUSAGE_SELF)
    counters={key:0 for key in ('train_forward_attempts','train_forwards_completed','backward_attempts','backwards_completed','Adam_attempts','Adam_steps_completed',
        'original_serving_forward_attempts','original_serving_forwards_completed','restored_serving_forward_attempts','restored_serving_forwards_completed',
        'original_constructor_attempts','original_constructors_completed','prior_probe_attempts','prior_probes_completed')}
    record=dict(status='started',qualification_passed=False,source_only_preparation=False,centered_prior_probe_passed=False,centered_update_qualified=False,
        full_graph=True,all_TRAIN_rows=True,validation_metric_access=False,scientific_metric_function_called=False,TEST_truth_present=False,
        automatic_retry=False,four_tape_reference=False,numerical_drift_gate=False,context_regularizer=0.0,
        scientific_full_fits_or_VALID_selected_checkpoints_created=False,
        identity=dict(source_seal_sha256=release['source_seal_sha256'],manifest_sha256=read(centered/'SEAL.json')['manifest_sha256'],
            qualifier_source_sha256=sha(__file__),root_release_sha256=sha(args.release),configuration_id=config['id'],configuration=config,
            optimizer=admit['optimizer'],admission_receipt_sha256=sha(args.admission),roles_sha256=release['roles_sha256'],role_metadata_sha256=release['role_metadata_sha256'],
            execution_source_commit=release['execution_source_commit'],device=release['device'],deterministic_algorithms=release['deterministic_algorithms'],allow_tf32=False,
            centered_prior_coefficient=0.0005,fast_optimizer_decay=0.0,base_seed=7409))
    observer=None
    def persist():common.write_json(output/'QUALIFICATION.json',dict(record,counters=counters,per_backward_gradient_observations=[] if observer is None else observer.records))
    persist();stage='runtime';model=opt=None;device=torch=None
    try:
        import numpy as np
        import torch
        versions={name:importlib.metadata.version(name) for name in q.VERSIONS}
        require(versions==release['expected_runtime_versions'] and str(torch.__version__)=='2.1.2+cu118' and np.__version__=='1.26.4','Exact existing qualified runtime')
        record['identity']['runtime_versions']=versions
        device=torch.device(release['device']);require(torch.cuda.device_count()==1,'One root-owned visible GPU')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        torch.use_deterministic_algorithms(release['deterministic_algorithms']);torch.cuda.init();torch.cuda.set_device(device);torch.cuda.reset_peak_memory_stats(device)
        stage='full_input_roles';arrays,meta,metadata_sha=common.read_roles(np,args.roles)
        require(metadata_sha==release['role_metadata_sha256'] and meta.get('exposure_classification')=='original_paper_benchmark_exploratory','Same full exploratory roles')
        del arrays['valid_y'],arrays['valid_index']
        data={name:torch.from_numpy(value).to(device) for name,value in arrays.items()};data['cpu_edge_index']=torch.from_numpy(arrays['edge_index'])
        common.write_json(output/'ROLE_METADATA.json',meta)
        adapter=common.load_adapter();cfg=admit['optimizer'];helpers.seed_all(np,torch,7409)
        native_args=dict(config['native_args'],graph_size=data['x'].shape[0],input_dim=data['x'].shape[1],output_dim=2,device=str(device))
        factory=placement.make_native_placed_factory(torch,adapter,data['cpu_edge_index'],data['edge_index'],native_args)
        def counted_factory():
            counters['original_constructor_attempts']+=1;value=factory();counters['original_constructors_completed']+=1;return value
        def fresh_bank():return bank.make_bank(torch,adapter,counted_factory)
        stage='centered_full_input_bank';model=fresh_bank();opt=fit.optimizer(torch,model,cfg)
        require([group['weight_decay'] for group in opt.param_groups]==[0.0005,0.0005,0.0],'Unchanged native slow decay and zero fast optimizer decay')
        record['initial_structure']=q.structural(torch,model,opt,data,meta,initial=True);topology=q.topology_snapshot(torch,model)
        original_topology=topology
        stage='prior_only_nonzero_away_probe';counters['prior_probe_attempts']+=1;persist()
        record['prior_probe']=probe.probe(np,torch,model,cfg,fit,old,helpers);counters['prior_probes_completed']+=1
        record['centered_prior_probe_passed']=record['prior_probe']['centered_prior_probe_passed'];persist()
        stage='one_full_TRAIN_centered_step';observer=q.GradientObserver(torch,model,persist)
        own_nll,own_prior=fit.train_update(torch,model,opt,data,counters,cfg['sheaf_decay'],observer)
        helpers.synchronize(torch,device)
        record.update(own_TRAIN_NLL=own_nll,member_weighted_centered_prior=own_prior,total_weighted_centered_prior=sum(own_prior),
            centered_update_qualified=True,after_update_structure=q.structural(torch,model,opt,data,meta),
            update_cache_ownership=q.cache_ownership(torch,model,data,meta),update_topology=q.verify_topology(torch,model,topology))
        state=dict(schema='one-step-centered-prior-engineering-v1',identity=record['identity'],model_state=old.cpu_tree(torch,model.state_dict()),
            optimizer_state=old.cpu_tree(torch,opt.state_dict()),training_rng=helpers.capture_rng(np,torch),TEST_truth_saved=False,VALID_truth_saved=False)
        stage='original_label_free_serving';state['label_free_serving']=q.label_free_serving(np,torch,helpers,old,model,data,7409,counters,'original_serving')
        record.update(original_serving_cache_ownership=q.cache_ownership(torch,model,data,meta),original_serving_topology=q.verify_topology(torch,model,topology))
        stage='engineering_state_serialization';tick=time.perf_counter();path=output/'ENGINEERING_STATE.pt';temporary=path.with_suffix('.tmp')
        torch.save(state,temporary);temporary.replace(path);record.update(state_write_seconds=time.perf_counter()-tick,state_bytes=path.stat().st_size,state_sha256=sha(path),server_only_engineering_state=True)
        del state;model=opt=None;gc.collect();torch.cuda.empty_cache()
        stage='fresh_exact_centered_reconstruction';saved=torch.load(path,map_location='cpu',weights_only=True)
        require(saved['identity']==record['identity'] and saved['TEST_truth_saved'] is False and saved['VALID_truth_saved'] is False,'Exact owned engineering identity')
        model=fresh_bank();model.load_state_dict(saved['model_state'],strict=True)
        require(old.exact(torch,old.cpu_tree(torch,model.state_dict()),saved['model_state']),'Exact centered parameter/buffer reconstruction')
        opt=fit.optimizer(torch,model,cfg);opt.load_state_dict(saved['optimizer_state'])
        require(old.exact(torch,old.cpu_tree(torch,opt.state_dict()),saved['optimizer_state']) and [group['weight_decay'] for group in opt.param_groups]==[0.0005,0.0005,0.0],
                'Exact centered optimizer state and group policy')
        helpers.restore_rng(np,torch,saved['training_rng']);require(old.exact(torch,helpers.capture_rng(np,torch),saved['training_rng']),'Exact centered TRAIN RNG reconstruction')
        record.update(model_parameter_buffer_restore_exact=True,optimizer_restore_exact=True,training_RNG_restore_exact=True,
            restored_structure=q.structural(torch,model,opt,data,meta),reconstructed_topology_exact=q.equivalent_topology(torch,model,original_topology))
        topology=q.topology_snapshot(torch,model);stage='restored_label_free_serving'
        serving=q.label_free_serving(np,torch,helpers,old,model,data,7409,counters,'restored_serving')
        record.update(label_free_serving_materiality=q.serving_materiality(torch,saved['label_free_serving'],serving),
            restored_cache_ownership=q.cache_ownership(torch,model,data,meta),restored_topology=q.verify_topology(torch,model,topology),finite_reconstructed_serving=True)
        require(counters['train_forwards_completed']==counters['backwards_completed']==counters['original_serving_forwards_completed']==counters['restored_serving_forwards_completed']==4
            and counters['Adam_steps_completed']==1 and counters['original_constructors_completed']==2,'Exact minimal full-input work counts')
        record.update(status='complete',qualification_passed=True)
    except Exception as error:record.update(status='failed',qualification_passed=False,failure=common.failure_record(error,stage))
    except BaseException as error:
        record.update(status='interrupted',qualification_passed=False,failure=common.failure_record(error,stage));raise
    finally:
        try:
            if torch is not None and device is not None:
                helpers.synchronize(torch,device);record.update(peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_CUDA_reserved_bytes=torch.cuda.max_memory_reserved(device))
                model=opt=None;gc.collect();torch.cuda.empty_cache()
        except Exception as error:record.update(status='failed',qualification_passed=False,cost_or_cleanup_failure=common.failure_record(error,'finalize'))
        usage=resource.getrusage(resource.RUSAGE_SELF);record.update(complete_attempt_seconds=time.perf_counter()-started,CPU_user_seconds=usage.ru_utime-usage0.ru_utime,
            CPU_system_seconds=usage.ru_stime-usage0.ru_stime,process_RSS_high_water_cumulative_bytes=common.process_peak_rss_bytes(),
            actual_TRAIN_forwards=counters['train_forwards_completed'],all_native_forwards=counters['train_forwards_completed']+counters['original_serving_forwards_completed']+counters['restored_serving_forwards_completed'],
            completed_prior_only_autograd_calls=counters['prior_probes_completed'],
            cost_includes_all_constructors_clones_probe_step_state_serving_serialization_and_cleanup=True,comparative_opening_authorized=False)
        persist()
    require(record['qualification_passed'],'Minimal centered full-input qualification failed; retain attempt/cost, no retry')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true')
    for name in ('release','admission','roles','output'):parser.add_argument('--'+name,type=Path)
    args=parser.parse_args()
    if not args.execute:print(json.dumps(dict(inactive=True,numeric_model_or_role_import=False)));return
    require(args.release and args.admission and args.roles and args.output,'Explicit root release/admission/roles/fresh output required');run(args)


if __name__=='__main__':main()
