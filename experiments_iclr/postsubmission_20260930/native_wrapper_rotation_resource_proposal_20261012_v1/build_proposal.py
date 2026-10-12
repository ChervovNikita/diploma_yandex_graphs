"""Compile ten exact actual-input resource cases from frozen metadata only."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

PACKETS=('common_native_wrapper_packet_20261012_v2','private_graph_rotation_q21_successor_20261012_v1')
CAP={'path':'private_source_zero_update_helpers_20261012_v2/runtime/qa/practical_zero_update_resources.py','bytes':34227,
     'sha256':'effa70e31e64881322ccfb5a6e7e3903ccdb6f887c694a97f6b63d0bd2e267e0'}
TENSOR={'path':'private_source_zero_update_helpers_20261012_v2/runtime/qa/private_source_zero_update_resources.py','bytes':30957,
        'sha256':'259ea644b251012da75c7010746e231cc70243cfd462eaef9b04464350adab41'}
RUNTIME='master_controller_source_20261012_v5/runtime'


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def fingerprint(value):return hashlib.sha256(canonical(value).encode()).hexdigest()
def receipt(root,path):
    path=Path(path).resolve();raw=path.read_bytes()
    return {'path':str(path.relative_to(root)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def build(root,output):
    root=Path(root).resolve();output=Path(output).resolve();cases=[];metadata=[];kept=0
    for name in PACKETS:
        packet=root/name;entry=json.loads((packet/'MASTER_INPUT_ENTRY.json').read_text());plan=json.loads((root/entry['plan']['path']).read_text())
        if fingerprint(plan)!=entry['plan_hash'] or plan['scientific_launch_enabled'] is not False:raise ValueError('Exact disabled frozen packet required')
        metadata.append(receipt(root,packet/'MASTER_INPUT_ENTRY.json'))
        common=name.startswith('common');family='common_native_wrapper' if common else 'private_rotation'
        arm='exchange' if common else 'shared4_paired_local';field='wrapper_arm' if common else 'rotation_arm';seed=88101 if common else 77101
        for row in plan['studies']:
            path=root/row['path'];study=json.loads(gzip.decompress(path.read_bytes()))
            if fingerprint(study)!=row['spec_hash']:raise ValueError('Frozen study differs')
            candidates=[r for r in study['runs'] if r['kind']=='model_fit' and r['config'].get(field)==arm and r['config']['stage']=='search'
                        and r['seed']==seed and r['config']['dropout']==.4 and r['config']['learning_rate']==.002]
            if len(candidates)!=1:raise ValueError('One exact largest fixed source search point required')
            run=candidates[0];c=run['config'];shape=study['data']['evidence']['model_graph']
            if c['members']!=4 or c['width']!=128 or c['depth']!=2 or run['body']!=0 or run['split']!=0:raise ValueError('Exact shared M4/H128/L2/mask0 shape required')
            if any(k in c for k in ('bootstrap_weights','execution_engineering')) or c.get('is_original_gnnm'):raise ValueError('No already-qualified weighted/recompute/baseline cases')
            custody=next(v for v in study['source']['files'] if Path(v['path']).name=='CUSTODY.json')
            # Q21 source metadata lives on allocation; use the exact byte copy
            # owned by the common packet to inspect role counts locally.
            copy_path=root/'common_native_wrapper_packet_20261012_v2/metadata'/study['dataset']/'CUSTODY.json'
            observed=receipt(root,copy_path)
            if (observed['bytes'],observed['sha256'])!=(custody['bytes'],custody['sha256']):raise ValueError('Custody metadata copy differs')
            provider=json.loads(copy_path.read_text());train_shapes=sorted({s['role_counts']['train'] for s in provider['splits']})
            expected_rows=580 if study['dataset']=='wikics' else 160
            if train_shapes!=[expected_rows]:raise ValueError('Mask0 must attain maximum TRAIN rows on all frozen masks')
            imported=[v for v in study['source']['files'] if v['path'].startswith(name+'/') and Path(v['path']).suffix=='.py']
            for v in imported:
                if receipt(root,root/v['path'])!=v:raise ValueError('Frozen source import changed')
            factory=receipt(root,root/run['adapter'].split(':')[0])
            if factory not in imported:raise ValueError('Exact factory must belong to pinned imports')
            actions={'model_forwards':4,'native_body_forward_calls':0 if common else 4,'native_stem_calls':4,
              'native_residual0_calls':4,'native_residual1_calls':4,'native_graph_block_calls':8,'native_head_calls':4,
              'coupled_route_block_calls':1 if common else 0,'corrected_dense_map_calls':0 if common else 8,
              'Householder_reflection_calls':0 if common else 16,'backward_calls':1,'autograd_grad_VJP_calls':0,
              'optimizer_attempts':1,'actual_optimizer_updates':0,'optimizer_kernels':0}
            sources=c['wrapper_sources'] if common else c['rotation_sources']
            cases.append({'case_id':family+'-'+study['dataset']+'-'+c['backbone'].lower()+'-'+arm,'family':family,'dataset':study['dataset'],
              'backbone':c['backbone'],'arm':arm,'study_id':study['study_id'],'study_receipt':receipt(root,path),'study_spec_hash':row['spec_hash'],
              'source_run_id':run['run_id'],'source_run_hash':fingerprint(run),'frozen_factory':factory,'factory_entrypoint':run['adapter'],
              'source_import_receipts':imported,'source_bindings':sources,'provider_custody':custody,'provider_custody_local_copy':observed,
              'model_graph_shape':shape,'train_rows':expected_rows,'all_mask_TRAIN_shape_set':train_shapes,
              'readable_role_inputs':{k:c['node_inputs'][k] for k in ('graph','train')},
              'shape_config':{k:c[k] for k in ('width','depth','classes','dropout','learning_rate','members')},
              'rank':16 if common else None,'seed':run['seed'],'body':run['body'],'split':run['split'],
              'cold_cycles':1,'warm_cycles':3,'timeout_seconds':90,'expected_actions_per_cycle':actions,
              'expected_label_free_serving':{'logits_shape':[4,shape['nodes'],c['classes']],'full_native_route_trajectories':4,
                'native_graph_block_calls':8,'coupled_route_block_calls':int(common),'corrected_dense_map_calls':0 if common else 8},
              'literal_extra_parameter_geometry':{'shared_exchange_parameters':8192 if common else 0,
                'paired_local_parameters':0 if common else 2048 if c['backbone']=='GAT' else 4096},
              'selection_policy':'maximal declared H128/L2/M4 shape;dropout.4;fixed seed/LR tie;mask0 attains maximumTRAIN;no scores used',
              'source_callback':'exact frozen factory ClassifierAdapter.train_step/custom_train; full native stem/both residuals/head; source private RNG; one denied step',
              'coupled_native_path':'literal four aligned live prefixes,one shared exchange,then four complete continuing tails' if common else 'literal four complete native routes with two local-pair corrected maps each',
              'source_STEP_denial_and_state_restore':True,'scientific_fit':False,'emit_losses_logits_predictions_or_scores':False})
            metadata.append(receipt(root,path));kept+=sum(r['kind']=='model_fit' and r['config'].get('is_original_gnnm',False) for r in study['runs'])
    if len(cases)!=10 or len({c['case_id'] for c in cases})!=10 or kept!=630:raise ValueError('Exact six-wrapper/four-local roster and baseline custody required')
    runtime_receipts=[receipt(root,root/RUNTIME/name) for name in ('engine.py','run_study.py','spec.py','registry.py')]
    probe=receipt(root,output/'qualify_wrapper_rotation.py')
    proposal={'schema':'wrapper-rotation-ten-case-resource-proposal-v1','requested_cases':cases,'case_counts':{'common_native_wrapper':6,'private_rotation':4},
      'launch_enabled':False,'root_review_required':True,'scientific_GO':False,'actual_optimizer_updates_authorized':0,'scientific_fits_authorized':0,
      'inherited_helpers':{'role_optimizer_cap_helper':CAP,'tensor_commit_tree_helper':TENSOR},'runtime_package':RUNTIME,'runtime_dependency_receipts':runtime_receipts,
      'probe_receipt':probe,'metadata_receipts':metadata,'original_GNNM_fits_preserved_in_ten_studies':kept,'added_original_GNNM_fits':0,
      'initial_callbacks':10,'warm_callbacks':30,'callback_count':40,'expected_total_callback_work':{
        k:sum(c['expected_actions_per_cycle'][k]*4 for c in cases) for k in cases[0]['expected_actions_per_cycle']},
      'label_free_full_graph_serving_calls':10,'disposable_engine_checkpoint_roundtrips':10,'dummy_CPU_bank_roundtrips':10,
      'child_deadline_sum_seconds':900,'runtime_ETA':None,'CUDA_executions':0,'real_data_arrays_opened':False,'numerical_model_construction':False,
      'two_process_pairs_proposed':0,'automatic_retry':False,'cap_policy':'existing h:min(fresh_free-4GiB,total*0.4);current child only',
      'coverage_limits':['Ten source-configuration engineering cases only;no broad master or concurrent-worker admission.',
        'Full shape/layout and synthetic manual Adam moments;optimizer kernels denied,so native optimizer-kernel latency/workspace remains unmeasured.',
        'Four views of one shared body;no duplicate bootstrap or sign-only controls;no separately initialized I4 GPU fits or SELECT/CAL/evaluate access.',
        'Mask0 reaches declared maximumTRAIN counts;root decides any transfer of this source/memory envelope to other seeds/masks/settings.',
        'Dummy CPU bank serializer uses only custody role counts;actual endpoint targets/IDs/logits/accuracy/temperatures are not opened or emitted.']}
    (output/'PROPOSAL.json').write_text(json.dumps(proposal,indent=2,sort_keys=True)+'\n')
    return proposal


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project-root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();v=build(a.project_root,a.output);print(canonical({'cases':len(v['requested_cases']),'callbacks':v['callback_count'],
      'expected_work':v['expected_total_callback_work'],'CUDA':0,'fits':0,'GO':False}))
