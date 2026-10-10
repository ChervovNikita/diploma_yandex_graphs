"""Disabled root CLI for the sealed CMCL API; no replacement learning code."""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
PHASE=HERE.parent
REPO=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
SERVER_PHASE=REPO/'experiments_iclr/postsubmission_20260930'
SOURCE=PHASE/'private_cmcl_polyformer_source_20261010_v1'
SOURCE_SHA='1a7e7ebff90d4e9dfdec41eb1e6f2038dd4ee6cae205f2fceb162eaa7b69915b'
REFERENCE=PHASE/'combination_pubmed_strong_reference_source_20261010_v2'
REFERENCE_SHA='a695504452e82203671e11f40c417024184a6c61e4bbe7f107cb5d1b2702e32a'
CONDITIONS=('own_floor','private_cmcl','all_block_cmcl','vanilla_cmcl','private_uniform','private_constant_credit')
SEEDS=(9101,9203,9307)
GPU='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
LIMITS=dict(GPU_bytes=32*1024**3,RSS_bytes=16*1024**3,output_bytes=512*1024**2,
            log_bytes=8*1024**2,external_active_seconds=9000,external_cleanup_seconds=10,automatic_retry=False)
ENGINEERING_LIMITS=dict(LIMITS,external_active_seconds=600)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inside(path,existing=True):
    path=Path(path).resolve(strict=existing)
    if path==PHASE or not path.is_relative_to(PHASE):raise ValueError('Owned phase path required')
    return path


def bind(row):
    path=inside(row['path'])
    if sha(path)!=row['sha256']:raise ValueError('Exact bound artifact changed')
    return path


def write(path,value):
    path=inside(path,existing=False)
    tmp=path.with_name(path.name+'.partial');tmp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');tmp.replace(path)


def verify_payload(folder,manifest_name,digest):
    path=folder/manifest_name
    if sha(path)!=digest:raise ValueError('Exact sealed manifest required')
    for row in json.loads(path.read_text())['files']:
        file=(folder/row['path']).resolve(strict=True)
        if not file.is_relative_to(folder) or sha(file)!=row['sha256'] or file.stat().st_size!=row['bytes']:
            raise ValueError('Sealed payload changed')


def package():
    name='_root_exact_private_cmcl'
    spec=importlib.util.spec_from_file_location(name,SOURCE/'__init__.py',submodule_search_locations=[str(SOURCE)])
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return {n:importlib.import_module(name+'.'+n) for n in ('caps','gate','data_interface','costs','session','fit')}


def reference_io():
    """Reuse the already reviewed exact full-array/provider loader, no new loader."""
    verify_payload(REFERENCE,'SOURCE_MANIFEST.json',REFERENCE_SHA)
    names=('admission','source','reference_plan')
    previous={n:sys.modules.pop(n,None) for n in names};previous_path=list(sys.path)
    try:
        sys.path.insert(0,str(REFERENCE))
        spec=importlib.util.spec_from_file_location('_cmcl_reviewed_reference_io',REFERENCE/'train.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
    finally:
        sys.path[:]=previous_path
        for n,v in previous.items():
            if v is None:sys.modules.pop(n,None)
            else:sys.modules[n]=v


def require_terminal(row,release_sha):
    raw=json.loads(bind(row).read_text());absent=raw.get('owned_absence',{})
    if raw.get('complete') is not True or raw.get('directly_waited') is not True or raw.get('child_exit_code')!=0 or raw.get('cap_or_owner_failure') is not None or release_sha not in raw.get('argv',[]):
        raise ValueError('Actually waited successful engineering owner required')
    if absent.get('owned_process_absence_verified') is not True or absent.get('owned_CUDA_absence_verified') is not True:
        raise ValueError('Owned process/CUDA absence required')


def admit(args):
    if socket.gethostname()!='anogena-2-0' or PHASE!=SERVER_PHASE or Path.cwd().resolve()!=REPO:
        raise ValueError('Exact allocation project route required')
    if subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()!=[GPU] or os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU:
        raise ValueError('Exact sole allocation GPU required')
    path=inside(args.release)
    if not path.is_relative_to(HERE) or sha(path)!=args.release_sha256:raise ValueError('Exact actual root release required')
    spec=json.loads(path.read_text())
    purpose=args.mode
    if spec.get('schema')!='private-CMCL-'+purpose+'-release-v1' or spec.get('purpose')!=purpose:
        raise ValueError('Exact purpose release required')
    for flag in ('enabled','root_authorized','source_review_approved','source_delta_assessment_approved',
                 'complete_roster_frozen','complete_input_custody_verified','VALID_custody_verified',
                 'external_hard_bound_confirmed','fresh_resource_readiness_confirmed','ordinary_runtime_confirmed','post_screen_exploration'):
        if spec.get(flag) is not True:raise ValueError('Inactive root admission: '+flag)
    for flag in ('TEST_access','CORE','masking','HPO','automatic_retry','resume','further18_activated','paper_score_recalculation'):
        if spec.get(flag) is not False:raise ValueError('Closed scope: '+flag)
    if spec.get('VALID_access') is not True:raise ValueError('Sealed fit API uses complete VALID even in qualification; declare it honestly')
    if any(k in spec for k in ('test_bundle','test_ids','test_y','full_y','split_ids_bundle')):raise ValueError('No TEST/full-label interface')
    if spec.get('condition') not in CONDITIONS or spec.get('seed') not in SEEDS or spec.get('record_id')!=f"seed{spec['seed']}__{spec['condition']}":
        raise ValueError('Frozen condition/paired seed required')
    if purpose=='engineering' and (spec['seed']!=9101 or spec.get('science_enabled') is not False):raise ValueError('One-update engineering only')
    if purpose=='science' and spec.get('science_enabled') is not True:raise ValueError('Separate full-science authority required')
    if spec.get('source_manifest_sha256')!=SOURCE_SHA or spec.get('source_seal_sha256')!=sha(SOURCE/'SEAL.json'):
        raise ValueError('Exact sealed learning API required')
    verify_payload(SOURCE,'MANIFEST.json',SOURCE_SHA)
    verify_payload(HERE,'SOURCE_MANIFEST.json',spec['adapter_manifest_sha256'])
    if spec.get('roster_sha256')!=sha(HERE/'ROSTER.json'):raise ValueError('Frozen18-cell roster required')
    review=json.loads(bind(spec['source_review']).read_text())
    if review.get('approved') is not True or review.get('source_manifest_sha256')!=SOURCE_SHA or review.get('adapter_manifest_sha256')!=spec['adapter_manifest_sha256']:
        raise ValueError('Root must review exact source and API adapter')
    bind(spec['source_delta_assessment']);bind(spec['resource_readiness_evidence'])
    data=json.loads((HERE/'DATA_AND_RUNTIME.json').read_text())
    for k in ('train_bundle','valid_bundle','split_custody','validation_custody','runtime','frozen_providers'):
        if spec.get(k)!=data[k]:raise ValueError('Exact existing full-task data/runtime receipt required')
    spec['_train_custody']=json.loads(bind(spec['split_custody']).read_text())
    spec['_validation_custody']=json.loads(bind(spec['validation_custody']).read_text())
    bind(spec['train_bundle']);bind(spec['valid_bundle'])
    if spec['_train_custody'].get('train_bundle_sha256')!=spec['train_bundle']['sha256'] or spec['_validation_custody'].get('valid_bundle_sha256')!=spec['valid_bundle']['sha256'] or spec['_validation_custody'].get('train_custody_sha256')!=spec['split_custody']['sha256'] or spec['_validation_custody'].get('TEST_labels_loaded') is not False:
        raise ValueError('Exact separate TRAIN/VALID role associations without TEST required')
    limits=ENGINEERING_LIMITS if purpose=='engineering' else LIMITS
    if spec.get('limits')!=limits or spec.get('max_epochs')!=2000 or spec.get('patience')!=250:
        raise ValueError('Fixed cap/horizon/selector required')
    owner=json.loads(bind(spec['external_owner_release']).read_text())
    if owner.get('enabled') is not True or owner.get('record_id')!=spec['record_id'] or owner.get('limits')!=limits or owner.get('automatic_retry') is not False:
        raise ValueError('Separately reviewed bounded owner required')
    for k in ('separate_process_group','direct_wait_required','resource_caps_enforced','output_and_log_caps_enforced'):
        if owner.get(k) is not True:raise ValueError('Owner fact missing')
    bind(owner['owner_source']);bind(owner['owner_review'])
    runtime=spec['runtime'];python=Path(os.path.abspath(runtime['python']['path']))
    if not python.is_relative_to(PHASE) or not python.is_file() or sha(python)!=runtime['python']['sha256'] or python.resolve()!=Path(sys.executable).resolve() or os.environ.get('PYTHONPATH','')!=runtime['PYTHONPATH']:
        raise ValueError('Exact existing qualified runtime required')
    if purpose=='science':
        root=json.loads(bind(spec['root_admission']).read_text())
        if root.get('enabled') is not True or root.get('source_manifest_sha256')!=SOURCE_SHA or root.get('adapter_manifest_sha256')!=spec['adapter_manifest_sha256'] or root.get('roster_sha256')!=sha(HERE/'ROSTER.json'):
            raise ValueError('Exact full-six-condition root adoption required')
        if set(root.get('qualifications',{}))!=set(CONDITIONS):raise ValueError('All six API paths need actual complete-input qualification')
        for condition,row in root['qualifications'].items():
            result=json.loads(bind(row['complete']).read_text())
            if result.get('complete') is not True or result.get('schema')!='private-CMCL-engineering-complete-v1' or result.get('condition')!=condition or result.get('seed')!=9101 or result.get('source_manifest_sha256')!=SOURCE_SHA or result.get('adapter_manifest_sha256')!=spec['adapter_manifest_sha256'] or result.get('updates')!=1 or result.get('fresh_reconstruction_verified') is not True or result.get('VALID_access') is not True or result.get('science_enabled') is not False:
                raise ValueError('Actual one-full-update/fresh-replay engineering result required')
            require_terminal(row['raw_terminal'],result['release_sha256'])
    output=inside(spec['output'],existing=False)
    if not output.is_relative_to(HERE) or output.exists():raise ValueError('Fresh root-confined output required')
    spec['_release_sha256']=args.release_sha256
    return spec,output


def finite_tree(torch,value):
    if isinstance(value,torch.Tensor):return bool(torch.isfinite(value).all())
    if isinstance(value,dict):return all(finite_tree(torch,v) for v in value.values())
    if isinstance(value,(list,tuple)):return all(finite_tree(torch,v) for v in value)
    return True


def execute(spec,output,started):
    output.mkdir(parents=True,exist_ok=False);session=None
    try:
        api=package()
        caps=api['caps'].Caps(source_bound=True,model=True,data=True,runtime=True,
                              scientific=spec['purpose']=='science',root_review_sha256=spec['source_review']['sha256'])
        identity=dict(source_seal_sha256=spec['source_seal_sha256'],root_review_sha256=spec['source_review']['sha256'],
                      input_custody_sha256=spec['split_custody']['sha256'],TRAIN_custody_sha256=spec['split_custody']['sha256'],
                      VALID_custody_sha256=spec['validation_custody']['sha256'])
        api['gate'].source_gate(identity,caps)
        io=reference_io();np,torch,providers,tensor,roles=io.runtime_and_data(spec)
        device=torch.device('cuda:0');costs=api['costs'].Costs(output,torch,device,caps)
        train=api['data_interface'].Role(*roles['TRAIN'],spec['split_custody']['sha256'])
        valid=api['data_interface'].Role(*roles['VALID'],spec['validation_custody']['sha256'])
        inputs=api['data_interface'].prepare(torch,tensor['x'],tensor['edge_index'],train,valid,
            spec['split_custody']['sha256'],device,costs,caps)
        session=api['session'].Session(spec['condition'],spec['seed'],inputs,device,costs,identity,caps)
        with costs.measure('inclusive_existing_fit_API_not_additive_to_nested_scopes'):
            fit=api['fit'].fit(session,output,numerical_qualification=spec['purpose']=='engineering',caps=caps)
        counters=dict(session.counters)
        if not finite_tree(torch,session.model.state_dict()) or not finite_tree(torch,session.optimizer.state_dict()):
            raise FloatingPointError('Nonfinite full model or original Adam state')
        updates=len(fit['history']);assignment=spec['condition'] in ('private_cmcl','all_block_cmcl','vanilla_cmcl')
        private=spec['condition'] in ('private_cmcl','private_uniform','private_constant_credit')
        expected=dict(updates=updates,optimizer_steps=updates,score_fullgraph_forwards=4*updates if assignment else 0,
                      gradient_fullgraph_forwards=4*updates,own_gradient_VJPs=4*updates if private else 0,
                      auxiliary_gradient_VJPs=4*updates if private else 0,combined_gradient_VJPs=0 if private else 4*updates,
                      serving_fullgraph_forwards=4*updates)
        if counters!=expected or (spec['purpose']=='engineering' and updates!=1):raise ValueError('Complete exact API work counts required')
        if spec['purpose']=='science' and updates<2000 and updates-1-fit['selected_epoch']!=250:
            raise ValueError('Private scientific horizon shortened')
        session=None;gc.collect()
        with costs.measure('selected_checkpoint_metadata_read'):
            checkpoint=output/'SELECTED_STATE.pt';selected=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if not finite_tree(torch,selected['model']) or not finite_tree(torch,selected['optimizer']):raise FloatingPointError('Nonfinite selected model/Adam')
        immutable_scores=selected['scores'];del selected
        with costs.measure('inclusive_existing_fresh_reconstruction_API_not_additive_to_nested_scopes'):
            replay=api['fit'].load_and_reconstruct(checkpoint,inputs,device,costs,caps)
        if time.monotonic()-started>=spec['limits']['external_active_seconds']:raise TimeoutError('Active cap before completion')
        stored_bytes=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
        if stored_bytes>spec['limits']['output_bytes']:raise RuntimeError('Output cap before completion')
        result=dict(schema='private-CMCL-'+spec['purpose']+'-complete-v1',complete=True,condition=spec['condition'],seed=spec['seed'],
                    record_id=spec['record_id'],release_sha256=spec['_release_sha256'],source_manifest_sha256=SOURCE_SHA,
                    adapter_manifest_sha256=spec['adapter_manifest_sha256'],identity=identity,providers=providers,
                    TRAIN_count=11829,VALID_count=3942,graph_nodes=19717,edge_count=88648,
                    updates=updates,selected_epoch_zero_based=fit['selected_epoch'],max_epochs=2000,patience=250,
                    selector=fit['selector'],selected_scores=immutable_scores,reconstruction=replay,
                    fresh_reconstruction_verified=True,member_argmax_identity_claim=False,
                    original_fit_counters=counters,fresh_reconstruction_serving_forwards=4,
                    predictor_parameters=2125647,shared_parameters=2069875,private_parameters=55772,decoder_parameters=0,
                    checkpoint=dict(path=str(checkpoint),sha256=sha(checkpoint),bytes=checkpoint.stat().st_size,
                                    contains_TRAIN_VALID_role_outputs_and_truth=True,server_only=True),
                    fit_record=dict(path=str(output/'FIT_RECORD.json'),sha256=sha(output/'FIT_RECORD.json')),
                    cost_events=dict(path=str(output/'COST_EVENTS.jsonl'),sha256=sha(output/'COST_EVENTS.jsonl')),
                    costs=dict(inclusive_seconds=time.monotonic()-started,CPU_user_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime,
                        CPU_system_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_stime,
                        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                        peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(),peak_CUDA_reserved_bytes=torch.cuda.max_memory_reserved(),
                        output_bytes_before_COMPLETE=stored_bytes,nested_scope_seconds_not_additive=True,
                        history_rewrite_IO_and_replay_charged_by_inclusive_owner_wall=True),
                    VALID_access=True,TEST_access=False,science_enabled=spec['purpose']=='science',
                    engineering_quality_is_accuracy_evidence=False,post_screen_exploration=True,
                    CORE=False,masking=False,HPO=False,automatic_retry=False,further18_activated=False,
                    paper_score_recalculation=False,partial_family_comparison_allowed=False,
                    current_reference_quality_read=False,owner_success_not_inferred=True)
        write(output/'COMPLETE.json',result);return result
    except BaseException as error:
        write(output/'FAILURE.json',dict(complete=False,record_id=spec['record_id'],error_type=type(error).__name__,error=str(error),
            inclusive_seconds=time.monotonic()-started,counters=None if session is None else session.counters,
            VALID_access=True,TEST_access=False,automatic_retry=False,partial_family_comparison_allowed=False))
        raise


def main():
    started=time.monotonic();p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('engineering','science'),required=True)
    p.add_argument('--release',type=Path,required=True);p.add_argument('--release-sha256',required=True)
    args=p.parse_args();spec,output=admit(args);result=execute(spec,output,started)
    print(json.dumps(dict(complete=result['complete'],record_id=spec['record_id'],owner_success_not_inferred=True)))


if __name__=='__main__':main()
