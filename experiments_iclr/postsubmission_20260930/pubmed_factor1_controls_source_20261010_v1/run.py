"""Disabled M1-factor controls using exact existing reference fit/data routines."""
import argparse
from contextlib import contextmanager
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

HERE=Path(__file__).resolve().parent;PHASE=HERE.parent
REPO=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
SERVER_PHASE=REPO/'experiments_iclr/postsubmission_20260930'
REFERENCE=PHASE/'combination_pubmed_strong_reference_source_20261010_v2'
REFERENCE_SHA='a695504452e82203671e11f40c417024184a6c61e4bbe7f107cb5d1b2702e32a'
CONDITIONS=('factor1_native','factor1_mean4_dropout');SEEDS=(9101,9203,9307)
ALIASES=dict(factor1_native='single_native',factor1_mean4_dropout='single_mean4_dropout')
GPU='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
SELECTOR='first strict maximum own raw-logit max(1)[1] VALID accuracy'
LIMITS=dict(GPU_bytes=32*1024**3,RSS_bytes=16*1024**3,output_bytes=512*1024**2,log_bytes=8*1024**2,
            external_active_seconds=9000,external_cleanup_seconds=10,automatic_retry=False)
ENGINEERING_LIMITS=dict(LIMITS,external_active_seconds=600)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inside(path,existing=True):
    path=Path(path).resolve(strict=existing)
    if path==PHASE or not path.is_relative_to(PHASE):raise ValueError('Exact owned project path required')
    return path


def bind(row):
    path=inside(row['path'])
    if sha(path)!=row['sha256']:raise ValueError('Bound artifact changed')
    return path


def write(path,value):
    path=inside(path,existing=False);tmp=path.with_name(path.name+'.partial')
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');tmp.replace(path)


def verify(folder,digest):
    path=folder/'SOURCE_MANIFEST.json'
    if sha(path)!=digest:raise ValueError('Exact sealed source manifest required')
    for row in json.loads(path.read_text())['files']:
        file=(folder/row['path']).resolve(strict=True)
        if not file.is_relative_to(folder) or sha(file)!=row['sha256'] or file.stat().st_size!=row['bytes']:
            raise ValueError('Sealed source payload changed')


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


@contextmanager
def reference_surface():
    """Scoped existing fit imports; explicit M1 factory replaces only construction."""
    verify(REFERENCE,REFERENCE_SHA)
    names=('source','admission','reference_plan','adapter','metrics')
    prior={n:sys.modules.pop(n,None) for n in names};old_path=list(sys.path)
    try:
        sys.path.insert(0,str(REFERENCE))
        engine=load('_exact_M1_reference_engine',REFERENCE/'train.py')
        base=load('_exact_M1_reference_factory',REFERENCE/'adapter.py')
        metrics=load('metrics',REFERENCE/'metrics.py')
        factor=load('_exact_M1_factor_factory',HERE/'adapter.py')
        factor.REFERENCE_FACTORY=base.fresh_single
        sys.modules['adapter']=factor
        yield engine,metrics,factor
    finally:
        sys.path[:]=old_path
        for n,v in prior.items():
            if v is None:sys.modules.pop(n,None)
            else:sys.modules[n]=v


def terminal(row,release_sha):
    raw=json.loads(bind(row).read_text());absence=raw.get('owned_absence',{})
    if raw.get('complete') is not True or raw.get('directly_waited') is not True or raw.get('child_exit_code')!=0 or raw.get('cap_or_owner_failure') is not None or release_sha not in raw.get('argv',[]):
        raise ValueError('Actual successfully waited qualifier required')
    if absence.get('owned_process_absence_verified') is not True or absence.get('owned_CUDA_absence_verified') is not True:
        raise ValueError('Confirmed owned process/CUDA absence required')


def admit(args):
    if socket.gethostname()!='anogena-2-0' or PHASE!=SERVER_PHASE or Path.cwd().resolve()!=REPO:
        raise ValueError('Exact allocation route/cwd required')
    if subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()!=[GPU] or os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU:
        raise ValueError('Exact sole allocation GPU required')
    path=inside(args.release)
    if not path.is_relative_to(HERE) or sha(path)!=args.release_sha256:raise ValueError('Exact root release required')
    spec=json.loads(path.read_text());purpose=args.mode
    if spec.get('schema')!='PubMed-factor1-'+purpose+'-release-v1' or spec.get('purpose')!=purpose:raise ValueError('Exact purpose release required')
    for k in ('enabled','root_authorized','source_review_approved','source_delta_assessment_approved','complete_roster_frozen',
              'complete_input_custody_verified','external_hard_bound_confirmed','fresh_resource_readiness_confirmed','ordinary_runtime_confirmed','post_screen_exploration'):
        if spec.get(k) is not True:raise ValueError('Inactive root admission: '+k)
    for k in ('TEST_access','CORE','masking','HPO','automatic_retry','resume','further18_activated','paper_score_recalculation','pure_parameter_match','confirmation_claim'):
        if spec.get(k) is not False:raise ValueError('Closed scope: '+k)
    if spec.get('condition') not in CONDITIONS or spec.get('seed') not in SEEDS or spec.get('record_id')!=f"seed{spec['seed']}__{spec['condition']}":raise ValueError('Frozen six-cell identity required')
    if spec.get('internal_reference_alias')!=ALIASES[spec['condition']] or spec.get('max_epochs')!=2000 or spec.get('patience')!=250 or spec.get('selector')!=SELECTOR:
        raise ValueError('Transparent exact ordinary/four-loss alias and own2000/250 selector required')
    if any(k in spec for k in ('test_bundle','test_y','test_ids','split_ids_bundle','full_y')):raise ValueError('No TEST or full-label interface')
    if spec.get('source_manifest_sha256')!=sha(HERE/'SOURCE_MANIFEST.json') or spec.get('roster_sha256')!=sha(HERE/'ROSTER.json'):raise ValueError('Exact new control source/roster required')
    verify(HERE,spec['source_manifest_sha256'])
    review=json.loads(bind(spec['source_review']).read_text())
    if review.get('approved') is not True or review.get('source_manifest_sha256')!=spec['source_manifest_sha256']:raise ValueError('Root exact-source review required')
    bind(spec['source_delta_assessment']);bind(spec['resource_readiness_evidence'])
    data=json.loads((HERE/'DATA_AND_RUNTIME.json').read_text())
    for k in ('train_bundle','split_custody','runtime','frozen_providers'):
        if spec.get(k)!=data[k]:raise ValueError('Exact existing data/runtime required')
    bind(spec['train_bundle']);spec['_train_custody']=json.loads(bind(spec['split_custody']).read_text())
    if purpose=='engineering':
        if spec['seed']!=9101 or spec.get('VALID_access') is not False or spec.get('science_enabled') is not False or any(k in spec for k in ('valid_bundle','validation_custody')):raise ValueError('Fresh TRAIN-only qualification required')
    else:
        if spec.get('VALID_access') is not True or spec.get('science_enabled') is not True or spec.get('VALID_custody_verified') is not True:raise ValueError('Full native-science VALID role required')
        for k in ('valid_bundle','validation_custody'):
            if spec.get(k)!=data[k]:raise ValueError('Exact actual VALID projection required')
        bind(spec['valid_bundle']);spec['_validation_custody']=json.loads(bind(spec['validation_custody']).read_text())
        root=json.loads(bind(spec['root_admission']).read_text())
        if root.get('enabled') is not True or root.get('source_manifest_sha256')!=spec['source_manifest_sha256'] or root.get('roster_sha256')!=sha(HERE/'ROSTER.json') or set(root.get('qualifications',{}))!=set(CONDITIONS):raise ValueError('Exact complete two-interface root adoption required')
        for c,row in root['qualifications'].items():
            result=json.loads(bind(row['complete']).read_text())
            if result.get('schema')!='PubMed-factor1-engineering-complete-v1' or result.get('complete') is not True or result.get('condition')!=c or result.get('source_manifest_sha256')!=spec['source_manifest_sha256'] or result.get('updates')!=1 or result.get('VALID_access') is not False or result.get('fresh_weight_restore_verified') is not True:raise ValueError('Actual whole-TRAIN update and fresh M1 weight restore required')
            terminal(row['raw_terminal'],result['release_sha256'])
    limits=ENGINEERING_LIMITS if purpose=='engineering' else LIMITS
    if spec.get('limits')!=limits:raise ValueError('Frozen finite limits required')
    owner=json.loads(bind(spec['external_owner_release']).read_text())
    if owner.get('enabled') is not True or owner.get('record_id')!=spec['record_id'] or owner.get('limits')!=limits or owner.get('automatic_retry') is not False:raise ValueError('New finite owner required')
    for k in ('separate_process_group','direct_wait_required','resource_caps_enforced','output_and_log_caps_enforced'):
        if owner.get(k) is not True:raise ValueError('Owner fact missing')
    bind(owner['owner_source']);bind(owner['owner_review'])
    runtime=spec['runtime'];python=Path(os.path.abspath(runtime['python']['path']))
    if not python.is_relative_to(PHASE) or not python.is_file() or sha(python)!=runtime['python']['sha256'] or python.resolve()!=Path(sys.executable).resolve() or os.environ.get('PYTHONPATH','')!=runtime['PYTHONPATH']:raise ValueError('Exact existing qualified Python/PYTHONPATH required')
    output=inside(spec['output'],existing=False)
    if not output.is_relative_to(HERE) or output.exists():raise ValueError('Fresh root-confined output required')
    spec['_release_sha256']=args.release_sha256;return spec,output


def execute(spec,output,started):
    output.mkdir(parents=True,exist_ok=False);session=None;timings={}
    try:
        with reference_surface() as (engine,metrics,factory):
            tick=time.monotonic()
            np,torch,providers,tensor,roles=engine.runtime_and_data(spec,valid=spec['purpose']=='science')
            engine.add(timings,'numeric_imports_and_complete_input',dict(wall_seconds=time.monotonic()-tick))
            alias=ALIASES[spec['condition']]
            if spec['purpose']=='engineering':
                session,clock=engine.timed(torch,lambda:factory.fresh_single(alias,spec['seed'],tensor));engine.add(timings,'construction_preprocessing_factor_Adam',clock)
                training,clock=engine.timed(torch,lambda:session.train_step(audit=False));engine.add(timings,'full_TRAIN_update',clock)
                (probabilities,logits),clock=engine.timed(torch,session.factual_probabilities);engine.add(timings,'fullgraph_factual_serving',clock)
                signature=metrics.signatures(logits,probabilities,*roles['TRAIN']);selected=metrics.classification(logits,probabilities,*roles['TRAIN'])
                counters=dict(session.counters);views=4 if alias=='single_mean4_dropout' else 1
                if counters!=dict(updates=1,factual_forwards=views+1,masked_forwards=0,backwards=views,optimizer_steps=1,serving_forwards=1,preprocessing_banks=1):raise ValueError('Full TRAIN work counts changed')
                tick=time.monotonic();state={k:v.detach().cpu().clone() for k,v in session.bodies[0].state_dict().items()}
                checkpoint=output/'QUALIFIED_M1_WEIGHTS.pt';torch.save(state,checkpoint)
                engine.add(timings,'qualification_weight_snapshot_write',dict(wall_seconds=time.monotonic()-tick))
                session=None;del probabilities,logits;gc.collect()
                session,clock=engine.timed(torch,lambda:factory.fresh_single(alias,spec['seed'],tensor));engine.add(timings,'fresh_restore_construction_preprocessing_factor_Adam',clock)
                def restore():session.bodies[0].load_state_dict(torch.load(checkpoint,map_location='cpu'),strict=True)
                _,clock=engine.timed(torch,restore);engine.add(timings,'fresh_selected_weight_restore',clock)
                (probabilities,logits),clock=engine.timed(torch,session.factual_probabilities);engine.add(timings,'fresh_restore_factual_serving',clock)
                if metrics.signatures(logits,probabilities,*roles['TRAIN'])!=signature:raise ValueError('Restored TRAIN predictions/correct masks changed')
                metrics.verify_floats(metrics.classification(logits,probabilities,*roles['TRAIN']),selected)
                result=dict(updates=1,training=training,counters=counters,fresh_restore_counters=dict(session.counters),fresh_weight_restore_verified=True,
                            TRAIN_count=11829,graph_nodes=19717,TRAIN_quality_is_accuracy_evidence=False,model_constructions=2,preprocessing_banks=2,Adam_constructions=4)
            else:
                run_spec=dict(spec,condition=alias)
                body=dict(body_id='body0',initialization_seed=spec['seed'],factual_dropout_seeds=[spec['seed']+300001+1009*v for v in range(4 if alias=='single_mean4_dropout' else 1)],max_epochs=2000,patience=250,selector=SELECTOR,fresh_full_native_fit=True,reuse_other_record=False)
                session,logits,state,row=engine.fit_body(run_spec,body,tensor,roles,output/'body0',started,timings)
                del state
                probabilities=logits.softmax(-1).mean(0)
                readouts={name:dict(classification=metrics.classification(logits,probabilities,*role),repair=metrics.repair_diagnostics(logits,probabilities,*role),prediction_signatures=metrics.signatures(logits,probabilities,*role)) for name,role in roles.items()}
                benchmark=engine.serving_benchmark(session,readouts['VALID']['prediction_signatures'],roles,timings)
                tick=time.monotonic();payload=output/'SELECTED_MEMBER_LOGITS.npz';np.savez_compressed(payload,factual_member_logits=logits.detach().cpu().numpy())
                engine.add(timings,'selected_fullgraph_logit_write',dict(wall_seconds=time.monotonic()-tick))
                result=dict(updates=row['epochs_executed'],body=row,selected_epoch=row['selected_epoch'],selected_VALID=readouts['VALID']['classification'],
                            selected_prediction_signatures=readouts['VALID']['prediction_signatures'],selected_readouts=readouts,serving_benchmark=benchmark,
                            complete_saved_member_logits=dict(path=str(payload),sha256=sha(payload),bytes=payload.stat().st_size,factual_shape=[1,19717,3],server_only=True,contains_labels_or_role_ids=False),
                            final_session_counters=dict(session.counters),model_constructions=1,preprocessing_banks=1,Adam_constructions=2,
                            fresh_training_trajectory=True,pooling='one native factual predictor',whole_six_comparison_required=True)
            engine.check_bounds(spec,output,started)
            result.update(schema='PubMed-factor1-'+spec['purpose']+'-complete-v1',complete=True,record_id=spec['record_id'],condition=spec['condition'],seed=spec['seed'],
                internal_reference_alias=alias,source_manifest_sha256=spec['source_manifest_sha256'],release_sha256=spec['_release_sha256'],providers=providers,
                predictor_parameters=2083818,native_slow_parameters=2069875,extra_factor_coordinates=13943,factor_sites=23,factor_members=1,unit_factor_initialization=True,
                native_slow_initialization_unchanged=True,pure_parameter_match=False,own_native_selector=SELECTOR,max_epochs=2000,patience=250,timings=timings,
                inclusive_seconds=time.monotonic()-started,CPU_user_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime,CPU_system_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_stime,
                peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_CUDA_bytes=dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved()),
                output_bytes_before_COMPLETE=sum(p.stat().st_size for p in output.rglob('*') if p.is_file()),VALID_access=spec['purpose']=='science',TEST_access=False,
                science_enabled=spec['purpose']=='science',post_screen_exploration=True,confirmation_claim=False,CORE=False,masking=False,HPO=False,automatic_retry=False,
                paper_score_recalculation=False,further18_activated=False,current_CMCL_or_77_quality_read=False,owner_success_not_inferred=True)
            write(output/'COMPLETE.json',result);return result
    except BaseException as error:
        write(output/'FAILURE.json',dict(complete=False,record_id=spec['record_id'],error_type=type(error).__name__,error=str(error),inclusive_seconds=time.monotonic()-started,
            counters=None if session is None else session.counters,timings=timings,automatic_retry=False,TEST_access=False,partial_family_comparison_allowed=False))
        raise


def main():
    started=time.monotonic();p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('engineering','science'),required=True);p.add_argument('--release',type=Path,required=True);p.add_argument('--release-sha256',required=True)
    args=p.parse_args();spec,output=admit(args);result=execute(spec,output,started)
    print(json.dumps(dict(complete=result['complete'],record_id=spec['record_id'],owner_success_not_inferred=True)))


if __name__=='__main__':main()
