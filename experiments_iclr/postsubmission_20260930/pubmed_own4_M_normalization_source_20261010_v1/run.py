"""Disabled qualification/three fresh own4 fits/closed stored-logit comparison."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import sys
import time

HERE=Path(__file__).resolve().parent;PHASE=HERE.parent
FULL=PHASE/'private_hop_credit_pubmed_fullfit_source_20261010_v1'
DIAG=PHASE/'combination_three_bank_stored_error_source_20261010_v1'
I4=PHASE/'pubmed_factorized_I4_reference_source_20261010_v1'
CONDITION='shared4_own_M_normalized';SEEDS=(9101,9203,9307)
QUAL='seed9101__own4_M_normalization_TRAIN_qualification';COMPARE='own4_M_normalization_complete3_comparison'


def surface():
    import importlib.util
    spec=importlib.util.spec_from_file_location('_own4_normalization_existing_entry',FULL/'run.py')
    entry=importlib.util.module_from_spec(spec);sys.modules[spec.name]=entry;spec.loader.exec_module(entry)
    return entry


def record_ids(mode):
    return [QUAL] if mode=='qualification' else [COMPARE] if mode=='comparison' else [f'seed{s}__{CONDITION}' for s in SEEDS]


def limits(base,mode):
    if mode=='science':return dict(base.LIMITS)
    if mode=='qualification':return dict(base.LIMITS,external_active_seconds=600)
    return dict(base.LIMITS,external_active_seconds=600,GPU_bytes=0,RSS_bytes=1073741824,output_bytes=8388608,log_bytes=2097152)


def admit(args,entry):
    entry.qualification_surface().route(owner=args.mode=='comparison')
    base=entry.load('_own4_normalization_reference',PHASE/'pubmed_factor1_controls_source_20261010_v1/run.py')
    ic=entry.load('_own4_normalization_existing_checks',I4/'common.py')
    path=args.release.resolve(strict=True)
    if not path.is_relative_to(HERE) or entry.sha(path)!=args.release_sha256:raise ValueError('Exact new root release required')
    spec=json.loads(path.read_text())
    if spec.get('schema')!='PubMed-own4-M-normalization-release-v1' or spec.get('purpose')!=args.mode or spec.get('record_id') not in record_ids(args.mode):
        raise ValueError('Fixed normalization stage/record required')
    for key in ('enabled','root_authorized','source_review_approved','complete_roster_frozen','native_shared_groups_unchanged','fresh_resource_readiness_confirmed','ordinary_runtime_confirmed','external_hard_bound_confirmed'):
        if spec.get(key) is not True:raise ValueError('Disabled normalization admission: '+key)
    for key in ('TEST_access','HPO','automatic_retry','resume','arm_selection','coefficient_selection','novelty_claim','paper_score_recalculation'):
        if spec.get(key) is not False:raise ValueError('Closed normalization scope: '+key)
    if any(k in spec for k in ('test_bundle','test_y','test_ids','full_y')):raise ValueError('No TEST inputs')
    if spec['condition']!=CONDITION or spec['M']!=4 or spec['private_decay_divisor']!=4 or spec['private_epsilon_divisor']!=4 or spec['max_epochs']!=2000 or spec['patience']!=250 or spec['selector']!=entry.M4_SELECTOR:
        raise ValueError('Exact one M-derived own4 control required')
    if spec['source_manifest_sha256']!=entry.sha(HERE/'SOURCE_MANIFEST.json'):raise ValueError('Exact reviewed control source required')
    ic.verify(HERE,spec['source_manifest_sha256'])
    for row in json.loads((HERE/'SOURCE_BINDINGS.json').read_text())['files']:
        file=(PHASE/row['path']).resolve(strict=True)
        if not file.is_relative_to(PHASE) or entry.sha(file)!=row['sha256']:raise ValueError('Reused source changed')
    data=json.loads((HERE/'DATA_AND_RUNTIME.json').read_text())
    for key in ('train_bundle','split_custody','runtime','frozen_providers','edge_shape','split_identity'):
        if spec[key]!=data[key]:raise ValueError('Exact native task/runtime required')
    if spec.get('VALID_access') is not (args.mode!='qualification'):raise ValueError('Declared VALID scope required')
    if args.mode!='qualification':
        for key in ('valid_bundle','validation_custody'):
            if spec[key]!=data[key]:raise ValueError('Exact VALID projection required')
        spec['_validation_custody']=json.loads(base.bind(spec['validation_custody']).read_text())
    elif any(k in spec for k in ('valid_bundle','validation_custody')):raise ValueError('Qualification is TRAIN-only')
    spec['_train_custody']=json.loads(base.bind(spec['split_custody']).read_text())
    review=json.loads(base.bind(spec['root_review']).read_text());base.bind(spec['resource_readiness_evidence'])
    if review.get('approved') is not True or review.get('source_manifest_sha256')!=spec['source_manifest_sha256']:raise ValueError('Exact current root review required')
    if spec['limits']!=limits(base,args.mode):raise ValueError('Existing finite stage bounds required')
    owner=json.loads(base.bind(spec['external_owner_release']).read_text())
    if owner.get('enabled') is not True or owner.get('record_id')!=spec['record_id'] or owner.get('limits')!=spec['limits'] or owner.get('automatic_retry') is not False or any(owner.get(k) is not True for k in ('separate_process_group','direct_wait_required','resource_caps_enforced','output_and_log_caps_enforced')):
        raise ValueError('Exact existing finite owner required')
    if base.bind(owner['owner_source'])!=HERE/'queue.py':raise ValueError('Exact reused-owner wrapper required')
    owner_review=json.loads(base.bind(owner['owner_review']).read_text())
    if owner_review.get('approved') is not True or owner_review.get('owner_sha256')!=entry.sha(HERE/'queue.py'):raise ValueError('Exact owner review required')
    python=Path(spec['runtime']['python']['path'])
    if entry.sha(python)!=spec['runtime']['python']['sha256'] or python.resolve()!=Path(sys.executable).resolve() or os.environ.get('PYTHONPATH','')!=spec['runtime']['PYTHONPATH']:
        raise ValueError('Exact original native Python/PYTHONPATH required')
    if args.mode!='qualification':
        q=json.loads(base.bind(spec['qualification_complete']).read_text());base.terminal(spec['qualification_terminal'],q['release_sha256'])
        if q.get('schema')!='PubMed-own4-M-normalization-qualification-complete-v1' or q.get('complete') is not True or q.get('source_manifest_sha256')!=spec['source_manifest_sha256'] or q.get('genuine_full_TRAIN_updates')!=1 or q.get('changed_private_groups')!=46 or q.get('unchanged_native_groups')!=52 or q.get('VALID_access') is not False or q.get('TEST_access') is not False:
            raise ValueError('Actual one full-TRAIN group-inventory qualification required')
    if args.mode in ('qualification','science') and spec['seed'] not in ((9101,) if args.mode=='qualification' else SEEDS):raise ValueError('Fixed seed required')
    if args.mode=='science' and spec['record_id']!=f"seed{spec['seed']}__{CONDITION}":raise ValueError('Exact seed/record pairing required')
    output=Path(spec['output']).resolve()
    if not output.is_relative_to(HERE) or output.exists():raise ValueError('Fresh confined output required')
    spec['_release_sha256']=args.release_sha256
    return base,ic,spec,output,data


def closed_science(entry,base,spec):
    plan=json.loads(base.bind(spec['science_owner_plan']).read_text());expected=record_ids('science')
    family=json.loads((HERE/'science/FAMILY_COMPLETE.json').read_text())
    if plan.get('enabled') is not True or plan.get('automatic_retry') is not False or plan.get('owner_sha256')!=entry.sha(HERE/'queue.py') or plan.get('purpose')!='science' or plan.get('source_manifest_sha256')!=spec['source_manifest_sha256'] or [r['record_id'] for r in plan['records']]!=expected or family.get('complete') is not True or family.get('records')!=expected or family.get('all_records_directly_waited') is not True or family.get('source_manifest_sha256')!=spec['source_manifest_sha256']:
        raise ValueError('Whole three-control closure required')
    values={}
    for row in plan['records']:
        rid=row['record_id'];folder=HERE/'science/cells'/rid;path=folder/'COMPLETE.json';raw=HERE/'owners'/('science__'+rid)/'RAW_OWNER_TERMINAL.json'
        base.terminal(dict(path=str(raw),sha256=entry.sha(raw)),row['release_sha256'])
        custody=json.loads((raw.parent/'TERMINAL_CUSTODY.json').read_text());v=json.loads(path.read_text())
        if custody.get('actual') is not True or custody.get('complete') is not True or custody.get('record_id')!=rid or custody.get('complete_sha256')!=entry.sha(path) or custody.get('source_manifest_sha256')!=spec['source_manifest_sha256'] or custody.get('release_sha256')!=row['release_sha256'] or custody.get('raw_owner_terminal')!=dict(path=str(raw),sha256=entry.sha(raw)) or v.get('complete') is not True or v.get('source_manifest_sha256')!=spec['source_manifest_sha256'] or v.get('condition')!=CONDITION or v.get('release_sha256')!=row['release_sha256'] or v.get('max_epochs')!=2000 or v.get('patience')!=250 or v.get('selector')!=spec['selector'] or v.get('providers')!=spec['frozen_providers'] or v.get('TEST_access') is not False:
            raise ValueError('Exact control result/owner custody required')
        values[v['seed']]=(v,folder)
    if set(values)!=set(SEEDS):raise ValueError('Every fixed control seed required')
    return values


def compare(entry,base,ic,spec,output,data,started):
    diag=entry.load('_normalization_existing_stored_reader',DIAG/'collect.py')
    norm=closed_science(entry,base,spec)
    full=diag.closed18(entry,base,spec['full18_owner_plan'],entry.sha(FULL/'SOURCE_MANIFEST.json'),data)
    b,idata,roster,reuse,im=ic.frozen();ib=json.loads((DIAG/'I4_BINDING.json').read_text())
    if im!=ib['source_manifest_sha256']:raise ValueError('Exact completed I4 source required')
    for stage in ('admission','qualification','science','assembly','comparison'):ic.closed_stage(stage,roster,im)
    receipt=json.loads(ic.stage_result('comparison','factorized_I4_complete_comparison').read_text())
    if receipt.get('comparison')!=ib['comparison'] or receipt.get('complete') is not True:raise ValueError('Exact I4 comparison closure required')
    closed=json.loads(base.bind(b['closed_M1_comparison']).read_text())
    if closed.get('complete_six') is not True:raise ValueError('Complete original M1 reference required')
    for key in ('valid_bundle','validation_custody','runtime','frozen_providers'):
        if idata[key]!=data[key]:raise ValueError('Identical role/runtime required')
    import numpy as np
    import torch
    providers=dict(numpy=str(np.__version__),torch=str(torch.__version__))
    for name in ('scipy','torch-geometric','torch-scatter','torch-sparse'):providers[name]=importlib.metadata.version(name)
    if providers!=data['frozen_providers']:raise ValueError('Exact admitted native providers required')
    torch.set_num_threads(2)
    with base.reference_surface() as (engine,metrics,_unused_factory):
        from source import fingerprint
        with np.load(base.bind(data['valid_bundle']),allow_pickle=False) as archive:
            if set(archive.files)!={'valid_ids','valid_y'}:raise ValueError('VALID projection only')
            arrays={k:archive[k].copy() for k in archive.files}
        if {k:fingerprint(v) for k,v in arrays.items()}!=spec['_validation_custody']['array_fingerprints']:raise ValueError('Exact VALID identities required')
        ids,truth=torch.from_numpy(arrays['valid_ids']),torch.from_numpy(arrays['valid_y'])
        paired=entry.load('_normalization_original_pair_metrics',FULL/'error_changes.py');rows=[]
        for seed in SEEDS:
            v,folder=norm[seed];p=v['complete_saved_member_logits'];candidate=diag.selected(np,torch,metrics,base,dict(path=str(folder/p['path']),sha256=p['sha256'],bytes=p['bytes']),v['selected_readouts']['VALID'],ids,truth)
            old,oldfolder=full[f'seed{seed}__shared4_own'];p=old['complete_saved_member_logits']
            own=diag.selected(np,torch,metrics,base,dict(path=str(oldfolder/p['path']),sha256=p['sha256'],bytes=p['bytes']),old['selected_readouts']['VALID'],ids,truth)
            committee=diag.selected(np,torch,metrics,base,ib['banks'][str(seed)],ib['selected_readouts'][str(seed)]['VALID'],ids,truth)
            m1=committee[:1];m1prob=m1.softmax(-1).mean(0);m1read=metrics.classification(m1,m1prob,ids,truth)
            expected=closed['all_selected_readouts'][f'seed{seed}__factor1_native']['VALID']
            metrics.verify_counts(m1read,expected);metrics.verify_floats(m1read,expected)
            anchor=next(r for r in reuse['reused_records'] if r['seed_block']==seed)
            if metrics.signatures(m1,m1prob,ids,truth)!=anchor['selected_prediction_signatures']:raise ValueError('Exact admitted original M1 body0 identity required')
            current=v['selected_readouts']['VALID']['classification']
            for name,bank in (('fresh_full18_own',own),('factorM1',m1),('factorized_I4',committee)):
                reference=metrics.classification(bank,bank.softmax(-1).mean(0),ids,truth)
                rows.append(dict(seed=seed,reference=name,classification=current,reference_classification=reference,
                    delta=dict(pooled_accuracy_pp=100*(current['pooled']['accuracy']-reference['pooled']['accuracy']),NLL=current['pooled']['NLL']-reference['pooled']['NLL'],
                        mean_member_accuracy_pp=100*(current['mean_member_accuracy']-reference['mean_member_accuracy']),worst_member_accuracy_pp=100*(current['worst_member_accuracy']-reference['worst_member_accuracy'])),
                    paired_errors=paired.compare(torch,metrics,bank,candidate,ids,truth)))
            engine.check_bounds(spec,output,started)
        means={name:{key:sum(r['delta'][key] for r in rows if r['reference']==name)/3 for key in rows[0]['delta']} for name in ('fresh_full18_own','factorM1','factorized_I4')}
        base.write(output/'COMPARISON.json',dict(complete=True,rows=rows,mean_contrasts=means,all3_controls_and_full18_and_I4_closed_before_logits=True,
            original_M1_is_exact_admitted_I4_body0=True,models=0,new_model_forwards=0,TEST_access=False,no_causal_regularization_or_novelty_claim=True))
    return dict(complete=True,comparison=dict(path=str(output/'COMPARISON.json'),sha256=entry.sha(output/'COMPARISON.json')),new_model_forwards=0,mean_contrasts=means)


def execute(mode,entry,base,ic,spec,output,data,started):
    adapter=entry.load('_normalized_private_native_groups',HERE/'adapter.py')
    if mode=='comparison':output.mkdir(parents=True,exist_ok=False);return compare(entry,base,ic,spec,output,data,started)
    with base.reference_surface() as (engine,_metrics,_unused_factory):
        inventory={}
        def factory(method,condition,seed,tensor):
            if condition!=CONDITION:raise ValueError('Only fixed normalized own4')
            session=adapter.fresh(method,seed,tensor,numerical_admitted=True);inventory.update(adapter.verify(session));return session
        if mode=='science':
            fit=entry.load('_normalization_exact_existing_M4_fit',FULL/'m4_fit.py')
            result=fit.fit(spec,output,started,factory,adapter.work,engine.serving_benchmark)
            result.update(schema='PubMed-own4-M-normalization-science-complete-v1',normalization_inventory=inventory,
                no_new_architecture=True,no_novelty_claim=True,whole_three_controls_and_fresh_full18_anchors_required=True)
            base.write(output/'COMPLETE.json',result);return result
        output.mkdir(parents=True,exist_ok=False);timings={};tick=time.monotonic()
        np,torch,providers,tensor,_roles=engine.runtime_and_data(spec,valid=False)
        timings['runtime_and_full_TRAIN_seconds']=time.monotonic()-tick
        method=sys.modules['source'].load_v3(numerical=True)['method']
        session,clock=engine.timed(torch,lambda:factory(method,CONDITION,9101,tensor));timings['native_construction_and_groups']=clock
        training,clock=engine.timed(torch,lambda:session.train_step(audit=False));timings['one_whole_TRAIN_native_update']=clock
        adapter.verify(session)
        if session.counters!=adapter.work(session,1,evaluations=0):raise ValueError('One complete native full-TRAIN update required')
        engine.check_bounds(spec,output,started)
        result=dict(schema='PubMed-own4-M-normalization-qualification-complete-v1',complete=True,record_id=QUAL,
            source_manifest_sha256=spec['source_manifest_sha256'],release_sha256=spec['_release_sha256'],providers=providers,
            genuine_full_TRAIN_updates=1,TRAIN_count=11829,graph_shape=[19717,500],edge_shape=[2,88648],
            changed_private_groups=46,unchanged_native_groups=52,normalization_inventory=inventory,post_update_group_inventory_verified=True,
            native_mean_own_train_step_unchanged=True,counters=dict(session.counters),training=training,timings=timings,
            peak_CUDA_bytes=dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved()),
            VALID_access=False,TEST_access=False,quality_evidence=False,new_learning_method=False,automatic_retry=False)
        base.write(output/'COMPLETE.json',result);return result


def main():
    started=time.monotonic();p=argparse.ArgumentParser();p.add_argument('--mode',choices=('qualification','science','comparison'),required=True)
    p.add_argument('--release',type=Path,required=True);p.add_argument('--release-sha256',required=True)
    args=p.parse_args();entry=surface();base,ic,spec,output,data=admit(args,entry)
    try:
        result=execute(args.mode,entry,base,ic,spec,output,data,started)
        result.update(purpose=args.mode,record_id=spec['record_id'],source_manifest_sha256=spec['source_manifest_sha256'],release_sha256=spec['_release_sha256'],
            inclusive_seconds=time.monotonic()-started,CPU_user_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime,
            CPU_system_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_stime,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            TEST_access=False,automatic_retry=False,no_novelty_claim=True)
        base.write(output/'COMPLETE.json',result)
    except BaseException as error:
        if output.exists():base.write(output/'FAILURE.json',dict(complete=False,error_type=type(error).__name__,error=str(error),inclusive_seconds=time.monotonic()-started,TEST_access=False,automatic_retry=False))
        raise
    print(json.dumps(dict(complete=True,record_id=spec['record_id'],TEST_access=False)))


if __name__=='__main__':main()
