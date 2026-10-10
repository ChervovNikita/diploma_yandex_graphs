"""Exact paired ranking/coverage changes after complete-family closure only."""
import argparse
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time


def compare(torch,metrics,baseline,candidate,ids,labels):
    """Arguments are two complete full-node factual logit banks, no TEST role."""
    bp,bm=metrics.predictions(baseline,baseline.softmax(-1).mean(0),ids)
    cp,cm=metrics.predictions(candidate,candidate.softmax(-1).mean(0),ids)
    bg=bm.eq(labels[None,:]);cg=cm.eq(labels[None,:])
    bpool=bp.eq(labels);cpool=cp.eq(labels);bu=bg.any(0);cu=cg.any(0)
    def common(bank):
        values=bank[:,ids];truth=values.gather(2,labels[None,:,None].expand(len(bank),-1,1))
        rivals=(values>truth).all(0);rivals.scatter_(1,labels[:,None],False)
        return rivals.any(1)
    br,cr=common(baseline),common(candidate)
    masks=dict(pool_repairs=~bpool&cpool,pool_harms=bpool&~cpool,
               acquired_correct_member=~bu&cu,acquired_correct_member_served=~bu&cu&cpool,
               acquired_correct_member_lost=~bu&cu&~cpool,correct_alternative_lost=bu&~cu,
               cleared_all_member_wrong=~bu&cu,introduced_all_member_wrong=bu&~cu,
               cleared_strict_common_false_rival=br&~cr,introduced_strict_common_false_rival=~br&cr,
               pool_lost_alternative_cleared=bu&~bpool&~(cu&~cpool),
               pool_lost_alternative_introduced=cu&~cpool&~(bu&~bpool))
    count=lambda value:int(value.sum().item())
    return dict(count=len(labels),counts={name:count(mask) for name,mask in masks.items()},
                exact_node_ids={name:ids[mask].detach().cpu().tolist() for name,mask in masks.items()},
                per_class=[dict(class_id=c,counts={name:count(mask&labels.eq(c)) for name,mask in masks.items()}) for c in range(3)],
                baseline_union_correct=count(bu),candidate_union_correct=count(cu),
                baseline_pool_lost_correct=count(bu&~bpool),candidate_pool_lost_correct=count(cu&~cpool),
                candidate_within_bank=metrics.repair_diagnostics(candidate,candidate.softmax(-1).mean(0),ids,labels),
                baseline_within_bank=metrics.repair_diagnostics(baseline,baseline.softmax(-1).mean(0),ids,labels),
                labels_or_TEST_exposed=False,affects_selection=False)


def main():
    # All source/closure/owner metadata is checked before numerical imports or logit reads.
    loader=importlib.util.spec_from_file_location('_private_hop_comparison_entry',Path(__file__).resolve().parent/'run.py')
    entry=importlib.util.module_from_spec(loader);sys.modules[loader.name]=entry;loader.loader.exec_module(entry)
    entry.qualification_surface().route(owner=True)
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['comparison'],required=True)
    parser.add_argument('--release',type=Path,required=True);parser.add_argument('--release-sha256',required=True)
    args=parser.parse_args();path=args.release.resolve(strict=True)
    if not path.is_relative_to(entry.HERE) or entry.sha(path)!=args.release_sha256:
        raise ValueError('Exact new root comparison release required')
    spec=json.loads(path.read_text())
    if spec.get('schema')!='private-hop-fullfit-comparison-release-v1' or spec.get('purpose')!='comparison':
        raise ValueError('Exact fixed whole-family comparison required')
    for key in ('enabled','root_authorized','source_review_approved','all_eighteen_closed','finite_owner_verified','ordinary_runtime_confirmed','fresh_resource_readiness_confirmed','VALID_access'):
        if spec.get(key) is not True:raise ValueError('Disabled comparison admission: '+key)
    for key in ('TEST_access','automatic_retry','HPO','affects_selection','partial_family_comparison_allowed'):
        if spec.get(key) is not False:raise ValueError('Closed comparison scope: '+key)
    if any(key in spec for key in ('test_bundle','test_ids','test_y','full_y','split_ids_bundle')):
        raise ValueError('No TEST/full-label surface')
    if spec['source_manifest_sha256']!=entry.sha(entry.HERE/'SOURCE_MANIFEST.json') or spec['roster_sha256']!=entry.sha(entry.HERE/'ROSTER.json'):
        raise ValueError('Exact current full-study source/roster required')
    for row in json.loads((entry.HERE/'SOURCE_MANIFEST.json').read_text())['files']:
        source=(entry.HERE/row['path']).resolve(strict=True)
        if not source.is_relative_to(entry.HERE) or entry.sha(source)!=row['sha256']:
            raise ValueError('Full-study source changed')
    for row in json.loads((entry.HERE/'SOURCE_BINDINGS.json').read_text())['files']:
        source=(entry.PHASE/row['path']).resolve(strict=True)
        if not source.is_relative_to(entry.PHASE) or entry.sha(source)!=row['sha256']:
            raise ValueError('Reused source changed')
    base=entry.load('_private_hop_closed_comparison_surface',entry.PHASE/'pubmed_factor1_controls_source_20261010_v1/run.py')
    review=json.loads(base.bind(spec['root_review']).read_text())
    if review.get('approved') is not True or review.get('source_manifest_sha256')!=spec['source_manifest_sha256']:
        raise ValueError('Exact root source review required')
    base.bind(spec['resource_readiness_evidence'])
    expected=[f'seed{s}__{c}' for s in entry.SEEDS for c in entry.CONDITIONS]
    family=json.loads(base.bind(spec['family_complete']).read_text())
    if (family.get('complete') is not True or family.get('records')!=expected or
            family.get('source_manifest_sha256')!=spec['source_manifest_sha256'] or
            family.get('all_records_directly_waited') is not True or
            family.get('partial_family_comparison_allowed') is not False):
        raise ValueError('Actual complete fixed18 family required')
    if [row['record_id'] for row in spec['records']]!=expected:
        raise ValueError('All18 completion and owner descriptors required')
    data=json.loads((entry.HERE/'DATA_AND_RUNTIME.json').read_text())
    for key in ('train_bundle','split_custody','valid_bundle','validation_custody','runtime','frozen_providers'):
        if spec[key]!=data[key]:raise ValueError('Exact current native roles/runtime required')
    python=Path(spec['runtime']['python']['path'])
    import os
    if (not python.is_relative_to(entry.PHASE) or entry.sha(python)!=spec['runtime']['python']['sha256'] or
            python.resolve()!=Path(sys.executable).resolve() or os.environ.get('PYTHONPATH','')!=spec['runtime']['PYTHONPATH']):
        raise ValueError('Exact current native Python/PYTHONPATH required')
    completed={};owner_costs={}
    for row in spec['records']:
        complete_path=base.bind(row['complete']);result=json.loads(complete_path.read_text())
        if not complete_path.is_relative_to(entry.HERE/'science'/'cells'/row['record_id']):
            raise ValueError('Exact owned full-fit cell required')
        condition=result.get('condition');seed=result.get('seed')
        selector=entry.M1_SELECTOR if condition=='factor1_allview' else entry.M4_SELECTOR
        if (result.get('schema')!='private-hop-fullfit-complete-v1' or result.get('complete') is not True or
                result.get('record_id')!=row['record_id'] or condition not in entry.CONDITIONS or seed not in entry.SEEDS or
                row['record_id']!=f'seed{seed}__{condition}' or result.get('source_manifest_sha256')!=spec['source_manifest_sha256'] or
                result.get('providers')!=spec['frozen_providers'] or result.get('max_epochs')!=2000 or result.get('patience')!=250 or
                result.get('selector')!=selector or result.get('TEST_access') is not False or result.get('TEST_scored') is not False or
                result.get('train_bundle_sha256')!=data['train_bundle']['sha256'] or
                result.get('valid_bundle_sha256')!=data['valid_bundle']['sha256'] or
                not 1<=result.get('selected_epoch',0)<=result.get('epochs_executed',0)<=2000):
            raise ValueError('Actual unchanged full-fit completion required')
        base.terminal(row['raw_terminal'],result['release_sha256'])
        terminal=json.loads(base.bind(row['terminal']).read_text())
        if (terminal.get('actual') is not True or terminal.get('complete') is not True or
                terminal.get('purpose')!='science' or terminal.get('record_id')!=row['record_id'] or
                terminal.get('owner_id')!='science__'+row['record_id'] or
                terminal.get('complete_sha256')!=row['complete']['sha256'] or
                terminal.get('raw_owner_terminal')!=row['raw_terminal'] or
                terminal.get('release_sha256')!=result['release_sha256'] or
                terminal.get('source_manifest_sha256')!=spec['source_manifest_sha256'] or
                terminal.get('owned_process_absence_verified') is not True or terminal.get('owned_CUDA_absence_verified') is not True):
            raise ValueError('Full-fit result/release/source/owner custody differ')
        raw=json.loads(base.bind(row['raw_terminal']).read_text());owner_costs[row['record_id']]=raw
        completed[row['record_id']]=(result,complete_path.parent)
    if spec['limits']!=dict(base.LIMITS,external_active_seconds=600):
        raise ValueError('Existing finite comparison caps required')
    owner=json.loads(base.bind(spec['external_owner_release']).read_text())
    if (owner.get('enabled') is not True or owner.get('record_id')!=spec['record_id'] or owner.get('limits')!=spec['limits'] or owner.get('automatic_retry') is not False):
        raise ValueError('New finite comparison owner required')
    for key in ('separate_process_group','direct_wait_required','resource_caps_enforced','output_and_log_caps_enforced'):
        if owner.get(key) is not True:raise ValueError('Comparison owner fact missing: '+key)
    if base.bind(owner['owner_source'])!=entry.PHASE/'pubmed_factor1_controls_source_20261010_v1/queue.py':
        raise ValueError('Exact existing directly waited comparison owner required')
    owner_review=json.loads(base.bind(owner['owner_review']).read_text())
    if owner_review.get('approved') is not True or owner_review.get('owner_sha256')!=entry.sha(entry.PHASE/'pubmed_factor1_controls_source_20261010_v1/queue.py'):
        raise ValueError('Exact existing comparison owner review required')
    output=Path(spec['output']).resolve()
    if not output.is_relative_to(entry.HERE) or output.exists():raise ValueError('Fresh owned comparison output required')
    output.mkdir(parents=True,exist_ok=False);started=time.monotonic();timings={}
    try:
        spec['_train_custody']=json.loads(base.bind(spec['split_custody']).read_text())
        spec['_validation_custody']=json.loads(base.bind(spec['validation_custody']).read_text())
        with base.reference_surface() as (engine,metrics,_unused_factory):
            tick=time.monotonic();np,torch,providers,_unused_tensor,roles=engine.runtime_and_data(spec,valid=True)
            timings['runtime_and_exact_roles_seconds']=time.monotonic()-tick
            banks={};tick=time.monotonic()
            for record_id,(result,folder) in completed.items():
                payload=result['complete_saved_member_logits'];file=(folder/payload['path']).resolve(strict=True)
                shape=[1 if result['condition']=='factor1_allview' else 4,19717,3]
                if (not file.is_relative_to(folder) or entry.sha(file)!=payload['sha256'] or file.stat().st_size!=payload['bytes'] or
                        payload['keys']!=['factual_member_logits'] or payload['factual_shape']!=shape or
                        payload.get('contains_labels_or_role_ids') is not False):
                    raise ValueError('Complete selected full-node logit payload changed')
                with np.load(file,allow_pickle=False) as archive:
                    if set(archive.files)!={'factual_member_logits'}:raise ValueError('Factual logits only')
                    array=archive['factual_member_logits'].copy()
                if array.dtype!=np.float32 or list(array.shape)!=shape or not np.isfinite(array).all():
                    raise ValueError('Full finite float32 selected bank required')
                bank=torch.from_numpy(array).to('cuda:0');banks[record_id]=bank
                probability=bank.softmax(-1).mean(0)
                for name,role in roles.items():
                    actual=metrics.classification(bank,probability,*role)
                    recorded=result['selected_readouts'][name]
                    metrics.verify_counts(actual,recorded['classification']);metrics.verify_floats(actual,recorded['classification'])
                    if metrics.signatures(bank,probability,*role)!=recorded['prediction_signatures']:
                        raise ValueError('Stored complete selected prediction identity changed')
            torch.cuda.synchronize();timings['selected_payload_and_identity_seconds']=time.monotonic()-tick
            tick=time.monotonic();rows=[]
            for seed in entry.SEEDS:
                for reference in ('shared4_own','factor1_allview'):
                    for condition in entry.CONDITIONS:
                        if condition==reference:continue
                        left=banks[f'seed{seed}__{reference}'];right=banks[f'seed{seed}__{condition}']
                        for role,values in roles.items():
                            paired=compare(torch,metrics,left,right,*values)
                            a=completed[f'seed{seed}__{reference}'][0]['selected_readouts'][role]['classification']
                            b=completed[f'seed{seed}__{condition}'][0]['selected_readouts'][role]['classification']
                            delta={key:100*(b[key]-a[key]) for key in ('mean_member_accuracy','worst_member_accuracy','macro_accuracy')}
                            delta['pooled_accuracy_pp']=100*(b['pooled']['accuracy']-a['pooled']['accuracy'])
                            delta['pooled_NLL']=b['pooled']['NLL']-a['pooled']['NLL']
                            if paired['counts']['pool_repairs']-paired['counts']['pool_harms']!=b['pooled']['correct']-a['pooled']['correct']:
                                raise ValueError('Exact paired pool repairs/harms do not reconcile')
                            rows.append(dict(seed=seed,reference=reference,condition=condition,role=role,delta=delta,**paired))
            timings['exact_paired_error_changes_seconds']=time.monotonic()-tick
            costs={record_id:dict(worker_inclusive_seconds=result['inclusive_seconds'],worker_timings=result['timings'],
                         counters=result['counters'],epochs_executed=result['epochs_executed'],selected_epoch=result['selected_epoch'],
                         peak_RSS_bytes=result['peak_RSS_bytes'],peak_CUDA_bytes=result['peak_CUDA_bytes'],
                         serving_benchmark=result['serving_benchmark'],owner=owner_costs[record_id])
                   for record_id,(result,_folder) in completed.items()}
            engine.check_bounds(spec,output,started)
            base.write(output/'EXACT_PAIRED_ERROR_CHANGES.json',dict(schema='private-hop-exact-paired-errors-v1',rows=rows,TEST_access=False,affects_selection=False))
            base.write(output/'ACTUAL_COSTS.json',dict(schema='private-hop-actual-full18-costs-v1',records=costs,
                       total_worker_seconds=sum(row['worker_inclusive_seconds'] for row in costs.values()),
                       total_owner_inclusive_seconds=sum(row['owner']['inclusive_seconds'] for row in costs.values()),
                       totals_are_inclusive_not_isolated_speed_evidence=True,failures_omitted=False))
            result=dict(schema='private-hop-fullfit-comparison-complete-v1',complete=True,record_id=spec['record_id'],
                source_manifest_sha256=spec['source_manifest_sha256'],release_sha256=args.release_sha256,
                records=expected,paired_rows=len(rows),providers=providers,all_eighteen_closed_before_logits=True,
                independent_factorized_I4_reference_pending_separate_root_admission=True,
                source_ready_or_qualification_is_not_quality_evidence=True,TEST_access=False,affects_selection=False,
                inclusive_seconds=time.monotonic()-started,timings=timings,
                peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                peak_CUDA_bytes=dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved()),
                automatic_retry=False,partial_family_comparison_allowed=False)
            base.write(output/'COMPLETE.json',result)
    except BaseException as error:
        base.write(output/'FAILURE.json',dict(complete=False,record_id=spec['record_id'],error_type=type(error).__name__,error=str(error),
                   inclusive_seconds=time.monotonic()-started,timings=timings,TEST_access=False,automatic_retry=False));raise
    print(json.dumps(dict(complete=True,record_id=spec['record_id'],paired_rows=result['paired_rows'],TEST_access=False)))


if __name__=='__main__':
    main()
