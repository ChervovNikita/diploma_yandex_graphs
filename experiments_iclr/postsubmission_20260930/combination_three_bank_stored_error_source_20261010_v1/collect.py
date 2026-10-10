"""Disabled CPU stored-logit diagnostic; no models, checkpoints, forwards or updates."""
import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time

HERE=Path(__file__).resolve().parent
PHASE=HERE.parent
FULL=PHASE/'private_hop_credit_pubmed_fullfit_source_20261010_v1'
I4=PHASE/'pubmed_factorized_I4_reference_source_20261010_v1'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def closed18(entry,base,row,manifest,data):
    """Reuse actual owner terminal checks; inspect no prediction payload here."""
    plan=json.loads(base.bind(row).read_text());expected=[f'seed{s}__{c}' for s in entry.SEEDS for c in entry.CONDITIONS]
    if (plan.get('enabled') is not True or plan.get('purpose')!='science' or plan.get('automatic_retry') is not False or
            plan.get('source_manifest_sha256')!=manifest or plan.get('owner_sha256')!=entry.sha(FULL/'queue.py') or
            [r['record_id'] for r in plan['records']]!=expected):raise ValueError('Exact full18 owner plan required')
    family=json.loads((FULL/'science/FAMILY_COMPLETE.json').read_text())
    if (family.get('complete') is not True or family.get('records')!=expected or
            family.get('all_records_directly_waited') is not True or family.get('source_manifest_sha256')!=manifest or
            family.get('partial_family_comparison_allowed') is not False):raise ValueError('Actual whole18 closure required')
    completed={}
    for r in plan['records']:
        release=json.loads(base.bind(dict(path=str(PHASE/r['release']),sha256=r['release_sha256'])).read_text())
        rid=r['record_id'];folder=FULL/'science/cells'/rid;complete=folder/'COMPLETE.json'
        raw=FULL/'owners'/('science__'+rid)/'RAW_OWNER_TERMINAL.json'
        if (r['owner_id']!='science__'+rid or r['entrypoint']!=dict(path=FULL.name+'/run.py',sha256=entry.sha(FULL/'run.py')) or
                r['entry_args']!=['--mode','science'] or r['limits']!=base.LIMITS or release['output']!=str(folder)):
            raise ValueError('Exact fixed full18 worker/owner identity required')
        base.terminal(dict(path=str(raw),sha256=entry.sha(raw)),r['release_sha256'])
        waited=json.loads(raw.read_text())
        expected_argv=[release['runtime']['python']['path'],'-B','-P',str(PHASE/r['entrypoint']['path']),
                       '--mode','science','--release',str(PHASE/r['release']),'--release-sha256',r['release_sha256']]
        if waited['argv']!=expected_argv:raise ValueError('Exact full18 child command required')
        custody=json.loads((raw.parent/'TERMINAL_CUSTODY.json').read_text())
        if (custody.get('actual') is not True or custody.get('complete') is not True or custody.get('record_id')!=rid or
                custody.get('purpose')!='science' or custody.get('owner_id')!='science__'+rid or
                custody.get('source_manifest_sha256')!=manifest or custody.get('release_sha256')!=r['release_sha256'] or
                custody.get('raw_owner_terminal')!=dict(path=str(raw),sha256=entry.sha(raw)) or
                custody.get('complete_sha256')!=entry.sha(complete)):raise ValueError('Exact full18 result/wait custody required')
        value=json.loads(complete.read_text());condition=value['condition'];seed=value['seed']
        selector=entry.M1_SELECTOR if condition=='factor1_allview' else entry.M4_SELECTOR
        if (value.get('complete') is not True or value.get('schema')!='private-hop-fullfit-complete-v1' or
                value.get('record_id')!=rid or rid!=f'seed{seed}__{condition}' or value.get('source_manifest_sha256')!=manifest or
                value.get('release_sha256')!=r['release_sha256'] or value.get('providers')!=data['frozen_providers'] or
                value.get('max_epochs')!=2000 or value.get('patience')!=250 or value.get('selector')!=selector or
                value.get('TEST_access') is not False or value.get('TEST_scored') is not False or
                value.get('train_bundle_sha256')!=data['train_bundle']['sha256'] or value.get('valid_bundle_sha256')!=data['valid_bundle']['sha256']):
            raise ValueError('Unchanged complete full18 endpoint required')
        completed[rid]=(value,folder)
    return completed


def selected(np,torch,metrics,base,row,expected,ids,truth):
    with np.load(base.bind(row),allow_pickle=False) as archive:
        if set(archive.files)!={'factual_member_logits'}:raise ValueError('Factual logit bank only')
        array=archive['factual_member_logits'].copy()
    if array.dtype!=np.float32 or array.shape!=(4,19717,3) or not np.isfinite(array).all():
        raise ValueError('Complete finite float32 four-member bank required')
    logits=torch.from_numpy(array);probability=logits.softmax(-1).mean(0)
    actual=metrics.classification(logits,probability,ids,truth)
    metrics.verify_counts(actual,expected['classification']);metrics.verify_floats(actual,expected['classification'])
    if metrics.signatures(logits,probability,ids,truth)!=expected['prediction_signatures']:
        raise ValueError('CPU stored-output selected predictions/correct masks changed')
    return logits


def summarize(report):
    """Render only completed diagnostic scalars; never read predictions here."""
    lines=['# Fixed three-bank complementary repairs','',
           '| Seed | A-only repairs | B-only repairs | Shared repairs | A harms | B harms |',
           '|---|---:|---:|---:|---:|---:|']
    for row in report['rows']:
        c=row['counts'];lines.append(f"|{row['seed']}|{c['A_only']}|{c['B_only']}|{c['shared']}|{c['harms_A']}|{c['harms_B']}|")
    lines+=['','| Seed | Bank | Pool accuracy % | NLL | Mean member % | Worst member % |',
            '|---|---|---:|---:|---:|---:|']
    for row in report['rows']:
        for bank in ('baseline','A','B'):
            m=row['selected_readouts'][bank]['classification']
            lines.append(f"|{row['seed']}|{bank}|{100*m['pooled']['accuracy']:.6f}|{m['pooled']['NLL']:.6f}|{100*m['mean_member_accuracy']:.6f}|{100*m['worst_member_accuracy']:.6f}|")
    lines+=['',f"Diagnostic eligibility: **{'PASS' if report['gate']['passed'] else 'FAIL'}**.",
            'This is an inactive interaction rationale, not an A+B fit or quality gate.',
            'Exact repairs, acquisition/serving partitions, class counts and cross-seed support are in JOINT_ERROR_DIAGNOSTIC.json.',
            '', 'No models, checkpoint loads, forwards, gradients, training or oracle training targets were used. TEST stayed closed.']
    return '\n'.join(lines)+'\n'


def main():
    started=time.monotonic();entry=load('_three_bank_fullfit_entry',FULL/'run.py')
    entry.qualification_surface().route(owner=True)
    parser=argparse.ArgumentParser();parser.add_argument('--release',type=Path,required=True);parser.add_argument('--release-sha256',required=True)
    args=parser.parse_args();path=args.release.resolve(strict=True)
    if not path.is_relative_to(HERE) or entry.sha(path)!=args.release_sha256:raise ValueError('Exact new root release required')
    spec=json.loads(path.read_text())
    if spec.get('schema')!='three-bank-stored-error-release-v1' or spec.get('record_id')!='fixed_three_bank_VALID_complementarity':
        raise ValueError('Fixed diagnostic identity required')
    for key in ('enabled','root_authorized','source_review_approved','all18_closed','exact_I4_closed','external_finite_bound_confirmed','VALID_access'):
        if spec.get(key) is not True:raise ValueError('Disabled diagnostic: '+key)
    for key in ('TEST_access','automatic_retry','HPO','arm_selection','coefficient_selection','oracle_training_target'):
        if spec.get(key) is not False:raise ValueError('Closed diagnostic scope: '+key)
    if any(k in spec for k in ('test_bundle','test_ids','test_y','full_y','train_bundle')):raise ValueError('VALID stored-output scope only')
    ic=load('_three_bank_I4_common',I4/'common.py');ic.route()
    if spec['source_manifest_sha256']!=entry.sha(HERE/'SOURCE_MANIFEST.json'):raise ValueError('Exact collector source required')
    ic.verify(HERE,spec['source_manifest_sha256'])
    bindings=json.loads((HERE/'SOURCE_BINDINGS.json').read_text())
    for row in bindings['files']:
        file=(PHASE/row['path']).resolve(strict=True)
        if not file.is_relative_to(PHASE) or entry.sha(file)!=row['sha256']:raise ValueError('Existing readout source changed')
    if entry.sha(HERE/'FIXED_SPEC.json')!=bindings['fixed_spec_sha256']:raise ValueError('Original fixed diagnostic specification changed')
    fixed=json.loads((HERE/'FIXED_SPEC.json').read_text());ib=json.loads((HERE/'I4_BINDING.json').read_text())
    ic.verify(FULL,bindings['full18_source_manifest_sha256'])
    if spec['full18_source_manifest_sha256']!=bindings['full18_source_manifest_sha256']:raise ValueError('Exact full18 source required')
    data=json.loads((FULL/'DATA_AND_RUNTIME.json').read_text());base=load('_three_bank_reference_surface',PHASE/'pubmed_factor1_controls_source_20261010_v1/run.py')
    review=json.loads(base.bind(spec['root_review']).read_text())
    if review.get('approved') is not True or review.get('source_manifest_sha256')!=spec['source_manifest_sha256']:raise ValueError('Exact root review required')
    if spec['runtime']!=data['runtime'] or entry.sha(data['runtime']['python']['path'])!=data['runtime']['python']['sha256'] or Path(sys.executable).resolve()!=Path(data['runtime']['python']['path']).resolve() or os.environ.get('PYTHONPATH','')!=data['runtime']['PYTHONPATH']:
        raise ValueError('Exact existing native runtime required')
    full=closed18(entry,base,spec['full18_owner_plan'],spec['full18_source_manifest_sha256'],data)
    _b,idata,roster,_reuse,imanifest=ic.frozen()
    if imanifest!=ib['source_manifest_sha256']:raise ValueError('Exact completed I4 source required')
    for stage in ('admission','qualification','science','assembly','comparison'):ic.closed_stage(stage,roster,imanifest)
    receipt=json.loads(ic.stage_result('comparison','factorized_I4_complete_comparison').read_text())
    if (receipt.get('complete') is not True or receipt.get('source_manifest_sha256')!=imanifest or
            receipt.get('comparison')!=ib['comparison'] or receipt.get('TEST_access') is not False):
        raise ValueError('Actual I4 waited completion must bind the exact comparison')
    comparison=json.loads(base.bind(ib['comparison']).read_text())
    if (comparison.get('complete_family') is not True or comparison.get('three_complete_banks') is not True or
            comparison.get('three_exact_admitted_body0_anchors') is not True or comparison.get('TEST_access') is not False):
        raise ValueError('Exact completed I4 comparison required')
    i4={}
    for seed in entry.SEEDS:
        value=json.loads(ic.stage_result('assembly',f'seed{seed}__factorized_I4_native__committee').read_text())
        if (value.get('complete') is not True or value.get('source_manifest_sha256')!=imanifest or value['complete_saved_member_logits']!=ib['banks'][str(seed)] or
                value['selected_readouts']!=ib['selected_readouts'][str(seed)]):raise ValueError('Exact selected I4 bank changed')
        i4[seed]=value
    for key in ('valid_bundle','validation_custody','runtime','frozen_providers'):
        if idata[key]!=data[key]:raise ValueError('Same exact task/provider role required')
    valid_custody=json.loads(base.bind(data['validation_custody']).read_text())
    expected_limits=dict(external_active_seconds=600,external_cleanup_seconds=10,RSS_bytes=1073741824,output_bytes=8388608,log_bytes=2097152,GPU_bytes=0)
    if spec['limits']!=expected_limits:raise ValueError('Unchanged predeclared CPU diagnostic envelope required')
    output=Path(spec['output']).resolve()
    if not output.is_relative_to(HERE) or output.exists():raise ValueError('Fresh root-confined diagnostic output required')
    output.mkdir(parents=True,exist_ok=False);timings={'source_role_and_all_closure_checks_seconds':time.monotonic()-started}
    try:
        # Numerical imports and payload reads occur only after all full18 and I4 closure checks above.
        import numpy as np
        import torch
        providers=dict(numpy=str(np.__version__),torch=str(torch.__version__))
        for name in ('scipy','torch-geometric','torch-scatter','torch-sparse'):providers[name]=importlib.metadata.version(name)
        if providers!=data['frozen_providers']:raise ValueError('Exact existing providers required')
        torch.set_num_threads(2)
        with base.reference_surface() as (_unused_engine,metrics,_unused_factory):
            from source import fingerprint
            with np.load(base.bind(data['valid_bundle']),allow_pickle=False) as archive:
                if set(archive.files)!={'valid_ids','valid_y'}:raise ValueError('Projected VALID only')
                arrays={k:archive[k].copy() for k in archive.files}
            if {k:fingerprint(v) for k,v in arrays.items()}!=valid_custody['array_fingerprints']:raise ValueError('Exact VALID role changed')
            ids=torch.from_numpy(arrays['valid_ids']);truth=torch.from_numpy(arrays['valid_y'])
            if ids.dtype!=torch.int64 or truth.dtype!=torch.int64 or ids.shape!=(3942,) or truth.shape!=ids.shape or len(np.unique(arrays['valid_ids']))!=3942 or np.bincount(arrays['valid_y'],minlength=3).tolist()!=[820,1547,1575]:
                raise ValueError('Complete exact VALID IDs/populations required')
            old_common=sys.modules.get('common');old_path=list(sys.path);sys.modules['common']=ic
            try:i4read=load('_three_bank_existing_I4_readout',I4/'run.py')
            finally:
                sys.path[:]=old_path
                if old_common is None:sys.modules.pop('common',None)
                else:sys.modules['common']=old_common
            paired=load('_three_bank_existing_pair_flows',FULL/'error_changes.py')
            rows=[];seed_masks=[];tick=time.monotonic()
            count=lambda mask:int(mask.sum().item())
            for seed in entry.SEEDS:
                banks={};expected={}
                for name,condition in (('baseline','shared4_own'),('A','private_missinghop')):
                    result,folder=full[f'seed{seed}__{condition}'];payload=result['complete_saved_member_logits']
                    path=(folder/payload['path']).resolve(strict=True)
                    if not path.is_relative_to(folder) or payload['factual_shape']!=[4,19717,3]:raise ValueError('Exact selected hop/own payload required')
                    expected[name]=result['selected_readouts']['VALID']
                    banks[name]=selected(np,torch,metrics,base,dict(path=str(path),sha256=payload['sha256'],bytes=payload['bytes']),expected[name],ids,truth)
                expected['B']=i4[seed]['selected_readouts']['VALID']
                banks['B']=selected(np,torch,metrics,base,ib['banks'][str(seed)],expected['B'],ids,truth)
                masks={name:i4read.masks(torch,metrics,bank,{'VALID':(ids,truth)})['VALID'] for name,bank in banks.items()}
                g,a,b=[masks[name]['pool_correct'] for name in ('baseline','A','B')];ua,ub=[masks[name]['coverage'] for name in ('A','B')]
                ra,rb=~g&a,~g&b
                flows=dict(R_A=ra,R_B=rb,A_only=ra&~rb,B_only=rb&~ra,shared=ra&rb,
                    harms_A=g&~a,harms_B=g&~b,shared_harms=g&~a&~b,
                    acquisition_complement_A=ra&~rb&ua&~ub,acquisition_complement_B=rb&~ra&ub&~ua,
                    serving_complement_A=ra&~rb&ub&~b,serving_complement_B=rb&~ra&ua&~a,
                    A_only_other_strict_rival=ra&~rb&masks['B']['strict_common_false_rival'],
                    B_only_other_strict_rival=rb&~ra&masks['A']['strict_common_false_rival'],
                    correct_member_union_intersection=ua&ub)
                for name,other in (('A','B'),('B','A')):
                    exclusive=flows[name+'_only'];acquired=flows['acquisition_complement_'+name];served=flows['serving_complement_'+name]
                    flows[name+'_only_pooled_only_rescue']=exclusive&~masks[name]['coverage']&~masks[other]['coverage']
                    if count(exclusive)!=count(acquired)+count(served)+count(flows[name+'_only_pooled_only_rescue']):raise ValueError('Exclusive repair partition does not reconcile')
                bits=g.to(torch.int64)*4+a.to(torch.int64)*2+b.to(torch.int64)
                patterns=[dict(pattern=f'{k:03b}',count=count(bits.eq(k)),node_ids=ids[bits.eq(k)].tolist(),
                    classes=[dict(class_id=c,count=count(bits.eq(k)&truth.eq(c)),node_ids=ids[bits.eq(k)&truth.eq(c)].tolist()) for c in range(3)]) for k in range(8)]
                row=dict(seed=seed,counts={k:count(v) for k,v in flows.items()},node_ids={k:ids[v].tolist() for k,v in flows.items()},
                    per_class=[dict(class_id=c,counts={k:count(v&truth.eq(c)) for k,v in flows.items()}) for c in range(3)],
                    joint_pool_correctness=patterns,selected_readouts=expected,
                    existing_paired_flows={name:paired.compare(torch,metrics,banks['baseline'],banks[name],ids,truth) for name in ('A','B')})
                rows.append(row);seed_masks.append({k:v for k,v in flows.items() if k in ('A_only','B_only')})
                if time.monotonic()-started>=600:raise TimeoutError('Predeclared finite CPU cap')
            timings['stored_logits_identity_and_joint_counts_seconds']=time.monotonic()-tick
            persistence={}
            for name in ('A_only','B_only'):
                support=torch.stack([m[name] for m in seed_masks]).sum(0)
                persistence[name]=dict(seed_instance_count=int(support.sum().item()),unique_node_count=count(support.gt(0)),
                    support_counts={str(k):count(support.eq(k)) for k in (1,2,3)},
                    node_membership=[dict(node_id=int(ids[i]),support=int(support[i]),seeds=[entry.SEEDS[j] for j,m in enumerate(seed_masks) if bool(m[name][i])]) for i in support.gt(0).nonzero().flatten().tolist()])
            minimum=fixed['predeclared_diagnostic_gate']['minimum_exclusive_served_repairs_per_component_per_seed']
            each_seed=all(row['counts'][name]>=minimum for row in rows for name in ('A_only','B_only'))
            persistent=all(persistence[name]['support_counts']['2']+persistence[name]['support_counts']['3']>=1 for name in persistence)
            result=dict(schema='fixed-three-bank-stored-error-complete-v1',complete=True,record_id=spec['record_id'],
                source_manifest_sha256=spec['source_manifest_sha256'],release_sha256=args.release_sha256,rows=rows,persistence=persistence,
                gate=dict(passed=each_seed and persistent,exclusive_minimum_per_seed=minimum,each_seed_passed=each_seed,persistence_passed=persistent,
                    eligibility_only=True,does_not_release_fits=True,not_statistical_significance=True),
                fixed_spec_sha256=entry.sha(HERE/'FIXED_SPEC.json'),role='VALID',TEST_access=False,oracle_training_target=False,
                all18_and_I4_closed_before_payload_reads=True,models=0,checkpoint_loads=0,model_forwards=0,backwards=0,Adam_steps=0,
                timings=timings,inclusive_seconds=time.monotonic()-started,
                CPU_user_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime,CPU_system_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_stime,
                peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                automatic_retry=False,server_only_exact_VALID_node_evidence=True)
            base.write(output/'JOINT_ERROR_DIAGNOSTIC.json',result)
            (output/'SUMMARY.md').write_text(summarize(result))
            base.write(output/'SUMMARY.json',dict(complete=True,gate=result['gate'],per_seed=[dict(seed=r['seed'],counts=r['counts'],
                classification={k:v['classification'] for k,v in r['selected_readouts'].items()}) for r in rows],
                persistence={k:{f:v[f] for f in ('seed_instance_count','unique_node_count','support_counts')} for k,v in persistence.items()},
                TEST_access=False,does_not_release_fits=True))
            result['output_bytes']=sum(f.stat().st_size for f in output.iterdir() if f.is_file())
            if result['output_bytes']>spec['limits']['output_bytes'] or result['peak_RSS_bytes']>spec['limits']['RSS_bytes'] or time.monotonic()-started>=600:
                raise RuntimeError('Predeclared diagnostic resource envelope exceeded')
            result['inclusive_seconds']=time.monotonic()-started
            result['CPU_user_seconds']=resource.getrusage(resource.RUSAGE_SELF).ru_utime
            result['CPU_system_seconds']=resource.getrusage(resource.RUSAGE_SELF).ru_stime
            result['completion_cost_includes_diagnostic_and_summary_writes']=True
            base.write(output/'COMPLETE.json',{k:v for k,v in result.items() if k not in ('rows','persistence')})
    except BaseException as error:
        base.write(output/'FAILURE.json',dict(complete=False,error_type=type(error).__name__,error=str(error),inclusive_seconds=time.monotonic()-started,
            TEST_access=False,automatic_retry=False,models=0,model_forwards=0));raise
    print(json.dumps(dict(complete=True,record_id=spec['record_id'],diagnostic_gate=result['gate']['passed'],TEST_access=False)))


if __name__=='__main__':
    main()
