"""Inactive post-all21 CPU analysis of exact fresh serving refs; no model forward."""
import argparse
import csv
import datetime
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent
PHASE=HERE.parent
SEEDS=[7409,8501,9607]
FAMILIES=('vanilla','centered','independent')
ROLES=('train','valid')
METRICS=('auroc','nll','accuracy','brier')
CONTRASTS=('vanilla_pool_minus_independent_pool','vanilla_pool_minus_independent_member0',
           'centered_pool_minus_independent_pool','centered_pool_minus_independent_member0')
VANILLA_SEAL='bc6264234cc003c5e89cc46bedf045e0edfbf097ec35b79a8d194fd0c3d11a22'
CENTERED_SEAL='b151054b833b059998376baff89a7ac55ee664024f47d5d704c9fbd6a0fc8ef0'
SCOPE_SHA='24111e6cc00ba233ed3f1d801dba4ab0166254dde0fbd49120b582c9787032f3'
PROTOCOL_SHA='7ae12351e83f5fb34b7b5f5e147ca354669377430b021276f50d1785fa92b86a'


def require(value,message):
    if not value:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):digest.update(block)
    return digest.hexdigest()


def bound(row):
    path=Path(row['path']).resolve(strict=True)
    require(path.is_relative_to(PHASE) and sha(path)==row['sha256'],'Exact declared source/status/custody/fresh-serving binding')
    if 'bytes' in row:require(path.stat().st_size==row['bytes'],'Exact declared bytes')
    return path


def status_key(row):return row['kind'],row['base_seed'],row.get('member',-1)


def same_binding(first,second):
    return Path(first['path']).resolve()==Path(second['path']).resolve() and first['sha256']==second['sha256']


def json_sha(value):
    """Canonical JSON provenance bytes only; no metric arithmetic or float parity test."""
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def completed_record_sha(row):
    # Both sealed shared producers persist counters in RESULT.json but return the
    # record without counters to COMPLETE/CENTERED_RECORDS. No other field is omitted.
    return json_sha({name:value for name,value in row.items()
                     if not (name=='counters' and row['kind'] in ('shared_fit','centered_shared_fit'))})


def completed_result_path(root,key):
    kind,seed,member=key
    if kind in ('shared_fit','centered_shared_fit'):return root/'shared'/('seed'+str(seed))/'RESULT.json'
    folder=root/'independent'/('base'+str(seed))
    return folder/('POOL_RESULT.json' if kind=='independent_pool' else 'MEMBER'+str(member)+'.json')


def provenance(release,original,centered):
    """Anchor all declared files to the closed executions before numeric imports."""
    original_root=Path(release['original_complete']['path']).resolve().parent
    centered_root=Path(release['centered_closure']['path']).resolve().parent
    origins=read(bound(release['original18_origins']))
    require(same_binding(origins['original_complete'],release['original_complete'])
            and centered['original18_origins_sha256']==release['original18_origins']['sha256']
            and origins['comparative_outcomes_unopened'] is True and origins['all_original_costs_charged'] is True,
            'Same exact original COMPLETE and origin receipt reused by this centered closure')
    original_index={status_key(value):value for value in original['records']}
    require(len(origins['records'])==18 and {status_key(value) for value in origins['records']}==set(original_index)
            and all(value['status']=='complete' and value['costs_charged_in_full'] is True for value in origins['records']),
            'All eighteen exact complete origin records and charged bodies retained')
    origin_index={status_key(value):value for value in origins['records']};results={};artifacts={}
    for key,attempt in original_index.items():
        origin=origin_index[key];path=bound(origin['result'])
        require(path==completed_result_path(original_root,key),'Original result belongs to this exact completed execution')
        result=read(path)
        require(status_key(result)==key and result['status']=='complete'
                and completed_record_sha(result)==completed_record_sha(attempt),
                'Exact complete result record provenance, including the charged costs; no metric recalculation')
        results[key]=result
        rows=origin['artifacts'];artifacts[key]={value['label']:value for value in rows}
        require(len(artifacts[key])==len(rows),'Unique original artifact labels')
        if key[0] in ('shared_fit','independent_member'):
            checkpoint=artifacts[key]['checkpoint'];history=artifacts[key]['history']
            checkpoint_path=path.parent/'SELECTED_STATE.pt' if key[0]=='shared_fit' else Path(attempt['checkpoint_path']).resolve()
            require(Path(checkpoint['path']).resolve()==checkpoint_path and checkpoint['sha256']==attempt['checkpoint_sha256']
                    and Path(history['path']).resolve()==checkpoint_path.parent/'HISTORY.jsonl',
                    'Exact selected checkpoint hash and complete history origin of the charged attempt')
        if key[0] in ('shared_fit','independent_pool'):
            serving=artifacts[key]['serving_reference']
            require(Path(serving['path']).resolve()==path.parent/'SELECTED_SERVING_REFERENCE.pt'
                    and serving['sha256']==attempt['serving_reference_sha256'],'Exact original fresh serving artifact origin')
        if key[0]=='independent_pool':
            require([value['member'] for value in attempt['components']]==list(range(4))
                    and all(completed_record_sha(value)==completed_record_sha(original_index[('independent_member',key[1],value['member'])])
                            for value in attempt['components']),
                    'Pool components are these exact four complete charged bodies')
    centered_path=bound(release['centered_records'])
    require(centered_path==centered_root/'CENTERED_RECORDS.json','Final centered records belong to the exact all21 closure directory')
    centered_records=read(centered_path)
    require(len(centered_records)==3 and {status_key(value) for value in centered_records}=={('centered_shared_fit',seed,-1) for seed in SEEDS}
            and all(value['status']=='complete' and value['identity']['original18_origins_sha256']==release['original18_origins']['sha256']
                    for value in centered_records),'Exact three centered completed attempts reuse the same original18 origins')
    attempts=dict(original_index);attempts.update({status_key(value):value for value in centered_records})
    for row in release['references']:
        family,seed=row['family'],row['base_seed'];kind={'vanilla':'shared_fit','centered':'centered_shared_fit','independent':'independent_pool'}[family]
        key=(kind,seed,-1);attempt=attempts[key];root=centered_root if family=='centered' else original_root
        result_path=bound(row['result'])
        require(result_path==completed_result_path(root,key) and Path(row['path']).resolve()==result_path.parent/'SELECTED_SERVING_REFERENCE.pt'
                and row['sha256']==attempt['serving_reference_sha256'],'Fresh reference/result paths and hash belong to this exact completed attempt')
        if family=='centered':
            result=read(result_path)
            require(status_key(result)==key and result['status']=='complete'
                    and completed_record_sha(result)==completed_record_sha(attempt),'Exact centered result and charged completed record provenance')
            results[key]=result
            require(len(row['histories'])==1 and Path(row['histories'][0]['path']).resolve()==result_path.parent/'HISTORY.jsonl',
                    'Centered history belongs to this exact closed attempt')
        else:
            require(same_binding(row['result'],origin_index[key]['result'])
                    and same_binding(row,artifacts[key]['serving_reference']),'Reference and result match the exact reused original origin bindings')
            if family=='independent':
                require([value['member'] for value in row['histories']]==list(range(4))
                        and all(same_binding(value,artifacts[('independent_member',seed,m)]['history']) for m,value in enumerate(row['histories'])),
                        'Every independent curve is the exact history of its charged completed body')
            else:
                require(len(row['histories'])==1 and same_binding(row['histories'][0],artifacts[key]['history']),
                        'Vanilla curve is the exact history of the original completed bank')
    return origins,centered_records,attempts,results


def opening(release):
    """No numeric library/reference/role access until all status and custody gates."""
    require(release['enabled'] is True and release['release_owner']=='root' and release['action']=='post_all21_fresh_serving_analysis'
            and release['analysis_source_sha256']==sha(__file__),'Explicit exact root analysis release')
    scope=read(bound(release['scope']));protocol=read(bound(release['protocol']))
    require(release['scope']['sha256']==SCOPE_SHA and scope['all21_required_before_comparative_opening'] is True
            and scope['declared_before_any_scientific_fit'] is True and scope['no_best_of_two_relabelled_single_primary'] is True,
            'Exact prospectively frozen two-recipe scope')
    require(release['protocol']['sha256']==PROTOCOL_SHA and protocol['co_primary_contrasts']==list(CONTRASTS) and protocol['seeds']==SEEDS
            and protocol['required_logical_records']['total']==21,'All four frozen contrasts and full family')
    custody=read(bound(release['root_custody']))
    require(custody['release_owner']=='root' and custody['all21_complete_verified'] is True
            and set(custody['families'])=={'original18','centered3'},'Root closed all21 before this analysis')
    for family in custody['families'].values():
        require(all(family[key] is True for key in ('actual_parent_absent','actual_parent_group_absent','actual_worker_absent','actual_group_absent','actual_worker_CUDA_absent','reaped'))
                and family['exit_code']==0,'Exact actual science parent/worker/group/CUDA closure, successful reaping')
        require(all(type(family[key]) is int and family[key]>0 for key in ('parent_pid','parent_birth','parent_group','worker_pid','worker_birth','worker_group')),
                'Exact owner/worker birth and group identities retained')
        launch=read(bound(family['launch']))
        require(launch['parent']['pid']==family['parent_pid'] and launch['parent']['start_ticks']==family['parent_birth']
                and launch['parent']['group']==family['parent_group'] and launch['parent']['boot_id']==family['boot_id'],
                'Same exact launch parent birth/group/boot identity')
        terminal=read(bound(family['terminal']))
        require(terminal['complete'] is True and terminal['exit_code']==0 and terminal['reaped'] is True
                and terminal['actual_worker_absent'] is True and terminal['actual_worker_CUDA_absent'] is True
                and terminal['child']['pid']==family['worker_pid'] and terminal['child']['start_ticks']==family['worker_birth']
                and terminal['child']['group']==family['worker_group'] and terminal['child']['boot_id']==family['boot_id'],
                'Same exact terminal custody, no stale/unrelated worker')
    # The centered closure is status/cost only; verify it before decoding original result records.
    require(Path(release['centered_closure']['path']).name=='ALL21_CLOSURE.json'
            and Path(release['original_complete']['path']).name=='COMPLETE.json'
            and Path(release['original_complete']['path']).resolve().parent.name==scope['original18_output'],
            'Exact centered closure role and prospectively fixed original18 execution directory')
    centered=read(bound(release['centered_closure']))
    require(centered['complete'] is True and centered['original18_complete'] is True and centered['centered3_complete'] is True
            and centered['required_logical_records']==21 and len(centered['logical_records'])==21
            and all(row['status']=='complete' for row in centered['logical_records']),'Actual status-only all21 closure before original result access')
    original=read(bound(release['original_complete']))
    expected={('shared_fit',s,-1) for s in SEEDS}|{('independent_pool',s,-1) for s in SEEDS}|{('independent_member',s,m) for s in SEEDS for m in range(4)}
    require(original['complete'] is True and len(original['records'])==18 and {status_key(row) for row in original['records']}==expected
            and all(row['status']=='complete' for row in original['records']),'Every original18 attempt complete, no survivor panel')
    expected|={('centered_shared_fit',s,-1) for s in SEEDS}
    require(centered['complete'] is True and centered['original18_complete'] is True and centered['centered3_complete'] is True
            and centered['required_logical_records']==21 and len(centered['logical_records'])==21
            and {status_key(row) for row in centered['logical_records']}==expected
            and all(row['status']=='complete' for row in centered['logical_records']),'Both exact recipes and all21 complete')
    require(len(release['references'])==9 and {(row['family'],row['base_seed']) for row in release['references']}==set(itertools.product(FAMILIES,SEEDS)),
            'Nine exact fresh-serving references, no selectively retained seeds')
    return protocol,custody,original,centered


def paired(values):
    require(len(values)==3 and all(math.isfinite(value) for value in values),'Three finite paired seed differences')
    mean=statistics.mean(values);sd=statistics.stdev(values);half=4.302652729911275*sd/math.sqrt(3)
    return dict(seeds=SEEDS,differences=values,mean=mean,sample_SD=sd,descriptive_df2_t95_interval=[mean-half,mean+half],
                interpretation='optimizer-seed variation on one selected development split; not graph-population uncertainty or confirmation')


def subset(np,mask,nll,true_probability):
    count=int(mask.sum())
    return dict(rows=count,mean_NLL=None if not count else float(nll[mask].mean()),
                mean_true_class_probability=None if not count else float(true_probability[mask].mean()))


def errors(np,member_logp,pool_logp,y,member_probability,pool_probability):
    labels=y.astype(np.int64,copy=False);members=np.stack(member_logp);pooled=pool_logp
    member_wrong=members.argmax(-1)!=labels[None,:];pool_wrong=pooled.argmax(-1)!=labels
    intersection=member_wrong.all(0);coverage=~intersection;count=len(labels)
    member_nll=-members[:,np.arange(count),labels];member_true=np.stack(member_probability)[:,np.arange(count),labels]
    pool_nll=-pooled[np.arange(count),labels];pool_true=pool_probability[np.arange(count),labels]
    per_member=[]
    for m in range(4):
        others=np.delete(member_wrong,m,axis=0)
        per_member.append(dict(member=m,wrong_rows=int(member_wrong[m].sum()),
            unique_correct_coverage_rows=int(((~member_wrong[m])&others.all(0)).sum()),
            pool_rescue_rows=int((member_wrong[m]&~pool_wrong).sum()),pool_harm_rows=int((~member_wrong[m]&pool_wrong).sum()),
            own_wrong=subset(np,member_wrong[m],member_nll[m],member_true[m]),
            own_correct=subset(np,~member_wrong[m],member_nll[m],member_true[m]),
            wrong_when_pool_rescues=subset(np,member_wrong[m]&~pool_wrong,member_nll[m],member_true[m]),
            correct_when_pool_harms=subset(np,~member_wrong[m]&pool_wrong,member_nll[m],member_true[m])))
    pairs=[]
    for i,j in itertools.combinations(range(4),2):
        a,b=member_wrong[i],member_wrong[j]
        pairs.append(dict(members=[i,j],both_wrong=int((a&b).sum()),only_first_wrong=int((a&~b).sum()),
                          only_second_wrong=int((~a&b).sum()),neither_wrong=int((~a&~b).sum()),
                          argmax_disagreement_rows=int((members[i].argmax(-1)!=members[j].argmax(-1)).sum())))
    return dict(population_rows=count,positive_rows=int((labels==1).sum()),negative_rows=int((labels==0).sum()),
        all_member_wrong_rows=int(intersection.sum()),any_member_correct_rows=int(coverage.sum()),
        all_member_wrong_fraction=float(intersection.mean()),any_member_correct_fraction=float(coverage.mean()),pool_wrong_rows=int(pool_wrong.sum()),
        coverage_not_converted_to_pool_correct_rows=int((coverage&pool_wrong).sum()),
        pool_correct_on_all_member_wrong_rows=int((intersection&~pool_wrong).sum()),
        correct_member_count_histogram=np.bincount((~member_wrong).sum(0),minlength=5).tolist(),
        members=per_member,within_bank_pairs=pairs,pool_wrong=subset(np,pool_wrong,pool_nll,pool_true),
        pool_correct=subset(np,~pool_wrong,pool_nll,pool_true),decisions='unchanged stored argmax, no calibrated/tuned threshold')


def rank_disagreement(np,probabilities,y):
    """All positive-negative pairs, including ties; bounded temporary blocks only."""
    probability=np.stack([value[:,1] for value in probabilities])
    positive=probability[:,y==1];negative=probability[:,y==0]
    pairs=list(itertools.combinations(range(len(probabilities)),2))
    rows={pair:dict(paths=list(pair),pairs=0,rank_disagreement_including_ties=0,opposite_strict_order=0,
        exactly_one_tie=0,both_tie=0,first_correct_second_wrong=0,second_correct_first_wrong=0) for pair in pairs}
    for a in range(0,positive.shape[1],256):
        for b in range(0,negative.shape[1],256):
            signs=np.sign(positive[:,a:a+256,None]-negative[:,None,b:b+256]).astype(np.int8)
            for pair in pairs:
                i,j=pair;first,second=signs[i],signs[j];row=rows[pair]
                row['pairs']+=first.size
                row['rank_disagreement_including_ties']+=int((first!=second).sum())
                row['opposite_strict_order']+=int((first*second==-1).sum())
                row['exactly_one_tie']+=int(((first==0)^(second==0)).sum())
                row['both_tie']+=int(((first==0)&(second==0)).sum())
                row['first_correct_second_wrong']+=int(((first>0)&(second<0)).sum())
                row['second_correct_first_wrong']+=int(((second>0)&(first<0)).sum())
    for row in rows.values():
        row['disagreement_fraction']=None if not row['pairs'] else row['rank_disagreement_including_ties']/row['pairs']
    return dict(paths='0..3 member paths;4 served pool; every pair stays within one bank/seed',all_positive_negative_pairs=True,
                temporary_block_shape=[len(probabilities),256,256],rows=list(rows.values()))


def overlap(np,a,b,y):
    first=a.argmax(-1)!=y;second=b.argmax(-1)!=y
    return dict(rows=len(y),both_wrong=int((first&second).sum()),only_first_wrong=int((first&~second).sum()),
                only_second_wrong=int((~first&second).sum()),neither_wrong=int((~first&~second).sum()),
                first_rescues_second=int((~first&second).sum()),second_rescues_first=int((first&~second).sum()))


def histories(path,family):
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    require(bool(rows) and [row['epoch'] for row in rows]==list(range(1,len(rows)+1)),'Complete contiguous learning history')
    if family=='independent':
        return [dict(epoch=row['epoch'],eval_TRAIN_NLL=row['train']['nll'],eval_VALID_AUROC=row['valid']['auroc']) for row in rows]
    return [[dict(epoch=row['epoch'],eval_TRAIN_NLL=row['member_scores'][m]['train']['nll'],eval_VALID_AUROC=row['member_scores'][m]['valid']['auroc'],
                  pre_update_TRAIN_own_NLL=row['own_TRAIN_nll'][m],
                  weighted_centered_prior=None if family=='vanilla' else row['member_weighted_centered_prior'][m]) for row in rows] for m in range(4)]


def run(args):
    release=read(args.release);protocol,custody,original,centered=opening(release)
    origins,centered_records,attempts,completed_results=provenance(release,original,centered)
    output=args.output.resolve()
    require(str(output)==release['output_directory'] and output.is_relative_to(PHASE) and not output.exists()
            and not output.is_relative_to(HERE)
            and not output.is_relative_to(Path(release['original_complete']['path']).resolve().parent)
            and not output.is_relative_to(Path(release['centered_closure']['path']).resolve().parent),'Fresh analysis output outside all original results')
    # All numeric imports and all role/reference arrays occur strictly after all21/custody opening.
    import numpy as np
    import torch
    require(np.__version__==release['numpy_version']=='1.26.4' and str(torch.__version__)==release['torch_version']=='2.1.2+cu118',
            'Same qualified CPU array/probability arithmetic; no drift rechecks')
    common_path=PHASE/'private_sheaf_train_valid_runner_20261009_v2/common.py'
    require(sha(common_path)==release['role_helper_sha256'],'Exact original six-key role validation helper')
    spec=importlib.util.spec_from_file_location('_all21_original_roles',common_path);common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
    require(sha(release['roles']['path'])==release['roles']['sha256'],'Exact original official roles')
    arrays,meta,metadata_sha=common.read_roles(np,release['roles']['path'])
    require(metadata_sha==release['role_metadata_sha256'],'Exact role metadata')
    labels={role:arrays[role+'_y'] for role in ROLES}
    del arrays
    sources={};refs={};numerical={};quality=[];diagnostics={};curves={};contrasts={};component={};costs=[]
    for row in release['references']:
        path=bound(row);require(path.name=='SELECTED_SERVING_REFERENCE.pt','Fresh serving reference only; no model checkpoint')
        family,seed=row['family'],row['base_seed'];reference=torch.load(path,map_location='cpu',weights_only=True)
        identity=reference['identity'];expected_seal=CENTERED_SEAL if family=='centered' else VANILLA_SEAL
        require(reference['base_seed']==seed and reference['score_source']=='fresh_reconstructed_selected_serving' and reference['TEST_truth'] is False
                and reference['server_only'] is True and identity['source_seal_sha256']==expected_seal
                and identity['role_archive_sha256']==release['roles']['sha256'] and identity['role_metadata_sha256']==release['role_metadata_sha256']
                and identity['configuration']['id']=='d4_f16_L4','Same fresh selected-serving source/state/role/config scope')
        kind={'vanilla':'shared_fit','centered':'centered_shared_fit','independent':'independent_pool'}[family]
        attempt=attempts[(kind,seed,-1)];result=completed_results[(kind,seed,-1)]
        require(result['kind']==kind and result['base_seed']==seed and result['status']=='complete'
                and result['serving_reference_sha256']==attempt['serving_reference_sha256']==row['sha256'],
                'Exact completed result binds this actual fresh serving reference')
        attempt_identity=attempt['components'][0]['native_result']['identity'] if family=='independent' else attempt['identity']
        require(json_sha(identity)==json_sha(attempt_identity),'Reference identity is the exact charged completed execution identity')
        if family=='independent':
            bindings=reference['member_bindings']
            require(len(bindings)==4 and {value['member'] for value in bindings}==set(range(4))
                    and all(value['seed']==seed+1000003*value['member'] for value in bindings),'Same exact four offset-seed own selected bodies')
            require([value['member'] for value in bindings]==list(range(4)),'Original fixed member order preserved')
            own_epochs=[value['own_selected_epoch'] for value in bindings]
            bodies=[attempts[('independent_member',seed,member)] for member in range(4)]
            require(len(bodies)==4 and {value['member'] for value in bodies}==set(range(4)),'Every charged own independent body retained')
            require(all(value['checkpoint_sha256']==bodies[m]['checkpoint_sha256']
                        and value['own_selected_epoch']==bodies[m]['native_result']['selected_epoch']
                        and value['seed']==bodies[m]['seed'] for m,value in enumerate(bindings)),
                    'Every fresh independent member is the exact own selected checkpoint/epoch of its charged completed body')
            costs.append(dict(family=family,seed=seed,total_complete_fit_and_serving_seconds=sum(value['complete_attempt_seconds'] for value in bodies)+attempt['selected_serving_seconds'],
                body_attempt_seconds=[value['complete_attempt_seconds'] for value in sorted(bodies,key=lambda v:v['member'])],
                selected_serving_seconds=attempt['selected_serving_seconds'],active_parameters=attempt['complete_independent_active_parameters'],
                parameter_bytes=attempt['complete_independent_parameter_bytes'],serving_residency=attempt['selected_serving_model_residency'],
                selected_serving_peak_CUDA_allocated_bytes=attempt['selected_serving_peak_CUDA_allocated_bytes'],
                selected_serving_peak_CUDA_reserved_bytes=attempt['selected_serving_peak_CUDA_reserved_bytes'],
                native_body_attempt_CUDA_allocated_peaks=[value['native_result']['complete_attempt_cuda_peak_allocated_bytes'] for value in sorted(bodies,key=lambda v:v['member'])],
                every_original_cost_charged=True,no_reference_replay=True))
        else:
            require(reference['checkpoint_sha256']==result['checkpoint_sha256']==attempt['checkpoint_sha256']
                    and reference['selected_epoch']==result['selected_epoch']==attempt['selected_epoch'],'Same exact completed own selected shared checkpoint identity')
            own_epochs=[reference['selected_epoch']]*4
            costs.append(dict(family=family,seed=seed,total_complete_fit_and_serving_seconds=attempt['complete_attempt_seconds'],
                parameters=attempt['parameters'],peak_CUDA_allocated_bytes=attempt['peak_CUDA_allocated_bytes'],peak_CUDA_reserved_bytes=attempt['peak_CUDA_reserved_bytes'],
                CPU_user_seconds=attempt['CPU_user_seconds'],CPU_system_seconds=attempt['CPU_system_seconds'],
                checkpoint_bytes=attempt['checkpoint_bytes'],serving_reference_bytes=attempt['serving_reference_bytes'],
                serving_residency='shared complete bank with four persistent native cache sets',every_original_cost_charged=True))
        sources[(family,seed)]=row;refs[(family,seed)]=reference
        for role in ROLES:
            pooled=reference['pooled_role_logp'][role];members=[value[role] for value in reference['member_role_logp']]
            require(len(members)==4 and all(value.shape==(len(labels[role]),2) and torch.isfinite(value).all().item() for value in [pooled]+members),
                    'Finite complete all-row binary fresh serving arrays')
            logp=[value.detach().numpy() for value in members];pool=pooled.detach().numpy()
            probabilities=[value.exp().detach().numpy() for value in members];pool_probability=pooled.exp().detach().numpy()
            numerical[(family,seed,role)]=(logp,pool)
            own=reference['member_scores'];pool_scores=reference['scores'][role]
            for member in range(4):quality.append(dict(family=family,seed=seed,role=role,path='member'+str(member),selected_epoch=own_epochs[member],**own[member][role]))
            quality.append(dict(family=family,seed=seed,role=role,path='served_pool',selected_epoch=None if family=='independent' else own_epochs[0],**pool_scores))
            diagnostics[(family,seed,role)]=dict(errors=errors(np,logp,pool,labels[role],probabilities,pool_probability),
                ranks=rank_disagreement(np,probabilities+[pool_probability],labels[role]),
                own_metric_mean={metric:statistics.mean(value[role][metric] for value in own) for metric in METRICS},
                own_metric_worst={metric:(min if metric in ('auroc','accuracy') else max)(value[role][metric] for value in own) for metric in METRICS},
                member_minus_pool=[{metric:value[role][metric]-pool_scores[metric] for metric in METRICS} for value in own])
        require(len(row['histories'])==(4 if family=='independent' else 1),'All actual member learning histories required')
        if family=='independent':
            require([value['member'] for value in row['histories']]==list(range(4)),'All four own history origins in original member order')
            require(all(bound(value).parent==Path(bodies[m]['checkpoint_path']).resolve().parent for m,value in enumerate(row['histories'])),
                    'Each complete curve belongs to its own selected independent body')
            curves[(family,seed)]=[histories(bound(value),family) for value in row['histories']]
        else:
            require(bound(row['histories'][0]).parent==Path(row['result']['path']).resolve().parent,'Shared complete curve belongs to this selected bank')
            curves[(family,seed)]=histories(bound(row['histories'][0]),family)
        require(all(1<=own_epochs[m]<=len(curves[(family,seed)][m]) for m in range(4)),'Selected epochs lie in complete own histories')
    for name in CONTRASTS:
        family='vanilla' if name.startswith('vanilla') else 'centered';single=name.endswith('member0')
        contrasts[name]={}
        for role in ROLES:
            contrasts[name][role]={metric:paired([refs[(family,seed)]['scores'][role][metric]-(refs[('independent',seed)]['member_scores'][0][role][metric]
                if single else refs[('independent',seed)]['scores'][role][metric]) for seed in SEEDS]) for metric in METRICS}
    gate=protocol['pilot_gate'];gates={}
    for family in ('vanilla','centered'):
        names=[name for name in CONTRASTS if name.startswith(family)];rows={}
        for name in names:
            auc=contrasts[name]['valid']['auroc'];nll=contrasts[name]['valid']['nll']
            rows[name]=dict(all_three_AUROC_gains_strictly_above_frozen_zero=all(value>gate['each_seed_signed_AUROC_gain_gt'] for value in auc['differences']),
                mean_AUROC_gain_meets_frozen_gate=auc['mean']>=gate['mean_signed_AUROC_gain_at_least'],
                mean_pooled_NLL_delta_meets_frozen_gate=nll['mean']<=gate['mean_pooled_NLL_delta_at_most'])
        gates[family]=dict(contrasts=rows,numerical_quality_gate_pass=all(all(row.values()) for row in rows.values()),
            root_member_competence_review_required=True,method_claim_or_confirmation_authorized=False)
    for seed in SEEDS:
        component[seed]={}
        for role in ROLES:
            component[seed][role]=dict(centered_minus_vanilla_pool={metric:refs[('centered',seed)]['scores'][role][metric]-refs[('vanilla',seed)]['scores'][role][metric] for metric in METRICS},
                pooled_error_overlap={name:overlap(np,numerical[(a,seed,role)][1],numerical[(b,seed,role)][1],labels[role])
                    for name,a,b in (('vanilla_vs_independent','vanilla','independent'),('centered_vs_independent','centered','independent'),('centered_vs_vanilla','centered','vanilla'))},
                mean_own_quality_delta_to_independent={family:{metric:diagnostics[(family,seed,role)]['own_metric_mean'][metric]-diagnostics[('independent',seed,role)]['own_metric_mean'][metric] for metric in METRICS}
                    for family in ('vanilla','centered')})
    result=dict(schema='complete-all21-fresh-serving-exploratory-analysis-v2',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        protocol=protocol,release_sha256=sha(args.release),source_sha256=sha(__file__),root_custody=custody,
        all21_verified_before_numeric_or_reference_access=True,fresh_serving_only=True,model_forwards=0,new_predictions=0,TEST_truth_access=False,
        quality=quality,contrasts=contrasts,unchanged_pilot_gates=gates,mandatory_centered_vs_vanilla_component=component,
        populations={str(key):value for key,value in diagnostics.items()},learning_curves={str(key):value for key,value in curves.items()},
        source_bindings=[sources[key] for key in sorted(sources)],
        original18_charged_cost_and_records=original,original18_origin_receipt=origins,
        original18_origins_sha256=release['original18_origins']['sha256'],exact_completed_result_reference_history_provenance=True,
        centered3_charged_cost_and_closure=centered,centered3_charged_cost_and_records=centered_records,per_seed_family_complete_costs=costs,
        interpretation='TRAIN eval-NLL/VALID member quality diagnose learning deficits; error intersection, unique coverage, rescue/harm and rank disagreements diagnose complementarity and its conversion by the served mean. These observations do not identify a causal private-geometry mechanism or a novel loss.',
        no_best_of_two_primary=True,no_selected_examples=True,no_calibration_or_threshold_tuning=True,no_member_pairing_across_banks_or_seeds=True,
        descriptive_intervals_not_confirmatory=True,additional_capacity_and_tied_path_controls_required=protocol['controls_required_before_method_claim'],
        selected_checkpoint_policy_difference='shared members inherit one pooled selected epoch; independent bodies each own selected epoch',
        factor_norm_and_actual_operator_response_claim=False,
        mechanistic_limit='Fresh serving refs do not contain learned factor tensors/native operator responses; causal capacity/prior/sharing/geometry attribution is not established by these prediction diagnostics.')
    output.mkdir(parents=True,exist_ok=False)
    (output/'ANALYSIS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (output/'EVERY_MEMBER_QUALITY.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['family','seed','role','path','selected_epoch',*METRICS]);writer.writeheader();writer.writerows(quality)
    summary=['# Complete all21 exploratory comparison','',
        'All four predeclared contrasts are reported; intervals describe three optimizer seeds on the already-used split. No confirmation or mechanism claim follows.','',
        '| Contrast | VALID AUROC paired differences | Mean | Sample SD | Descriptive df2 95% interval | Mean VALID NLL delta |',
        '|---|---|---:|---:|---|---:|']
    for name in CONTRASTS:
        a=contrasts[name]['valid']['auroc'];n=contrasts[name]['valid']['nll']
        summary.append('| '+name+' | '+', '.join(format(value,'.8g') for value in a['differences'])+' | '+format(a['mean'],'.8g')+' | '+format(a['sample_SD'],'.8g')+' | '+str(a['descriptive_df2_t95_interval'])+' | '+format(n['mean'],'.8g')+' |')
    summary+=['','## Unchanged pilot gates','',
        'Each of the three AUROC gains must be strictly positive, mean gain at least0.003, and mean VALID NLL delta at most0 for both contrasts of each predeclared recipe. Numerical gate passage still requires root member-competence/cost review and the frozen additional controls.','',
        '| Recipe | Both frozen numerical contrasts pass | Root competence/cost review |', '|---|---|---|']
    for family in ('vanilla','centered'):summary.append('| '+family+' | '+str(gates[family]['numerical_quality_gate_pass'])+' | required |')
    summary+=['','## Member learning and complementarity','',
        'Every member, its mean/worst quality and complete TRAIN eval learning curves are retained. Full-population errors report common failures, unique correct coverage, pool rescue/harm and confidence on each error set. All positive-negative rank pairs, including ties, are processed in bounded blocks.','',
        'Higher shared TRAIN eval NLL alongside lower VALID member quality supports a learning deficit. Low common-error intersection or distinct correct coverage supports complementary predictions; coverage lost by the pool and rescue/harm show how much the fixed probability mean converts it. These descriptive signals cannot assign cause to capacity, sharing, factor priors or geometry without the frozen additional controls.','']
    summary+=['| Recipe/seed | Mean TRAIN NLL minus independent mean | Mean VALID AUROC minus independent mean | All4 wrong fraction | Any member correct fraction | Correct coverage lost by pool rows | Pool AUROC minus own mean |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for family in ('vanilla','centered'):
        for seed in SEEDS:
            d=diagnostics[(family,seed,'valid')];e=d['errors'];train=component[seed]['train']['mean_own_quality_delta_to_independent'][family]['nll']
            auc=component[seed]['valid']['mean_own_quality_delta_to_independent'][family]['auroc']
            benefit=refs[(family,seed)]['scores']['valid']['auroc']-d['own_metric_mean']['auroc']
            summary.append('| '+family+'/'+str(seed)+' | '+' | '.join(format(value,'.8g') for value in (train,auc,e['all_member_wrong_fraction'],e['any_member_correct_fraction'],e['coverage_not_converted_to_pool_correct_rows'],benefit))+' |')
    summary+=['','Per-member rescue/harm, confidence, unique coverage, all rank pairs and complete comparable eval-TRAIN curves are in ANALYSIS.json; every member and selected epoch is in EVERY_MEMBER_QUALITY.csv. Cost records retain all original independent fitting costs and the differing serving residency.','',
              'Shared paths use a pooled selected epoch; independent paths use their own selected epochs. Learned factor tensors and actual operator responses are absent from serving references, so this reader makes no causal factor/geometry or operator-response claim.','']
    (output/'REPORT.md').write_text('\n'.join(summary))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true');parser.add_argument('--release',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if not args.execute:print(json.dumps(dict(inactive=True,numeric_or_role_or_reference_import=False,model_forwards=0)));return
    require(args.release and args.output,'Explicit root post-all21 release and fresh analysis output required');run(args)


if __name__=='__main__':main()
