"""Disabled stdlib reader: complete M1-six and immutable closed PubMed12 scalars."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent;PHASE=HERE.parent
SERVER_PHASE=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930')
GPU='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
SEEDS=(9101,9203,9307)
M1=('factor1_native','factor1_mean4_dropout')
REFERENCES=('single_native','single_mean4_dropout','shared4_own','independent4_own')
PAIRS=(('factor1_native','single_native'),('factor1_mean4_dropout','single_mean4_dropout'),
       ('shared4_own','factor1_native'),('shared4_own','factor1_mean4_dropout'),
       ('factor1_native','independent4_own'),('factor1_mean4_dropout','independent4_own'),
       ('factor1_mean4_dropout','factor1_native'))
LIMITS=dict(external_active_seconds=120,external_cleanup_seconds=10,RSS_bytes=512*1024**2,
            GPU_bytes=0,output_bytes=8*1024**2,log_bytes=1024**2,automatic_retry=False)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def path_in_phase(path,existing=True):
    path=Path(path).resolve(strict=existing)
    if path==PHASE or not path.is_relative_to(PHASE):raise ValueError('Owned project phase path required')
    return path


def bound(row):
    path=path_in_phase(row['path'])
    if sha(path)!=row['sha256']:raise ValueError('Exact scalar/source artifact binding changed')
    return path


def read(row):return json.loads(bound(row).read_text())


def write(path,value):
    path=path_in_phase(path,existing=False);tmp=path.with_name(path.name+'.partial')
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');tmp.replace(path)


def actual_wait(row,release_sha,expected_argv):
    value=read(row);absence=value.get('owned_absence',{})
    if value.get('complete') is not True or value.get('directly_waited') is not True or value.get('child_exit_code')!=0 or value.get('cap_or_owner_failure') is not None:
        raise ValueError('Every actual raw terminal must be successfully and directly waited')
    if absence.get('owned_process_absence_verified') is not True or absence.get('owned_CUDA_absence_verified') is not True:
        raise ValueError('Every owned process/CUDA absence must be confirmed')
    if value.get('argv')!=expected_argv or expected_argv[-2:]!=['--release-sha256',release_sha]:raise ValueError('Exact actual command/release absent from waited argv')
    if value.get('automatic_retry') is not False:raise ValueError('No owner automatic retry')
    return value


def admit(path,digest):
    if socket.gethostname()!='anogena-2-0' or PHASE!=SERVER_PHASE:raise ValueError('Exact allocation project route required')
    if subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()!=[GPU]:raise ValueError('Exact sole allocation GPU route required')
    path=path_in_phase(path)
    if sha(path)!=digest:raise ValueError('Exact separate root scalar-reader release required')
    spec=json.loads(path.read_text())
    if spec.get('schema')!='PubMed-factor1-complete6-scalar-release-v1':raise ValueError('New complete-six scalar release required')
    for flag in ('enabled','root_authorized','source_review_approved','all_six_owned_closed','all_six_roster_frozen',
                 'closed_reference12_verified','external_hard_bound_confirmed','ordinary_runtime_confirmed','post_screen_exploration'):
        if spec.get(flag) is not True:raise ValueError('Inactive complete-family reader: '+flag)
    for flag in ('TEST_access','new_model_calls','new_array_reads','new_score_recalculation','current_CMCL_or_77_quality_read',
                 'paper_score_recalculation','confirmation_claim','pure_M4_parameter_match','CORE_continuation_eligible','further18_activated','automatic_retry'):
        if spec.get(flag) is not False:raise ValueError('Closed reader scope: '+flag)
    if any(k in spec for k in ('train_bundle','valid_bundle','test_bundle','logit_payload','checkpoint','full_y')):raise ValueError('Scalar reader cannot accept numerical payload interfaces')
    manifest=HERE/'SOURCE_MANIFEST.json'
    if spec.get('source_manifest_sha256')!=sha(manifest):raise ValueError('Exact sealed scalar-reader source required')
    for row in json.loads(manifest.read_text())['files']:
        file=(HERE/row['path']).resolve(strict=True)
        if not file.is_relative_to(HERE) or sha(file)!=row['sha256'] or file.stat().st_size!=row['bytes']:raise ValueError('Scalar-reader source changed')
    bind_review=bound(spec['source_review']);review=json.loads(bind_review.read_text())
    if review.get('approved') is not True or review.get('source_manifest_sha256')!=spec['source_manifest_sha256']:raise ValueError('Root exact scalar-reader approval required')
    bindings=json.loads((HERE/'BINDINGS.json').read_text())
    for k in ('M1_source_manifest','M1_roster','M1_source_protocol'):
        if spec.get(k)!=bindings[k]:raise ValueError('Exact frozen M1 source/roster/protocol binding required')
        bound(spec[k])
    if spec.get('closed_comparison',{}).get('sha256')!=bindings['closed_comparison']['sha256'] or spec.get('closed_costs',{}).get('sha256')!=bindings['closed_costs']['sha256']:
        raise ValueError('Exact already closed PubMed12 report/costs required')
    expected=[f'seed{s}__{c}' for s in SEEDS for c in M1]
    if set(spec.get('records',{}))!=set(expected):raise ValueError('All six M1 descriptors required')
    roster=read(spec['M1_roster'])
    if [r['record_id'] for r in roster]!=expected:raise ValueError('Fixed original six-record order required')
    m1_root=bound(spec['M1_source_manifest']).parent
    if Path(spec['science_owner_plan']['path'])!=m1_root/'SCIENCE_OWNER_PLAN.json' or Path(spec['science_family_complete']['path'])!=m1_root/'science'/'FAMILY_COMPLETE.json':raise ValueError('Original M1 science plan/family locations required')
    plan=read(spec['science_owner_plan']);family=read(spec['science_family_complete'])
    manifest_rows=read(spec['M1_source_manifest'])['files']
    queue_sha=next(r['sha256'] for r in manifest_rows if r['path']=='queue.py')
    run_sha=next(r['sha256'] for r in manifest_rows if r['path']=='run.py')
    if plan.get('schema')!='PubMed-factor1-finite-owner-plan-v1' or plan.get('enabled') is not True or plan.get('automatic_retry') is not False or plan.get('purpose')!='science' or plan.get('source_manifest_sha256')!=bindings['M1_source_manifest']['sha256'] or plan.get('roster_sha256')!=bindings['M1_roster']['sha256'] or plan.get('owner_sha256')!=queue_sha or [r['record_id'] for r in plan['records']]!=expected:
        raise ValueError('Exact original full-six science owner plan required')
    if family.get('complete') is not True or family.get('purpose')!='science' or family.get('all_records_directly_waited') is not True or family.get('records')!=expected or family.get('quality_fields_opened') is not False:
        raise ValueError('Actual entire six-owner family must close first')
    owners={}
    # This pass reads only owner facts. No actual M1 COMPLETE quality is read
    # until every direct wait, exact release argv and owned absence succeeds.
    for row in plan['records']:
        record=row['record_id'];owner_id='science__'+record
        release=m1_root/'releases'/(owner_id+'.json')
        if row['owner_id']!=owner_id or row['entry_args']!=['--mode','science'] or PHASE/row['release']!=release or row['entrypoint']!=dict(path=str(m1_root/'run.py'),sha256=run_sha):raise ValueError('Original six owner commands required')
        bound(dict(path=str(release),sha256=row['release_sha256']));bound(row['entrypoint'])
        if Path(spec['records'][record]['complete']['path'])!=m1_root/'science'/'cells'/record/'COMPLETE.json' or Path(spec['records'][record]['raw_terminal']['path'])!=m1_root/'owners'/owner_id/'RAW_OWNER_TERMINAL.json':raise ValueError('Original complete/raw owner locations required')
        argv=[bindings['runtime']['python']['path'],'-B','-P',str(m1_root/'run.py'),'--mode','science','--release',str(release),'--release-sha256',row['release_sha256']]
        owners[record]=actual_wait(spec['records'][record]['raw_terminal'],row['release_sha256'],argv)
    if spec.get('limits')!=LIMITS:raise ValueError('Frozen scalar-only bounds required')
    contract=read(spec['external_owner_release'])
    if contract.get('enabled') is not True or contract.get('record_id')!='factor1_complete6_scalar_comparison' or contract.get('limits')!=LIMITS or contract.get('automatic_retry') is not False:raise ValueError('Root finite scalar-reader owner required')
    for k in ('separate_process_group','direct_wait_required','resource_caps_enforced','output_and_log_caps_enforced'):
        if contract.get(k) is not True:raise ValueError('Bound owner fact missing')
    bound(contract['owner_source']);owner_review=read(contract['owner_review'])
    if owner_review.get('approved') is not True or owner_review.get('owner_sha256')!=contract['owner_source']['sha256']:raise ValueError('Exact approved finite reader owner required')
    runtime=bindings['runtime'];python=Path(os.path.abspath(runtime['python']['path']))
    if spec.get('runtime')!=runtime or not python.is_relative_to(PHASE) or not python.is_file() or sha(python)!=runtime['python']['sha256'] or python.resolve()!=Path(sys.executable).resolve():raise ValueError('Existing qualified interpreter required')
    if set(spec.get('engineering_records',{}))!=set(M1):raise ValueError('Both actual acquisition qualifier costs required')
    output=path_in_phase(spec['output'],existing=False)
    if output.exists() or output.is_relative_to(HERE):raise ValueError('Fresh separate root-owned comparison output required')
    spec['_release_sha256']=digest
    return spec,output,bindings,plan,owners


def check_classification(value,role,members):
    population=3942 if role=='VALID' else 11829
    class_counts=[820,1547,1575] if role=='VALID' else [2461,4643,4725]
    def row(r,n):
        if type(r.get('correct')) is not int or r.get('count')!=n or not 0<=r['correct']<=n or r.get('accuracy')!=r['correct']/n or not math.isfinite(r['NLL']):raise ValueError('Stored count/accuracy/NLL scalar inconsistency')
    row(value['pooled'],population)
    if len(value['members'])!=members or len(value['classes'])!=3:raise ValueError('Complete member/class readout required')
    for r in value['members']:row(r,population)
    for c,(r,n) in enumerate(zip(value['classes'],class_counts)):
        if r['class_id']!=c or len(r['members'])!=members:raise ValueError('Authoritative ordered class/member readout required')
        row(r,n)
        for member in r['members']:row(member,n)
    for field in ('mean_member_accuracy','worst_member_accuracy','macro_accuracy'):
        if not math.isfinite(value[field]):raise ValueError('Finite immutable scalar summary required')
    if sum(r['correct'] for r in value['classes'])!=value['pooled']['correct'] or any(sum(r['members'][m]['correct'] for r in value['classes'])!=value['members'][m]['correct'] for m in range(members)):raise ValueError('Stored class/pooled/member counts differ')
    if value['mean_member_accuracy']!=sum(r['accuracy'] for r in value['members'])/members or value['worst_member_accuracy']!=min(r['accuracy'] for r in value['members']) or value['macro_accuracy']!=sum(r['accuracy'] for r in value['classes'])/3:raise ValueError('Stored immutable summaries differ from their scalars')


def check_repair(value,classification,role,members):
    population=3942 if role=='VALID' else 11829
    fields=('oracle_union_correct','all_members_correct','all_members_wrong','pool_correct','all_wrong_repaired_by_pool',
            'pool_wrong_despite_some_member_correct','member0_errors_repaired_by_pool','member0_correct_lost_by_pool','strict_common_false_rival_nodes')
    if value.get('count')!=population or value.get('member_correct')!=[r['correct'] for r in classification['members']]:raise ValueError('Stored repair population/member identity differs')
    if any(type(value.get(k)) is not int or not 0<=value[k]<=population for k in fields):raise ValueError('Finite in-range stored repair counts required')
    if value['pool_correct']!=classification['pooled']['correct'] or value['all_members_wrong']!=population-value['oracle_union_correct'] or value['pool_correct']!=value['oracle_union_correct']+value['all_wrong_repaired_by_pool']-value['pool_wrong_despite_some_member_correct']:raise ValueError('Stored repair count identities differ')
    if value.get('affects_selection') is not False or len(value.get('pair_disagreement',[]))!=members*(members-1)//2:raise ValueError('Complete descriptive within-bank repair readout required')


def m1_record(value,declared,source_sha,release_sha,providers):
    if value.get('schema')!='PubMed-factor1-science-complete-v1' or value.get('complete') is not True or value.get('record_id')!=declared['record_id'] or value.get('condition')!=declared['condition'] or value.get('seed')!=declared['seed'] or value.get('source_manifest_sha256')!=source_sha or value.get('release_sha256')!=release_sha or value.get('providers')!=providers:raise ValueError('Exact full M1 scientific completion required')
    for k in ('TEST_access','CORE','masking','HPO','automatic_retry','paper_score_recalculation','further18_activated','confirmation_claim','pure_parameter_match','current_CMCL_or_77_quality_read'):
        if value.get(k) is not False:raise ValueError('M1 completion left frozen exploratory scope')
    if any(value.get(k) is not True for k in ('VALID_access','science_enabled','fresh_training_trajectory','post_screen_exploration','unit_factor_initialization','native_slow_initialization_unchanged')):raise ValueError('Whole fresh full-science control required')
    if (value.get('predictor_parameters'),value.get('native_slow_parameters'),value.get('extra_factor_coordinates'),value.get('factor_sites'),value.get('factor_members'))!=(2083818,2069875,13943,23,1):raise ValueError('Exact one-predictor factorized reparameterization required')
    if value.get('internal_reference_alias')!=declared['internal_reference_alias'] or value.get('max_epochs')!=2000 or value.get('patience')!=250 or value.get('own_native_selector')!=declared['selector']:raise ValueError('Fixed transparent alias/own2000/250 selector required')
    body=value['body'];epochs=value['updates'];selected=value['selected_epoch']
    if type(epochs) is not int or not 1<=selected<=epochs<=2000 or body['epochs_executed']!=epochs or body['selected_epoch']!=selected or body.get('own_selector') is not True or body.get('common_bank_selector') is not False or (epochs<2000 and epochs-selected!=250):raise ValueError('Complete honest own-selected horizon required')
    if value['selected_VALID']!=value['selected_readouts']['VALID']['classification'] or value['selected_prediction_signatures']!=value['selected_readouts']['VALID']['prediction_signatures'] or value['selected_prediction_signatures']!=body['selected_prediction_signatures']:raise ValueError('Immutable selected scalar/signature identity differs')
    for role in ('TRAIN','VALID'):
        check_classification(value['selected_readouts'][role]['classification'],role,1)
        check_repair(value['selected_readouts'][role]['repair'],value['selected_readouts'][role]['classification'],role,1)


def contrast(a,b):
    return dict(accuracy_pp=100*(a['pooled']['accuracy']-b['pooled']['accuracy']),NLL=a['pooled']['NLL']-b['pooled']['NLL'],
                macro_accuracy_pp=100*(a['macro_accuracy']-b['macro_accuracy']),
                mean_member_accuracy_pp=100*(a['mean_member_accuracy']-b['mean_member_accuracy']),
                worst_member_accuracy_pp=100*(a['worst_member_accuracy']-b['worst_member_accuracy']),
                mean_member_NLL=sum(r['NLL'] for r in a['members'])/len(a['members'])-sum(r['NLL'] for r in b['members'])/len(b['members']),
                classes=[dict(class_id=c,accuracy_pp=100*(a['classes'][c]['accuracy']-b['classes'][c]['accuracy']),NLL=a['classes'][c]['NLL']-b['classes'][c]['NLL']) for c in range(3)])


def compare(spec,output,bindings,plan,owners,started):
    output.mkdir(parents=True,exist_ok=False)
    try:
        roster=read(spec['M1_roster']);raw={};readouts={};costs={}
        for declared,row in zip(roster,plan['records']):
            record=declared['record_id'];value=read(spec['records'][record]['complete'])
            m1_record(value,declared,bindings['M1_source_manifest']['sha256'],row['release_sha256'],bindings['frozen_providers'])
            raw[record]=value
            readouts[record]=dict(TRAIN=value['selected_readouts']['TRAIN']['classification'],VALID=value['selected_VALID'],
                TRAIN_repair=value['selected_readouts']['TRAIN']['repair'],VALID_repair=value['selected_readouts']['VALID']['repair'],
                selected_epochs=[value['selected_epoch']],serving_benchmark=value['serving_benchmark'],
                stored_selected_scalar_identity_verified=True,extra_factor_coordinates=13943,predictor_parameters=2083818)
            keys=('body','updates','selected_epoch','timings','inclusive_seconds','CPU_user_seconds','CPU_system_seconds','peak_RSS_bytes','peak_CUDA_bytes',
                  'output_bytes_before_COMPLETE','final_session_counters','model_constructions','preprocessing_banks','Adam_constructions','predictor_parameters','extra_factor_coordinates','serving_benchmark','complete_saved_member_logits')
            costs[record]=dict(immutable_result_costs={k:value[k] for k in keys},actual_raw_waited_owner=owners[record])
        prior=read(spec['closed_comparison']);prior_costs=read(spec['closed_costs'])
        if prior.get('schema')!='PubMed-strong-reference-complete12-comparison-v1' or prior.get('complete12') is not True or prior.get('source_manifest_sha256')!=bindings['closed_reference_source_manifest_sha256'] or prior.get('TEST_scored') is not False or prior.get('new_model_forwards')!=0:raise ValueError('Exact closed complete12 selected-scalar report required')
        expected={f'seed{s}__{c}' for s in SEEDS for c in REFERENCES}
        if set(prior['readouts'])!=expected:raise ValueError('Every closed selected reference/anchor required')
        for record,value in prior['readouts'].items():
            members=4 if record.endswith(('shared4_own','independent4_own')) else 1
            check_classification(value['VALID'],'VALID',members);check_classification(value['TRAIN'],'TRAIN',members)
            check_repair(value['VALID_repair'],value['VALID'],'VALID',members);check_repair(value['TRAIN_repair'],value['TRAIN'],'TRAIN',members)
            readouts[record]=value
        engineering={}
        for c,row in spec['engineering_records'].items():
            m1_root=bound(spec['M1_source_manifest']).parent;record='seed9101__'+c;owner_id='engineering__'+record
            if Path(row['complete']['path'])!=m1_root/'engineering'/'cells'/record/'COMPLETE.json' or Path(row['raw_terminal']['path'])!=m1_root/'owners'/owner_id/'RAW_OWNER_TERMINAL.json':raise ValueError('Original two qualifier cost locations required')
            result=read(row['complete'])
            if result.get('schema')!='PubMed-factor1-engineering-complete-v1' or result.get('complete') is not True or result.get('record_id')!=record or result.get('condition')!=c or result.get('seed')!=9101 or result.get('source_manifest_sha256')!=bindings['M1_source_manifest']['sha256'] or result.get('providers')!=bindings['frozen_providers'] or result.get('updates')!=1 or result.get('fresh_weight_restore_verified') is not True or result.get('VALID_access') is not False or result.get('TEST_access') is not False:raise ValueError('Actual complete TRAIN-only qualifier cost result required')
            release=m1_root/'releases'/(owner_id+'.json')
            bound(dict(path=str(release),sha256=result['release_sha256']))
            argv=[bindings['runtime']['python']['path'],'-B','-P',str(m1_root/'run.py'),'--mode','engineering','--release',str(release),'--release-sha256',result['release_sha256']]
            engineering[c]=dict(immutable_result=result,actual_raw_waited_owner=actual_wait(row['raw_terminal'],result['release_sha256'],argv))
        paired=[]
        repair_fields=('oracle_union_correct','all_members_correct','all_members_wrong','pool_wrong_despite_some_member_correct','strict_common_false_rival_nodes','all_wrong_repaired_by_pool','member0_errors_repaired_by_pool','member0_correct_lost_by_pool')
        for seed in SEEDS:
            contrasts={};repairs={}
            for a,b in PAIRS:
                av,bv=readouts[f'seed{seed}__{a}'],readouts[f'seed{seed}__{b}'];key=a+'_minus_'+b
                contrasts[key]=contrast(av['VALID'],bv['VALID'])
                repairs[key]={f:av['VALID_repair'][f]-bv['VALID_repair'][f] for f in repair_fields}
            paired.append(dict(seed=seed,contrasts=contrasts,within_bank_count_differences=repairs))
        means={name:{f:sum(row['contrasts'][name][f] for row in paired)/3 for f in paired[0]['contrasts'][name] if f!='classes'} for name in paired[0]['contrasts']}
        for name in means:
            means[name]['classes']=[dict(class_id=c,**{f:sum(row['contrasts'][name]['classes'][c][f] for row in paired)/3 for f in ('accuracy_pp','NLL')}) for c in range(3)]
        if time.monotonic()-started>=LIMITS['external_active_seconds']:raise TimeoutError('Frozen scalar-reader cap')
        report=dict(schema='PubMed-factor1-complete6-plus-closed12-scalar-comparison-v1',complete_six=True,complete_reference12=True,
                    source_manifest_sha256=spec['source_manifest_sha256'],comparison_release_sha256=spec['_release_sha256'],
                    M1_source_manifest_sha256=bindings['M1_source_manifest']['sha256'],closed_reference_comparison=spec['closed_comparison'],
                    paired_seeds=paired,mean_contrasts=means,all_selected_readouts=readouts,new_M1_custody=spec['records'],
                    scalar_count_consistency_verified=True,new_array_reads=0,new_model_calls=0,new_scores_from_arrays=0,
                    post_screen_exploration=True,confirmation_claim=False,pure_M4_parameter_match=False,
                    parameter_counts=dict(native=2069875,factor1=2083818,shared4=2125647,independent4=8279500),
                    interpretation='M1 reparameterization plus13943 coordinates and coordinate-wise Adam/decay; not a fullM4 parameter-count match or pure causal coupling estimate. Same encountered graph/split/seeds and selected VALID endpoints.',
                    repair_limit='Stored within-bank coverage/repair readouts and their count differences are retained. Exact cross-predictor repair/new-error intersections are unavailable from scalars and are not inferred.',
                    scientific_gate_tolerance_added=False,score_or_selector_recalculation=False,TEST_access=False,
                    CORE_continuation_eligible=False,further18_activated=False,current_CMCL_or_77_quality_read=False,paper_score_recalculation=False)
        ledger=dict(schema='PubMed-factor1-complete6-plus-prior-actual-cost-ledger-v1',new_M1_scientific=costs,new_M1_engineering=engineering,
                    existing_closed_reference_ledger=prior_costs,existing_closed_ledger_binding=spec['closed_costs'],
                    reader=dict(inclusive_seconds=time.monotonic()-started,CPU_user_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime,
                        CPU_system_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_stime,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,GPU_model_or_tensor_calls=0),
                    additive_rule='Keep the already incurred closed12 ledger once; add new2 qualifiers/new6 science owners and this reader owner once. Do not add historical anchor subsets twice or add nested/inclusive phase times together. Every failure, owner cleanup, output/log and transport cost remains reportable.',
                    reader_external_wait_cleanup_transport_to_be_appended_by_root=True,serving_limit='M1 and new native references have the frozen3/10 warm benchmark. Historical shared anchors do not; no measured end-to-end shared/M1 latency ratio is certified.')
        write(output/'COMPARISON.json',report);write(output/'COSTS.json',ledger)
        if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>LIMITS['output_bytes']:raise RuntimeError('Scalar output cap')
        write(output/'COMPLETE.json',dict(complete=True,schema='PubMed-factor1-scalar-reader-complete-v1',comparison_sha256=sha(output/'COMPARISON.json'),costs_sha256=sha(output/'COSTS.json'),
            comparison_release_sha256=spec['_release_sha256'],inclusive_seconds=time.monotonic()-started,all_six_owned_closed=True,scalar_only=True,owner_success_not_inferred=True))
        return report
    except BaseException as error:
        write(output/'FAILURE.json',dict(complete=False,error_type=type(error).__name__,error=str(error),inclusive_seconds=time.monotonic()-started,
              partial_family_comparison_allowed=False,automatic_retry=False,TEST_access=False,current_CMCL_or_77_quality_read=False))
        raise


def main():
    started=time.monotonic();p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release',type=Path,required=True);p.add_argument('--release-sha256',required=True)
    args=p.parse_args();spec,output,bindings,plan,owners=admit(args.release,args.release_sha256)
    result=compare(spec,output,bindings,plan,owners,started)
    print(json.dumps(dict(complete_six=result['complete_six'],output=str(output),scalar_only=True,owner_success_not_inferred=True)))


if __name__=='__main__':main()
