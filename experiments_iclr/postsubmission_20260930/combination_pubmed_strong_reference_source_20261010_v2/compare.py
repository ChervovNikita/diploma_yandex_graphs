"""Disabled complete12 stored-prediction reader, after every new fit/owner closes."""
import argparse
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time
from source import HERE, PHASE, bind, inside, sha, verify_manifest, write
from reference_plan import CONDITIONS, SEEDS, body_roster, SERVER_HOST, SERVER_GPU, SERVER_PHASE

LIMITS=dict(GPU_bytes=2*1024**3,RSS_bytes=2*1024**3,output_bytes=8*1024**2,
            log_bytes=1024**2,external_active_seconds=300,external_cleanup_seconds=10,automatic_retry=False)


def terminal(row,release_sha):
    value=json.loads(bind(row).read_text())
    if 'argv' not in value:
        if value.get('owned_process_absence_verified') is not True or value.get('owned_CUDA_absence_verified') is not True:
            raise ValueError('Historical terminal custody requires confirmed owned absence')
        raw=json.loads(bind(value['raw_owner_terminal']).read_text())
        for k in ('directly_waited','child_exit_code','cap_or_owner_failure'):
            if raw.get(k) != value.get(k):raise ValueError('Hash-bound raw/custody terminal mismatch')
        if raw.get('complete') is not True:raise ValueError('Successful actual raw owner terminal required')
        value=raw
    if value.get('directly_waited') is not True or value.get('child_exit_code') != 0 or value.get('cap_or_owner_failure') is not None:
        raise ValueError('Every actual child must have successful waited owner custody')
    if release_sha not in value.get('argv',[]):raise ValueError('Actual scientific release absent from waited argv')
    return value


def admit_comparison(path,digest):
    if sha(inside(path)) != digest:raise ValueError('Exact new comparison release required')
    spec=json.loads(inside(path).read_text())
    if spec.get('schema') != 'PubMed-strong-reference-complete12-comparison-release-v1':raise ValueError('New complete12 reader release required')
    for k in ('enabled','root_authorized','source_review_approved','all_nine_new_owners_closed','all_three_anchor_owners_closed',
              'complete_reference_roster_frozen','post_screen_exploration','external_hard_bound_confirmed','fresh_resource_readiness_confirmed'):
        if spec.get(k) is not True:raise ValueError('Inactive complete12 reader: '+k)
    for k in ('TEST_access','CORE_continuation_eligible','further18_activated','paper_score_recalculation','new_model_forwards','HPO','automatic_retry'):
        if spec.get(k) is not False:raise ValueError('Reader must preserve closed scope: '+k)
    verify_manifest(spec['source_manifest_sha256'])
    data=json.loads((HERE/'DATA_AND_RUNTIME.json').read_text())
    for k in ('train_bundle','valid_bundle','split_custody','validation_custody','runtime','frozen_providers'):
        if spec.get(k) != data[k]:raise ValueError('Exact data/runtime reader binding differs')
    spec['_train_custody']=json.loads(bind(spec['split_custody']).read_text())
    spec['_validation_custody']=json.loads(bind(spec['validation_custody']).read_text())
    bind(spec['train_bundle']);bind(spec['valid_bundle']);bind(spec['source_review'])
    expected={f'seed{s}__{c}' for s in SEEDS for c in CONDITIONS}
    if set(spec.get('new_records',{})) != expected:raise ValueError('All nine new references must close before any numerical or VALID read')
    frozen_anchors=json.loads((HERE/'ANCHORS.json').read_text())
    if spec.get('anchors') != frozen_anchors['anchors']:raise ValueError('Exact retained prior3 shared-own anchors required')
    if spec.get('anchors_sha256') != sha(HERE/'ANCHORS.json'):raise ValueError('Frozen anchor custody digest required')
    records={};owners={};payloads={}
    for record_id,row in {**spec['new_records'],**spec['anchors']}.items():
        result=json.loads(bind(row['complete']).read_text())
        if result.get('complete') is not True or result.get('record_id') != record_id or result.get('TEST_scored') is not False:
            raise ValueError('Complete declared record without TEST scoring required')
        if record_id in expected:
            if result.get('schema') != 'PubMed-strong-reference-complete-v1' or result.get('source_manifest_sha256') != spec['source_manifest_sha256'] or result.get('private_horizons_complete') is not True:
                raise ValueError('New whole own-selected references required')
            seed,condition=result['seed'],result['condition']
            if record_id != f'seed{seed}__{condition}':raise ValueError('Frozen paired group identity differs')
            frozen=body_roster(seed,condition)
            if len(result['bodies']) != len(frozen):raise ValueError('Every own-selected body required')
            for body,declared in zip(result['bodies'],frozen):
                if any(body.get(k)!=v for k,v in declared.items()):raise ValueError('Private native body provenance changed')
                if not (1<=body['selected_epoch']<=body['epochs_executed']<=2000) or body.get('own_selector') is not True or body.get('common_bank_selector') is not False:
                    raise ValueError('Every full private selector must be honest')
                if body['epochs_executed']<2000 and body['epochs_executed']-body['selected_epoch'] !=250:raise ValueError('Private horizon shortened')
                checkpoint=dict(body['selected_predictor']);checkpoint['path']=str(bind(row['complete']).parent/checkpoint['path']);bind(checkpoint)
        else:
            if result.get('schema') != 'masked-context-PubMed-stage1-complete-v1' or result.get('condition') != 'shared4_own' or result.get('source_manifest_sha256') != frozen_anchors['source_manifest_sha256']:
                raise ValueError('Exact historical own-only anchor required')
        owners[record_id]=terminal(row['terminal_custody'],result['release_sha256'])
        payload=dict(result['complete_saved_member_logits']);payload['path']=str(bind(row['complete']).parent/payload['path'])
        payloads[record_id]=bind(payload);records[record_id]=result
    if spec.get('limits') != LIMITS:raise ValueError('Frozen complete12 reader bounds required')
    owner=json.loads(bind(spec['external_owner_release']).read_text())
    if owner.get('enabled') is not True or owner.get('limits') != LIMITS or owner.get('record_id') != 'strong_reference_complete12':raise ValueError('Separate finite reader owner required')
    for k in ('separate_process_group','direct_wait_required','resource_caps_enforced','output_and_log_caps_enforced'):
        if owner.get(k) is not True:raise ValueError('Reader owner fact missing')
    if owner.get('automatic_retry') is not False:raise ValueError('No reader automatic retry')
    bind(owner['owner_source']);bind(spec['resource_readiness_evidence'])
    if socket.gethostname()!=SERVER_HOST or str(PHASE)!=SERVER_PHASE:raise ValueError('Exact allocation project reader route required')
    uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()
    if uuids!=[SERVER_GPU] or os.environ.get('CUDA_VISIBLE_DEVICES')!=SERVER_GPU:raise ValueError('Exact allocation GPU required')
    runtime=spec['runtime'];python_path=Path(os.path.abspath(runtime['python']['path']))
    if not python_path.is_relative_to(PHASE) or not python_path.is_file() or sha(python_path)!=runtime['python']['sha256'] or Path(sys.executable).resolve()!=python_path.resolve() or os.environ.get('PYTHONPATH','')!=runtime['PYTHONPATH']:
        raise ValueError('Exact reader provider runtime required')
    qualifications=spec.get('engineering_records',{})
    if set(qualifications)!=set(CONDITIONS):raise ValueError('All new engineering acquisition costs required')
    engineering={}
    for condition,row in qualifications.items():
        result=json.loads(bind(row['complete']).read_text())
        if result.get('schema')!='PubMed-strong-reference-TRAIN-only-engineering-v1' or result.get('complete') is not True or result.get('condition')!=condition or result.get('source_manifest_sha256')!=spec['source_manifest_sha256']:
            raise ValueError('Exact complete new engineering result required')
        engineering[condition]=dict(result=result,terminal=terminal(row['terminal_custody'],result['engineering_release_sha256']))
    output=inside(spec['output'],existing=False)
    if output.exists() or output.is_relative_to(HERE):raise ValueError('Fresh separate reader output required')
    spec['_release_sha256']=digest
    return spec,output,records,owners,payloads,engineering


def contrast(a,b):
    fields=dict(accuracy_pp=lambda r:100*r['pooled']['accuracy'],NLL=lambda r:r['pooled']['NLL'],
                macro_accuracy_pp=lambda r:100*r['macro_accuracy'],
                mean_member_accuracy_pp=lambda r:100*r['mean_member_accuracy'],
                worst_member_accuracy_pp=lambda r:100*r['worst_member_accuracy'])
    return {k:f(a)-f(b) for k,f in fields.items()}


def compare(spec,output,records,owners,payloads,engineering,started):
    output.mkdir(parents=True,exist_ok=False)
    try:
        from train import runtime_and_data,check_bounds
        np,torch,providers,tensor,roles=runtime_and_data(spec)
        del tensor
        from metrics import classification,repair_diagnostics,signatures,verify_counts
        readouts={}
        # Complete all12 metadata/owner/payload custody above before first array.
        for record_id,result in records.items():
            check_bounds(spec,output,started)
            with np.load(payloads[record_id],allow_pickle=False) as archive:
                if set(archive.files)!= {'factual_member_logits'}:raise ValueError('Factual-only selected native/anchor payload required')
                array=archive['factual_member_logits'].copy()
            expected=1 if result['condition'] in ('single_native','single_mean4_dropout') else 4
            if array.dtype!=np.float32 or array.shape!=(expected,19717,3):raise ValueError('Complete exact saved fullgraph native bank required')
            logits=torch.from_numpy(array).to('cuda:0');probabilities=logits.softmax(-1).mean(0)
            valid=classification(logits,probabilities,*roles['VALID'])
            if signatures(logits,probabilities,*roles['VALID'])!=result['selected_prediction_signatures']:
                raise ValueError('Saved selected predictions/correct-mask signatures changed; preserve failure')
            verify_counts(valid,result['selected_VALID'])
            # Use immutable selected float metrics for paired scores. Repeat
            # calculations establish exact identity/counts and error diagnostics.
            readouts[record_id]=dict(VALID=result['selected_VALID'],
                VALID_repair=repair_diagnostics(logits,probabilities,*roles['VALID']),
                TRAIN=classification(logits,probabilities,*roles['TRAIN']),
                TRAIN_repair=repair_diagnostics(logits,probabilities,*roles['TRAIN']),
                selected_epochs=[b['selected_epoch'] for b in result['bodies']] if 'bodies' in result else [result['selected_epoch']],
                serving_benchmark=result.get('serving_benchmark'),
                native_raw_logit_single_selector=True,stored_prediction_identity_verified=True)
            del logits,probabilities,array
        conditions=CONDITIONS+('shared4_own',)
        pairs=(('single_mean4_dropout','single_native'),('independent4_own','single_native'),
               ('shared4_own','single_native'),('shared4_own','single_mean4_dropout'),('shared4_own','independent4_own'))
        paired=[]
        for seed in SEEDS:
            rows={c:readouts[f'seed{seed}__{c}']['VALID'] for c in conditions}
            contrasts={a+'_minus_'+b:contrast(rows[a],rows[b]) for a,b in pairs}
            classes={a+'_minus_'+b:[dict(class_id=c,accuracy_pp=100*(rows[a]['classes'][c]['accuracy']-rows[b]['classes'][c]['accuracy']),
                                           NLL=rows[a]['classes'][c]['NLL']-rows[b]['classes'][c]['NLL']) for c in range(3)] for a,b in pairs}
            paired.append(dict(seed=seed,contrasts=contrasts,classes=classes))
        means={name:{field:sum(r['contrasts'][name][field] for r in paired)/3 for field in paired[0]['contrasts'][name]}
               for name in paired[0]['contrasts']}
        result=dict(schema='PubMed-strong-reference-complete12-comparison-v1',complete12=True,
                    source_manifest_sha256=spec['source_manifest_sha256'],comparison_release_sha256=spec['_release_sha256'],
                    readouts=readouts,paired_seeds=paired,mean_contrasts=means,all_record_custody={**spec['new_records'],**spec['anchors']},
                    primary_question='Four-dropout stochastic-gradient averaging versus the ordinary-reference selection/competence gap',
                    post_screen_exploration=True,new_body_trajectories=18,independent_seed_groups=3,
                    independently_selected_native_I4=True,CORE_continuation_eligible=False,further18_activated=False,
                    HPO=False,TEST_scored=False,paper_score_recalculation=False,accuracy_tolerance_added=False,
                    new_model_forwards=0,new_score_gate_or_confirmation_claim=False,
                    claim_limit='Same encountered graph/split/seeds and retained selected anchors; exploratory paired differences only. Extra shared factor coordinates and different learning opportunities remain. No pure causal sharing claim, unused confirmation, novelty or paper-score recalculation.')
        check_bounds(spec,output,started)
        cost_keys=('bodies','timings','assembly_counters','assembly_preprocessing_seconds','final_session_counters',
                   'fresh_training_body_trajectories','body_constructions','preprocessing_banks','predictor_parameters',
                   'decoder_parameters','serving_benchmark','inclusive_seconds','CPU_user_seconds','CPU_system_seconds',
                   'peak_RSS_bytes','peak_CUDA_bytes','output_payload_bytes_before_COMPLETE','complete_saved_member_logits')
        costs=dict(schema='PubMed-strong-reference-complete-costs-v1',
            new_scientific={k:dict(result_costs={f:r[f] for f in cost_keys if f in r},actual_waited_owner=owners[k])
                            for k,r in records.items() if r['condition'] in CONDITIONS},
            new_engineering=engineering,historical_closed_stage1_all9=json.loads((HERE/'PRIOR_COSTS.json').read_text()),
            historical_anchor_subset_nonadditive={k:owners[k] for k,r in records.items() if r['condition']=='shared4_own'},
            reader=dict(inclusive_seconds=time.monotonic()-started,CPU_user_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime,
                        CPU_system_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_stime,
                        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                        peak_CUDA_bytes=dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved())),
            additive_rule='Sum new engineering owners + new scientific owners + reader owner once; report the already incurred entire stage1 screen separately. Never subtract duplicated deterministic fits or count the anchor historical subset twice.',
            serving_limit='New reference benchmarks use the identical frozen3-warmup/10-measured protocol. Historical shared anchors lack that benchmark; their historical serving timings are descriptive and cannot certify an end-to-end latency ratio.',
            CUDA_event_limit='Instrumented CUDA event duration measures elapsed streams in recorded regions, not device occupancy or an exhaustive GPU-utilization integral; inclusive owner wall, CPU, counters and measured peaks remain separately reported.',
            costs_of_failures_and_interrupted_attempts_required=True,external_reader_cleanup_and_transport_costs_root_owned=True)
        write(output/'COMPARISON.json',result);write(output/'COSTS.json',costs)
        return result
    except BaseException as error:
        write(output/'FAILURE.json',dict(complete12=False,error_type=type(error).__name__,error=str(error),
              inclusive_seconds=time.monotonic()-started,partial_family_comparison_allowed=False,automatic_retry=False,
              TEST_access=False,CORE_continuation_eligible=False,further18_activated=False))
        raise


def main():
    started=time.monotonic();p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release',type=Path,required=True);p.add_argument('--release-sha256',required=True)
    args=p.parse_args();spec,output,records,owners,payloads,engineering=admit_comparison(args.release,args.release_sha256)
    result=compare(spec,output,records,owners,payloads,engineering,started)
    print(json.dumps(dict(complete12=result['complete12'],output=str(output),CORE_continuation_eligible=False)))


if __name__=='__main__':main()
