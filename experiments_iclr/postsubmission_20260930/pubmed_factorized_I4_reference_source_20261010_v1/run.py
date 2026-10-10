"""Disabled exact M1 fits plus new four-separate-session committee serving."""
import argparse
import gc
import json
from pathlib import Path
import resource
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import HERE,PHASE,SEEDS,SELECTOR,admit,anchor_static,bind,closed_stage,descriptor,load,read,sha,stage_result,write


def readouts(metrics,logits,roles):
    probabilities=logits.softmax(-1).mean(0)
    return {role:dict(classification=metrics.classification(logits,probabilities,*ids),
                      repair=metrics.repair_diagnostics(logits,probabilities,*ids),
                      prediction_signatures=metrics.signatures(logits,probabilities,*ids)) for role,ids in roles.items()}


def verify_selected(metrics,actual,expected):
    if set(actual)!=set(expected):raise ValueError('Complete selected role roster required')
    for role in expected:
        if actual[role]['prediction_signatures']!=expected[role]['prediction_signatures']:raise ValueError('Exact selected prediction/correct-mask identity changed')
        metrics.verify_counts(actual[role]['classification'],expected[role]['classification'])
        metrics.verify_floats(actual[role]['classification'],expected[role]['classification'])


def saved_logits(np,torch,row,members):
    with np.load(bind(row),allow_pickle=False) as archive:
        if set(archive.files)!={'factual_member_logits'}:raise ValueError('Factual-only saved logit payload required')
        array=archive['factual_member_logits'].copy()
    if array.dtype!=np.float32 or array.shape!=(members,19717,3) or not np.isfinite(array).all():raise ValueError('Complete finite original float32 bank required')
    return torch.from_numpy(array).to('cuda:0')


def save_logits(np,logits,path):
    np.savez_compressed(path,factual_member_logits=logits.detach().cpu().numpy())
    return dict(**descriptor(path),factual_shape=list(logits.shape),server_only=True,contains_labels_or_role_ids=False)


def restore(session,torch,checkpoint,identity):
    selected=torch.load(bind(checkpoint),map_location='cpu')
    for k,v in identity.items():
        if selected.get(k)!=v:raise ValueError('Original selected checkpoint identity changed')
    if selected.get('schema')!='PubMed-own-selected-native-body-v1' or selected.get('resumable') is not False or selected.get('labels_included') is not False or selected.get('optimizer_history_included') is not False:raise ValueError('Exact own-selected weight-only state required')
    session.bodies[0].load_state_dict(selected['body'],strict=True)
    return selected


class M1Committee:
    """No learning operation: four exact private M1-native sessions, local row0."""
    def __init__(self,torch,sessions):
        if len(sessions)!=4:raise ValueError('Four complete M1 sessions required')
        self.torch,self.sessions=torch,sessions;self.bank_calls=0
        ids=[];storage=[]
        for s in sessions:
            body=s.bodies[0];parameters=list(body.parameters())
            maps=[m for m in body.modules() if isinstance(m,s.factors.FactorLinear)]
            if len(s.bodies)!=1 or len(s.optimizers)!=1 or s.decoder is not None or len(maps)!=23 or any(m.r.shape[0]!=1 or m.s.shape[0]!=1 for m in maps):raise ValueError('Four separate exact M1-native row0 interfaces required')
            if sum(p.numel() for p in parameters)!=2083818 or {id(p) for g in s.optimizers[0].param_groups for p in g['params']}!={id(p) for p in parameters}:raise ValueError('Own M1 parameters/Adam required')
            ids.append({id(p) for p in parameters});storage.append({p.untyped_storage().data_ptr() for p in parameters})
        if any(ids[a]&ids[b] or storage[a]&storage[b] for a in range(4) for b in range(a)):raise ValueError('No trainable parameter/storage sharing between reference bodies')

    def factual_probabilities(self):
        values=[]
        for s in self.sessions:
            with s.factors.member_context(s.bodies[0],0):
                _,logits=s.factual_probabilities()
            if logits.shape!=(1,19717,3) or logits.dtype!=self.torch.float32:raise ValueError('Complete native FP32 member required')
            values.append(logits[0])
        bank=self.torch.stack(values);self.bank_calls+=1
        return bank.softmax(-1).mean(0),bank


def admit_anchors(spec,output,b,data,reuse,engine,metrics,factory,np,torch,tensor,roles,timings):
    originals=anchor_static(b,data,reuse);receipts=[]
    for r in reuse['reused_records']:
        seed=r['seed_block'];value=originals[seed]
        session,clock=engine.timed(torch,lambda:factory.fresh_single('single_native',seed,tensor));engine.add(timings,'anchor_constructor_preprocessing_two_Adams',clock)
        selected,clock=engine.timed(torch,lambda:restore(session,torch,r['selected_predictor'],dict(record_id=r['record_id'],body_id='body0',initialization_seed=seed,factual_dropout_seeds=[seed+300001],selected_epoch=r['selected_epoch'],selector=SELECTOR)))
        engine.add(timings,'anchor_original_selected_restore',clock)
        if selected['selected_VALID']!=r['selected_VALID'] or selected['selected_prediction_signatures']!=r['selected_prediction_signatures']:raise ValueError('Original anchor checkpoint scalar/signature changed')
        (_,logits),clock=engine.timed(torch,session.factual_probabilities);engine.add(timings,'anchor_fresh_selected_replay',clock)
        actual=readouts(metrics,logits,roles);verify_selected(metrics,actual,value['selected_readouts'])
        stored=saved_logits(np,torch,r['complete_saved_member_logits'],1)
        verify_selected(metrics,readouts(metrics,stored,roles),value['selected_readouts'])
        receipts.append(dict(seed_block=seed,committee_body_id='body0',original_record_id=r['record_id'],original_checkpoint_identity_preserved=True,
                             complete=r['complete_custody']['complete'],raw_terminal=r['complete_custody']['raw_terminal'],selected_predictor=r['selected_predictor'],saved_logits=r['complete_saved_member_logits'],
                             initialization_seed=seed,factual_dropout_seeds=[seed+300001],selected_epoch=r['selected_epoch'],selected_readouts=value['selected_readouts'],
                             exact_prediction_and_count_replay=True,original_restore_float_tolerance=2e-6,max_abs_replay_saved_logit=float((logits-stored).abs().max().item())))
        del session,selected,logits,stored;gc.collect();torch.cuda.empty_cache()
    return dict(all_three_admitted=True,anchors=receipts,model_constructions=3,preprocessing_banks=3,Adam_constructions=6,new_updates=0,
                fresh12_fallback=False,VALID_access=True,replay_is_not_new_quality_or_selection=True)


def qualify(spec,output,roster,engine,metrics,factory,np,torch,tensor,roles,timings):
    declared=[r for r in roster if r['seed_block']==9101];sessions=[];weights=[]
    for r in declared:
        s,clock=engine.timed(torch,lambda:factory.fresh_single('single_native',r['initialization_seed'],tensor));engine.add(timings,'four_body_constructor_preprocessing_two_Adams',clock);sessions.append(s)
    bank=M1Committee(torch,sessions)
    for s in sessions:
        _,clock=engine.timed(torch,lambda:s.train_step(audit=False));engine.add(timings,'four_private_full_TRAIN_updates',clock)
        if s.counters['updates']!=1 or s.counters['backwards']!=1 or s.counters['optimizer_steps']!=1:raise ValueError('One unchanged private M1 update per body required')
        weights.append({k:v.detach().cpu().clone() for k,v in s.bodies[0].state_dict().items()})
    (_,logits),clock=engine.timed(torch,bank.factual_probabilities);engine.add(timings,'first_four_body_TRAIN_serving',clock)
    expected=readouts(metrics,logits,roles);update_counters=[dict(s.counters) for s in sessions];del logits
    torch.save(weights,output/'QUALIFIED_FOUR_M1_WEIGHTS.pt');weights=None;sessions=None;bank=None;s=None;gc.collect();torch.cuda.empty_cache()
    weights=torch.load(output/'QUALIFIED_FOUR_M1_WEIGHTS.pt',map_location='cpu');sessions=[]
    for r,state in zip(declared,weights):
        s,clock=engine.timed(torch,lambda:factory.fresh_single('single_native',r['initialization_seed'],tensor));engine.add(timings,'fresh_four_body_constructor_preprocessing_two_Adams',clock)
        _,clock=engine.timed(torch,lambda:s.bodies[0].load_state_dict(state,strict=True));engine.add(timings,'fresh_four_body_weight_restore',clock);sessions.append(s)
    bank=M1Committee(torch,sessions);(_,logits),clock=engine.timed(torch,bank.factual_probabilities);engine.add(timings,'fresh_four_body_TRAIN_replay',clock)
    verify_selected(metrics,readouts(metrics,logits,roles),expected)
    return dict(new_four_body_assembly_restore_verified=True,TRAIN_count=11829,VALID_access=False,TRAIN_quality_is_accuracy_evidence=False,
                model_constructions=8,preprocessing_banks=8,Adam_constructions=16,TRAIN_forwards=4,backwards=4,optimizer_steps=4,serving_or_replay_forwards=8,
                parameter_count=8335272,private_factor_coordinates=55772,all_local_factor_row_indices_zero=True,ordinary_learning_requalification=False,
                updated_session_counters=update_counters,fresh_restore_session_counters=[dict(s.counters) for s in sessions])


def fit(spec,output,engine,metrics,factory,np,torch,tensor,roles,started,timings):
    body=dict(body_id=spec['body_id'],initialization_seed=spec['initialization_seed'],factual_dropout_seeds=spec['factual_dropout_seeds'],
              max_epochs=2000,patience=250,selector=SELECTOR,fresh_full_native_fit=True,reuse_other_record=False)
    run_spec=dict(spec,condition='single_native')
    session,logits,state,row=engine.fit_body(run_spec,body,tensor,roles,output/spec['body_id'],started,timings);del state
    selected=readouts(metrics,logits,roles)
    benchmark=engine.serving_benchmark(session,selected['VALID']['prediction_signatures'],roles,timings)
    payload=save_logits(np,logits,output/'SELECTED_MEMBER_LOGITS.npz')
    checkpoint=dict(row['selected_predictor']);checkpoint['path']=str(output/checkpoint['path'])
    return dict(seed_block=spec['seed_block'],body_id=spec['body_id'],body_index=spec['body_index'],initialization_seed=spec['initialization_seed'],
                factual_dropout_seeds=spec['factual_dropout_seeds'],body=row,selected_epoch=row['selected_epoch'],updates=row['epochs_executed'],
                selected_predictor=checkpoint,selected_readouts=selected,serving_benchmark=benchmark,complete_saved_member_logits=payload,
                predictor_parameters=2083818,native_slow_parameters=2069875,extra_factor_coordinates=13943,model_constructions=1,preprocessing_banks=1,Adam_constructions=2,
                final_session_counters=dict(session.counters),own_selector=True,common_bank_selector=False,VALID_access=True)


def slot_records(spec,b,roster,reuse,manifest_sha):
    closed_stage('science',roster,manifest_sha)
    originals={r['seed_block']:r for r in reuse['reused_records']};slots=[]
    for r in roster:
        if r['seed_block']!=spec['seed_block']:continue
        if r['body_index']==0:
            old=originals[r['seed_block']];value=read(old['complete_custody']['complete'])
            slots.append(dict(roster=r,checkpoint=old['selected_predictor'],payload=old['complete_saved_member_logits'],expected=value['selected_readouts'],
                              checkpoint_identity=dict(record_id=old['record_id'],body_id='body0',initialization_seed=old['initialization_seed'],factual_dropout_seeds=old['factual_dropout_seeds'],selected_epoch=old['selected_epoch'],selector=SELECTOR),selected_epoch=old['selected_epoch'],acquisition='immutable_old_body0'))
        else:
            value=json.loads(stage_result('science',r['record_id']).read_text())
            if value.get('complete') is not True or value.get('schema')!='PubMed-factorized-I4-science-complete-v1' or any(value.get(k)!=r[k] for k in ('seed_block','body_id','body_index','initialization_seed','factual_dropout_seeds')) or value.get('source_manifest_sha256')!=manifest_sha:raise ValueError('Exact new full own-selected M1 body required')
            body=value['body'];epochs=body['epochs_executed'];selected=body['selected_epoch']
            if not 1<=selected<=epochs<=2000 or body['own_selector'] is not True or body['common_bank_selector'] is not False or (epochs<2000 and epochs-selected!=250):raise ValueError('Honest full native own horizon required')
            slots.append(dict(roster=r,checkpoint=value['selected_predictor'],payload=value['complete_saved_member_logits'],expected=value['selected_readouts'],
                              checkpoint_identity=dict(record_id=r['record_id'],body_id=r['body_id'],initialization_seed=r['initialization_seed'],factual_dropout_seeds=r['factual_dropout_seeds'],selected_epoch=selected,selector=SELECTOR),selected_epoch=selected,acquisition='new_full_M1_native'))
    if [s['roster']['body_index'] for s in slots]!=[0,1,2,3]:raise ValueError('All four ordered bodies required')
    for s in slots:bind(s['checkpoint']);bind(s['payload'])
    return slots


def masks(torch,metrics,logits,roles):
    probabilities=logits.softmax(-1).mean(0);out={}
    for role,(ids,truth) in roles.items():
        pool,members=metrics.predictions(logits,probabilities,ids);good=members.eq(truth[None,:])
        true=logits[:,ids].gather(2,truth[None,:,None].expand(len(logits),-1,1));rivals=(logits[:,ids]>true).all(0);rivals.scatter_(1,truth[:,None],False)
        out[role]=dict(ids=ids,truth=truth,pool_predictions=pool,member_predictions=members,member_correct=good,pool_correct=pool.eq(truth),
                       coverage=good.any(0),no_correct=~good.any(0),strict_common_false_rival=rivals.any(1))
    return out


def evidence(np,torch,metrics,logits,roles,output):
    values=masks(torch,metrics,logits,roles);report={};arrays={}
    for role,row in values.items():
        good=row['member_correct'];bits=(good.to(torch.int64)*torch.tensor([1,2,4,8],device=good.device)[:,None]).sum(0)
        count=lambda mask:int(mask.sum().item())
        hist=lambda mask:torch.bincount(bits[mask],minlength=16).detach().cpu().tolist()
        report[role]=dict(correct_member_bitmask_histogram=hist(torch.ones_like(row['truth'],dtype=torch.bool)),
                         class_correct_member_bitmask_histograms=[dict(class_id=c,counts=hist(row['truth']==c)) for c in range(3)],
                         unique_correct_by_member=[count(bits==2**m) for m in range(4)],member_correct=[count(g) for g in good],
                         oracle_coverage=count(row['coverage']),no_correct=count(row['no_correct']),pool_correct=count(row['pool_correct']),
                         served_harm=count(row['coverage']&~row['pool_correct']),all_wrong_pool_rescues=count(row['no_correct']&row['pool_correct']),
                         strict_common_false_rival=count(row['strict_common_false_rival']),
                         pair_disagreement=[dict(a=a,b=b,count=count(row['member_predictions'][a]!=row['member_predictions'][b])) for a in range(4) for b in range(a)])
        for k,v in row.items():arrays[role+'_'+k]=v.detach().cpu().numpy()
    target=output/'SERVER_ONLY_PREDICTION_EVIDENCE.npz';np.savez_compressed(target,**arrays)
    return report,dict(**descriptor(target),server_only=True,contains_only_bound_TRAIN_VALID_truth=True,TEST_truth_or_membership=False)


def assemble(spec,output,b,roster,reuse,manifest_sha,engine,metrics,factory,np,torch,tensor,roles,timings):
    slots=slot_records(spec,b,roster,reuse,manifest_sha);sessions=[]
    for slot in slots:
        s,clock=engine.timed(torch,lambda:factory.fresh_single('single_native',slot['roster']['initialization_seed'],tensor));engine.add(timings,'four_selected_M1_constructors_preprocessing_two_Adams',clock)
        _,clock=engine.timed(torch,lambda:restore(s,torch,slot['checkpoint'],slot['checkpoint_identity']));engine.add(timings,'four_exact_own_weight_restores',clock);sessions.append(s)
    bank=M1Committee(torch,sessions);(_,logits),clock=engine.timed(torch,bank.factual_probabilities);engine.add(timings,'first_complete_four_body_serving',clock)
    for m,slot in enumerate(slots):
        verify_selected(metrics,readouts(metrics,logits[m:m+1],roles),slot['expected'])
        stored=saved_logits(np,torch,slot['payload'],1);verify_selected(metrics,readouts(metrics,stored,roles),slot['expected']);del stored
    selected=readouts(metrics,logits,roles);diagnostics,prediction_evidence=evidence(np,torch,metrics,logits,roles,output)
    benchmark=engine.serving_benchmark(bank,selected['VALID']['prediction_signatures'],roles,timings)
    payload=save_logits(np,logits,output/'SELECTED_MEMBER_LOGITS.npz')
    return dict(seed_block=spec['seed_block'],selected_epochs=[s['selected_epoch'] for s in slots],selected_readouts=selected,within_bank_evidence=diagnostics,
                complete_saved_member_logits=payload,server_only_prediction_evidence=prediction_evidence,slots=[{k:v for k,v in s.items() if k!='expected'} for s in slots],
                serving_benchmark=benchmark,assembly_bank_calls=bank.bank_calls,final_session_counters=[dict(s.counters) for s in sessions],
                predictor_parameters=8335272,native_slow_parameters=8279500,extra_factor_coordinates=55772,model_constructions=4,preprocessing_banks=4,Adam_constructions=8,
                no_training_during_assembly=True,all_local_factor_row_indices_zero=True,original_body0_checkpoint_identity_preserved=True,VALID_access=True)


def flow(torch,a,b):
    if not torch.equal(a['ids'],b['ids']) or not torch.equal(a['truth'],b['truth']):raise ValueError('Exact paired role identity required')
    def counts(mask):
        count=lambda value:int(value.sum().item())
        acquired=mask&a['coverage']
        return dict(count=count(mask),acquired_correct_alternative=count(acquired),acquired_and_served=count(acquired&a['pool_correct']),
                    acquired_but_lost=count(acquired&~a['pool_correct']))
    count=lambda value:int(value.sum().item())
    return dict(served_repairs=count(a['pool_correct']&~b['pool_correct']),introduced_errors=count(~a['pool_correct']&b['pool_correct']),
                net_served_correct=count(a['pool_correct'])-count(b['pool_correct']),net_coverage=count(a['coverage'])-count(b['coverage']),
                coverage_gains=count(a['coverage']&~b['coverage']),coverage_losses=count(~a['coverage']&b['coverage']),
                baseline_no_correct=counts(b['no_correct']),baseline_strict_common_false_rival=counts(b['strict_common_false_rival']),
                new_no_correct_outside_baseline=count(a['no_correct']&~b['no_correct']),
                new_strict_common_false_rival_outside_baseline=count(a['strict_common_false_rival']&~b['strict_common_false_rival']),
                class_flows=[dict(class_id=c,repairs=count((a['truth']==c)&a['pool_correct']&~b['pool_correct']),
                                  harms=count((a['truth']==c)&~a['pool_correct']&b['pool_correct'])) for c in range(3)])


def contrast(a,b):
    return dict(accuracy_pp=100*(a['pooled']['accuracy']-b['pooled']['accuracy']),NLL=a['pooled']['NLL']-b['pooled']['NLL'],
                macro_accuracy_pp=100*(a['macro_accuracy']-b['macro_accuracy']),mean_member_accuracy_pp=100*(a['mean_member_accuracy']-b['mean_member_accuracy']),
                worst_member_accuracy_pp=100*(a['worst_member_accuracy']-b['worst_member_accuracy']),
                classes=[dict(class_id=c,accuracy_pp=100*(a['classes'][c]['accuracy']-b['classes'][c]['accuracy']),NLL=a['classes'][c]['NLL']-b['classes'][c]['NLL']) for c in range(3)])


def compare(spec,output,b,roster,manifest_sha,engine,metrics,np,torch,roles):
    _,admission_owners=closed_stage('admission',roster,manifest_sha);_,qualification_owners=closed_stage('qualification',roster,manifest_sha)
    _,scientific_owners=closed_stage('science',roster,manifest_sha);_,assembly_owners=closed_stage('assembly',roster,manifest_sha)
    closed=read(b['closed_M1_comparison']);prior=read(b['closed_reference_comparison']);old_costs=read(b['closed_M1_costs'])
    if closed.get('complete_six') is not True or prior.get('complete12') is not True:raise ValueError('Immutable complete M1/reference reports required')
    old_terminal=load('_factorized_I4_old_terminal',PHASE/'combination_pubmed_strong_reference_source_20261010_v2'/'compare.py').terminal
    paired=[];assembly_results={};joint={}
    for seed in SEEDS:
        record=f'seed{seed}__factorized_I4_native__committee';value=json.loads(stage_result('assembly',record).read_text())
        if value.get('complete') is not True or value.get('source_manifest_sha256')!=manifest_sha or value.get('seed_block')!=seed:raise ValueError('Every exact assembled bank must close')
        logits=saved_logits(np,torch,value['complete_saved_member_logits'],4);verify_selected(metrics,readouts(metrics,logits,roles),value['selected_readouts'])
        candidate=masks(torch,metrics,logits,roles);flows={'factor1_native':{role:flow(torch,row,masks(torch,metrics,logits[:1],roles)[role]) for role,row in candidate.items()}}
        arrays={}
        for role,row in candidate.items():
            for k,v in row.items():arrays['factorizedI4_'+role+'_'+k]=v.detach().cpu().numpy()
        for condition in ('shared4_own','independent4_own'):
            key=f'seed{seed}__{condition}';custody=b['closed_reference_custody'][key]
            try:
                original=read(custody['complete']);old_terminal(custody['terminal_custody'],original['release_sha256'])
                payload=dict(original['complete_saved_member_logits']);payload['path']=str(bind(custody['complete']).parent/payload['path'])
                old_logits=saved_logits(np,torch,payload,4);actual=readouts(metrics,old_logits,roles)
                if actual['VALID']['prediction_signatures']!=original['selected_prediction_signatures']:raise ValueError('Original selected reference signature changed')
                for role in roles:
                    metrics.verify_counts(actual[role]['classification'],prior['readouts'][key][role]);metrics.verify_floats(actual[role]['classification'],prior['readouts'][key][role])
                old_masks=masks(torch,metrics,old_logits,roles);flows[condition]={role:flow(torch,candidate[role],old_masks[role]) for role in roles}
                for role,row in old_masks.items():
                    for k,v in row.items():arrays[condition+'_'+role+'_'+k]=v.detach().cpu().numpy()
                del old_logits,old_masks
            except (OSError,ValueError,KeyError) as error:
                flows[condition]=dict(exact_intersections_available=False,reason=type(error).__name__+': '+str(error),frozen_scalar_contrast_retained=True,no_overlap_inferred=True)
        target=output/f'seed{seed}_SERVER_ONLY_EXACT_JOINT_EVIDENCE.npz';np.savez_compressed(target,**arrays);joint[seed]=dict(**descriptor(target),server_only=True,known_TRAIN_VALID_only=True)
        valid=value['selected_readouts']['VALID']['classification'];contrasts={}
        for condition in ('factor1_native','shared4_own','independent4_own','factor1_mean4_dropout'):
            old=closed['all_selected_readouts'][f'seed{seed}__{condition}']['VALID'];contrasts[condition]=contrast(valid,old)
        paired.append(dict(seed_block=seed,contrasts=contrasts,exact_prediction_flows=flows));assembly_results[seed]=value
        del logits,candidate,arrays
    means={c:{f:sum(row['contrasts'][c][f] for row in paired)/3 for f in paired[0]['contrasts'][c] if f!='classes'} for c in paired[0]['contrasts']}
    report=dict(schema='PubMed-factorized-I4-complete9plus3-comparison-v1',complete_family=True,nine_new_fits=True,three_exact_admitted_body0_anchors=True,three_complete_banks=True,
                paired_seeds=paired,mean_contrasts=means,selected_bank_readouts={s:r['selected_readouts'] for s,r in assembly_results.items()},selected_epochs={s:r['selected_epochs'] for s,r in assembly_results.items()},
                server_only_joint_evidence=joint,new_model_forwards=0,novelty='none',TEST_access=False,paper_score_recalculation=False,confirmation_claim=False,pure_causal_sharing_estimate=False,
                old_prediction_unavailability_rule='Exact old intersections unavailable when artifact/identity cannot be admitted; immutable scalar contrasts retained, never overlaps inferred.',
                claim_limit='One encountered graph/split and three optimization blocks; retrospective body0 reuse and independent own selection. Four slow bodies differ in capacity/work from sharedM4. No novelty, parameter match, unused confirmation or model-success claim.')
    write(output/'COMPARISON.json',report)
    new_fits={r['record_id']:dict(result=json.loads(stage_result('science',r['record_id']).read_text()),actual_waited_owner=scientific_owners[r['record_id']]) for r in roster if r['body_index']>0}
    costs=dict(schema='PubMed-factorized-I4-complete-actual-cost-ledger-v1',new_scientific=new_fits,new_assembly={s:dict(result=r,actual_waited_owner=assembly_owners[r['record_id']]) for s,r in assembly_results.items()},
               existing_closed_M1_six_and_complete12_ledger_once=old_costs,existing_ledger_binding=b['closed_M1_costs'],
               additive_rule='Existing M1-six ledger already contains complete12 once. Add new admission/qualification/nine fits/three assemblies/reader owners once; do not add nested phase timings or old body0 fits twice.',
               new_admission=dict(result=json.loads(stage_result('admission','factorized_I4_anchor_admission').read_text()),actual_waited_owner=admission_owners['factorized_I4_anchor_admission']),
               new_qualification=dict(result=json.loads(stage_result('qualification','factorized_I4_assembly_qualification').read_text()),actual_waited_owner=qualification_owners['factorized_I4_assembly_qualification']),
               root_appends_actual_reader_wait_cleanup_transport=True,serving_limit='Historical shared anchors lack the new3/10 benchmark; no shared/committee latency ratio certified.')
    write(output/'COSTS.json',costs)
    return dict(complete_family=True,comparison=descriptor(output/'COMPARISON.json'),costs=descriptor(output/'COSTS.json'),new_model_forwards=0,VALID_access=True)


def main():
    started=time.monotonic();p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('admission','qualification','science','assembly','comparison'),required=True);p.add_argument('--release',type=Path,required=True);p.add_argument('--release-sha256',required=True)
    args=p.parse_args();spec,output,b,data,roster,reuse=admit(args);output.mkdir(parents=True,exist_ok=False);timings={}
    try:
        m1=load('_factorized_I4_exact_M1',bind(b['M1_source_manifest']).parent/'run.py')
        with m1.reference_surface() as (engine,metrics,factory):
            tick=time.monotonic();np,torch,providers,tensor,roles=engine.runtime_and_data(spec,valid=args.mode!='qualification');engine.add(timings,'exact_full_input_and_numeric_imports',dict(wall_seconds=time.monotonic()-tick))
            if args.mode=='admission':result=admit_anchors(spec,output,b,data,reuse,engine,metrics,factory,np,torch,tensor,roles,timings)
            elif args.mode=='qualification':result=qualify(spec,output,roster,engine,metrics,factory,np,torch,tensor,roles,timings)
            elif args.mode=='science':result=fit(spec,output,engine,metrics,factory,np,torch,tensor,roles,started,timings)
            elif args.mode=='assembly':result=assemble(spec,output,b,roster,reuse,spec['source_manifest_sha256'],engine,metrics,factory,np,torch,tensor,roles,timings)
            else:
                del tensor;result=compare(spec,output,b,roster,spec['source_manifest_sha256'],engine,metrics,np,torch,roles)
            engine.check_bounds(spec,output,started)
            result.update(schema='PubMed-factorized-I4-'+args.mode+'-complete-v1',complete=True,purpose=args.mode,record_id=spec['record_id'],source_manifest_sha256=spec['source_manifest_sha256'],release_sha256=spec['_release_sha256'],
                          providers=providers,selector=SELECTOR,max_epochs=2000,patience=250,timings=timings,inclusive_seconds=time.monotonic()-started,
                          CPU_user_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime,CPU_system_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_stime,
                          peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_CUDA_bytes=dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved()),
                          output_bytes_before_COMPLETE=sum(x.stat().st_size for x in output.rglob('*') if x.is_file()),TEST_access=False,automatic_retry=False,novelty='none',post_screen_exploration=True,
                          paper_score_recalculation=False,confirmation_claim=False,pure_parameter_match=False,owner_success_not_inferred=True)
            write(output/'COMPLETE.json',result)
        print(json.dumps(dict(complete=True,purpose=args.mode,record_id=spec['record_id'],novelty='none',owner_success_not_inferred=True)))
    except BaseException as error:
        write(output/'FAILURE.json',dict(complete=False,purpose=args.mode,record_id=spec['record_id'],error_type=type(error).__name__,error=str(error),inclusive_seconds=time.monotonic()-started,
                                      fresh12_fallback=False,partial_family_comparison_allowed=False,automatic_retry=False,TEST_access=False))
        raise


if __name__=='__main__':main()
