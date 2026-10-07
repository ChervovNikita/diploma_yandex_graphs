#!/usr/bin/env python3
"""Disabled prospective full-data donor/fit recipes; independent root gates required."""
import copy
import json
from pathlib import Path
import shutil
import time
from common import ROOT,PHASE,guard,runtime,load_data,bound,write,sha,deadline


def main():
    args,job,output=guard(__file__,'fit');torch,versions=runtime()
    from method import (ARMS,NATIVE_ARGUMENTS,construct_native,native_step,cpu_tree,
        ResidualBank,PathSingle,bank_epoch,single_epoch,native_hidden_logits,active_native)
    plan=json.loads(bound(job['plan']).read_text())
    if plan['arms']!=list(ARMS) or plan['native_arguments']!=NATIVE_ARGUMENTS or job.get('fits_authorized') is not True or job.get('VALID_values_access') is not True:
        raise ValueError('Exact representative fixed recipe and fitting roles required')
    data,authority=load_data(torch,job,False)
    x,edge,ids,labels=(data[k] for k in ('x','edge_index','train_ids','train_y'))
    valid_ids,valid_y=data['valid_ids'],data['valid_y'];device=x.device
    output.mkdir();started=time.monotonic();counters=[]

    def finite_update(model,active=None):
        """Inspect the actual ordinary update; no reference forward or extra update."""
        active_count=inactive_count=0
        for name,parameter in model.named_parameters():
            if not bool(torch.isfinite(parameter).all()):
                raise FloatingPointError('Nonfinite updated parameter: '+name)
            if not parameter.requires_grad:continue
            if active is None or active(name):
                if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                    raise FloatingPointError('Disconnected/nonfinite ordinary active gradient: '+name)
                active_count+=1
            else:
                if parameter.grad is not None:raise ValueError('Inactive native parameter has gradient: '+name)
                inactive_count+=1
        return {'finite_updated_parameters':True,'finite_active_gradients':True,
            'inactive_native_gradients_absent':True,'active_tensors':active_count,'inactive_tensors':inactive_count,
            'reference_parity_tested':False,'extra_training_updates':0}

    def score(probabilities):
        if not bool(torch.isfinite(probabilities).all()):raise FloatingPointError('Nonfinite serving probabilities')
        values=probabilities.detach().cpu()[:,valid_ids];pool=values.mean(0)
        yy=torch.nn.functional.one_hot(valid_y,10).double()
        def metric(p):
            p=p.double();selected=p[torch.arange(len(valid_y)),valid_y]
            return {'correct':int((p.argmax(1)==valid_y).sum()),'nodes':len(valid_y),
                    'accuracy':float((p.argmax(1)==valid_y).double().mean()),
                    'NLL':float(-selected.clamp_min(1e-300).log().mean()),
                    'Brier':float((p-yy).square().sum(1).mean())}
        return {'pool':metric(pool),'members':[metric(p) for p in values]}

    def native_probabilities(model):
        model.eval()
        with torch.no_grad():return torch.softmax(model(x,edge),1).unsqueeze(0)

    def row(path):return {'path':str(path.relative_to(PHASE)),'sha256':sha(path)}

    def fit_native(folder,seed,total,resume=None):
        folder.mkdir();model,opt,stream=construct_native(seed,device)
        selected=folder/'selected_checkpoint.pt';first=0;best=-1
        if resume is None:
            torch.save({'model':cpu_tree(model.state_dict()),'stage_global':False,'seed':seed},folder/'INITIAL_STATE.pt')
        else:
            end=torch.load(bound(resume['end']),map_location='cpu',weights_only=True)
            model.load_state_dict(end['model']);opt.load_state_dict(end['optimizer']);stream=copy.deepcopy(end['stream'])
            model._global=end['stage_global'];first=end['epoch']
            if first!=1100 or model._global is not True or end['seed']!=seed:raise ValueError('Complete live native1100 continuation required')
            shutil.copyfile(bound(resume['selected']),selected)
            initial_best=torch.load(selected,map_location='cpu',weights_only=True);best=initial_best['metrics']['pool']['correct']
            write(folder/'CONTINUATION.json',{'live_end_model_Adam_RNG_preserved':True,'eligible_prior_selector_preserved':True,
                'start_epoch':first,'end':resume['end'],'selected':resume['selected']})
        torch.cuda.reset_peak_memory_stats()
        with (folder/'HISTORY.jsonl').open('x') as history:
            for epoch in range(first,total):
                before=time.monotonic()
                if epoch==100:
                    checkpoint=torch.load(selected,map_location='cpu',weights_only=True)
                    if checkpoint['stage_global']:raise ValueError('Selected local transition required')
                    model.load_state_dict(checkpoint['model']);opt.load_state_dict(checkpoint['optimizer'])
                    model._global=True
                    write(folder/'TRANSITION.json',{'selected_local_epoch':checkpoint['epoch'],'model_Adam_restored':True,
                        'live_end_local_RNG_retained':True,'selector_reset':False})
                result=native_step(model,opt,stream,x,edge,ids,labels)
                checks=finite_update(model,lambda name:active_native(name,model._global))
                if epoch in (first,100):
                    write(folder/('TRAIN_SMOKE_'+('global' if model._global else 'local')+'.json'),
                        {'complete':True,'epoch':epoch+1,'seed':seed,'TRAIN':result,'checks':checks,
                         'ordinary_training_epoch':True,'TEST_access':False})
                metrics=score(native_probabilities(model))
                if metrics['pool']['correct']>best:
                    best=metrics['pool']['correct']
                    torch.save({'model':cpu_tree(model.state_dict()),'optimizer':cpu_tree(opt.state_dict()),
                        'stream':cpu_tree(stream),'stage_global':model._global,'epoch':epoch+1,'seed':seed,
                        'metrics':metrics,'native_arguments':NATIVE_ARGUMENTS},selected)
                torch.cuda.synchronize();history.write(json.dumps({'epoch':epoch+1,'stage_global':model._global,
                    'TRAIN':result,'complete_VALID':metrics,'seconds':time.monotonic()-before,
                    'CUDA_peak_allocated':torch.cuda.max_memory_allocated(),'CUDA_peak_reserved':torch.cuda.max_memory_reserved()})+'\n');history.flush()
                write(output/'PROGRESS.json',{'native_seed':seed,'epoch':epoch+1,'complete_target':total,'scores_read_by_owner':False})
                deadline(started,job)
        end_path=folder/'END_STATE.pt'
        torch.save({'model':cpu_tree(model.state_dict()),'optimizer':cpu_tree(opt.state_dict()),'stream':cpu_tree(stream),
            'stage_global':model._global,'epoch':total,'seed':seed,'native_arguments':NATIVE_ARGUMENTS},end_path)
        chosen=torch.load(selected,map_location='cpu',weights_only=True)
        model.load_state_dict(chosen['model']);model._global=chosen['stage_global']
        probabilities=native_probabilities(model)
        if not bool(torch.isfinite(probabilities).all()):raise FloatingPointError('Nonfinite selected native probabilities')
        write(folder/'SELECTED_REPLAY.json',{'selected_recorded':chosen['metrics'],'replayed':score(probabilities),
            'mode_explicitly_restored':True,'reselection':False,'replay_agreement_is_not_training_gate':True,
            'custody':'Selected checkpoint is retained unchanged; output hashes are recorded in NATIVE_FREEZE.'})
        freeze={'complete':True,'epochs':total,'seed':seed,'fresh_constructor_acquisition':resume is None,
            'source_native_sha256':sha(ROOT/'vendor/native_polynormer.py'),'native_arguments':NATIVE_ARGUMENTS,
            'data_manifest_sha256':job['data_manifest']['sha256'],'selected':row(selected),'end':row(end_path),
            'selected_epoch':chosen['epoch'],'selected_global':chosen['stage_global'],'own_selector':True,
            'own_CE_only':True,'pooled_training_or_selection':False,'program_sha256':sha(__file__),
            'source_manifest_sha256':job['source_manifest_sha256'],'selector_evaluations':total,
            'ordinary_Adam_updates_total':total,'TEST_access':False,'kind':'native_fit',
            'integrated_TRAIN_checks':True,'native_reference_parity_pass_claimed':False,
            'engineering_waiver_sha256':plan['engineering_waiver']['sha256'],
            'selected_replay':row(folder/'SELECTED_REPLAY.json')}
        write(folder/'NATIVE_FREEZE.json',freeze);counters.append({'native_seed':seed,'new_native_epochs':total-first,
            'total_native_epochs':total,'source_mode':chosen['stage_global']})
        del model,opt,chosen
        return freeze,probabilities

    def donor_records(expected):
        if len(job['donor_freezes'])!=expected:raise ValueError('Exact donor bank size required')
        records=[]
        for member,item in enumerate(job['donor_freezes']):
            record=json.loads(bound(item).read_text());seed=job['seed']+1009*member if expected==4 else job['seed']
            if (record.get('complete') is not True or record['epochs']!=1100 or record['seed']!=seed
                or record.get('fresh_constructor_acquisition') is not True or record['own_CE_only'] is not True
                or record['pooled_training_or_selection'] is not False or record['own_selector'] is not True
                or record['source_native_sha256']!=sha(ROOT/'vendor/native_polynormer.py')
                or record['native_arguments']!=NATIVE_ARGUMENTS or record['data_manifest_sha256']!=job['data_manifest']['sha256']
                or record['source_manifest_sha256'] not in (job['source_manifest_sha256'],plan['native_donor_manifest_sha256'])):
                raise ValueError('Donor must be independently acquired full native1100 and source VALID-selected')
            external_native=record['source_manifest_sha256']==plan['native_donor_manifest_sha256']
            expected_policy=plan['native_training_engineering_policy_sha256'] if external_native else plan['engineering_waiver']['sha256']
            expected_program=plan['native_donor_program_sha256'] if external_native else sha(__file__)
            if (
                record.get('kind')!='native_fit' or record.get('integrated_TRAIN_checks') is not True
                or record.get('native_reference_parity_pass_claimed') is not False
                or record.get('engineering_waiver_sha256')!=expected_policy
                or record.get('program_sha256')!=expected_program):
                raise ValueError('Exact reviewed actual-fit source and finite-smoke policy required')
            bound(record['selected']);bound(record['end']);records.append(record)
        return records

    def selected_donor(record):
        state=torch.load(bound(record['selected']),map_location='cpu',weights_only=True)
        model,opt,stream=construct_native(record['seed'],device)
        model.load_state_dict(state['model']);model._global=state['stage_global']
        if state['epoch']!=record['selected_epoch'] or model._global!=record['selected_global']:
            raise ValueError('Selected donor mode/epoch mismatch')
        del opt,stream
        return model,state

    def snapshot(model):
        value={'model':cpu_tree(model.state_dict()),'arm':job['arm'],'seed':job['seed']}
        if isinstance(model,ResidualBank):
            value.update(donor_modes=[d._global for d in model.donors],streams=cpu_tree(model.streams),
                optimizers=[cpu_tree(o.state_dict()) for o in model.optimizers],cached_only_frozen_donors=True)
        else:value.update(donor_mode=model.donor._global,base_stream=cpu_tree(model.base_stream),streams=cpu_tree(model.streams),
            native_optimizer=cpu_tree(model.native_optimizer.state_dict()),new_optimizer=cpu_tree(model.new_optimizer.state_dict()),cache=False)
        return value

    try:
        if job['arm'] in ('DONOR_COMMON','DONOR_INDEPENDENT'):
            count=4 if job['arm']=='DONOR_INDEPENDENT' else 1;freezes=[]
            for m in range(count):
                freeze,prob=fit_native(output/('member_'+str(m)),job['seed']+1009*m,1100)
                freezes.append(row(output/('member_'+str(m))/'NATIVE_FREEZE.json'));del prob;torch.cuda.empty_cache()
            write(output/'DONOR_FREEZE.json',{'complete':True,'members':freezes,'kind':'native1100_donors','arm':job['arm'],
                'seed':job['seed'],'TEST_access':False,'inclusive_seconds':time.monotonic()-started,'costs':counters})
            return
        if job['arm'] in ('S_continue','I_native'):
            count=4 if job['arm']=='I_native' else 1;records=donor_records(count);served=[];native_freezes=[]
            for m,record in enumerate(records):
                freeze,prob=fit_native(output/('member_'+str(m)),record['seed'],1200 if count==4 else 1500,resume=record)
                served.append(prob[0].detach().cpu());native_freezes.append(freeze);del prob;torch.cuda.empty_cache()
            bank=torch.stack(served);final_metrics=score(bank)
            selection={'mode':'native_first_max_spanning_source_stages_and_continuation','members':native_freezes,
                       'independent_optimizers':count==4,'pooled_training_or_selection':False}
        else:
            records=donor_records(4 if job['arm']=='U_stage' else 1)
            models=[];states=[]
            for record in records:
                model,state=selected_donor(record);models.append(model);states.append(state)
            if job['arm']=='S_paths':
                live=torch.load(bound(records[0]['end']),map_location='cpu',weights_only=True)
                model=PathSingle(models[0],states[0]['optimizer'],live['stream'],job['seed'],device)
                schedule=[list(range(4))]*100
            else:
                model=ResidualBank(models,job['seed'],device)
                if job['arm'] in ('E_joint','E_own') and job.get('frozen_cache') is None:
                    raise ValueError('Matched common-base controls must reuse the exact pre-correction donor cache')
                if job.get('frozen_cache') is not None:
                    cache_authority=json.loads(bound(job['frozen_cache']).read_text())
                    if job['arm'] not in ('E_joint','E_own'):
                        raise ValueError('Only the two matched common-cache controls load an external cache')
                    origin_record=job.get('cache_origin_E_stage_job')
                    if not origin_record:raise ValueError('Root must bind the exact canonical E_stage producer job')
                    origin_job=json.loads(bound(origin_record).read_text())
                    expected_authority=(Path(origin_job['output_directory'])/'CACHE_FREEZE.json').resolve()
                    if (origin_job.get('root_execution_authorized') is not True or origin_job.get('source_review_approved') is not True
                        or origin_job.get('arm')!='E_stage' or origin_job.get('seed')!=job['seed']
                        or origin_job.get('source_manifest_sha256')!=job['source_manifest_sha256']
                        or origin_job.get('data_manifest')!=job['data_manifest'] or origin_job.get('plan')!=job['plan']
                        or origin_job.get('donor_freezes')!=job['donor_freezes']
                        or bound(job['frozen_cache'])!=expected_authority
                        or cache_authority.get('producer_job')!=origin_record
                        or cache_authority.get('producer_arm')!='E_stage' or cache_authority.get('producer_seed')!=job['seed']
                        or cache_authority.get('producer_program_sha256')!=sha(__file__)
                        or cache_authority.get('source_manifest_sha256')!=job['source_manifest_sha256']
                        or cache_authority.get('frozen_parameters_only') is not True
                        or cache_authority.get('created_before_correction_or_endpoint_scoring') is not True):
                        raise ValueError('Cache must be the exact source/root-pinned pre-correction E_stage export')
                    if cache_authority['donor_selected']!=[r['selected'] for r in records] or cache_authority['donor_modes']!=[d._global for d in model.donors] or cache_authority['data_manifest_sha256']!=job['data_manifest']['sha256']:
                        raise ValueError('Common cache donor/mode/data provenance differs')
                    caches=torch.load(bound(cache_authority['cache']),map_location='cpu',weights_only=True)
                    if len(caches)!=len(model.donors):raise ValueError('Frozen cache bank size differs')
                    model.cache=[]
                    for h,z in caches:
                        if tuple(h.shape)!=(11701,512) or tuple(z.shape)!=(11701,10) or h.dtype!=torch.float32 or z.dtype!=torch.float32 or h.requires_grad or z.requires_grad or not bool(torch.isfinite(h).all() and torch.isfinite(z).all()):
                            raise ValueError('Full finite native cache geometry/type differs')
                        model.cache.append((h.to(device),z.to(device)))
                else:model.make_cache(x,edge)
                cache_file=output/'FROZEN_DONOR_CACHE.pt';torch.save(cpu_tree(model.cache),cache_file)
                write(output/'CACHE_FREEZE.json',{'cache':row(cache_file),'donor_selected':[r['selected'] for r in records],
                    'donor_modes':[d._global for d in model.donors],'data_manifest_sha256':job['data_manifest']['sha256'],
                    'frozen_parameters_only':True,'created_before_correction_or_endpoint_scoring':True,
                    'producer_arm':job['arm'],'producer_seed':job['seed'],'producer_program_sha256':sha(__file__),
                    'source_manifest_sha256':job['source_manifest_sha256'],'producer_job':row(args.job.resolve())})
                initial=model.serving(x,edge)
                expected=torch.stack([torch.softmax(model.donor_cache(m)[1],1) for m in range(4)])
                if not torch.equal(initial,expected):raise ValueError('Zero-residual cached native inclusion is not literal')
                write(output/'INITIAL_INCLUSION.json',{'all_members_literal_cached_donor_probabilities':True,
                    'same_captured_cache_bug_check':True,'native_repeat_reference_is_optional_diagnostic':True,
                    'frozen_donor_modes':[d._global for d in model.donors]})
                schedule=[list(range(4))]*100 if job['arm']=='E_joint' else [[m] for m in range(4) for _ in range(100)]
                del initial,expected
            del states,models
            torch.save(snapshot(model),output/'INITIAL_STATE.pt');torch.cuda.reset_peak_memory_stats()
            smoked_members=set()
            with (output/'CORRECTION_HISTORY.jsonl').open('x') as history:
                for step,members in enumerate(schedule):
                    before=time.monotonic()
                    result=single_epoch(model,x,edge,ids,labels) if job['arm']=='S_paths' else bank_epoch(
                        model,x,edge,ids,labels,members,include_residual=job['arm']!='E_own',diagnostics=step==0 or (step+1)%100==0)
                    if job['arm']=='S_paths':
                        checks=finite_update(model,lambda name:not name.startswith('donor.') or active_native(name[6:],model.donor._global))
                        if step==0:
                            write(output/'TRAIN_SMOKE_single.json',{'complete':True,'correction_epoch':step+1,
                                'TRAIN':result,'checks':checks,'ordinary_training_epoch':True,'TEST_access':False})
                    else:
                        for m in members:
                            checks=finite_update(model.paths[m])
                            if m not in smoked_members:
                                write(output/('TRAIN_SMOKE_member_'+str(m)+'.json'),{'complete':True,
                                    'correction_epoch':step+1,'member':m,'TRAIN':result,'checks':checks,
                                    'ordinary_training_epoch':True,'TEST_access':False})
                                smoked_members.add(m)
                    torch.cuda.synchronize();history.write(json.dumps({'correction_epoch':step+1,'TRAIN':result,
                        'seconds':time.monotonic()-before,'CUDA_peak_allocated':torch.cuda.max_memory_allocated(),
                        'CUDA_peak_reserved':torch.cuda.max_memory_reserved(),'VALID_scoring':False})+'\n');history.flush()
                    write(output/'PROGRESS.json',{'arm':job['arm'],'correction_epoch':step+1,'target_epochs':len(schedule),'scores_read_by_owner':False})
                    deadline(started,job)
            # Freeze every changed predictor before its first endpoint VALID score.
            endpoint=output/'FINAL_STATE.pt';torch.save(snapshot(model),endpoint)
            write(output/'MODEL_FREEZE.json',{'complete':True,'fixed_endpoint':True,'model':row(endpoint),
                'VALID_endpoint_scoring_started':False,'no_correction_checkpoint_selection':True,
                'donor_freezes':job['donor_freezes'],'source_manifest_sha256':job['source_manifest_sha256']})
            if isinstance(model,ResidualBank):
                frozen_model=json.loads((output/'MODEL_FREEZE.json').read_text())
                frozen_model.update(cache_authority=row(output/'CACHE_FREEZE.json'),cache_payload=row(output/'FROZEN_DONOR_CACHE.pt'),
                    imported_cache_authority=job.get('frozen_cache'),cache_origin_E_stage_job=job.get('cache_origin_E_stage_job'))
                write(output/'MODEL_FREEZE.json',frozen_model)
            bank=model.serving(x,edge).detach().cpu();final_metrics=score(bank)
            selection={'mode':'fixed_final_endpoint','correction_path_updates':400,
                'native_prefix_updates':1100*(4 if job['arm']=='U_stage' else 1),
                'cache_only_frozen_donors':job['arm']!='S_paths','all_four_paths_paid':True,
                'extra_checkpoint_rematerialization_paid':job['arm']=='S_paths'}
            del model
        probabilities=output/'FINAL_VALID_PROBABILITIES.pt'
        torch.save({'member_VALID_probabilities':bank[:,valid_ids],'mean_VALID_probabilities':bank.mean(0)[valid_ids],
            'selection':selection,'data_manifest_sha256':job['data_manifest']['sha256']},probabilities)
        write(output/'FREEZE.json',{'complete':True,'arm':job['arm'],'seed':job['seed'],'final_VALID':final_metrics,
            'selection':selection,'probabilities':row(probabilities),'source_manifest_sha256':job['source_manifest_sha256'],
            'program_sha256':sha(__file__),'job_sha256':sha(args.job),'plan_sha256':job['plan']['sha256'],
            'native_costs':counters,'inclusive_seconds':time.monotonic()-started,'TEST_access':False,'retry':False,
            'integrated_TRAIN_checks':True,'strict_reference_parity_pass_claimed':False,
            'engineering_waiver':plan['engineering_waiver']})
        if (output/'MODEL_FREEZE.json').is_file():
            closed=json.loads((output/'FREEZE.json').read_text());closed['model_freeze']=row(output/'MODEL_FREEZE.json')
            write(output/'FREEZE.json',closed)
    except BaseException as error:
        write(output/'FAILURE.json',{'error':type(error).__name__+': '+str(error),'partial_outputs_preserved':True,
            'TEST_access':False,'retry':False,'inclusive_seconds':time.monotonic()-started})
        raise


if __name__=='__main__':main()
