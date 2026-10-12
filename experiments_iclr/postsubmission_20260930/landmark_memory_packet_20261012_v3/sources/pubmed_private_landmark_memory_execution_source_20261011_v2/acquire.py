"""Complete fresh LM4 roster; SELECT fitting and label-free CAL/DEV exports."""
import json
from pathlib import Path
import time
from learning import decision,own_steps,serve,timed,update
from runtime import ARMS,HERE,NOMINATION_SHA,SEEDS,Runtime,checked,owner_closed,receipt,require,write

ENDPOINTS=tuple('F1_b'+str(b) for b in range(4))+ARMS+('GA4',)

def token(value):return value.replace('[','_').replace(']','')

def add(costs,phase,clock):
    target=costs.setdefault(phase,{k:0. for k in clock})
    for key,value in clock.items():target[key]+=value

def selector(rt,logits,probability,roles):
    predicted=decision(rt,logits,probability,roles['select_ids'])
    correct=predicted.eq(roles['select_y'])
    return int(correct.sum().item()),rt.fp(dict(native_member_argmax=logits.max(-1)[1],native_pool_argmax=probability.argmax(-1),SELECT_correct=correct))

def checkpoint(rt,path,model,optimizer,streams,meta,costs):
    temporary=path.with_suffix('.partial.pt')
    digest,clock=timed(rt,lambda:rt.save(temporary,model,optimizer,streams,meta));temporary.replace(path)
    add(costs,'selected_checkpoint_write',clock);return digest

def fit(rt,config,roles,output,panel,group,row,model,optimizer,streams,forward,train,*,source=False):
    """Own selector/stop/Adam/state; independent body calls are serialized."""
    output.mkdir();row.update(status='running',costs={},operation_counts=dict(updates=0,backwards=0,Adam_steps=0,selector_forwards=0,selected_checkpoint_writes=0,restore_forwards=0))
    maximum,patience=(2000,250) if source else (500,100)
    path=output/'SELECTED_STATE.pt';best=-1;best_step=None;best_signature=None;digest=None;completed=0
    base=dict(schema='PubMed-LM4-own-selected-full-state-v1',seed=group['seed'],endpoint_id=group['endpoint_id'],body=row['body'],stage=group['stage'],source_manifest=config['source_manifest'],prepared_custody=config['prepared_custody'],nomination_manifest_sha256=NOMINATION_SHA)
    def observed(step):
        nonlocal best,best_step,best_signature,digest
        (logits,probability),clock=timed(rt,forward);add(row['costs'],'SELECT_full_serving',clock);row['operation_counts']['selector_forwards']+=1
        correct,signature=selector(rt,logits,probability,roles);improved=correct>best
        if improved:
            best,best_step,best_signature=correct,step,signature
            digest=checkpoint(rt,path,model,optimizer,streams,dict(base,selected_step=step),row['costs']);row['operation_counts']['selected_checkpoint_writes']+=1
        del logits,probability
        return dict(step=step,SELECT_correct=correct,SELECT_count=1313,selected=improved,best_step=best_step,best_correct=best)
    try:
        with (output/'TRACE.jsonl').open('x') as trace:
            if not source:
                observation=observed(0);trace.write(json.dumps(observation,allow_nan=False)+'\n');trace.flush()
            for step in range(1,maximum+1):
                actual,clock=timed(rt,train);add(row['costs'],'own_TRAIN_updates',clock);own_steps(rt,optimizer,step)
                completed=step;row['completed_updates']=step;row['operation_counts']['updates']+=1
                row['operation_counts']['backwards']+=1 if source else model.views;row['operation_counts']['Adam_steps']+=1
                observation=observed(step);observation['TRAIN']=actual
                trace.write(json.dumps(observation,allow_nan=False)+'\n');trace.flush()
                row.update(selected_step=best_step,selected_SELECT_correct=best);write(Path(config['output'])/'PANEL.json',panel);rt.guard()
                if step-best_step>=patience:break
        require(best_step is not None and completed>=1 and (1<=best_step<=completed if source else 0<=best_step<=completed),'Declared earliest strict SELECT maximum and full own stopping')
        restored,clock=timed(rt,lambda:rt.restore(path,model,optimizer,streams,dict(base,selected_step=best_step)));add(row['costs'],'strict_selected_restore',clock)
        require(restored==digest,'Selected whole-state fingerprint differs');own_steps(rt,optimizer,best_step) if best_step else require(not optimizer.state,'Selected checkpoint0 empty Adam')
        (logits,probability),clock=timed(rt,forward);add(row['costs'],'restored_full_serving',clock);row['operation_counts']['restore_forwards']+=1
        correct,signature=selector(rt,logits,probability,roles);require(correct==best and signature==best_signature,'Own selected full native decisions/SELECT masks changed after exact restore')
        row.update(status='selected_complete',checkpoint=receipt(path),checkpoint_fingerprint=digest,selected_step=best_step,completed_updates=completed,selected_SELECT_correct=best,SELECT_count=1313,selected_signature=signature,selected_restore_verified=True,stopped_by='patience'+str(patience) if completed-best_step>=patience else 'max'+str(maximum),max_updates=maximum,patience=patience,checkpoint0_included=not source,selector='first strict own raw-logit SELECT integer maximum' if source or model.views==1 else 'first strict own FP32 probability-mean SELECT integer maximum',pooled_independent_selector=False,trace=receipt(output/'TRACE.jsonl'))
        write(output/'FIT_COMPLETE.json',row);return logits,probability
    except BaseException as error:
        row.update(status='failed_retained',completed_updates=completed,failure=dict(type=type(error).__name__,message=str(error)));raise

def export(rt,roles,output,seed,endpoint_id,logits,probability,view_body_ids):
    """Native FP32 class/body/probability authority; never export role targets."""
    output.mkdir();tick=time.monotonic();views=len(view_body_ids);body_ids=sorted(set(view_body_ids));require(body_ids==list(range(len(body_ids))),'Contiguous native body mapping')
    require(logits.shape==(views,19717,3) and probability.shape==(19717,3) and logits.dtype==probability.dtype==rt.t.float32 and rt.t.isfinite(logits).all().item() and rt.t.isfinite(probability).all().item(),'Complete finite native FP32 endpoint')
    native_members=logits.max(-1)[1];native_pool=native_members[0] if views==1 else probability.argmax(-1);member_probabilities=logits.softmax(-1)
    native_bodies=rt.t.stack([member_probabilities[[v for v,b in enumerate(view_body_ids) if b==body]].mean(0) for body in body_ids])
    native_body_decisions=rt.t.stack([native_members[view_body_ids.index(body)] if view_body_ids.count(body)==1 else native_bodies[body].argmax(-1) for body in body_ids])
    result={}
    for role,count in (('cal',1313),('dev',1316)):
        ids=roles[role+'_ids_numpy'];index=rt.t.from_numpy(ids).to(rt.device)
        arrays=dict(ids=ids,member_log_probs=logits[:,index].log_softmax(-1).cpu().numpy(),native_member_argmax=native_members[:,index].cpu().numpy(),native_pool_probabilities=probability[index].cpu().numpy(),native_pool_argmax=native_pool[index].cpu().numpy(),view_body_ids=rt.np.asarray(view_body_ids,dtype=rt.np.int64),native_body_probabilities=native_bodies[:,index].cpu().numpy(),native_body_argmax=native_body_decisions[:,index].cpu().numpy())
        require(arrays['ids'].shape==(count,) and arrays['member_log_probs'].shape==(views,count,3) and arrays['native_body_probabilities'].shape==(len(body_ids),count,3),'Whole role/member/body export dimensions')
        path=output/(role+'.npz');rt.np.savez(path,**arrays);result[role]=receipt(path)
    rt.guard();return result,dict(label_free_export_seconds=time.monotonic()-tick,roles=['cal','dev'],role_targets_accessed=False)

def inventory():
    groups=[];trajectories=[]
    for seed in SEEDS:
        for endpoint_id in ENDPOINTS[:-1]:
            source=endpoint_id.startswith('F1_');bodies=[int(endpoint_id[-1])] if source else (list(range(4)) if endpoint_id.startswith('I4[') else [0])
            fits=[dict(trajectory_id=str(seed)+'/'+endpoint_id+'/body'+str(body),seed=seed,endpoint_id=endpoint_id,body=body,status='unstarted',completed_updates=0) for body in bodies]
            groups.append(dict(group_id=str(seed)+'/'+endpoint_id,seed=seed,endpoint_id=endpoint_id,arm='F1' if source else endpoint_id,stage='stage1' if source else 'stage2',status='unstarted',fits=fits));trajectories.extend(fits)
    return groups,trajectories

def run(config,custody):
    output=Path(config['output']);output.mkdir(parents=True,exist_ok=False);rt=None;groups,trajectories=inventory()
    endpoints=[];counts=dict(stage1_fit_banks=12,stage1_trajectories=12,stage2_fit_banks=24,stage2_trajectories=42,total_new_fit_banks=36,total_new_trajectories=54,CAL_endpoints=39,CAL_scalar_updates=19500,inherited_fits_or_CAL=0)
    panel=dict(schema='PubMed-LM4-complete-fresh-panel-v1',complete=False,terminal=False,status='running',retry=False,custody=custody,config=config['_config_receipt'],source_manifest=config['source_manifest'],root_decision=config['root_decision'],engineering_policy=config['engineering_policy'],prepared_custody=config['prepared_custody'],prepare_owner_end=config['prepare_owner_end'],qualification=config['qualification'],qualification_owner_end=config['qualification_owner_end'],owner_start=receipt(Path(config['owner_output'])/'OWNER_START.json'),expected_owner_end_path=str(Path(config['owner_output'])/'OWNER_END.json'),nomination_manifest_sha256=NOMINATION_SHA,seeds=list(SEEDS),arms=list(ARMS),endpoint_ids=list(ENDPOINTS),groups=groups,trajectories=trajectories,endpoints=endpoints,records=endpoints,counts=counts,expected_counts=dict(fresh_banks=36,optimizer_trajectories=54,endpoints=39,CAL_updates=19500),CAL_DEVELOPMENT_truth_read=False,CAL_DEV_truth_read=False,TEST_read=False,inherited_fits_or_CAL=0,qualification_ancestors_reused=False,serving='mean_class_probabilities',stage1_closed_before_stage2=False)
    write(output/'PANEL.json',panel)
    try:
        ended=owner_closed(config['qualification_owner_end']);require(ended['stage']=='qualify' and ended['source_manifest']==config['source_manifest'] and ended['produced_artifact']==config['qualification'],'Qualifier direct owner/source/produced report linkage');qualification=json.loads(checked(config['qualification']).read_text())
        require(qualification['complete'] and qualification['status']=='pass' and qualification['source_manifest']==config['source_manifest'] and qualification['prepared_custody']==config['prepared_custody'] and qualification['prepare_owner_end']==config['prepare_owner_end'] and qualification['interfaces']==8 and qualification['source_updates']==4 and qualification['stage2_updates']==26 and qualification['checkpoint0_empty_moments_and_full_restore'] and qualification['all_memory_second_step_CE_gradients_verified'],'One successful same-source material full-input qualifier before any fit')
        require(qualification['core_source']==receipt(HERE/'core.py') and qualification['learning_source']==receipt(HERE/'learning.py'),'Qualified core/update bytes changed')
        rt=Runtime(config);roles=rt.roles(select=True,export=True);panel.update(runtime_providers=rt.providers,graph=rt.graph,geometry_custody=config['prepared_custody'],prior_paid_costs=dict(prepare=rt.prepared['costs'],geometry=rt.prepared['geometry_costs'],qualification=qualification['costs']))
        by_key={(g['seed'],g['endpoint_id']):g for g in groups};ancestors={s:[] for s in SEEDS};caches={s:[] for s in SEEDS};source_logits={s:[] for s in SEEDS};source_endpoints={s:[] for s in SEEDS}
        # Close all twelve fresh stage1 acquisitions before any stage2 fit.
        for seed in SEEDS:
            for body in range(4):
                endpoint_id='F1_b'+str(body);group=by_key[seed,endpoint_id];group['status']='running';row=group['fits'][0]
                session,audit=rt.fresh_source(seed,body);model=session.bodies[0];optimizer=session.optimizers[0];streams=session.streams.states
                require(not optimizer.state and session.counters['updates']==0,'Fresh never-fitted source body/Adam')
                def forward():
                    outside=rt.fp(rt.rng());probability,logits=session.factual_probabilities();require(rt.fp(rt.rng())==outside,'Source eval changed outside RNG');rt.guard();return logits,probability
                def train():
                    outside=rt.fp(rt.rng());value=session.train_step(audit=False);require(rt.fp(rt.rng())==outside,'Source update changed outside RNG');rt.guard();return value
                logits,probability=fit(rt,config,roles,output/(str(seed)+'_'+endpoint_id),panel,group,row,model,optimizer,streams,forward,train,source=True)
                row['operation_counts']['exact_native_counters']=dict(session.counters)
                cache,original,cache_record=rt.cache_source(session,seed,body,row['checkpoint']);require(rt.t.equal(original.max(-1)[1],logits[0].max(-1)[1]),'Selected source/cache native prediction custody')
                ancestors[seed].append(model);caches[seed].append(cache);source_logits[seed].append(original)
                exports,costs=export(rt,roles,output/(str(seed)+'_'+endpoint_id+'_exports'),seed,endpoint_id,logits,probability,[0])
                endpoint=dict(seed=seed,endpoint_id=endpoint_id,arm='F1',stage='stage1',origin='fresh_source',status='selected_complete',actual_served_views=1,actual_native_bodies=1,view_body_ids=[0],probability_pool_authority='native FP32 mean of per-view softmax',fits=group['fits'],exports=exports,ancestors=[],construction=audit,cache=cache_record,costs=costs,provenance=dict(source_body=body,fresh_factorized_M1=True,selected_on_new_SELECT=True,qualification_reuse=False))
                group.update(status='selected_complete',construction=audit,cache=cache_record);panel['endpoints'].append(endpoint);source_endpoints[seed].append(endpoint);write(output/'PANEL.json',panel)
                del session,optimizer,streams,logits,probability,original
        require(sum(g['status']=='selected_complete' for g in groups if g['stage']=='stage1')==12,'All twelve fresh own-selected sources closed before stage2')
        panel['all12_stage1_closed_before_stage2']=panel['stage1_closed_before_stage2']=True
        for seed in SEEDS:
            source_bank=rt.t.stack(source_logits[seed]);exports,costs=export(rt,roles,output/(str(seed)+'_GA4_exports'),seed,'GA4',source_bank,source_bank.softmax(-1).mean(0),list(range(4)))
            ancestry=[dict(body=b,endpoint_id='F1_b'+str(b),checkpoint=source_endpoints[seed][b]['fits'][0]['checkpoint']) for b in range(4)]
            panel['endpoints'].append(dict(seed=seed,endpoint_id='GA4',arm='GA4',stage='derived',origin='derived_no_fit',status='selected_complete',actual_served_views=4,actual_native_bodies=4,view_body_ids=list(range(4)),probability_pool_authority='native FP32 mean of four own-selected source softmax',fits=[],exports=exports,ancestors=ancestry,construction=dict(extra_fit=False,pooled_selector=False),costs=costs,provenance=dict(fresh_sources=True,extra_optimizer_updates=0)))
            del source_bank
            for arm in ARMS:
                group=by_key[seed,arm];group['status']='running';tick=time.monotonic();core,optimizers,audit=rt.construct(arm,seed,ancestors[seed],caches[seed]);construction_seconds=time.monotonic()-tick;restored_logits=[]
                for body,(model,optimizer,row) in enumerate(zip(core['models'],optimizers,group['fits'])):
                    streams=rt.streams(seed,body,model.views);require(not optimizer.state,'Fresh independent own stage2 Adam0')
                    logits,probability=fit(rt,config,roles,output/(str(seed)+'_'+token(arm)+'_body'+str(body)),panel,group,row,model,optimizer,streams,lambda:serve(rt,model),lambda:update(rt,model,optimizer,streams,roles))
                    restored_logits.append(logits);del probability,streams
                def assembled():
                    values=rt.t.cat([serve(rt,model)[0] for model in core['models']],dim=0);return values,values.softmax(-1).mean(0)
                (logits,probability),clock=timed(rt,assembled)
                require(rt.t.equal(logits.max(-1)[1],rt.t.cat(restored_logits,dim=0).max(-1)[1]),'Complete bank preserves each own-restored native body decisions')
                mapping=list(range(4)) if core['genuine_independent'] else [0]*core['members'];exports,costs=export(rt,roles,output/(str(seed)+'_'+token(arm)+'_exports'),seed,arm,logits,probability,mapping)
                costs.update(construction_seconds=construction_seconds,complete_restored_serving=clock)
                used=range(4) if core['genuine_independent'] else (0,);ancestry=[dict(body=b,endpoint_id='F1_b'+str(b),checkpoint=source_endpoints[seed][b]['fits'][0]['checkpoint'],cache=source_endpoints[seed][b]['cache']) for b in used]
                endpoint=dict(seed=seed,endpoint_id=arm,arm=arm,stage='stage2',origin='fresh_stage2',status='selected_complete',actual_served_views=core['members'],actual_native_bodies=len(core['models']),view_body_ids=mapping,probability_pool_authority='native FP32 mean of per-view softmax',fits=group['fits'],exports=exports,ancestors=ancestry,construction=audit,costs=costs,provenance=dict(fresh_head_copy_and_memory=True,independent_own_selectors=core['genuine_independent'],qualification_reuse=False))
                group.update(status='selected_complete',construction=audit,costs=costs);panel['endpoints'].append(endpoint);write(output/'PANEL.json',panel)
                del core,optimizers,model,optimizer,logits,probability,restored_logits;rt.t.cuda.empty_cache();rt.guard()
        require(len(groups)==36 and len(trajectories)==54 and len(panel['endpoints'])==39 and all(g['status']=='selected_complete' for g in groups) and all(t['status']=='selected_complete' and t['selected_restore_verified'] for t in trajectories),'Whole fresh36-bank54-trajectory39-endpoint closure')
        require({(e['seed'],e['endpoint_id']) for e in panel['endpoints']}=={(s,e) for s in SEEDS for e in ENDPOINTS},'Every fixed fresh endpoint exactly once')
        panel.update(complete=True,terminal=True,status='acquisition_complete',actual_counts=dict(fresh_banks=36,optimizer_trajectories=54,endpoints=39,CAL_updates=0),CAL_reader_pending=True,source_cache_records=rt.cache_records,automatic_quality_interpretation=False)
    except BaseException as error:
        panel.update(status='failed_retained',failure=dict(type=type(error).__name__,message=str(error)))
        for group in groups:
            if group['status']=='running':group['status']='failed_retained'
        raise
    finally:
        if rt is not None:panel['costs']=rt.costs()
        write(output/'PANEL.json',panel)
    return panel
