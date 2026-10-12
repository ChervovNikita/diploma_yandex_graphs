"""Fixed9 new P1 then21 GRU/control banks; no CAL or DEVELOPMENT truth."""
import json
import math
from pathlib import Path
import time
from runtime import ARMS,BLOCKS,CORE_SHA,Runtime,checked,owner_closed,receipt,require,write

ENDPOINTS=('P1_b0','P1_b1','P1_b2','P1_b3')+ARMS+('GA4','inherited_WI4')
ACCOUNTING=dict(cohort_fit_banks=33,cohort_optimizer_trajectories=51,cohort_new_fit_banks=30,cohort_new_optimizer_trajectories=48,inherited_P1_fit_banks=3,inherited_P1_optimizer_trajectories=3,external_WI4_fit_banks=3,external_WI4_optimizer_trajectories=12,derived_GA4_rows=3,extra_P1_ancestor_diagnostic_rows=9,serving_rows=39)

def selected_forward(rt,model,roles,stage):
    model.eval()
    with rt.t.no_grad():
        out=rt.native_forward(model) if stage=='stage1' else rt.forward(model)
        views=1 if stage=='stage1' else 4
        require(out.shape==(views,34493,5) and rt.t.isfinite(out).all().item(),'Complete own SELECT forward')
        ce=rt.t.stack([rt.F.cross_entropy(z[roles['select_ids']],roles['select_y']) for z in out]).mean()
        penalty=model.regularizer();values=[float(v.item()) for v in (ce+penalty,ce,penalty)]
        require(all(math.isfinite(v) for v in values),'Finite own SELECT CE+native penalty')
    return out,values

def train_stage1(rt,model,optimizer,streams,roles):
    model.train();optimizer.zero_grad(set_to_none=True);out=rt.native_forward(model,streams)
    ce=rt.F.cross_entropy(out[0,roles['train_ids']],roles['train_y']);penalty=model.regularizer();total=ce+penalty
    values=[float(v.item()) for v in (total,ce,penalty)]
    require(all(math.isfinite(v) for v in values) and rt.t.isfinite(out).all().item(),'Finite ordinary P1 TRAIN objective')
    total.backward();require(all(p.grad is not None and rt.t.isfinite(p.grad).all().item() for p in model.parameters()),'Finite complete ordinary P1 gradients')
    optimizer.step();del out,total,ce,penalty;return values,1

def train_stage2(rt,model,optimizer,streams,roles):
    model.train();optimizer.zero_grad(set_to_none=True);own=[]
    for view in range(4):
        out=rt.view(model,view,streams);ce=rt.F.cross_entropy(out[roles['train_ids']],roles['train_y'])
        require(out.shape==(34493,5) and rt.t.isfinite(out).all().item() and rt.t.isfinite(ce).item(),'Finite complete live own view')
        own.append(float(ce.item()));(ce/4).backward();del out,ce;rt.guard()
    penalty=model.regularizer();value=float(penalty.item());require(math.isfinite(value),'Finite one/mean source native first penalty')
    regularizer_backward=int(penalty.requires_grad)
    if penalty.requires_grad:penalty.backward()
    require(all(p.grad is not None and rt.t.isfinite(p.grad).all().item() for p in model.parameters() if p.requires_grad),'Finite every stage2 live gradient')
    require(all(p.grad is None for p in model.parameters() if not p.requires_grad),'Frozen slow first parameters have no gradients')
    optimizer.step();del penalty;return [sum(own)/4+value,sum(own)/4,value],regularizer_backward

def fit(rt,model,optimizer,streams,roles,folder,meta,stage):
    folder.mkdir();views=1 if stage=='stage1' else 4;best=math.inf;selected=0;stale=0;tick=time.monotonic();selected_meta=None
    counts=dict(TRAIN_views=0,SELECT_views=0,selected_replay_views=0,backward_calls=0,regularizer_backward_calls=0,Adam_steps=0,checkpoint_writes=0,SELECT_checkpoint0_views=0)
    timings=dict(TRAIN_seconds=0.,SELECT_seconds=0.,checkpoint_seconds=0.)
    with (folder/'trace.jsonl').open('x') as trace:
        if stage=='stage2':
            require(optimizer.state['tf1_global']['updates']==0 and len(optimizer.state)==1,'Fresh stage2 Adam checkpoint0')
            start=time.monotonic();out,values=selected_forward(rt,model,roles,stage);del out
            rt.t.cuda.synchronize(rt.device);timings['SELECT_seconds']+=time.monotonic()-start;counts['SELECT_views']+=views;counts['SELECT_checkpoint0_views']=views
            best=values[0];selected_meta={**meta,'selected_step':0,'SELECT_total':values[0],'SELECT_CE':values[1],'SELECT_L2':values[2]}
            start=time.monotonic();rt.save(folder/'selected.pt',model,optimizer,streams,selected_meta);rt.t.cuda.synchronize(rt.device)
            timings['checkpoint_seconds']+=time.monotonic()-start;counts['checkpoint_writes']+=1
            trace.write(json.dumps(dict(update=0,TRAIN_total=None,TRAIN_CE=None,TRAIN_L2=None,SELECT_total=values[0],SELECT_CE=values[1],SELECT_L2=values[2],selected_step=0,stale=0,latest_tied_minimum_accepted=True),allow_nan=False)+'\n');trace.flush();rt.guard()
        for update in range(1,100001):
            rt.guard();outside=rt.fp(rt.rng());start=time.monotonic()
            train,regularizer_calls=(train_stage1 if stage=='stage1' else train_stage2)(rt,model,optimizer,streams,roles)
            require(optimizer.state['tf1_global']['updates']==update and rt.fp(rt.rng())==outside,'One own source Adam update and outside RNG unchanged')
            counts['TRAIN_views']+=views;counts['backward_calls']+=1 if stage=='stage1' else views+regularizer_calls
            counts['regularizer_backward_calls']+=0 if stage=='stage1' else regularizer_calls;counts['Adam_steps']+=1
            rt.t.cuda.synchronize(rt.device);timings['TRAIN_seconds']+=time.monotonic()-start
            start=time.monotonic();out,values=selected_forward(rt,model,roles,stage);del out
            counts['SELECT_views']+=views;rt.t.cuda.synchronize(rt.device);timings['SELECT_seconds']+=time.monotonic()-start
            accepted=values[0]<=best
            if accepted:
                best,selected,stale=values[0],update,0
                selected_meta={**meta,'selected_step':selected,'SELECT_total':values[0],'SELECT_CE':values[1],'SELECT_L2':values[2]}
                start=time.monotonic();rt.save(folder/'selected.pt',model,optimizer,streams,selected_meta);rt.t.cuda.synchronize(rt.device)
                timings['checkpoint_seconds']+=time.monotonic()-start;counts['checkpoint_writes']+=1
            else:stale+=1
            trace.write(json.dumps(dict(update=update,TRAIN_total=train[0],TRAIN_CE=train[1],TRAIN_L2=train[2],SELECT_total=values[0],SELECT_CE=values[1],SELECT_L2=values[2],selected_step=selected,stale=stale,latest_tied_minimum_accepted=accepted),allow_nan=False)+'\n');trace.flush()
            write(folder/'PROGRESS.json',dict(stage=stage,update=update,selected_step=selected,stale=stale));rt.guard()
            if stale>=50:break
    require(selected_meta is not None,'A selected own state must exist')
    start=time.monotonic();digest=rt.restore(folder/'selected.pt',model,optimizer,streams,selected_meta)
    require(optimizer.state['tf1_global']['updates']==selected,'Selected Adam iteration')
    if selected==0:require(len(optimizer.state)==1,'Selected checkpoint0 retains empty per-parameter moments')
    out,values=selected_forward(rt,model,roles,stage);counts['selected_replay_views']+=views
    require(math.isclose(values[0],selected_meta['SELECT_total'],rel_tol=1e-5,abs_tol=1e-7),'Restored SELECT total within inherited reduction tolerance')
    rt.t.cuda.synchronize(rt.device);timings['restoration_and_selected_forward_seconds']=time.monotonic()-start;rt.guard()
    require(counts['TRAIN_views']==views*update and counts['SELECT_views']==views*(update+int(stage=='stage2')) and counts['Adam_steps']==update,'Closed own fit work counts')
    forwards=counts['TRAIN_views']+counts['SELECT_views']+counts['selected_replay_views']
    memory=stage=='stage2' and model.arm!='B4'
    counts.update(native_first_layer_calls=forwards if stage=='stage1' or model.arm=='GN4' else 0,native_tail_layer_calls=forwards,GRU_direction_forward_API_calls=2*forwards if memory else 0,GRU_direction_backward_API_calls=2*counts['TRAIN_views'] if memory else 0)
    record=dict(status='selected_complete',stage=stage,selected_step=selected,completed_updates=update,selected_restore_verified=True,all_module_modes_restore_verified=True,private_and_global_rng_restore_verified=True,selected_state_fingerprint=digest,checkpoint=receipt(folder/'selected.pt'),SELECT_total=selected_meta['SELECT_total'],SELECT_total_replay=values[0],SELECT_total_replay_delta=values[0]-selected_meta['SELECT_total'],SELECT_total_replay_tolerance=dict(rel_tol=1e-5,abs_tol=1e-7),physical_backbones=1,members=views,operation_counts=counts,costs={**timings,'fit_wall_seconds':time.monotonic()-tick},stopping='patience50' if stale>=50 else 'max100000',selector_includes_checkpoint0=stage=='stage2')
    write(folder/'RESULT.json',record);return record,out.detach()

def export(rt,finals,roles,folder):
    tick=time.monotonic();logits=rt.t.cat(finals,dim=0);probabilities=logits.softmax(-1);pool=probabilities.mean(0);lp=logits.log_softmax(-1)
    body_probability=rt.t.stack([z.softmax(-1).mean(0) for z in finals]);mapping=[body for body,z in enumerate(finals) for _ in range(z.shape[0])]
    require(logits.shape[1:]==(34493,5) and body_probability.shape==(len(finals),34493,5),'Full selected actual body/view probability authority')
    exports={}
    for role in ('cal','dev'):
        ids=roles[role+'_ids_numpy'];idx=rt.tensor(ids);path=folder/('selected_'+role+'.npz')
        rt.np.savez(path,ids=ids,member_log_probs=lp[:,idx].cpu().numpy(),native_member_argmax=logits[:,idx].argmax(-1).cpu().numpy(),native_pool_argmax=pool[idx].argmax(-1).cpu().numpy(),native_pool_probabilities=pool[idx].cpu().numpy(),view_body_ids=rt.np.asarray(mapping,dtype=rt.np.int64),native_body_probabilities=body_probability[:,idx].cpu().numpy(),native_body_argmax=body_probability[:,idx].argmax(-1).cpu().numpy())
        exports[role]=receipt(path)
    rt.t.cuda.synchronize(rt.device);rt.guard();return exports,dict(export_seconds=time.monotonic()-tick),mapping

def inherited_panels(config):
    result={};expected_sources={'P1':'f83cb34bae29bda7bd542075bb3009d973fc7ab3bf494bd2cdfcf0801e704cd4','WI4':'ce7d2f5dbabd922a2722afe2f401f65b73c973e783ad6f863d5cf66c5f6dc94d'}
    require(set(config['inherited_sources'])=={'P1','WI4'},'Both mandatory legacy source families')
    for family,source in config['inherited_sources'].items():
        require(set(source)=={'panel','temperatures','acquisition_owner_end','calibration_owner_end','source_manifest'},'Exact inherited panel/CAL/source/owner provenance')
        owner_closed(source['acquisition_owner_end']);owner_closed(source['calibration_owner_end']);checked(source['source_manifest'])
        require(source['source_manifest']['sha256']==expected_sources[family],'Unchanged historical source seal')
        panel=json.loads(checked(source['panel']).read_text());temperatures=json.loads(checked(source['temperatures']).read_text())
        require(panel['complete'] and panel['terminal'] and panel['role_custody']==config['custody'] and temperatures['complete'],'Closed original complete acquisition and CAL')
        rows={}
        for block in BLOCKS:
            matches=[r for r in panel['records'] if r['block']==block and r['arm']==family]
            require(len(matches)==1,'Exact one inherited row per block/family');row=matches[0]
            require(row['status']=='exported_complete' and row['probability_pool_authority'] and len(row['fits'])==(1 if family=='P1' else 4),'Original inherited genuine learner count')
            for fit_record in row['fits']:
                require(fit_record['status']=='selected_complete' and fit_record['selected_restore_verified'],'Original own selected-state closure');checked(fit_record['checkpoint'])
            for item in row['exports'].values():checked(item)
            rows[block]=row
        result[family]=dict(rows=rows,source=source)
    return result

def legacy_record(config,legacy,block,family):
    source=legacy[family]['source'];old=legacy[family]['rows'][block];views=1 if family=='P1' else 4
    return dict(block=block,endpoint_id='P1_b0' if family=='P1' else 'inherited_WI4',arm=family,body=0 if family=='P1' else None,stage='stage1' if family=='P1' else 'external',origin='inherited_P1' if family=='P1' else 'inherited_WI4',status='exported_complete',export_schema='legacy_native_probability_members_v1',actual_served_views=views,actual_native_bodies=views,view_body_ids=list(range(views)),probability_pool_authority=True,fits=old['fits'],exports=old['exports'],ancestors=[],construction=old['construction'],costs=dict(inherited_fit_costs=[f['costs'] for f in old['fits']],new_fit_seconds=0.,new_CAL_steps=0,original_scalar_CAL_steps=500),provenance=dict(inherited_sources=source,original_arm=family,original_row_origin=old.get('origin'),paid_work_preserved=True,reexported=False,recalibrated=False,external_to_nominal_v2_counts=family=='WI4'))

def run(config,custody):
    output=Path(config['output']);output.mkdir(parents=True,exist_ok=False);rt=None
    result=dict(complete=False,terminal=False,retry=False,records=[],CAL_DEV_truth_read=False,TEST_read=False,serving='mean_class_probabilities',role_custody=config['custody'],custody=custody,source_manifest=config['source_manifest'],qualification=config['qualification'],qualification_owner_end=config['qualification_owner_end'],inherited_sources=config['inherited_sources'],expected_roster=[dict(block=b,endpoint_id=e) for b in BLOCKS for e in ENDPOINTS],completed_new_fit_banks=0,completed_new_optimizer_trajectories=0,stage1_closed_before_stage2=False,automatic_CAL_release=False)
    try:
        require(config.get('endpoint_ids')==list(ENDPOINTS),'Exact whole39 serving endpoint roster')
        owner_closed(config['qualification_owner_end']);qualified=json.loads(checked(config['qualification']).read_text());checked(config['qualification_source_manifest'])
        require(qualified['complete'] and qualified['status']=='pass' and qualified['fits']==0 and qualified['TRAIN_updates']==qualified['Adam_steps']==1 and qualified['restored_Adam_updates']==0 and qualified['restored_per_parameter_moment_states']==0 and qualified['SELECT_checkpoints']==2 and qualified['strict_model_optimizer_all_module_modes_global_and_private_rng_restore'] and qualified['cache_immutable'] and not qualified['CAL_DEV_truth_read'],'Closed sole actual full-input qualification')
        require(qualified['role_custody']==config['custody'] and qualified['source_manifest']==config['qualification_source_manifest'] and qualified['custody']['core_manifest_sha256']==CORE_SHA,'Identical passed qualifier core/source/roles')
        binding=json.loads(Path(__file__).with_name('BINDING.json').read_text())
        require(config['qualification_source_manifest']['sha256']==binding['qualification_source_manifest']['sha256'] and config['qualification_source_manifest']['bytes']==binding['qualification_source_manifest']['bytes'],'Exact bound sole full-input worker source')
        require(config['inherited_sources']==binding['inherited_sources'],'Exact preserved original P1 and external WI4 panel/CAL/source/owner receipts')
        require(qualified['backend_operations'].get('aten::_cudnn_rnn')==qualified['backend_operations'].get('aten::_cudnn_rnn_backward')==8,'Qualified full-input backend dispatch')
        legacy=inherited_panels(config);rt=Runtime(config,output);result.update(graph=rt.graph,tour_bindings=rt.tour_bindings,packing_bindings=rt.packing_bindings,canonical_CPU_capsule=rt.capsule_receipt,canonical_CPU_capsule_fingerprint=rt.capsule_fingerprint,persisted_packing_load_and_transfer_seconds=rt.packing_seconds)
        ancestors={};caches={};ancestor_receipts={};role_cache={};stage1_logits={};cache_records=[]
        for block in BLOCKS:
            roles=rt.roles(block);role_cache[block]=roles;model,hidden=rt.inherited_P1(block)
            baseline=legacy_record(config,legacy,block,'P1');require(baseline['fits'][0]['checkpoint']==rt.ancestor_records[-1]['checkpoint'],'Exact original P1 selected ancestor receipt')
            result['records'].append(baseline);result['records'].append(legacy_record(config,legacy,block,'WI4'))
            ancestors[block]=[model];caches[block]=[hidden];ancestor_receipts[block]=[baseline['fits'][0]['checkpoint']];stage1_logits[block]=[None]
            for body in (1,2,3):
                rt.guard();folder=output/f'block_{block}_P1_b{body}';folder.mkdir();core,optimizer,audit=rt.construct_stage1(block,body);model=core['models'][0];streams=rt.stage1_streams(block,body)
                record=dict(block=block,endpoint_id=f'P1_b{body}',arm='P1',body=body,stage='stage1',origin='new_stage1_P1',status='running',export_schema='native_probability_views_v1',actual_served_views=1,actual_native_bodies=1,view_body_ids=[0],probability_pool_authority=True,fits=[],exports={},ancestors=[],construction=audit,costs={},provenance=dict(source_manifest=config['source_manifest'],init_seed=audit['init_seed'],dropout_seed=audit['dropout_seed'],initial_private_stream_fingerprint=rt.fp(streams[0]),reference_only=True))
                result['records'].append(record);meta=dict(block=block,arm='P1',body=body,stage='stage1',role_custody=config['custody'],source_manifest=config['source_manifest'],core_manifest_sha256=CORE_SHA,construction=audit,graph=rt.graph)
                fitted,final=fit(rt,model,optimizer,streams,roles,folder/'fit_0',meta,'stage1');record['fits'].append(fitted)
                exports,costs,mapping=export(rt,[final],roles,folder);record.update(status='exported_complete',exports=exports,costs=costs,view_body_ids=mapping)
                ancestors[block].append(model);ancestor_receipts[block].append(fitted['checkpoint']);stage1_logits[block].append(final)
                result['completed_new_fit_banks']+=1;result['completed_new_optimizer_trajectories']+=1;write(folder/'BANK_RESULT.json',record)
                write(output/'ACQUISITION_PROGRESS.json',dict(stage='stage1',completed_new_fit_banks=result['completed_new_fit_banks'],completed_new_optimizer_trajectories=result['completed_new_optimizer_trajectories'],expected_new_fit_banks=30))
                del core,optimizer,streams;rt.guard()
        require(result['completed_new_fit_banks']==result['completed_new_optimizer_trajectories']==9 and all(len(v)==4 for v in ancestors.values()),'All12 logical selected P1 ancestors close before stage2 references')
        result['stage1_closed_before_stage2']=True;result['stage1_closure']=dict(new_fit_banks=9,new_optimizer_trajectories=9,inherited_fit_banks=3,inherited_optimizer_trajectories=3,selected_ancestor_checkpoints={str(b):ancestor_receipts[b] for b in BLOCKS})
        write(output/'STAGE1_CLOSURE.json',result['stage1_closure'])
        for block in BLOCKS:
            for body in (1,2,3):
                hidden,cache_record=rt.cache_selected_P1(ancestors[block][body],block,body);caches[block].append(hidden);cache_records.append(cache_record)
            roles=role_cache[block]
            for arm in ARMS:
                rt.guard();folder=output/f'block_{block}_{arm}';folder.mkdir();needed=4 if arm in ('GI16','GFI16') else 1
                core,optimizers,audit=rt.construct(arm,block,ancestors[block],caches[block]);views=core['members']
                record=dict(block=block,endpoint_id=arm,arm=arm,stage='stage2',origin='new_GRU_stage2',status='running',export_schema='native_probability_views_v1',actual_served_views=views,actual_native_bodies=needed,view_body_ids=[body for body in range(needed) for _ in range(4)],probability_pool_authority=True,fits=[],exports={},ancestors=ancestor_receipts[block][:needed],construction=audit,costs={},provenance=dict(source_manifest=config['source_manifest'],core_manifest_sha256=CORE_SHA,stage1_closure=receipt(output/'STAGE1_CLOSURE.json'),memory_effective_weights_fresh_per_view=True,first_factor_paths_live_uncached=arm=='GN4',immutable_H_cache=arm!='GN4'))
                result['records'].append(record);finals=[]
                for body,(model,optimizer) in enumerate(zip(core['models'],optimizers)):
                    streams=rt.streams(block,body)
                    meta=dict(block=block,arm=arm,body=body,stage='stage2',role_custody=config['custody'],source_manifest=config['source_manifest'],core_manifest_sha256=CORE_SHA,construction=audit,graph=rt.graph,ancestor=ancestor_receipts[block][body],tail_seed_law='block+459000151+1000003*b+1009*j')
                    fitted,final=fit(rt,model,optimizer,streams,roles,folder/f'fit_{body}',meta,'stage2');record['fits'].append(fitted);finals.append(final)
                    result['completed_new_optimizer_trajectories']+=1;del streams
                exports,costs,mapping=export(rt,finals,roles,folder);record.update(status='exported_complete',exports=exports,costs=costs,view_body_ids=mapping)
                result['completed_new_fit_banks']+=1;write(folder/'BANK_RESULT.json',record)
                write(output/'ACQUISITION_PROGRESS.json',dict(stage='stage2',completed_new_fit_banks=result['completed_new_fit_banks'],completed_new_optimizer_trajectories=result['completed_new_optimizer_trajectories'],expected_new_fit_banks=30))
                del core,optimizers,model,optimizer,final,finals;rt.t.cuda.empty_cache();rt.guard()
            folder=output/f'block_{block}_GA4';folder.mkdir();tick=time.monotonic()
            with rt.t.no_grad():stage1_logits[block][0]=rt.native_forward(ancestors[block][0])
            source_forward_seconds=time.monotonic()-tick;exports,costs,mapping=export(rt,stage1_logits[block],roles,folder)
            record=dict(block=block,endpoint_id='GA4',arm='GA4',stage='derived',origin='derived_GA4',status='exported_complete',export_schema='native_probability_views_v1',actual_served_views=4,actual_native_bodies=4,view_body_ids=mapping,probability_pool_authority=True,fits=[],exports=exports,ancestors=ancestor_receipts[block],construction=dict(resident_parameters=4*538917,optimizer_trajectories=0,genuine_independent_selected_P1_ancestors=True),costs={**costs,'inherited_ordinary_source_forward_seconds':source_forward_seconds,'new_fit_seconds':0.,'scalar_CAL_endpoint_required':True},provenance=dict(source_manifest=config['source_manifest'],derived_from_exact_selected_ordinary_P1_ancestors=True,extra_ancestors_reference_only=True))
            result['records'].append(record);write(folder/'BANK_RESULT.json',record);del stage1_logits[block];rt.guard()
        require((result['completed_new_fit_banks'],result['completed_new_optimizer_trajectories'],len(result['records']))==(30,48,39),'Whole fixed v2 acquisition/export closure')
        require({(r['block'],r['endpoint_id']) for r in result['records']}=={(b,e) for b in BLOCKS for e in ENDPOINTS} and all(r['status']=='exported_complete' for r in result['records']),'Exact whole39 endpoint rows, no partial release')
        new_fits=[f for r in result['records'] if r['origin'] in ('new_stage1_P1','new_GRU_stage2') for f in r['fits']]
        stage2_rows=[r for r in result['records'] if r['stage']=='stage2']
        recurrent_contexts=sum(r['actual_served_views'] for r in stage2_rows if r['arm']!='B4')
        no_memory_contexts=sum(r['actual_served_views'] for r in stage2_rows if r['arm']=='B4')
        require(len(stage2_rows)==21 and sum(len(r['fits']) for r in stage2_rows)==39 and recurrent_contexts==144 and no_memory_contexts==12,'Fixed stage2 banks/trajectories/context inventory')
        count_keys=set(k for f in new_fits for k in f['operation_counts'])
        result.update(complete=True,terminal=True,accounting=ACCOUNTING,stage2_inventory=dict(new_banks=21,new_optimizer_trajectories=39,recurrent_contexts=recurrent_contexts,no_memory_tail_contexts=no_memory_contexts),inherited_P1_restores=rt.ancestor_records,new_H_cache_records=cache_records,new_fit_operation_counts={k:sum(f['operation_counts'].get(k,0) for f in new_fits) for k in sorted(count_keys)},new_fit_costs={k:sum(f['costs'].get(k,0.) for f in new_fits) for k in sorted(set(k for f in new_fits for k in f['costs']))},nominal_CAL_accounting=dict(cohort_endpoints=36,new_endpoints=33,inherited_P1_endpoints=3,new_steps=16500,inherited_P1_steps=1500,total_cohort_steps=18000),external_WI4_accounting=dict(endpoints=3,inherited_steps=1500,new_fit_banks=0,new_optimizer_trajectories=0,new_CAL_steps=0,original_fit_banks=3,original_optimizer_trajectories=12),whole_serving_CAL_accounting=dict(endpoints=39,new_endpoints=33,inherited_endpoints=6,new_steps=16500,inherited_steps=3000,total_bound_steps=19500))
    except BaseException as error:
        result.update(terminal=True,failure=dict(type=type(error).__name__,message=str(error)))
        if result['records'] and result['records'][-1]['status']=='running':result['records'][-1]['status']='failed_retained'
        raise
    finally:
        if rt is not None:result['costs']=rt.costs()
        result['unstarted_retained']=[r for r in result['expected_roster'] if (r['block'],r['endpoint_id']) not in {(v['block'],v['endpoint_id']) for v in result['records']}]
        write(output/'COMPLETE_PANEL.json',result)
    return result
