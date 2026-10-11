"""Complete fixed 42-bank/69-body panel, then label-free selected export."""
import json
import math
from pathlib import Path
import time
from runtime import BLOCKS,HERE,Runtime,checked,load,receipt,require,write

POLICIES={'IID':'iid','sign_pair':'activation_sign_paired','uniform_balance':'balanced_08'}

def roster():
    proposal=json.loads((HERE/'STUDY_PROPOSAL.json').read_text())
    arms=proposal['roster']+proposal['references']
    require(len(arms)==14 and sum(a['optimizer_bodies_per_block'] for a in arms)==23,'Fixed scientific roster')
    return arms

def validate_readiness(config):
    ready=json.loads(checked(config['qualification']['result']).read_text())
    owner=json.loads(checked(config['qualification']['owner_end']).read_text())
    require(ready['complete'] and ready['full_input_candidate_updates']==1 and ready['full_input_coupled_rounds']==1 and ready['total_optimizer_updates']==5 and ready['total_backwards']==8 and ready['role_custody']==config['custody'],'Same fresh-role finite full-input qualification')
    require(ready['custody']['source_manifest']==config['qualification_manifest'],'Same sealed qualification source')
    require(owner['status']=='COMPLETE_EXIT_0' and owner['exit_code']==0 and owner['child_pid_absent'] and owner['owned_cuda_pid_absent'] and owner['direct_child_wait'],'Qualification owner closed/reaped')

def selector(rt,model,placement,members,roles):
    raw=rt.raw(model,placement,members)
    value=int(raw[:,roles['select_ids']].mean(0).argmax(-1).eq(roles['select_y']).sum().item())
    del raw;return value

def native_fit(rt,record,folder,model,optimizer,stream,generator,placement,members,roles,mode):
    """Original strict saved-best/off-cadence quirk, without changing its law."""
    counts=record['counts'];streams=[stream];best=-math.inf;stale=selected=0;tick=time.monotonic()
    sampler=None if generator is None else rt.samplers.make_mask_sampler(POLICIES[record['law']],torch=rt.t,generator=generator)
    with (folder/'trace.jsonl').open('x') as trace:
        for step in range(1,5001):
            rt.guard();record.update(status='running',completed_updates=step-1)
            if mode=='original':
                loss=rt.original.train_step_tabm(model,rt.data,optimizer,False);streams[0]=rt.rng()
            else:
                with rt.body_scope(streams,0):
                    if mode in ('joint','MS4'):
                        loss=rt.joint.train_step_joint_routes(model,rt.data,rt.data.x,roles['train_ids'],roles['train_y'],optimizer,placement=placement,torch=rt.t,functional=rt.F,mask_sampler=sampler,multisample_single=mode=='MS4')
                    else:
                        model.train();optimizer.zero_grad()
                        out=rt.joint.native_route_forward(model,rt.data,rt.data.x,0,placement=placement,torch=rt.t)
                        ce=rt.F.cross_entropy(out[roles['train_ids']],roles['train_y']);ce.backward();optimizer.step();loss=ce.item();del out,ce
            require(math.isfinite(loss),'Finite own TRAIN CE')
            copies=4 if mode in ('original','joint','MS4') else 1
            counts['TRAIN_routes']+=copies;counts['backwards']+=copies;counts['optimizer_updates']+=1
            counts['detached_prefix_routes']+=4 if mode in ('joint','MS4') else 0
            correct=selector(rt,model,placement,members,roles);counts['SELECT_routes']+=members
            improving=correct/150>best;stale=0 if improving else stale+1;saved=improving and step%10==0
            if saved:
                best,selected=correct/150,step
                meta=dict(block=record['block'],arm=record['arm'],body=record['body'],selected_step=step,SELECT_correct=correct,custody=record['custody'])
                rt.save(folder/'selected.pt',[model],[optimizer],streams,generator,meta);counts['checkpoint_writes']+=1
                record.update(selected_step=step,selected_meta=meta,checkpoint=receipt(folder/'selected.pt'))
            trace.write(json.dumps(dict(update=step,TRAIN_own_CE=loss,SELECT_correct=correct,strict_above_saved_best=improving,checkpoint_written=saved,selected_step=selected,stale=stale),allow_nan=False)+'\n');trace.flush()
            record.update(completed_updates=step,stale_updates=stale);write(folder/'PROGRESS.json',dict(update=step,selected_step=selected,stale=stale));rt.guard()
            if stale>=300:break
    require(selected>0,'Native own selector must close a checkpoint')
    record.update(status='selected_complete',stopping='300_stale' if stale>=300 else '5000_cap',acquisition_seconds=time.monotonic()-tick,own_optimizer=True)
    write(folder/'FIT.json',record)

def coupled_fit(rt,bank,folder,roles):
    models=[];optimizers=[];streams=[]
    for m in range(4):
        model,opt,stream=rt.construct(bank['block']+1000003*m,1);models.append(model);optimizers.append(opt);streams.append(stream)
    generator=rt.generator(bank['block']);sampler=rt.samplers.make_mask_sampler('activation_sign_paired',torch=rt.t,generator=generator)
    active=[True]*4;best=[-math.inf]*4;stale=[0]*4;selected=[0]*4;started=time.monotonic()
    traces=[(folder/f'body_{m}'/'trace.jsonl').open('x') for m in range(4)]
    try:
        for step in range(1,5001):
            rt.guard();flags=list(active)
            result=rt.independent.coupled_I4_round(models,optimizers,flags,rt.data,rt.data.x,roles['train_ids'],roles['train_y'],torch=rt.t,functional=rt.F,mask_sampler=sampler,native_route_forward=rt.joint.native_route_forward,body_rng_scope=lambda i:rt.body_scope(streams,i))
            bank['detached_prefix_routes']+=result['prefix_forwards'];bank['stopped_companion_prefix_routes']+=result['inactive_companion_prefix_forwards']
            for m in range(4):
                if not flags[m]:continue
                record=bank['fits'][m];record['status']='running';counts=record['counts'];loss=result['losses'][m]
                require(math.isfinite(loss),'Finite coupled full own CE')
                counts['TRAIN_routes']+=1;counts['backwards']+=1;counts['optimizer_updates']+=1
                correct=selector(rt,models[m],'first_normalized_sage_input',1,roles);counts['SELECT_routes']+=1
                improving=correct/150>best[m];stale[m]=0 if improving else stale[m]+1;saved=improving and step%10==0
                if saved:
                    best[m],selected[m]=correct/150,step
                    meta=dict(block=bank['block'],arm=bank['arm'],body=m,selected_step=step,SELECT_correct=correct,custody=record['custody'])
                    rt.save(folder/f'body_{m}'/'selected.pt',[models[m]],[optimizers[m]],[streams[m]],generator,meta);counts['checkpoint_writes']+=1
                    record.update(selected_step=step,selected_meta=meta,checkpoint=receipt(folder/f'body_{m}'/'selected.pt'))
                record.update(completed_updates=step,stale_updates=stale[m])
                traces[m].write(json.dumps(dict(round=step,TRAIN_own_CE=loss,SELECT_correct=correct,checkpoint_written=saved,selected_step=selected[m],stale=stale[m],active_before=flags,stopped_companions_use='LAST acquisition state'),allow_nan=False)+'\n');traces[m].flush()
                if stale[m]>=300 or step==5000:
                    require(selected[m]>0,'Coupled own selector must close')
                    active[m]=False;record.update(status='selected_complete',stopping='300_stale' if stale[m]>=300 else '5000_cap',acquisition_seconds=time.monotonic()-started,own_optimizer=True)
                    write(folder/f'body_{m}'/'FIT.json',record)
            write(folder/'PROGRESS.json',dict(round=step,active=active,updates=[f.get('completed_updates',0) for f in bank['fits']],prefix_routes=bank['detached_prefix_routes'],stopped_companion_prefix_routes=bank['stopped_companion_prefix_routes']))
            rt.guard()
            if not any(active):break
    finally:
        for trace in traces:trace.close()
    require(not any(active),'All four own stopping states close')
    # The live stopped models were never overwritten with selected checkpoints.
    del models,optimizers,streams,generator,model,opt,stream;rt.t.cuda.empty_cache()

def author_inputs(rt):
    t=rt.t
    features=rt.data.x.detach().cpu().to_sparse_coo().coalesce().to(rt.device)
    edges=rt.data.edge_index.detach().cpu()
    adjacency=t.sparse_coo_tensor(edges.flip(0),t.ones(530417),(34493,34493)).coalesce().to(rt.device)
    degrees=t.bincount(adjacency.indices()[0],minlength=34493).float().unsqueeze(1)
    require(adjacency._nnz()==530417 and degrees.min().item()>=1,'Author-port exact common support')
    return features,adjacency,degrees

def author_construct(rt,block,author):
    rt.common.set_seed(block);model=author.model_class(rt.t,rt.F)().to(rt.device);author.inspect_model(model)
    optimizer=author.optimizer_class(rt.t)(model.parameters())
    return model,optimizer,rt.rng()

def author_fit(rt,record,folder,roles,inputs,author):
    model,optimizer,stream=author_construct(rt,record['block'],author);streams=[stream];counts=record['counts'];best=math.inf;stale=selected=0;started=time.monotonic()
    with (folder/'trace.jsonl').open('x') as trace:
        for step in range(1,100001):
            rt.guard();record['status']='running'
            model.train();optimizer.zero_grad();raw=model(*inputs)
            ce=rt.F.cross_entropy(raw[roles['train_ids']],roles['train_y']);penalty=model.regularizer();loss=ce+penalty
            train_ce,train_l2,train_total=ce.item(),penalty.item(),loss.item();require(math.isfinite(train_total),'Finite author TRAIN CE+L2')
            loss.backward();optimizer.step();streams[0]=rt.rng();counts['TRAIN_routes']+=1;counts['backwards']+=1;counts['optimizer_updates']+=1;del raw,ce,penalty,loss
            model.eval()
            with rt.t.no_grad():
                raw=model(*inputs)[roles['select_ids']];select_ce=rt.F.cross_entropy(raw,roles['select_y']);select_l2=model.regularizer();total=(select_ce+select_l2).item();correct=int(raw.argmax(-1).eq(roles['select_y']).sum())
            require(rt.t.isfinite(raw).all().item() and math.isfinite(total),'Finite author SELECT total CE+L2');counts['SELECT_routes']+=1
            accepted=total<=best
            if accepted:
                best,selected,stale=total,step,0
                meta=dict(block=record['block'],arm=record['arm'],body=0,selected_step=step,SELECT_total=total,SELECT_correct=correct,custody=record['custody'])
                rt.save(folder/'selected.pt',[model],[optimizer],streams,None,meta);counts['checkpoint_writes']+=1
                record.update(selected_step=step,selected_meta=meta,checkpoint=receipt(folder/'selected.pt'))
            else:stale+=1
            trace.write(json.dumps(dict(update=step,TRAIN_CE=train_ce,TRAIN_L2=train_l2,TRAIN_total=train_total,SELECT_CE=select_ce.item(),SELECT_L2=select_l2.item(),SELECT_total=total,SELECT_correct=correct,accepted_minimum_including_ties=accepted,selected_step=selected,stale=stale),allow_nan=False)+'\n');trace.flush()
            del raw,select_ce,select_l2
            record.update(completed_updates=step,stale_updates=stale);write(folder/'PROGRESS.json',dict(update=step,selected_step=selected,stale=stale));rt.guard()
            if stale>=50:break
    require(selected>0,'Author own selector must close')
    record.update(status='selected_complete',stopping='50_stale' if stale>=50 else '100000_cap',fixed_source_point=author.POINT,acquisition_seconds=time.monotonic()-started,own_optimizer=True)
    write(folder/'FIT.json',record);del model,optimizer,streams;rt.t.cuda.empty_cache()

def export_bank(rt,bank,folder,roles,author,inputs):
    """Only called after all 69 acquisitions close; every selected body restores."""
    t=rt.t;raw_members=[];arm=bank['arm'];placement=bank['placement']
    for record in bank['fits']:
        if arm=='fresh_source_meanSAGE_M1':model,optimizer,stream=author_construct(rt,bank['block'],author)
        else:model,optimizer,stream=rt.construct(record['model_seed'],4 if arm.startswith('O4_') or arm=='fresh_native_original_sequential_O4' else 1)
        generator=rt.generator(bank['block']) if bank['controlled_sampler'] else None;streams=[stream]
        rt.restore(checked(record['checkpoint']),[model],[optimizer],streams,generator,record['selected_meta'])
        if arm=='fresh_source_meanSAGE_M1':
            require(optimizer.state['tf1_global']['updates']==record['selected_step'],'Author TF1Adam update powers restored')
            model.eval()
            with t.no_grad():
                raw=model(*inputs).unsqueeze(0);selected_total=(rt.F.cross_entropy(raw[0,roles['select_ids']],roles['select_y'])+model.regularizer()).item()
            saved_total=record['selected_meta']['SELECT_total']
            record['SELECT_total_replay_delta']=selected_total-saved_total
            record['SELECT_total_replay_tolerance']=dict(rel_tol=1e-5,abs_tol=1e-7)
            require(math.isclose(selected_total,saved_total,rel_tol=1e-5,abs_tol=1e-7),'Restored author SELECT criterion within disclosed reduction tolerance')
        else:raw=rt.raw(model,placement,4 if arm.startswith('O4_') or arm=='fresh_native_original_sequential_O4' else 1)
        require(t.isfinite(raw).all().item(),'Finite selected raw logits')
        correct=int(raw[:,roles['select_ids']].mean(0).argmax(-1).eq(roles['select_y']).sum())
        require(correct==record['selected_meta']['SELECT_correct'],'Restored own SELECT decision')
        raw_members.append(raw.detach().cpu());record['selected_restore_verified']=True;record['counts']['selected_routes']+=raw.shape[0]
        del model,optimizer,streams,generator,raw;rt.t.cuda.empty_cache();rt.guard()
    raw=t.cat(raw_members).numpy();require(raw.shape==(bank['actual_served_members'],34493,5),'Actual served member count, MS4 is one')
    bank['exports']={}
    for role in ('cal','dev'):
        ids=rt.np.load(checked(rt.custody['files'][f'block_{bank["block"]}/{role}_ids.npy']),allow_pickle=False)
        require(ids.dtype==rt.np.int64 and ids.shape==((150,) if role=='cal' else (34093,)),'Label-free role IDs')
        value=raw[:,ids];native_value=t.from_numpy(value).to(rt.device);native_mean=native_value.mean(0)
        mean=native_mean.cpu().numpy();decision=native_mean.argmax(-1).cpu().numpy()
        path=folder/(role+'_raw.npz');rt.np.savez(path,ids=ids,raw_logits=value,raw_logit_mean=mean,member_decision=value.argmax(-1),raw_logit_decision=decision)
        del native_value,native_mean
        bank['exports'][role]=receipt(path)
    bank.update(status='exported_complete',serving='dropout off; post-mean raw logits',calibration='one scalar after actual raw-logit mean',costs=rt.costs())
    write(folder/'BANK.json',bank)

def run(config,custody):
    validate_readiness(config);output=Path(config['output']);output.mkdir(parents=True,exist_ok=False);arms=roster()
    records=[];counts_keys=('TRAIN_routes','SELECT_routes','backwards','optimizer_updates','detached_prefix_routes','checkpoint_writes','selected_routes')
    for block in BLOCKS:
        for arm in arms:
            aid=arm['id'];n=arm['optimizer_bodies_per_block'];placement=arm.get('placement','stem_pre_gelu')
            bank=dict(block=block,arm=aid,status='unstarted_retained',actual_served_members=arm['actual_served_members'],placement=placement,controlled_sampler=aid.startswith('O4_') or '_MS4_' in aid or aid=='unshared_coupled_I4',detached_prefix_routes=0,stopped_companion_prefix_routes=0,fits=[])
            for m in range(n):
                seed=block+1000003*m if n==4 else block
                bank['fits'].append(dict(block=block,arm=aid,body=m,model_seed=seed,status='unstarted_retained',law=arm.get('law','native'),counts={k:0 for k in counts_keys},custody=dict(admission=custody,role_custody=config['custody'])))
            records.append(bank)
    result=dict(complete=False,records=records,retry=False,subset_rescue=False,CAL_DEV_truth_read=False,selected_before_exports=True,all_exports_before_CAL=True,custody=custody,role_custody=config['custody'])
    rt=None
    try:
        rt=Runtime(config,select=True);author=load('author_model');inputs=None
        for bank in records:
            rt.guard();bank['status']='acquiring';folder=output/(str(bank['block'])+'_'+bank['arm']);folder.mkdir()
            for record in bank['fits']:(folder/f'body_{record["body"]}').mkdir()
            roles=rt.roles(bank['block'],select=True);arm=bank['arm']
            if arm=='unshared_coupled_I4':coupled_fit(rt,bank,folder,roles)
            elif arm=='fresh_source_meanSAGE_M1':
                if inputs is None:inputs=author_inputs(rt)
                author_fit(rt,bank['fits'][0],folder/'body_0',roles,inputs,author)
            else:
                for record in bank['fits']:
                    members=4 if arm.startswith('O4_') or arm=='fresh_native_original_sequential_O4' else 1
                    model,optimizer,stream=rt.construct(record['model_seed'],members)
                    generator=rt.generator(bank['block']) if bank['controlled_sampler'] else None
                    mode='original' if arm=='fresh_native_original_sequential_O4' else ('MS4' if '_MS4_' in arm else ('joint' if arm.startswith('O4_') else 'single'))
                    native_fit(rt,record,folder/f'body_{record["body"]}',model,optimizer,stream,generator,bank['placement'],members,roles,mode)
                    del model,optimizer,stream,generator;rt.t.cuda.empty_cache()
            bank['status']='selected_complete';write(output/'ACQUISITION_PROGRESS.json',dict(closed_banks=sum(b['status']=='selected_complete' for b in records),closed_states=sum(f['status']=='selected_complete' for b in records for f in b['fits']),expected_banks=42,expected_states=69))
        require(all(b['status']=='selected_complete' for b in records) and sum(len(b['fits']) for b in records)==69,'All 69 fresh states before export')
        write(output/'SELECTED_CLOSURE.json',dict(complete=True,banks=42,selected_states=69,CAL_DEV_truth_read=False,checkpoint_receipts=[f['checkpoint'] for b in records for f in b['fits']]))
        for bank in records:
            roles=rt.roles(bank['block'],select=True);export_bank(rt,bank,output/(str(bank['block'])+'_'+bank['arm']),roles,author,inputs)
        require(len(records)==42 and all(b['status']=='exported_complete' for b in records),'All 42 label-free exports')
        result.update(complete=True,banks=42,selected_states=69,exports=42,costs=rt.costs())
        result['counts']={k:sum(f['counts'][k] for b in records for f in b['fits']) for k in counts_keys}
        result['counts']['detached_prefix_routes']+=sum(b['detached_prefix_routes'] for b in records)
        result['counts']['stopped_companion_prefix_routes']=sum(b['stopped_companion_prefix_routes'] for b in records)
        limits={'optimizer_updates':630000,'TRAIN_routes':990000,'SELECT_routes':900000,'detached_prefix_routes':480000}
        require(all(result['counts'][k]<=v for k,v in limits.items()),'Fixed theoretical operation ceilings')
    except BaseException as e:
        result['failure']=dict(type=type(e).__name__,message=str(e));result['costs']=None if rt is None else rt.costs()
        for bank in records:
            if bank['status']=='acquiring':bank['status']='failed_retained'
            for record in bank['fits']:
                if record['status']=='running':record['status']='failed_retained'
        write(output/'PANEL.json',result);raise
    write(output/'PANEL.json',result);return result
