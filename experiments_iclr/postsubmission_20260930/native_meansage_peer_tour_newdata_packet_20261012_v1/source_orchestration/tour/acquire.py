"""Fixed30banks:12prior/18new,21prior/27new trajectories; no DEV release."""
import json
import math
from pathlib import Path
import time
from runtime import ARMS,BLOCKS,NEW_ARMS,Runtime,checked,receipt,require,write

def criterion(rt,model,logits,ids,y):
    ces=rt.t.stack([rt.F.cross_entropy(z[ids],y) for z in logits])
    penalty=model.regularizer();return ces.mean()+penalty,ces.mean(),penalty

def fit(rt,model,optimizer,streams,roles,folder,meta):
    folder.mkdir();best=math.inf;stale=0;selected=0;tick=time.monotonic()
    counts=dict(TRAIN_routes=0,SELECT_routes=0,backwards=0,Adam_steps=0,selected_routes=0,checkpoint_writes=0)
    timings=dict(TRAIN_seconds=0.,SELECT_seconds=0.,checkpoint_seconds=0.)
    with (folder/'trace.jsonl').open('x') as trace:
        for update in range(1,100001):
            rt.guard();start=time.monotonic();model.train();optimizer.zero_grad()
            out=rt.forward(model,streams);counts['TRAIN_routes']+=model.members
            total,ce,penalty=criterion(rt,model,out,roles['train_ids'],roles['train_y'])
            train=[float(v.item()) for v in (total,ce,penalty)];require(all(math.isfinite(v) for v in train),'Finite native TRAIN objective')
            total.backward();counts['backwards']+=1;optimizer.step();counts['Adam_steps']+=1
            del out,total,ce,penalty;rt.t.cuda.synchronize(rt.device);timings['TRAIN_seconds']+=time.monotonic()-start
            model.eval();start=time.monotonic()
            with rt.t.no_grad():
                out=rt.forward(model);total,ce,penalty=criterion(rt,model,out,roles['select_ids'],roles['select_y'])
                select=[float(v.item()) for v in (total,ce,penalty)]
                require(out.shape==(model.members,34493,5) and rt.t.isfinite(out).all().item(),'Complete native SELECT forward')
            counts['SELECT_routes']+=model.members;require(all(math.isfinite(v) for v in select),'Finite SELECT total')
            del out,total,ce,penalty;rt.t.cuda.synchronize(rt.device);timings['SELECT_seconds']+=time.monotonic()-start
            accepted=select[0]<=best
            if accepted:
                best,selected,stale=select[0],update,0;selected_meta={**meta,'selected_step':selected,'SELECT_total':select[0],'SELECT_CE':select[1],'SELECT_L2':select[2]}
                start=time.monotonic();rt.save(folder/'selected.pt',model,optimizer,streams,selected_meta);rt.t.cuda.synchronize(rt.device)
                timings['checkpoint_seconds']+=time.monotonic()-start;counts['checkpoint_writes']+=1
            else:stale+=1
            trace.write(json.dumps(dict(update=update,TRAIN_total=train[0],TRAIN_CE=train[1],TRAIN_L2=train[2],SELECT_total=select[0],SELECT_CE=select[1],SELECT_L2=select[2],selected_step=selected,stale=stale,latest_tied_minimum_accepted=accepted),allow_nan=False)+'\n');trace.flush()
            write(folder/'PROGRESS.json',dict(update=update,selected_step=selected,stale=stale));rt.guard()
            if stale>=50:break
    start=time.monotonic();state=rt.restore(folder/'selected.pt',model,optimizer,streams,selected_meta)
    require(optimizer.state['tf1_global']['updates']==selected,'Selected source Adam iteration differs')
    model.eval()
    with rt.t.no_grad():
        final=rt.forward(model);total,ce,penalty=criterion(rt,model,final,roles['select_ids'],roles['select_y'])
        selected_replay=float(total.item());saved_select=selected_meta['SELECT_total']
        require(math.isclose(selected_replay,saved_select,rel_tol=1e-5,abs_tol=1e-7),'Restored SELECT total within disclosed reduction tolerance')
    counts['selected_routes']+=model.members;rt.t.cuda.synchronize(rt.device);timings['restoration_and_selected_forward_seconds']=time.monotonic()-start
    require(counts['TRAIN_routes']==counts['SELECT_routes']==model.members*update and counts['backwards']==counts['Adam_steps']==update,'Closed native work counts')
    counts['meanSAGE_layer_calls']=2*(counts['TRAIN_routes']+counts['SELECT_routes']+counts['selected_routes'])
    routes=counts['TRAIN_routes']+counts['SELECT_routes']+counts['selected_routes']
    counts['scan_direction_calls']=0 if model.scanner is None or model.arm=='R4' else 2*routes
    counts['scan_occurrence_mean_calls']=0 if model.scanner is None else routes
    counts['scan_live_backward_direction_calls']=0 if model.scanner is None or model.arm=='R4' else 2*counts['TRAIN_routes']
    record=dict(status='selected_complete',selected_step=selected,completed_updates=update,selected_restore_verified=True,checkpoint=receipt(folder/'selected.pt'),SELECT_total=selected_meta['SELECT_total'],SELECT_total_replay=selected_replay,SELECT_total_replay_delta=selected_replay-saved_select,SELECT_total_replay_tolerance=dict(rel_tol=1e-5,abs_tol=1e-7),physical_backbones=len(model.bodies),members=model.members,operation_counts=counts,costs={**timings,'fit_wall_seconds':time.monotonic()-tick},stopping='patience50' if stale>=50 else 'max100000')
    write(folder/'RESULT.json',record);return record,final.detach()

def run(config,custody):
    output=Path(config['output']);output.mkdir(parents=True,exist_ok=False);rt=None
    result=dict(complete=False,terminal=False,records=[],retry=False,banks=0,optimizer_trajectories=0,physical_backbones=0,exports=0,role_custody=config['custody'],custody=custody,CAL_DEV_truth_read=False,serving='mean_class_probabilities',expected_roster=[dict(block=b,arm=a) for b in BLOCKS for a in ARMS])
    try:
        from reader import owner_closed
        from reuse import references,operation_counts
        prior=references(config)
        result['records']=list(prior['records']);result.update(banks=12,optimizer_trajectories=21,physical_backbones=21,exports=12,prior_banks=12,new_banks=0,prior_optimizer_trajectories=21,new_optimizer_trajectories=0,prior_physical_bodies=21,new_physical_bodies=0,prior_receipts=prior['receipts'],reuse_audit=config['reuse_audit'])
        owner_closed(config['qualification_owner_end'])
        qualified=json.loads(checked(config['qualification']).read_text())
        require(qualified['complete'] and qualified['status']=='pass' and qualified['TRAIN_routes']==4 and qualified['Adam_steps']==1 and qualified['SELECT_label_free_routes']==4 and qualified['strict_model_optimizer_mode_global_and_route_rng_restore'] and qualified['role_custody']==config['custody'] and qualified['source_manifest']==config['source_manifest'] and qualified['custody']['binding']==custody['binding'],'One full-input source/resource qualification must close first')
        rt=Runtime(config);result['graph']=rt.graph
        require(all(rt.graph[k]==prior['panel']['graph'][k] for k in ('nodes','features','classes','edges','self_edges','feature_sha256','graph_sha256')),'Native original feature/support identity required for reuse')
        for b in BLOCKS:
            roles=rt.roles(b,select=True)
            for arm in NEW_ARMS:
                rt.guard();folder=output/f'block_{b}_{arm}';folder.mkdir();core,optimizers,audit=rt.construct(arm,b)
                record=dict(block=b,arm=arm,status='running',origin='new_tour_scanner_fit',construction=audit,fits=[],exports={},actual_served_members=core['members'])
                result['records'].append(record);finals=[]
                for i,(model,optimizer) in enumerate(zip(core['models'],optimizers)):
                    streams=rt.streams(b,model.members,start_index=i if core['genuine_independent'] else 0)
                    meta=dict(block=b,arm=arm,body=i,role_custody=config['custody'],source_manifest=config['source_manifest'],binding=custody['binding'],construction=audit,graph=rt.graph)
                    fitted,final=fit(rt,model,optimizer,streams,roles,folder/f'fit_{i}',meta);record['fits'].append(fitted);finals.append(final)
                    result['optimizer_trajectories']+=1;result['physical_backbones']+=fitted['physical_backbones'];result['new_optimizer_trajectories']+=1;result['new_physical_bodies']+=fitted['physical_backbones']
                    del streams
                logits=rt.t.cat(finals,dim=0);require(logits.shape==(core['members'],34493,5),'Actual complete served membership')
                probabilities=logits.softmax(-1);pool=probabilities.mean(0);lp=logits.log_softmax(-1)
                for role in ('cal','dev'):
                    ids=roles[role+'_ids_numpy'];idx=rt.tensor(ids);path=folder/('selected_'+role+'.npz')
                    rt.np.savez(path,ids=ids,member_log_probs=lp[:,idx].cpu().numpy(),native_member_argmax=logits[:,idx].argmax(-1).cpu().numpy(),native_pool_argmax=pool[idx].argmax(-1).cpu().numpy(),native_pool_probabilities=pool[idx].cpu().numpy())
                    record['exports'][role]=receipt(path)
                record.update(status='exported_complete',optimizer_trajectories=len(record['fits']),physical_backbones=sum(f['physical_backbones'] for f in record['fits']),probability_pool_authority=True)
                write(folder/'BANK_RESULT.json',record);result['banks']+=1;result['exports']+=1;result['new_banks']+=1
                write(output/'ACQUISITION_PROGRESS.json',dict(closed_banks=result['banks'],optimizer_trajectories=result['optimizer_trajectories'],physical_backbones=result['physical_backbones'],expected_banks=30))
                del core,optimizers,model,optimizer,final,finals,logits,probabilities,pool,lp;rt.t.cuda.empty_cache();rt.guard()
        require((result['banks'],result['optimizer_trajectories'],result['physical_backbones'],result['exports'],result['new_banks'],result['new_optimizer_trajectories'],result['new_physical_bodies'])==(30,48,48,30,18,27,27),'Whole frozen panel and21prior/27new accounting required')
        result.update(complete=True,terminal=True,selected_optimizer_states=48,selected_physical_bodies=48,logical_member_view_outputs=sum(r['actual_served_members'] for r in result['records']),counts=dict(prior=operation_counts(prior['records']),new=operation_counts([r for r in result['records'] if r['origin']=='new_tour_scanner_fit'])),costs=rt.costs(),prior_original_whole_peer_family_costs=prior['panel']['costs'])
    except BaseException as e:
        result.update(terminal=True,failure=dict(type=type(e).__name__,message=str(e)))
        if result['records'] and result['records'][-1]['status']=='running':result['records'][-1]['status']='failed_retained'
        raise
    finally:
        if rt is not None:result['costs']=rt.costs()
        result['unstarted_retained']=[r for r in result['expected_roster'] if (r['block'],r['arm']) not in {(x['block'],x['arm']) for x in result['records']}]
        write(output/'COMPLETE_PANEL.json',result)
    return result
