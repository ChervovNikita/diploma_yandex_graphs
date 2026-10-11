"""Sealed selected-fit function; only device synchronization is adapted."""
import json
import math
import time
from common import receipt,require,write

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
            del out,total,ce,penalty;rt.synchronize();timings['TRAIN_seconds']+=time.monotonic()-start
            model.eval();start=time.monotonic()
            with rt.t.no_grad():
                out=rt.forward(model);total,ce,penalty=criterion(rt,model,out,roles['select_ids'],roles['select_y'])
                select=[float(v.item()) for v in (total,ce,penalty)]
                require(out.shape==(model.members,34493,5) and rt.t.isfinite(out).all().item(),'Complete native SELECT forward')
            counts['SELECT_routes']+=model.members;require(all(math.isfinite(v) for v in select),'Finite SELECT total')
            del out,total,ce,penalty;rt.synchronize();timings['SELECT_seconds']+=time.monotonic()-start
            accepted=select[0]<=best
            if accepted:
                best,selected,stale=select[0],update,0;selected_meta={**meta,'selected_step':selected,'SELECT_total':select[0],'SELECT_CE':select[1],'SELECT_L2':select[2]}
                start=time.monotonic();rt.save(folder/'selected.pt',model,optimizer,streams,selected_meta);rt.synchronize()
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
    counts['selected_routes']+=model.members;rt.synchronize();timings['restoration_and_selected_forward_seconds']=time.monotonic()-start
    require(counts['TRAIN_routes']==counts['SELECT_routes']==model.members*update and counts['backwards']==counts['Adam_steps']==update,'Closed native work counts')
    counts['meanSAGE_layer_calls']=2*(counts['TRAIN_routes']+counts['SELECT_routes']+counts['selected_routes'])
    routes=counts['TRAIN_routes']+counts['SELECT_routes']+counts['selected_routes']
    counts['scan_direction_calls']=0 if model.scanner is None or model.arm=='R4' else 2*routes
    counts['scan_occurrence_mean_calls']=0 if model.scanner is None else routes
    counts['scan_live_backward_direction_calls']=0 if model.scanner is None or model.arm=='R4' else 2*counts['TRAIN_routes']
    record=dict(status='selected_complete',selected_step=selected,completed_updates=update,selected_restore_verified=True,checkpoint=receipt(folder/'selected.pt'),SELECT_total=selected_meta['SELECT_total'],SELECT_total_replay=selected_replay,SELECT_total_replay_delta=selected_replay-saved_select,SELECT_total_replay_tolerance=dict(rel_tol=1e-5,abs_tol=1e-7),physical_backbones=len(model.bodies),members=model.members,operation_counts=counts,costs={**timings,'fit_wall_seconds':time.monotonic()-tick},stopping='patience50' if stale>=50 else 'max100000')
    write(folder/'RESULT.json',record);return record,final.detach()
