"""Exact existing individual own-selector loop; factory/counters are the only changes."""
import json
import os
import time
from source import sha, write

def fit_body(spec,body_spec,tensor,roles,output,started,timings,factory,work):
    from metrics import classification,signatures,verify_counts,verify_floats
    import torch
    output.mkdir(exist_ok=False)
    session,clock=timed(torch,lambda:factory(spec['condition'],body_spec['initialization_seed'],tensor))
    add(timings,'fit_construction_and_preprocessing',clock)
    best_correct,best_epoch,best_metrics,best_signatures,epochs=-1,None,None,None,0
    writes=0;path=output/'SELECTED_PREDICTOR.pt'
    views=4 if spec['condition']=='single_mean4_dropout' else 1
    with (output/'EPOCHS.jsonl').open('x') as trace:
        for epoch in range(1,2001):
            check_bounds(spec,output.parent,started)
            training,clock=timed(torch,lambda:session.train_step(audit=False));add(timings,'native_updates',clock)
            (probabilities,logits),clock=timed(torch,session.factual_probabilities);add(timings,'regular_VALID_forwards',clock)
            tick=time.monotonic();metrics=classification(logits,probabilities,*roles['VALID'])
            # Author-native raw-logit maximum, never a pooled selector.
            correct=int(logits[0,roles['VALID'][0]].max(1)[1].eq(roles['VALID'][1]).sum().item())
            if correct != metrics['members'][0]['correct']:raise ValueError('Native selector count differs')
            add(timings,'regular_VALID_metrics',dict(wall_seconds=time.monotonic()-tick))
            improved=correct > best_correct
            if improved:
                best_correct,best_epoch,best_metrics=correct,epoch,metrics
                best_signatures=signatures(logits,probabilities,*roles['VALID'])
                tick=time.monotonic()
                selected=dict(schema='PubMed-own-selected-native-body-v1',record_id=spec['record_id'],
                              body_id=body_spec['body_id'],initialization_seed=body_spec['initialization_seed'],
                              factual_dropout_seeds=body_spec['factual_dropout_seeds'],selected_epoch=epoch,
                              selector=spec['selector'],selected_VALID=metrics,selected_prediction_signatures=best_signatures,
                              body={k:v.detach().cpu().clone() for k,v in session.bodies[0].state_dict().items()},
                              resumable=False,optimizer_history_included=False,labels_included=False)
                temporary=output/'SELECTED_PREDICTOR.partial.pt';torch.save(selected,temporary);os.replace(temporary,path)
                writes+=1;del selected
                add(timings,'checkpoint_write',dict(wall_seconds=time.monotonic()-tick))
            epochs=epoch
            tick=time.monotonic()
            trace.write(json.dumps(dict(epoch=epoch,TRAIN=training,VALID=metrics,selected=improved,best_epoch=best_epoch),allow_nan=False)+'\n');trace.flush()
            write(output.parent/'PROGRESS.json',dict(complete=False,record_id=spec['record_id'],body_id=body_spec['body_id'],
                  private_epoch=epoch,private_best_epoch=best_epoch,private_max_epochs=2000,
                  counters=session.counters,inclusive_seconds=time.monotonic()-started))
            add(timings,'trace_and_progress_write',dict(wall_seconds=time.monotonic()-tick))
            del probabilities,logits
            if epoch-best_epoch >=250:break
    def restore():
        selected=torch.load(path,map_location='cpu')
        for k,v in dict(record_id=spec['record_id'],body_id=body_spec['body_id'],initialization_seed=body_spec['initialization_seed'],selected_epoch=best_epoch,selector=spec['selector']).items():
            if selected.get(k)!=v:raise ValueError('Own-selected body identity changed')
        if selected['selected_prediction_signatures'] != best_signatures:raise ValueError('Selected signature custody changed')
        session.bodies[0].load_state_dict(selected['body'],strict=True)
        return selected
    selected,clock=timed(torch,restore);add(timings,'private_selected_restore',clock)
    (probabilities,logits),clock=timed(torch,session.factual_probabilities);add(timings,'private_selected_forwards',clock)
    metrics=classification(logits,probabilities,*roles['VALID'])
    signature=signatures(logits,probabilities,*roles['VALID'])
    if signature != best_signatures:raise ValueError('Restored exact native predictions/correct masks changed')
    verify_counts(metrics,best_metrics);verify_floats(metrics,best_metrics)
    if session.counters != work(session,epochs,evaluations=epochs+1):
        raise ValueError('Complete private native work counters differ')
    row=dict(**body_spec,epochs_executed=epochs,selected_epoch=best_epoch,
             stopped_by='patience250' if epochs-best_epoch>=250 else 'max2000',
             selected_VALID=metrics,selected_prediction_signatures=signature,counters=dict(session.counters),
             preprocessing_seconds=session.preparation_seconds,selected_checkpoint_writes=writes,
             selected_predictor=dict(path=str(path.relative_to(output.parent)),sha256=sha(path),bytes=path.stat().st_size),
             own_selector=True,common_bank_selector=False)
    write(output/'BODY_COMPLETE.json',row)
    return session,logits,selected['body'],row
